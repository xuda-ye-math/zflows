# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Verification script for zflows.utils sampling / optimization routines.

Run from the repo root:  python -m tests._verify_utils

Consolidates the earlier per-routine sanity scripts (_taming, _hmc, _lbfgs)
into one banner-separated harness. Sections:

   A. Tamed Langevin stability on the cubic-gradient quartic potential.
   B. HMC sanity: invariance, convergence, anisotropic mixing, NaN guard,
      enable_grad gating, chunk-equivalence, enable_eval fallback,
      super-linear (quartic) potential.
   C. L-BFGS sanity: well-conditioned and ill-conditioned convergence,
      chunk-determinism, enable_grad gating, Armijo line search,
      `optimization` alias.
"""

import math

import torch

from zflows.potential import Gaussian, Potential
from zflows.utils import (
    hmc, langevin, lbfgs, optimization,
    set_cache_size_limit, suppress_warnings,
)

# Silence Triton autotune / Inductor / Dynamo / Python warnings; give Dynamo
# generous cache headroom since multiple Gaussian / Quartic instances in
# §B / §C each trigger their own enable_grad / enable_eval compile.
suppress_warnings()
set_cache_size_limit(64)


def banner(s: str) -> None:
    print()
    print("═" * 60)
    print(s)
    print("═" * 60)


def section(s: str) -> None:
    print()
    print("─" * 60)
    print(s)
    print("─" * 60)


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


# ══════════════════════════════════════════════════════════════════
# A. Tamed Langevin stability
# ══════════════════════════════════════════════════════════════════
banner("A. Tamed Langevin on U(x) = ||x||^4 / 4  (cubic gradient)")
# step * ||grad U|| = step * ||x||^2 * ||x|| amplifies the norm whenever
# step * ||x||^2 > 2. With step=0.5, ~13% of N(0, I) particles start
# unstable; cubic-growth feedback then blows up plain ULA within a few
# iterations. Tamed drift G(x) = grad U / (1 + taming * ||grad U||)
# caps ||step * G|| <= step / taming.

class Quartic(Potential):
    """U(x) = (1/4) * ||x||^4 — grad U(x) = ||x||^2 * x grows super-linearly."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.25 * (x ** 2).sum(dim=-1) ** 2


N_T, D_T, STEP_T, ITERS_T, TAMING = 512, 2, 0.5, 200, 0.1

def fresh_init():
    torch.manual_seed(0)
    return torch.randn(N_T, D_T, device=device)

# A.1 — plain ULA blows up
section("A.1  plain ULA (taming=0) blows up")
u_q = Quartic().to(device).enable_grad()
x0 = fresh_init()
torch.manual_seed(1)
y_ula = langevin(x0, potential=u_q, step=STEP_T, iters=ITERS_T, taming=0.0)
n_bad = (~torch.isfinite(y_ula)).any(dim=-1).sum().item()
finite_mask = torch.isfinite(y_ula).all(dim=-1)
max_norm_ula = y_ula[finite_mask].norm(dim=-1).max().item() if finite_mask.any() else float("nan")
print(f"  NaN/Inf particles: {n_bad} / {N_T}")
print(f"  max ||y|| (finite particles only): {max_norm_ula:.3e}")
assert n_bad > 0 or max_norm_ula > 1e6, "ULA did not blow up — pick a larger step / more iters"
print("  [OK ] ULA is unstable, as expected")

# A.2 — tamed ULA stays bounded
section(f"A.2  tamed ULA (taming={TAMING}) stays bounded")
u_q = Quartic().to(device).enable_grad()
x0 = fresh_init()
torch.manual_seed(1)
y_tamed = langevin(x0, potential=u_q, step=STEP_T, iters=ITERS_T, taming=TAMING)
assert torch.isfinite(y_tamed).all(), "tamed ULA produced NaN/Inf"
max_norm_tamed = y_tamed.norm(dim=-1).max().item()
mean_norm_tamed = y_tamed.norm(dim=-1).mean().item()
print(f"  NaN/Inf particles: 0 / {N_T}")
print(f"  max  ||y||: {max_norm_tamed:.3e}")
print(f"  mean ||y||: {mean_norm_tamed:.3e}")
assert max_norm_tamed < 50, f"tamed ULA wandered too far: {max_norm_tamed}"
print("  [OK ] tamed chain stays bounded")

# A.3 — adjust=True + taming>0 forbidden
section("A.3  adjust=True together with taming>0 raises ValueError")
u_q = Quartic().to(device).enable_grad()
x0 = fresh_init()
raised = False
try:
    langevin(x0, potential=u_q, step=1e-2, iters=2, adjust=True, taming=TAMING)
except ValueError as e:
    raised = True
    print(f"  raised: {e}")
assert raised, "langevin must reject adjust=True with taming>0"
print("  [OK ] rejected as expected")


# ══════════════════════════════════════════════════════════════════
# B. HMC sanity
# ══════════════════════════════════════════════════════════════════
banner("B. HMC: invariance, convergence, NaN guard, chunk, super-linear")

u = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad().enable_eval()

# B.1 — invariance: start AT target, stay AT target
section("B.1  start at N(0, I), stay at N(0, I)")
N = 10000
torch.manual_seed(1)
x0 = u.samples(N)
mean0, std0 = x0.mean(0).tolist(), x0.std(0).tolist()
x = hmc(x0, potential=u, step=0.15, iters=10, burns=20)
mean1, std1 = x.mean(0).tolist(), x.std(0).tolist()
print(f"  before: mean = {[f'{v:+.3f}' for v in mean0]}, std = {[f'{v:.3f}' for v in std0]}")
print(f"   after: mean = {[f'{v:+.3f}' for v in mean1]}, std = {[f'{v:.3f}' for v in std1]}")
for m in mean1:
    assert abs(m) < 0.05, f"invariance broken: mean = {mean1}"
for s in std1:
    assert abs(s - 1.0) < 0.05, f"invariance broken: std = {std1}"
print("  [OK ] invariance preserved")

# B.2 — convergence: broad init → target
section("B.2  broad N(0, 25 I) init pulls onto N(0, I)")
torch.manual_seed(2)
x0 = torch.randn(5000, 2, device=device) * 5.0
mean0, std0 = x0.mean(0).tolist(), x0.std(0).tolist()
x = hmc(x0, potential=u, step=0.15, iters=15, burns=100)
mean1, std1 = x.mean(0).tolist(), x.std(0).tolist()
print(f"  before: mean = {[f'{v:+.3f}' for v in mean0]}, std = {[f'{v:.3f}' for v in std0]}")
print(f"   after: mean = {[f'{v:+.3f}' for v in mean1]}, std = {[f'{v:.3f}' for v in std1]}")
for m in mean1:
    assert abs(m) < 0.1, f"mean did not converge to 0: {mean1}"
for s in std1:
    assert abs(s - 1.0) < 0.1, f"std did not converge to 1: {std1}"
print("  [OK ] broad cloud contracts onto target")

# B.3 — anisotropic target
section("B.3  anisotropic target (variance ratio 4:1) — per-axis std")
u_aniso = Gaussian(mean=[0.0, 0.0], variance=[4.0, 1.0]).to(device).enable_grad().enable_eval()
torch.manual_seed(3)
x0 = u_aniso.samples(10000)
x = hmc(x0, potential=u_aniso, step=0.1, iters=15, burns=30)
std1 = x.std(0).tolist()
print(f"  per-axis std = {[f'{v:.3f}' for v in std1]}   (target [2.000, 1.000])")
assert abs(std1[0] - 2.0) < 0.1, f"wide axis std off: {std1[0]}"
assert abs(std1[1] - 1.0) < 0.05, f"narrow axis std off: {std1[1]}"
print("  [OK ] both axes mix correctly")

# B.4 — NaN guard
section("B.4  huge step rejects every trajectory, no NaN leaks")
x0 = torch.zeros(2000, 2, device=device)  # all at the mode of N(0, I)
x = hmc(x0, potential=u, step=10.0, iters=20, burns=5)
n_finite = torch.isfinite(x).all(-1).float().mean().item()
max_abs = x.abs().max().item()
print(f"  fraction finite = {n_finite:.4f}")
print(f"  max |x|         = {max_abs:.4f}  (0.0 = every trajectory rejected)")
assert n_finite == 1.0, "NaN leaked out of HMC"
assert max_abs == 0.0, "some divergent trajectories were accepted"
print("  [OK ] divergent leapfrog rejected, chain finite")

# B.5 — enable_grad gate
section("B.5  hmc() raises RuntimeError without enable_grad()")
u_noenable = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)
x0 = torch.randn(8, 2, device=device)
raised = False
try:
    hmc(x0, potential=u_noenable, step=0.1, iters=5, burns=3)
except RuntimeError as e:
    raised = True
    print(f"  raised: {e}")
assert raised, "hmc() must raise without enable_grad()"
print("  [OK ] rejected as expected")

# B.6 — chunk statistically matches chunk=1
section("B.6  chunk statistically matches chunk=1 (aggregate moments)")
torch.manual_seed(6)
x0 = u.samples(4000)
torch.manual_seed(60)
y1 = hmc(x0, potential=u, step=0.15, iters=10, burns=10, chunk=1)
torch.manual_seed(60)
y4 = hmc(x0, potential=u, step=0.15, iters=10, burns=10, chunk=4)
mean_err = (y1.mean(0) - y4.mean(0)).abs().max().item()
std_err  = (y1.std(0)  - y4.std(0) ).abs().max().item()
print(f"  max |mean(chunk=1) - mean(chunk=4)| = {mean_err:.4f}")
print(f"  max |std (chunk=1) - std (chunk=4)| = {std_err:.4f}")
# Different RNG consumption order -> different per-particle outputs but the
# same distribution. Aggregate moments agree within MC error ~ 0.02 for N=4000.
assert mean_err < 0.05, f"chunk moments diverged: mean err {mean_err}"
assert std_err  < 0.05, f"chunk moments diverged: std err {std_err}"
print("  [OK ] chunk produces statistically equivalent samples")

# B.7 — enable_eval optional
section("B.7  hmc() falls back to potential(x) without enable_eval()")
u_grad_only = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad()
# no .enable_eval() — hmc must route MH energy evals through __call__
torch.manual_seed(7)
x0 = u_grad_only.samples(5000)
x = hmc(x0, potential=u_grad_only, step=0.15, iters=10, burns=20)
mean1, std1 = x.mean(0).tolist(), x.std(0).tolist()
print(f"  fallback path: mean = {[f'{v:+.3f}' for v in mean1]}, std = {[f'{v:.3f}' for v in std1]}")
for m in mean1:
    assert abs(m) < 0.08, f"fallback path: mean off: {mean1}"
for s in std1:
    assert abs(s - 1.0) < 0.08, f"fallback path: std off: {std1}"
print("  [OK ] fallback produces correct samples")

# B.8 — super-linear (quartic) potential
section("B.8  U(x) = x^4 / 4 (cubic gradient): sampling + tail rejection")
# (8a) sampling correctness against the analytic moment.
# For 1D p(x) = exp(-x^4/4) / Z (Z = 2 * 4^(-3/4) * Gamma(1/4)), the
# substitution u = x^4 / 4 gives E[x^2] = 2*Gamma(3/4)/Gamma(1/4) ≈ 0.6760.
class Quartic1D(Potential):
    """U(x) = sum_i x_i^4 / 4 — gradient grows as x^3."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.25 * (x ** 4).sum(dim=-1)

u_q = Quartic1D().to(device).enable_grad().enable_eval()
target_std = math.sqrt(2 * math.gamma(0.75) / math.gamma(0.25))

torch.manual_seed(8)
x0 = torch.randn(10000, 1, device=device) * 0.5
x = hmc(x0, potential=u_q, step=0.1, iters=10, burns=100)
final_std = x.std(0).item()
final_mean = x.mean(0).item()
n_finite = torch.isfinite(x).all(-1).float().mean().item()
print(f"  target std (analytic)       = {target_std:.4f}")
print(f"  sampled std after HMC       = {final_std:.4f}")
print(f"  sampled mean after HMC      = {final_mean:+.4f}")
print(f"  fraction finite             = {n_finite:.4f}")
assert n_finite == 1.0, "NaN leaked from super-linear potential"
assert abs(final_std - target_std) < 0.03, \
    f"sampled std {final_std} far from analytic {target_std}"
assert abs(final_mean) < 0.03, f"sampled mean off: {final_mean}"

# (8b) aggressive step from the extreme tail: every trajectory diverges,
# MH rejects them all via isfinite guard, particles stay locked at x_start.
torch.manual_seed(80)
x_tail = torch.full((1000, 1), 10.0, device=device)
x_after = hmc(x_tail, potential=u_q, step=0.5, iters=10, burns=5)
tail_finite = torch.isfinite(x_after).all(-1).float().mean().item()
n_at_start = (x_after == 10.0).all(-1).float().mean().item()
print(f"  aggressive-step tail: finite = {tail_finite:.4f}, "
      f"stuck at x_start = {n_at_start:.4f}")
assert tail_finite == 1.0, "NaN leaked from divergent tail trajectories"
assert n_at_start > 0.99, f"too many divergent trajectories accepted: {n_at_start}"
print("  [OK ] super-linear chain sampled + tail rejected")


# ══════════════════════════════════════════════════════════════════
# C. L-BFGS sanity
# ══════════════════════════════════════════════════════════════════
banner("C. L-BFGS: well/ill-conditioned, chunk-determinism, Armijo, alias")

# C.1 — well-conditioned Gaussian
section("C.1  lbfgs reaches ~fp32 precision in ~30 iters")
x_star = torch.tensor([2.0, -1.5, 0.7], device=device)
u = Gaussian(mean=x_star.tolist(), variance=[1.0, 1.0, 1.0]).to(device).enable_grad()
torch.manual_seed(1)
x0 = torch.randn(2000, 3, device=device) * 5.0
init_err = (x0 - x_star).norm(dim=-1).max().item()
ITERS = 30
y = lbfgs(x0, potential=u, step=1.0, iters=ITERS, memory=6)
err = (y - x_star).norm(dim=-1).max().item()
print(f"  init max err: {init_err:.3e}")
print(f"  lbfgs ({ITERS:3d} iters) max err: {err:.3e}")
assert err < 1e-5, f"L-BFGS failed to converge: {err}"
print("  [OK ] superlinear convergence on isotropic Gaussian")

# C.2 — ill-conditioned diagonal Gaussian
section("C.2  ill-conditioned diag Gaussian (kappa=100)")
x_star2 = torch.zeros(4, device=device)
u2 = Gaussian(mean=x_star2.tolist(), variance=[100.0, 1.0, 100.0, 1.0]).to(device).enable_grad()
torch.manual_seed(2)
x0 = torch.randn(2000, 4, device=device) * 5.0
init_err = (x0 - x_star2).norm(dim=-1).max().item()
ITERS = 50
y = lbfgs(x0, potential=u2, step=1.0, iters=ITERS, memory=6)
err = (y - x_star2).norm(dim=-1).max().item()
print(f"  init max err: {init_err:.3e}")
print(f"  lbfgs ({ITERS:3d} iters) max err: {err:.3e}")
assert err < 1e-4, f"L-BFGS failed on ill-conditioned problem: {err}"
print("  [OK ] curvature pairs handle kappa=100")

# C.3 — chunk-determinism
section("C.3  chunk > 1 gives identical output to chunk == 1")
u3 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad()
torch.manual_seed(3)
x0 = torch.randn(128, 2, device=device) * 4.0
y1 = lbfgs(x0, potential=u3, step=1.0, iters=20, memory=6, chunk=1)
y4 = lbfgs(x0, potential=u3, step=1.0, iters=20, memory=6, chunk=4)
err = (y1 - y4).abs().max().item()
print(f"  max |y_chunk1 - y_chunk4| = {err:.3e}")
assert err < 1e-5, f"chunking changed the output: {err}"
print("  [OK ] bit-identical across chunkings")

# C.4 — enable_grad gate
section("C.4  lbfgs() raises RuntimeError without enable_grad()")
u4 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)
x0 = torch.randn(8, 2, device=device)
raised = False
try:
    lbfgs(x0, potential=u4, step=1.0, iters=10, memory=6)
except RuntimeError as e:
    raised = True
    print(f"  raised: {e}")
assert raised, "lbfgs() must raise without enable_grad()"
print("  [OK ] rejected as expected")

# C.5 — Armijo line search: gating + convergence + monotonicity
section("C.5  armijo line search — gating, convergence, monotonicity")
# (5a) armijo=True without enable_eval() must raise
u5_noeval = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad()
x0 = torch.randn(8, 2, device=device)
raised = False
try:
    lbfgs(x0, potential=u5_noeval, step=1.0, iters=5, memory=6, armijo=True)
except RuntimeError as e:
    raised = True
    print(f"  no-enable_eval raised: {e}")
assert raised, "lbfgs(armijo=True) must raise without enable_eval()"

# (5b) with enable_eval, armijo converges on the well-conditioned Gaussian
x_star5 = torch.tensor([2.0, -1.5, 0.7], device=device)
u5 = Gaussian(mean=x_star5.tolist(), variance=[1.0, 1.0, 1.0]).to(device).enable_grad().enable_eval()
torch.manual_seed(15)
x0 = torch.randn(2000, 3, device=device) * 5.0
y_arm = lbfgs(x0, potential=u5, step=1.0, iters=30, memory=6, armijo=True)
err_arm = (y_arm - x_star5).norm(dim=-1).max().item()
print(f"  armijo=True ({30:3d} iters) max ||y - x*||: {err_arm:.3e}")
assert err_arm < 1e-4, f"armijo L-BFGS failed to converge: {err_arm}"

# (5c) Armijo guarantees U does not increase across iterations.
torch.manual_seed(15)
x = torch.randn(256, 3, device=device) * 5.0
U_prev = u5.eval(x).mean().item()
worst_increase = 0.0
for k in range(20):
    x = lbfgs(x, potential=u5, step=1.0, iters=1, memory=6, armijo=True)
    U_now = u5.eval(x).mean().item()
    worst_increase = max(worst_increase, U_now - U_prev)
    U_prev = U_now
print(f"  worst U_after - U_before across 20 single-iter steps: {worst_increase:.3e}")
assert worst_increase <= 1e-4, f"armijo allowed U to grow by {worst_increase}"
print("  [OK ] gating + convergence + monotonicity all hold")

# C.6 — `optimization` alias
section("C.6  optimization is an alias of lbfgs")
print(f"  optimization is lbfgs: {optimization is lbfgs}")
assert optimization is lbfgs, "optimization should be the lbfgs symbol itself, not a wrapper"
print("  [OK ] alias confirmed")

# ─────────────────────────────────────────────────────────────────
print()
print("All utils verification checks passed.")
