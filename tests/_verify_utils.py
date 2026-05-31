# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Verification script for zflows.utils sampling / optimization routines.

Run from the repo root:  python -m tests._verify_utils

Consolidates the earlier per-routine sanity scripts (_taming, _hmc, _lbfgs)
into one banner-separated harness. Sections:

   A. Tamed Langevin stability on the cubic-gradient quartic potential
      (+ A.4 inverse-temperature beta tests).
   B. HMC sanity: invariance, convergence, anisotropic mixing, NaN guard,
      enable_grad gating, chunk-equivalence, enable_eval fallback,
      super-linear (quartic) potential (+ B.9 inverse-temperature beta).
   C. L-BFGS sanity: well-conditioned and ill-conditioned convergence,
      chunk-determinism, enable_grad gating, Armijo line search,
      `optimization` alias.
   D. Importance-weight reweighting: default vs (beta_source, beta_target)
      = (1, 1) byte-equality, linearity in each beta.
   E. Stochastic Heun integrator: convergence + anisotropic mixing,
      inverse-temperature beta scaling, reduced discretization bias vs
      Euler-Maruyama (ULA) at matched step, enable_grad gating,
      chunk-equivalence.
   F. Annealed importance sampling (flow-proposal SMC), _F and _G variants:
      a trained NSF maps source -> target, AIS lands the pushforward cloud
      tightly on the target (mean/std), the ladder beats a single hop on a poor
      proposal, target enable_grad gating (source needs none), chunk-equivalence,
      and the inverse-map _G twin (G = F.inv) reproduces the _F weight rule.
"""

import math

import torch

from zflows.potential import Gaussian, Potential, potential_from
from zflows.flow import NSF
from zflows.loss import reverse_KL
from zflows.utils import (
    hmc, langevin, stochastic_heun, lbfgs, optimization,
    importance_weights, importance_weights_F, importance_weights_G,
    importance_weights_log, importance_weights_log_F, importance_weights_log_G,
    annealed_importance_sampling_F, annealed_importance_sampling_G,
    compute_ESS_log, set_cache_size_limit, suppress_warnings,
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

# U(x) = (1/4) * ||x||^4 — grad U(x) = ||x||^2 * x grows super-linearly.
def Quartic_forward(x: torch.Tensor) -> torch.Tensor: # Tensor [N, d] -> Tensor [N]
    return 0.25 * (x ** 2).sum(dim=-1) ** 2


N_T, D_T, STEP_T, ITERS_T, TAMING = 512, 2, 0.5, 200, 0.1

def fresh_init():
    torch.manual_seed(0)
    return torch.randn(N_T, D_T, device=device)

# A.1 — plain ULA blows up
section("A.1  plain ULA (taming=0) blows up")
u_q = potential_from(Quartic_forward).to(device).enable_grad()
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
u_q = potential_from(Quartic_forward).to(device).enable_grad()
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
u_q = potential_from(Quartic_forward).to(device).enable_grad()
x0 = fresh_init()
raised = False
try:
    langevin(x0, potential=u_q, step=1e-2, iters=2, adjust=True, taming=TAMING)
except ValueError as e:
    raised = True
    print(f"  raised: {e}")
assert raised, "langevin must reject adjust=True with taming>0"
print("  [OK ] rejected as expected")

# A.4 — inverse-temperature beta: defaults are bit-compatible, and beta
# scales the stationary distribution correctly (Gaussian: var -> var/beta).
section("A.4  beta=1.0 reproduces default; beta>1 tightens the cloud")
u_beta = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad().enable_eval()

# (4a) beta defaulted vs beta=1.0 explicit: byte-identical for both ULA and MALA
torch.manual_seed(0)
x0 = torch.randn(256, 2, device=device)
torch.manual_seed(123)
y_def_ula = langevin(x0.clone(), potential=u_beta, step=1e-3, iters=50)
torch.manual_seed(123)
y_b1_ula  = langevin(x0.clone(), potential=u_beta, beta=1.0, step=1e-3, iters=50)
print(f"  ULA  default vs beta=1.0: equal={torch.equal(y_def_ula, y_b1_ula)}")
assert torch.equal(y_def_ula, y_b1_ula)

torch.manual_seed(123)
y_def_mala = langevin(x0.clone(), potential=u_beta, step=1e-3, iters=50, adjust=True)
torch.manual_seed(123)
y_b1_mala  = langevin(x0.clone(), potential=u_beta, beta=1.0, step=1e-3, iters=50, adjust=True)
print(f"  MALA default vs beta=1.0: equal={torch.equal(y_def_mala, y_b1_mala)}")
assert torch.equal(y_def_mala, y_b1_mala)

# (4b) MALA at beta=1 and beta=4 against N(0,I): variance should scale as 1/beta.
torch.manual_seed(4)
x0 = u_beta.samples(8000)
y_b1 = langevin(x0.clone(), potential=u_beta, beta=1.0, step=0.05, iters=300, adjust=True)
y_b4 = langevin(x0.clone(), potential=u_beta, beta=4.0, step=0.05, iters=300, adjust=True)
var1 = y_b1.var(dim=0).mean().item() # target ~1.0
var4 = y_b4.var(dim=0).mean().item() # target ~0.25
print(f"  MALA stationary var:  beta=1.0 -> {var1:.4f} (expect ~1.00)")
print(f"                        beta=4.0 -> {var4:.4f} (expect ~0.25)")
assert abs(var1 - 1.0)  < 0.06, f"beta=1 cloud variance off: {var1}"
assert abs(var4 - 0.25) < 0.04, f"beta=4 cloud variance off: {var4}"
print("  [OK ] beta=1.0 bit-compatible; beta=4.0 contracts variance 4x")


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
# U(x) = sum_i x_i^4 / 4 — gradient grows as x^3.
def Quartic1D_forward(x: torch.Tensor) -> torch.Tensor: # Tensor [N, d] -> Tensor [N]
    return 0.25 * (x ** 4).sum(dim=-1)

u_q = potential_from(Quartic1D_forward).to(device).enable_grad().enable_eval()
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


# B.9 — inverse-temperature beta: bit-compatible default + tempered variance
section("B.9  beta=1.0 reproduces default; beta>1 tightens the cloud")
u_b = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad().enable_eval()

# (9a) defaulted vs beta=1.0 explicit: byte-identical for the same seed.
torch.manual_seed(9)
x0 = u_b.samples(2000)
torch.manual_seed(90)
y_def = hmc(x0.clone(), potential=u_b, step=0.15, iters=10, burns=10)
torch.manual_seed(90)
y_b1  = hmc(x0.clone(), potential=u_b, beta=1.0, step=0.15, iters=10, burns=10)
print(f"  HMC default vs beta=1.0: equal={torch.equal(y_def, y_b1)}")
assert torch.equal(y_def, y_b1)

# (9b) on N(0, I), the stationary at inverse temperature beta is N(0, I/beta):
# per-axis variance should be 1/beta.
torch.manual_seed(91)
x0 = u_b.samples(8000)
y_b1 = hmc(x0.clone(), potential=u_b, beta=1.0, step=0.15, iters=10, burns=30)
y_b4 = hmc(x0.clone(), potential=u_b, beta=4.0, step=0.08, iters=10, burns=30)
var1 = y_b1.var(dim=0).mean().item() # target ~1.0
var4 = y_b4.var(dim=0).mean().item() # target ~0.25
print(f"  HMC stationary var:  beta=1.0 -> {var1:.4f} (expect ~1.00)")
print(f"                       beta=4.0 -> {var4:.4f} (expect ~0.25)")
assert abs(var1 - 1.0)  < 0.05, f"beta=1 cloud variance off: {var1}"
assert abs(var4 - 0.25) < 0.03, f"beta=4 cloud variance off: {var4}"
print("  [OK ] beta=1.0 bit-compatible; beta=4.0 contracts variance 4x")


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


# ══════════════════════════════════════════════════════════════════
# D. Importance-weight reweighting
# ══════════════════════════════════════════════════════════════════
banner("D. importance_weights / _log: defaults + beta linearity + F/G variants")

torch.manual_seed(0)
d = 3
flow_iw = NSF(a=[-5.0]*d, b=[5.0]*d).to(device)
F = flow_iw.t()
src = Gaussian(mean=[0.0]*d, variance=[1.0]*d, device=device)
tgt = Gaussian(mean=[0.5]*d, variance=[0.7]*d, device=device)
x_iw = torch.randn(2000, d, device=device)

# D.1 — defaults == (1.0, 1.0) byte-for-byte
section("D.1  default vs beta_source=beta_target=1.0 is byte-identical")
lw_def = importance_weights_log(x_iw, src, tgt, F)
lw_b11 = importance_weights_log(x_iw, src, tgt, F, beta_source=1.0, beta_target=1.0)
print(f"  log:     equal = {torch.equal(lw_def, lw_b11)}")
assert torch.equal(lw_def, lw_b11)

w_def = importance_weights(x_iw, src, tgt, F)
w_b11 = importance_weights(x_iw, src, tgt, F, beta_source=1.0, beta_target=1.0)
print(f"  linear:  equal = {torch.equal(w_def, w_b11)}")
assert torch.equal(w_def, w_b11)
print("  [OK ] defaults match explicit (1.0, 1.0)")

# D.2 — log-weight is linear in each beta, with the right coefficient
section("D.2  linearity:  d(log w)/d(beta_t) = -target(y),  d(log w)/d(beta_s) = +source(x)")
y_iw, _ = F.call_and_ladj(x_iw)
diff_bt = (importance_weights_log(x_iw, src, tgt, F, beta_target=2.0)
           - importance_weights_log(x_iw, src, tgt, F, beta_target=1.0))
diff_bs = (importance_weights_log(x_iw, src, tgt, F, beta_source=2.0)
           - importance_weights_log(x_iw, src, tgt, F, beta_source=1.0))
err_bt = (diff_bt + tgt(y_iw)).abs().max().item()
err_bs = (diff_bs - src(x_iw)).abs().max().item()
print(f"  beta_target 1->2:  max |diff + target(y)| = {err_bt:.3e}")
print(f"  beta_source 1->2:  max |diff - source(x)| = {err_bs:.3e}")
assert err_bt < 1e-4, f"beta_target slope wrong: {err_bt}"
assert err_bs < 1e-4, f"beta_source slope wrong: {err_bs}"
print("  [OK ] log-weights linear in each beta with correct sign")

# D.3 — linear wrapper forwards betas to the log-version
section("D.3  importance_weights forwards betas to importance_weights_log")
log_w = importance_weights_log(x_iw, src, tgt, F, beta_source=1.5, beta_target=0.4)
w_expected = (log_w - log_w.max()).exp()
w_actual   = importance_weights(x_iw, src, tgt, F, beta_source=1.5, beta_target=0.4)
err_fwd = (w_actual - w_expected).abs().max().item()
print(f"  max |w_actual - w_expected| = {err_fwd:.3e}")
assert err_fwd < 1e-6, f"linear wrapper drops the betas: {err_fwd}"
print("  [OK ] kwargs reach the log-space implementation")

# D.4 — F/G variants + aliases: importance_weights_log{,_F} and importance_weights{,_F}
#        are the same symbol; the _G inverse-map twin (G = F.inv) agrees with _F.
section("D.4  _F / _G variants and aliases")
assert importance_weights_log is importance_weights_log_F, "importance_weights_log must alias _F"
assert importance_weights is importance_weights_F, "importance_weights must alias _F"
G_iw = F.inv  # inverse map target -> source
lwF = importance_weights_log_F(x_iw, src, tgt, F,    beta_source=1.3, beta_target=0.6)
lwG = importance_weights_log_G(x_iw, src, tgt, G_iw, beta_source=1.3, beta_target=0.6)
err_fg = (lwF - lwG).abs().max().item()
print(f"  aliases: log->_F={importance_weights_log is importance_weights_log_F}, "
      f"lin->_F={importance_weights is importance_weights_F}")
print(f"  max |log w_F - log w_G|  (G = F.inv) = {err_fg:.3e}")
assert err_fg < 1e-3, f"_F / _G log-weights diverged: {err_fg}"
# linear _G twin agrees with _F too
wF = importance_weights_F(x_iw, src, tgt, F,    beta_source=1.3, beta_target=0.6)
wG = importance_weights_G(x_iw, src, tgt, G_iw, beta_source=1.3, beta_target=0.6)
assert (wG.max() <= 1.0 + 1e-6) and ((wF - wG).abs().max().item() < 1e-3), "linear _G twin disagrees"
print("  [OK ] _G twin reproduces _F; aliases point to the forward variant")

# D.5 — compiled fused maps (for_ladj / inv_ladj) match the raw transform path
section("D.5  compiled for_ladj / inv_ladj match the raw path")
F_c = flow_iw.t().enable_for_ladj(mode="default")        # forward map, compiled call_and_ladj
G_c = flow_iw.t().inv.enable_inv_ladj(mode="default")    # inverse map, compiled inv.call_and_ladj
lwF_c = importance_weights_log_F(x_iw, src, tgt, F_c, beta_source=1.3, beta_target=0.6)
lwG_c = importance_weights_log_G(x_iw, src, tgt, G_c, beta_source=1.3, beta_target=0.6)
err_Fc = (lwF_c - lwF).abs().max().item()
err_Gc = (lwG_c - lwG).abs().max().item()
print(f"  max |_F compiled - raw| = {err_Fc:.3e}")
print(f"  max |_G compiled - raw| = {err_Gc:.3e}")
assert err_Fc < 1e-3, f"_F compiled path drifted: {err_Fc}"
assert err_Gc < 1e-3, f"_G compiled path drifted: {err_Gc}"
print("  [OK ] compiled fused maps reproduce the raw importance weights")


# ══════════════════════════════════════════════════════════════════
# E. Stochastic Heun integrator
# ══════════════════════════════════════════════════════════════════
banner("E. stochastic_heun: convergence, beta, lower bias than ULA, gating")

u = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad()

# E.1 — convergence + anisotropic mixing: broad init contracts onto target
section("E.1  broad init pulls onto target; anisotropic axes mix")
u_aniso = Gaussian(mean=[1.0, -2.0], variance=[1.0, 4.0]).to(device).enable_grad()
torch.manual_seed(1)
x0 = torch.randn(20000, 2, device=device) * 5.0
x = stochastic_heun(x0, potential=u_aniso, step=5e-3, iters=3000, chunk=2)
mean1, std1 = x.mean(0).tolist(), x.std(0).tolist()
print(f"   mean = {[f'{v:+.3f}' for v in mean1]}   (target [+1.000, -2.000])")
print(f"   std  = {[f'{v:.3f}' for v in std1]}   (target [1.000, 2.000])")
assert abs(mean1[0] - 1.0) < 0.06 and abs(mean1[1] + 2.0) < 0.08, f"mean off: {mean1}"
assert abs(std1[0] - 1.0) < 0.06 and abs(std1[1] - 2.0) < 0.08, f"std off: {std1}"
assert torch.isfinite(x).all(), "stochastic_heun produced NaN/Inf"
print("  [OK ] both axes converge to the correct moments")

# E.2 — inverse-temperature beta: stationary of N(0, I) at beta is N(0, I/beta)
section("E.2  beta scales stationary variance as 1/beta")
torch.manual_seed(2)
x0 = u.samples(20000)
y_b1 = stochastic_heun(x0.clone(), potential=u, beta=1.0, step=5e-3, iters=2000)
y_b4 = stochastic_heun(x0.clone(), potential=u, beta=4.0, step=5e-3, iters=2000)
var1 = y_b1.var(dim=0).mean().item() # target ~1.0
var4 = y_b4.var(dim=0).mean().item() # target ~0.25
print(f"  stationary var:  beta=1.0 -> {var1:.4f} (expect ~1.00)")
print(f"                   beta=4.0 -> {var4:.4f} (expect ~0.25)")
assert abs(var1 - 1.0)  < 0.04, f"beta=1 cloud variance off: {var1}"
assert abs(var4 - 0.25) < 0.02, f"beta=4 cloud variance off: {var4}"
print("  [OK ] beta=4.0 contracts variance 4x")

# E.3 — headline property: the trapezoidal drift cancels the leading O(step)
# bias of Euler-Maruyama. On N(0, I) at step=0.2 the analytic ULA stationary
# variance is 1/(1 - step/2) = 1.1111 (an +0.11 bias); Heun's is ~0.989. So at
# a *matched* step, Heun's stationary variance must sit much closer to 1.0.
section("E.3  lower discretization bias than Euler-Maruyama at matched step")
BIG_STEP, ITERS_E, N_E = 0.2, 1500, 40000
torch.manual_seed(3)
x0 = u.samples(N_E)
torch.manual_seed(30)
y_ula  = langevin(x0.clone(), potential=u, step=BIG_STEP, iters=ITERS_E) # ULA
torch.manual_seed(30)
y_heun = stochastic_heun(x0.clone(), potential=u, step=BIG_STEP, iters=ITERS_E)
var_ula  = y_ula.var(dim=0).mean().item()
var_heun = y_heun.var(dim=0).mean().item()
bias_ula, bias_heun = abs(var_ula - 1.0), abs(var_heun - 1.0)
print(f"  Euler-Maruyama (ULA) stationary var = {var_ula:.4f}  (|bias| = {bias_ula:.4f}, analytic 1.1111)")
print(f"  stochastic Heun      stationary var = {var_heun:.4f}  (|bias| = {bias_heun:.4f})")
assert bias_ula > 0.08, f"ULA bias unexpectedly small ({bias_ula}) — test can't discriminate"
assert bias_heun < 0.5 * bias_ula, f"Heun did not reduce the ULA bias: {bias_heun} vs {bias_ula}"
print("  [OK ] Heun more than halves the Euler-Maruyama variance bias")

# E.4 — enable_grad gate
section("E.4  stochastic_heun() raises RuntimeError without enable_grad()")
u_noenable = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)
x0 = torch.randn(8, 2, device=device)
raised = False
try:
    stochastic_heun(x0, potential=u_noenable, step=1e-3, iters=5)
except RuntimeError as e:
    raised = True
    print(f"  raised: {e}")
assert raised, "stochastic_heun() must raise without enable_grad()"
print("  [OK ] rejected as expected")

# E.5 — chunk statistically matches chunk=1
section("E.5  chunk statistically matches chunk=1 (aggregate moments)")
torch.manual_seed(5)
x0 = u.samples(8000)
torch.manual_seed(50)
y1 = stochastic_heun(x0, potential=u, step=5e-3, iters=500, chunk=1)
torch.manual_seed(50)
y4 = stochastic_heun(x0, potential=u, step=5e-3, iters=500, chunk=4)
mean_err = (y1.mean(0) - y4.mean(0)).abs().max().item()
std_err  = (y1.std(0)  - y4.std(0) ).abs().max().item()
print(f"  max |mean(chunk=1) - mean(chunk=4)| = {mean_err:.4f}")
print(f"  max |std (chunk=1) - std (chunk=4)| = {std_err:.4f}")
# Different RNG consumption order -> different per-particle noise but the same
# distribution; aggregate moments agree within MC error for N=8000.
assert mean_err < 0.05, f"chunk moments diverged: mean err {mean_err}"
assert std_err  < 0.05, f"chunk moments diverged: std err {std_err}"
print("  [OK ] chunk produces statistically equivalent samples")


# ══════════════════════════════════════════════════════════════════
# F. Annealed importance sampling (flow-proposal SMC)
# ══════════════════════════════════════════════════════════════════
banner("F. annealed_importance_sampling_F / _G: flow proposal F_# source -> target")
# AIS takes a trained flow as the proposal and anneals along the geometric
# path between F_# source and the target, refreshing the latent pre-image for
# the importance weights and rejuvenating with Langevin in the target. The _F
# variant takes the forward map F (source -> target); the _G variant takes the
# inverse map G = F^{-1} (target -> source) and is otherwise identical.
# source = N(0, I), target = N([3, 3], 0.5 I); an NSF is reverse-KL trained so
# that F_# source ~~ target, then AIS should land the cloud on the target.
ais_source = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)        # forward-only, no enable_grad
ais_target = Gaussian(mean=[3.0, 3.0], variance=[0.5, 0.5]).to(device).enable_grad()
target_std = math.sqrt(0.5) # ~0.7071 per axis

torch.manual_seed(6)
ais_flow = NSF(a=[-6.0, -6.0], b=[6.0, 6.0], bins=8, transforms=4,
               hidden_features=(64, 64)).to(device)
opt = torch.optim.Adam(ais_flow.parameters(), lr=2e-3)
x_train = ais_source.samples(4000)
for _ep in range(60):
    for _s in range(0, 4000, 1000):
        loss = reverse_KL(x_train[_s:_s + 1000], target=ais_target, F=ais_flow.t())
        opt.zero_grad(); loss.backward(); opt.step()
ais_F = ais_flow.t()
with torch.no_grad():
    ess_prop = compute_ESS_log(
        importance_weights_log(ais_source.samples(8000), ais_source, ais_target, ais_F)
    ).item()
print(f"  trained proposal ESS (F_# source vs target) = {ess_prop:.3f}")

# F.1 — the ladder lands the flow-pushforward cloud on the target
section("F.1  AIS lands the pushforward cloud on the target N([3, 3], 0.5 I)")
torch.manual_seed(61)
x_src = ais_source.samples(8000)
y = annealed_importance_sampling_F(
    x_src, source=ais_source, target=ais_target, F=ais_F,
    ladder=12, step=3e-3, iters=30, chunk=2,
)
mean1 = y.mean(0).tolist()
std1 = y.std(0).tolist()
n_finite = torch.isfinite(y).all(dim=-1).float().mean().item()
print(f"   after mean = {[f'{v:+.3f}' for v in mean1]}  (target [3.000, 3.000])")
print(f"   after std  = {[f'{v:.3f}' for v in std1]}  (target [{target_std:.3f}, {target_std:.3f}])")
print(f"  fraction finite = {n_finite:.4f}")
assert n_finite == 1.0, "AIS leaked non-finite particles"
for v in mean1:
    assert abs(v - 3.0) < 0.1, f"AIS mean off target: {mean1}"
for v in std1:
    assert abs(v - target_std) < 0.1, f"AIS std off target: {std1}"
print("  [OK ] particles transported onto the target")

# F.2 — a poor proposal: the ladder beats a single hop (M>1 helps)
section("F.2  ladder beats a single hop on a poor (identity) proposal")
torch.manual_seed(62)
bad_flow = NSF(a=[-6.0, -6.0], b=[6.0, 6.0], bins=8, transforms=4,
               hidden_features=(32, 32)).to(device)
bad_flow.zeros()                 # identity init: F_# source = source, far from target
bad_F = bad_flow.t()
x_src = ais_source.samples(6000)
y_one = annealed_importance_sampling_F(x_src, ais_source, ais_target, bad_F,
                                     ladder=1,  step=2e-3, iters=60, chunk=2)
y_lad = annealed_importance_sampling_F(x_src, ais_source, ais_target, bad_F,
                                     ladder=20, step=2e-3, iters=60, chunk=2)
err_one = max(abs(v - 3.0) for v in y_one.mean(0).tolist())
err_lad = max(abs(v - 3.0) for v in y_lad.mean(0).tolist())
print(f"  M=1  mean = {[f'{v:.3f}' for v in y_one.mean(0).tolist()]}  |err| = {err_one:.3f}")
print(f"  M=20 mean = {[f'{v:.3f}' for v in y_lad.mean(0).tolist()]}  |err| = {err_lad:.3f}")
assert err_lad < err_one, "annealing ladder should beat a single hop on a poor proposal"
print("  [OK ] the ladder closes the gap a single hop cannot")

# F.3 — only the target needs enable_grad (Langevin runs in the target);
#        the source is forward-only and needs none.
section("F.3  target enable_grad gating; source needs none")
tgt_raw = Gaussian(mean=[3.0, 3.0], variance=[0.5, 0.5]).to(device)
x0 = ais_source.samples(64)
raised_tgt = False
try:
    annealed_importance_sampling_F(x0, ais_source, tgt_raw, ais_F, ladder=4)
except RuntimeError as e:
    raised_tgt = True
    print(f"  target not enabled -> raised: {e}")
assert raised_tgt, "must raise when target lacks enable_grad()"
# source without enable_grad must be fine (only ever called forward)
y_ok = annealed_importance_sampling_F(x0, ais_source, ais_target, ais_F, ladder=2, iters=10)
assert torch.isfinite(y_ok).all(), "source needs no enable_grad; run must succeed"
print("  [OK ] target gated on enable_grad(); raw source accepted")

# F.4 — chunk statistically matches chunk=1 (aggregate moments)
section("F.4  chunk statistically matches chunk=1 (aggregate moments)")
torch.manual_seed(7)
x_src = ais_source.samples(6000)
torch.manual_seed(70)
y1 = annealed_importance_sampling_F(x_src, ais_source, ais_target, ais_F,
                                  ladder=10, step=3e-3, iters=40, chunk=1)
torch.manual_seed(70)
y4 = annealed_importance_sampling_F(x_src, ais_source, ais_target, ais_F,
                                  ladder=10, step=3e-3, iters=40, chunk=4)
mean_err = (y1.mean(0) - y4.mean(0)).abs().max().item()
std_err  = (y1.std(0)  - y4.std(0) ).abs().max().item()
print(f"  max |mean(chunk=1) - mean(chunk=4)| = {mean_err:.4f}")
print(f"  max |std (chunk=1) - std (chunk=4)| = {std_err:.4f}")
# Different RNG consumption order in the per-rung Langevin -> different
# per-particle noise but the same distribution; moments agree to MC error.
assert mean_err < 0.05, f"chunk moments diverged: mean err {mean_err}"
assert std_err  < 0.05, f"chunk moments diverged: std err {std_err}"
print("  [OK ] chunk produces statistically equivalent samples")

# F.5 — the _G variant (inverse map G = F.inv) is the exact twin of _F:
#        same per-rung weight rule, so it lands on the same target.
section("F.5  _G (inverse map G = F.inv) matches _F")
ais_G = ais_F.inv # G = F^{-1}: G.inv(x) = F(x), G(y) = F^{-1}(y)
# (a) the incremental weight rule agrees with _F's to round-trip tolerance:
#     _F uses +log|det J_F(x)| (forward at x = F.inv(y)); _G uses -log|det J_G(y)|
#     (inverse at y); these are equal up to the bijection's round-trip error.
torch.manual_seed(75)
y_probe = ais_target.samples(4000)
with torch.no_grad():
    xF = ais_F.inv(y_probe); _, ladjF = ais_F.call_and_ladj(xF)
    wF = -ais_target(y_probe) + ais_source(xF) + ladjF
    xG, ladjG = ais_G.call_and_ladj(y_probe)
    wG = -ais_target(y_probe) + ais_source(xG) - ladjG
w_err = (wF - wG).abs().max().item()
print(f"  max |log w_F - log w_G| = {w_err:.2e}  (round-trip tolerance)")
assert w_err < 1e-3, f"_F / _G weight rule diverged: {w_err}"
# (b) _G lands the cloud on the target, just like _F.1.
torch.manual_seed(61)
x_src = ais_source.samples(8000)
yG = annealed_importance_sampling_G(x_src, ais_source, ais_target, ais_G,
                                    ladder=12, step=3e-3, iters=30, chunk=2)
meanG = yG.mean(0).tolist()
stdG = yG.std(0).tolist()
print(f"   _G mean = {[f'{v:+.3f}' for v in meanG]}  (target [3.000, 3.000])")
print(f"   _G std  = {[f'{v:.3f}' for v in stdG]}  (target [{target_std:.3f}, {target_std:.3f}])")
assert torch.isfinite(yG).all(), "_G leaked non-finite particles"
for v in meanG:
    assert abs(v - 3.0) < 0.1, f"_G mean off target: {meanG}"
for v in stdG:
    assert abs(v - target_std) < 0.1, f"_G std off target: {stdG}"
print("  [OK ] _G reproduces the weight rule and lands on the target")

# F.6 — compiled `.eval` energy fast path: enabling target/source.enable_eval()
#        routes the per-rung reweighting through the compiled forward, and the
#        run still lands on the target (same as the plain-__call__ path).
section("F.6  AIS uses target/source.enable_eval() and still lands on target")
src_eval = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_eval()
tgt_eval = Gaussian(mean=[3.0, 3.0], variance=[0.5, 0.5]).to(device).enable_grad().enable_eval()
assert tgt_eval._eval_fn is not None and src_eval._eval_fn is not None, "enable_eval did not set _eval_fn"
torch.manual_seed(63)
x_src = ais_source.samples(6000)
y_ev = annealed_importance_sampling_F(x_src, src_eval, tgt_eval, ais_F,
                                      ladder=10, step=3e-3, iters=30, chunk=2)
mean_ev, std_ev = y_ev.mean(0).tolist(), y_ev.std(0).tolist()
print(f"   mean = {[f'{v:+.3f}' for v in mean_ev]}  std = {[f'{v:.3f}' for v in std_ev]}  (target [3,3]/{target_std:.3f})")
assert torch.isfinite(y_ev).all(), "AIS .eval path leaked non-finite particles"
for v in mean_ev:
    assert abs(v - 3.0) < 0.1, f".eval path mean off target: {mean_ev}"
for v in std_ev:
    assert abs(v - target_std) < 0.1, f".eval path std off target: {std_ev}"
print("  [OK ] compiled .eval reweighting path lands on the target")


# ─────────────────────────────────────────────────────────────────
print()
print("All utils verification checks passed.")
