# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""
Sanity checks for the batched hmc() routine in zflows.utils.

`hmc` is the safer cousin of `langevin(adjust=True)`: each "burn" is
one full-momentum refresh p ~ N(0, I) followed by an efficient
leapfrog trajectory (iters + 1 grad calls, combined half-kicks) and
one MH accept/reject decision. Divergent trajectories produce a
non-finite log_alpha, which is clamped to -inf so the particle
reverts to its pre-trajectory position -- no NaN ever leaks out.

We verify:
  1. Invariance: starting FROM the target N(0, I), the sample
     distribution after many HMC trajectories is unchanged.
  2. Convergence: starting from a broad N(0, 25 * I) init, HMC pulls
     the cloud onto the target N(0, I).
  3. Anisotropic target (variance ratio 4:1): HMC's long trajectories
     mix both axes, recovering the per-axis std under invariance.
  4. NaN guard: an absurdly large step makes every trajectory
     divergent; every particle reverts to x_start and the returned
     tensor is finite (max|x| == 0 from a zero init).
  5. hmc() raises a clear RuntimeError if enable_grad() was not called.
  6. chunk > 1 is *statistically* equivalent to chunk == 1 (not bit-
     identical -- RNG state advances differently per-chunk -- but
     aggregate mean and std agree within MC error).
  7. enable_eval() is optional: hmc() falls back to potential(x) when
     the compiled forward fast path is not enabled.
"""

import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # cleaner logs

import torch
from zflows.potential import Gaussian, Potential
from zflows.utils import hmc

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# ----------------------------------------------------------------------
# Test 1: invariance -- start at N(0, I), the chain leaves it alone
# ----------------------------------------------------------------------
print("=" * 60)
print("Test 1: invariance -- start at N(0, I), stay at N(0, I)")
print("=" * 60)

u = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad().enable_eval()

N = 10000
torch.manual_seed(1)
x0 = u.samples(N)
mean0 = x0.mean(0).tolist()
std0 = x0.std(0).tolist()

x = hmc(x0, potential=u, step=0.15, iters=10, burns=20)
mean1 = x.mean(0).tolist()
std1 = x.std(0).tolist()

print(f"before: mean = {[f'{v:+.3f}' for v in mean0]}, std = {[f'{v:.3f}' for v in std0]}")
print(f" after: mean = {[f'{v:+.3f}' for v in mean1]}, std = {[f'{v:.3f}' for v in std1]}")
for m in mean1:
    assert abs(m) < 0.05, f"invariance broken: mean = {mean1}"
for s in std1:
    assert abs(s - 1.0) < 0.05, f"invariance broken: std = {std1}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 2: convergence -- broad init contracts onto the target
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 2: convergence -- broad N(0, 25 I) init pulls onto N(0, I)")
print("=" * 60)

torch.manual_seed(2)
x0 = torch.randn(5000, 2, device=device) * 5.0
mean0 = x0.mean(0).tolist()
std0 = x0.std(0).tolist()

# Need long enough trajectories + enough burns to forget the broad init.
x = hmc(x0, potential=u, step=0.15, iters=15, burns=100)
mean1 = x.mean(0).tolist()
std1 = x.std(0).tolist()

print(f"before: mean = {[f'{v:+.3f}' for v in mean0]}, std = {[f'{v:.3f}' for v in std0]}")
print(f" after: mean = {[f'{v:+.3f}' for v in mean1]}, std = {[f'{v:.3f}' for v in std1]}")
for m in mean1:
    assert abs(m) < 0.1, f"mean did not converge to 0: {mean1}"
for s in std1:
    assert abs(s - 1.0) < 0.1, f"std did not converge to 1: {std1}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 3: anisotropic target -- per-axis std recovered under invariance
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 3: anisotropic target -- per-axis std (variance ratio 4:1)")
print("=" * 60)

u_aniso = Gaussian(mean=[0.0, 0.0], variance=[4.0, 1.0]).to(device).enable_grad().enable_eval()

torch.manual_seed(3)
x0 = u_aniso.samples(10000)
x = hmc(x0, potential=u_aniso, step=0.1, iters=15, burns=30)
std1 = x.std(0).tolist()
print(f"per-axis std = {[f'{v:.3f}' for v in std1]}   (target [2.000, 1.000])")
assert abs(std1[0] - 2.0) < 0.1, f"wide axis std off: {std1[0]}"
assert abs(std1[1] - 1.0) < 0.05, f"narrow axis std off: {std1[1]}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 4: NaN guard -- huge step rejects every trajectory
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 4: NaN guard -- huge step rejects every trajectory, no NaN leaks")
print("=" * 60)

x0 = torch.zeros(2000, 2, device=device)  # all at the mode of N(0, I)
x = hmc(x0, potential=u, step=10.0, iters=20, burns=5)
n_finite = torch.isfinite(x).all(-1).float().mean().item()
max_abs = x.abs().max().item()
print(f"fraction finite = {n_finite:.4f}")
print(f"max |x|         = {max_abs:.4f}  (0.0 means every trajectory rejected)")
assert n_finite == 1.0, "NaN leaked out of HMC"
assert max_abs == 0.0, "some divergent trajectories were accepted"
print("PASSED")

# ----------------------------------------------------------------------
# Test 5: missing enable_grad() raises RuntimeError
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 5: hmc() raises RuntimeError without enable_grad()")
print("=" * 60)

u_noenable = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)
x0 = torch.randn(8, 2, device=device)
raised = False
try:
    hmc(x0, potential=u_noenable, step=0.1, iters=5, burns=3)
except RuntimeError as e:
    raised = True
    print(f"  raised: {e}")
assert raised, "hmc() must raise without enable_grad()"
print("PASSED")

# ----------------------------------------------------------------------
# Test 6: chunk statistically matches chunk=1 (mean/std agree within MC error)
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 6: chunk statistically matches chunk=1 (aggregate moments)")
print("=" * 60)

torch.manual_seed(6)
x0 = u.samples(4000)

torch.manual_seed(60)
y1 = hmc(x0, potential=u, step=0.15, iters=10, burns=10, chunk=1)
torch.manual_seed(60)
y4 = hmc(x0, potential=u, step=0.15, iters=10, burns=10, chunk=4)

mean_err = (y1.mean(0) - y4.mean(0)).abs().max().item()
std_err  = (y1.std(0)  - y4.std(0) ).abs().max().item()
print(f"max |mean(chunk=1) - mean(chunk=4)| = {mean_err:.4f}")
print(f"max |std (chunk=1) - std (chunk=4)| = {std_err:.4f}")
# Different RNG consumption order -> different per-particle outputs but the
# same distribution. Aggregate moments agree within MC error ~ 0.02 for N=4000.
assert mean_err < 0.05, f"chunk moments diverged: mean err {mean_err}"
assert std_err  < 0.05, f"chunk moments diverged: std err {std_err}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 7: enable_eval() is optional -- fallback to potential(x) works
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 7: hmc() falls back to potential(x) without enable_eval()")
print("=" * 60)

u_grad_only = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad()
# no .enable_eval() -- hmc must route MH energy evals through __call__

torch.manual_seed(7)
x0 = u_grad_only.samples(5000)
x = hmc(x0, potential=u_grad_only, step=0.15, iters=10, burns=20)
mean1 = x.mean(0).tolist()
std1 = x.std(0).tolist()
print(f"after fallback path: mean = {[f'{v:+.3f}' for v in mean1]}, std = {[f'{v:.3f}' for v in std1]}")
for m in mean1:
    assert abs(m) < 0.08, f"fallback path: mean off: {mean1}"
for s in std1:
    assert abs(s - 1.0) < 0.08, f"fallback path: std off: {std1}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 8: super-linearly growing potential (cubic gradient)
# ----------------------------------------------------------------------
# This is the regime where the "safer than ULA" claim matters: plain
# ULA on U = x^4/4 explodes whenever a particle lands in the tail
# (drift = step * x^3 overwhelms the diffusion). HMC's MH gate clamps
# divergent leapfrog trajectories to log_alpha = -inf, so the chain
# stays finite and unbiased.
print()
print("=" * 60)
print("Test 8: super-linear potential U(x) = x^4 / 4 (cubic gradient)")
print("=" * 60)

class Quartic(Potential):
    """U(x) = sum_i x_i^4 / 4 -- gradient grows as x^3."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.25 * (x ** 4).sum(dim=-1)

u_q = Quartic().to(device).enable_grad().enable_eval()

# (8a) Sampling correctness against the analytic moment.
# For 1D p(x) = exp(-x^4/4) / Z (Z = 2 * 4^(-3/4) * Gamma(1/4)), the
# substitution u = x^4 / 4 gives:
#     E[x^2] = 2 * Gamma(3/4) / Gamma(1/4)   ~  0.6760
#     std    = sqrt(E[x^2])                  ~  0.8222
import math
target_std = math.sqrt(2 * math.gamma(0.75) / math.gamma(0.25))

torch.manual_seed(8)
# Mild init in the stable region; HMC needs to find and sample the bulk.
x0 = torch.randn(10000, 1, device=device) * 0.5
x = hmc(x0, potential=u_q, step=0.1, iters=10, burns=100)
final_std = x.std(0).item()
final_mean = x.mean(0).item()
n_finite = torch.isfinite(x).all(-1).float().mean().item()

print(f"target std (analytic)       = {target_std:.4f}")
print(f"sampled std after HMC       = {final_std:.4f}")
print(f"sampled mean after HMC      = {final_mean:+.4f}")
print(f"fraction finite             = {n_finite:.4f}")
assert n_finite == 1.0, "NaN leaked from super-linear potential"
assert abs(final_std - target_std) < 0.03, \
    f"sampled std {final_std} far from analytic {target_std}"
assert abs(final_mean) < 0.03, f"sampled mean off: {final_mean}"

# (8b) Aggressive step from the extreme tail: every trajectory diverges
# (leapfrog at step=0.5, |x|=10 -> grad ~ 1000 -> blows up within 2-3
# steps), MH rejects them all via the isfinite guard, particles stay
# locked at x_start = 10. This is the "safer" path: no NaN, no silent
# corruption of the chain.
torch.manual_seed(80)
x_tail = torch.full((1000, 1), 10.0, device=device)
x_after = hmc(x_tail, potential=u_q, step=0.5, iters=10, burns=5)
tail_finite = torch.isfinite(x_after).all(-1).float().mean().item()
n_at_start = (x_after == 10.0).all(-1).float().mean().item()
print(f"aggressive-step tail: finite = {tail_finite:.4f}, "
      f"stuck at x_start = {n_at_start:.4f}")
assert tail_finite == 1.0, "NaN leaked from divergent tail trajectories"
assert n_at_start > 0.99, \
    f"too many divergent trajectories were accepted: {n_at_start}"
print("PASSED")

print()
print("All hmc() checks passed.")
