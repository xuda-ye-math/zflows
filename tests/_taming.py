# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false

"""
Stability sanity check for the tamed Langevin option in zflows.utils.langevin.

Setup: U(x) = (1/4) * ||x||^4, so grad U(x) = ||x||^2 * x grows as |x|^3.
The Euler-Maruyama iterate is
    x' = (1 - step * ||x||^2) * x + sqrt(2 * step) * xi,
which amplifies the norm whenever step * ||x||^2 > 2. With step=0.5 and
d=2, ~13% of N(0, I) initial particles already start in that unstable
regime; cubic-growth feedback then drives them to overflow within a few
iterations. The tamed drift
    G(x) = grad U(x) / (1 + taming * ||grad U(x)||)
caps ||step * G|| <= step / taming per iteration, so the chain stays
bounded no matter how far a particle wanders.

We verify:
  1. taming=0 (plain ULA) on this potential blows up (NaN/Inf or huge norm)
  2. taming=0.1 keeps the chain bounded (no NaN/Inf, finite max norm)
  3. adjust=True together with taming>0 raises ValueError
"""

import torch
from zflows.potential import Potential
from zflows.utils import langevin

device = "cuda" if torch.cuda.is_available() else "cpu"

class Quartic(Potential):
    """U(x) = (1/4) * ||x||^4 — grad U(x) = ||x||^2 * x grows super-linearly."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.25 * (x ** 2).sum(dim=-1) ** 2

N, D     = 512, 2
STEP     = 0.5
ITERS    = 200
TAMING   = 0.1

def fresh_init():
    torch.manual_seed(0)
    return torch.randn(N, D, device=device)

# ----------------------------------------------------------------------
# Test 1: plain ULA (taming=0) on the quartic blows up
# ----------------------------------------------------------------------
print("=" * 60)
print("Test 1: plain ULA (taming=0) on U(x) = ||x||^4 / 4")
print("=" * 60)

u = Quartic().to(device).enable_grad()
x0 = fresh_init()
torch.manual_seed(1)
y_ula = langevin(x0, potential=u, step=STEP, iters=ITERS, taming=0.0)

n_bad = (~torch.isfinite(y_ula)).any(dim=-1).sum().item()
finite_mask = torch.isfinite(y_ula).all(dim=-1)
max_norm_ula = y_ula[finite_mask].norm(dim=-1).max().item() if finite_mask.any() else float("nan")
print(f"NaN/Inf particles: {n_bad} / {N}")
print(f"max ||y|| (finite particles only): {max_norm_ula:.3e}")
assert n_bad > 0 or max_norm_ula > 1e6, "ULA did not blow up — pick a larger step / more iters"
print("PASSED (ULA is unstable here, as expected)")

# ----------------------------------------------------------------------
# Test 2: tamed ULA (taming=0.1) stays bounded
# ----------------------------------------------------------------------
print()
print("=" * 60)
print(f"Test 2: tamed ULA (taming={TAMING}) on the same potential")
print("=" * 60)

u = Quartic().to(device).enable_grad()
x0 = fresh_init()
torch.manual_seed(1)
y_tamed = langevin(x0, potential=u, step=STEP, iters=ITERS, taming=TAMING)

assert torch.isfinite(y_tamed).all(), "tamed ULA produced NaN/Inf"
max_norm_tamed = y_tamed.norm(dim=-1).max().item()
mean_norm_tamed = y_tamed.norm(dim=-1).mean().item()
print(f"NaN/Inf particles: 0 / {N}")
print(f"max  ||y||: {max_norm_tamed:.3e}")
print(f"mean ||y||: {mean_norm_tamed:.3e}")
# step * G is bounded by step/taming = 5 per iter; with the stabilising
# drift always pointing to the origin, the chain settles near the bulk
# of exp(-||x||^4 / 4).
assert max_norm_tamed < 50, f"tamed ULA wandered too far: {max_norm_tamed}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 3: adjust=True with taming>0 is forbidden
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 3: adjust=True together with taming>0 raises ValueError")
print("=" * 60)

u = Quartic().to(device).enable_grad()
x0 = fresh_init()
raised = False
try:
    _ = langevin(x0, potential=u, step=1e-2, iters=2, adjust=True, taming=TAMING)
except ValueError as e:
    raised = True
    print(f"  raised: {e}")
assert raised, "langevin must reject adjust=True with taming>0"
print("PASSED")

print()
print("All tamed Langevin stability checks passed.")
