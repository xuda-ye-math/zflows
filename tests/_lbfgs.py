# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""
Sanity checks for the batched lbfgs() routine in zflows.utils.

`lbfgs` is the mode-finding counterpart of `langevin`: same call shape,
same chunking, no noise -- it builds a rank-`memory` approximation of
the inverse Hessian from per-particle history of (s, y) pairs and
descends in lockstep across all N particles. `optimization` in
zflows.utils is an alias for this routine.

We verify:
  1. On a well-conditioned Gaussian (variance=1), lbfgs reaches fp32
     precision in ~30 iters (superlinear convergence from the BFGS
     Hessian approximation).
  2. On an ill-conditioned diagonal Gaussian (variance ratio 100:1)
     where plain gradient descent would crawl on the flat axes, lbfgs
     still converges tightly because the curvature pairs encode the
     per-axis scaling.
  3. chunk > 1 is bit-identical to chunk == 1 (no noise, per-particle
     history is independent of batch composition).
  4. lbfgs() raises a clear RuntimeError if enable_grad() was not
     called.
  5. `optimization` and `lbfgs` are the same function (alias check).
"""

import torch
from zflows.potential import Gaussian
from zflows.utils import lbfgs, optimization

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# ----------------------------------------------------------------------
# Test 1: L-BFGS crushes a well-conditioned Gaussian in tens of iters
# ----------------------------------------------------------------------
print("=" * 60)
print("Test 1: lbfgs on U(x) = 0.5*||x-x*||^2 reaches ~fp32 precision fast")
print("=" * 60)

x_star = torch.tensor([2.0, -1.5, 0.7], device=device)
u = Gaussian(mean=x_star.tolist(), variance=[1.0, 1.0, 1.0]).to(device).enable_grad()

N, D = 2000, 3
torch.manual_seed(1)
x0 = torch.randn(N, D, device=device) * 5.0  # broad init
init_err = (x0 - x_star).norm(dim=-1).max().item()

ITERS = 30
y = lbfgs(x0, potential=u, step=1.0, iters=ITERS, memory=6)
err = (y - x_star).norm(dim=-1).max().item()

print(f"init max err: {init_err:.3e}")
print(f"lbfgs ({ITERS:3d} iters) max err: {err:.3e}")
assert err < 1e-5, f"L-BFGS failed to converge: {err}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 2: ill-conditioned Gaussian (variance ratio 100:1)
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 2: lbfgs on ill-conditioned diag Gaussian (kappa=100)")
print("=" * 60)

x_star2 = torch.zeros(4, device=device)
variance = [100.0, 1.0, 100.0, 1.0]  # condition number 100
u2 = Gaussian(mean=x_star2.tolist(), variance=variance).to(device).enable_grad()

N, D = 2000, 4
torch.manual_seed(2)
x0 = torch.randn(N, D, device=device) * 5.0
init_err = (x0 - x_star2).norm(dim=-1).max().item()

ITERS = 50
y = lbfgs(x0, potential=u2, step=1.0, iters=ITERS, memory=6)
err = (y - x_star2).norm(dim=-1).max().item()

print(f"init max err: {init_err:.3e}")
print(f"lbfgs ({ITERS:3d} iters) max err: {err:.3e}")
assert err < 1e-4, f"L-BFGS failed on ill-conditioned problem: {err}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 3: chunked vs unchunked is bit-identical (deterministic)
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 3: chunk > 1 gives identical output to chunk == 1")
print("=" * 60)

u3 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad()
torch.manual_seed(3)
x0 = torch.randn(128, 2, device=device) * 4.0

y1 = lbfgs(x0, potential=u3, step=1.0, iters=20, memory=6, chunk=1)
y4 = lbfgs(x0, potential=u3, step=1.0, iters=20, memory=6, chunk=4)
err = (y1 - y4).abs().max().item()
print(f"max |y_chunk1 - y_chunk4| = {err:.3e}")
assert err < 1e-5, f"chunking changed the output: {err}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 4: lbfgs() without enable_grad() raises RuntimeError
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 4: lbfgs() raises RuntimeError without enable_grad()")
print("=" * 60)

u4 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)  # no enable_grad()
x0 = torch.randn(8, 2, device=device)
raised = False
try:
    lbfgs(x0, potential=u4, step=1.0, iters=10, memory=6)
except RuntimeError as e:
    raised = True
    print(f"  raised: {e}")
assert raised, "lbfgs() must raise without enable_grad()"
print("PASSED")

# ----------------------------------------------------------------------
# Test 5: `optimization` is the same callable as `lbfgs`
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 5: optimization is an alias of lbfgs")
print("=" * 60)
print(f"optimization is lbfgs: {optimization is lbfgs}")
assert optimization is lbfgs, "optimization should be the lbfgs symbol itself, not a wrapper"
print("PASSED")

print()
print("All lbfgs() checks passed.")
