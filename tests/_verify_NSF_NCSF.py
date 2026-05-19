"""Verification script for NSF / NCSF correctness.

Run from the repo root:  python -m tests._verify_NSF_NCSF

Checks:
  1. Construction + .to(device) cleanliness.
  2. Forward / inverse bijection round-trip for NSF and NCSF.
  3. zeros() initialisation gives the identity (y == x, ladj == 0).
  4. log|det J| via call_and_ladj matches the autograd-computed exact
     log|det J| for small batches.
  5. NSF preserves the box: x ∈ [a, b]^d ⇒ F(x) ∈ [a, b]^d.
  6. NCSF periodicity: F(x + (b - a) * k) == F(x) for integer k.
  7. AdditiveTransform-sandwich invariant: log|det J| equals the inner
     RQS' log|det J| (the two AdditiveTransforms contribute zero).
"""

import sys
from math import pi

import torch

from zflows.flow import NSF, NCSF


def banner(s: str) -> None:
    print()
    print("─" * 60)
    print(s)
    print("─" * 60)


def assert_close(actual, expected, atol, name):
    ok = torch.allclose(actual, expected, atol=atol)
    diff = (actual - expected).abs().max().item()
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name}: max |diff| = {diff:.3e}  (atol={atol})")
    if not ok:
        sys.exit(1)


def exact_log_abs_det_jacobian(F, x):
    """For a (1, d) input x, compute log|det Jacobian| via torch.autograd.functional.jacobian."""
    def f(z):
        return F(z.unsqueeze(0)).squeeze(0)
    J = torch.autograd.functional.jacobian(f, x.squeeze(0))
    sign, ladj = torch.linalg.slogdet(J)
    return ladj


torch.manual_seed(0)
device = "cpu"   # CPU for portability; check device mobility separately

# ─────────────────────────────────────────────────────────────────
banner("1. Construction + .to(device)")
nsf = NSF(a=[-4.0, -4.0], b=[4.0, 4.0], bins=8, transforms=4).to(device)
print(f"  NSF built: features=2, {sum(p.numel() for p in nsf.parameters())} params")

ncsf = NCSF(a=[-pi, -pi, -pi], b=[pi, pi, pi], bins=8, transforms=4).to(device)
print(f"  NCSF built: features=3, {sum(p.numel() for p in ncsf.parameters())} params")

# asymmetric box for NSF
nsf_asym = NSF(a=[1.0, -2.0, 0.0], b=[3.0, 2.0, 5.0], bins=6, transforms=3).to(device)
print(f"  NSF asymmetric built: features=3, halfwidth={nsf_asym.halfwidth.tolist()}")

# ─────────────────────────────────────────────────────────────────
banner("2. Bijection round-trip")
for name, flow, a, b in [
    ("NSF symmetric", nsf, [-4.0, -4.0], [4.0, 4.0]),
    ("NSF asymmetric", nsf_asym, [1.0, -2.0, 0.0], [3.0, 2.0, 5.0]),
    ("NCSF", ncsf, [-pi, -pi, -pi], [pi, pi, pi]),
]:
    F = flow.t()
    a_t = torch.tensor(a)
    b_t = torch.tensor(b)
    d = len(a)
    # sample uniformly in the box
    x = a_t + (b_t - a_t) * torch.rand(200, d)
    with torch.no_grad():
        y = F(x)
        x_back = F.inv(y)
    assert_close(x_back, x, atol=1e-4, name=f"{name}  F.inv(F(x)) ≈ x")

# ─────────────────────────────────────────────────────────────────
banner("3. zeros() initialisation → identity bijection")
for name, builder in [
    ("NSF zeros", lambda: NSF(a=[-4.0, -4.0], b=[4.0, 4.0], bins=8, transforms=4)),
    ("NSF asym zeros", lambda: NSF(a=[1.0, -2.0, 0.0], b=[3.0, 2.0, 5.0], bins=6, transforms=3)),
    ("NCSF zeros", lambda: NCSF(a=[-pi, -pi, -pi], b=[pi, pi, pi], bins=8, transforms=4)),
]:
    flow = builder().to(device)
    flow.zeros()
    a_, b_ = flow.a, flow.b
    d = a_.size(0)
    x = a_ + (b_ - a_) * torch.rand(200, d)
    F = flow.t()
    with torch.no_grad():
        y, ladj = F.call_and_ladj(x)
    assert_close(y, x, atol=1e-5, name=f"{name}  y ≈ x")
    assert_close(ladj, torch.zeros_like(ladj), atol=1e-5, name=f"{name}  ladj ≈ 0")

# ─────────────────────────────────────────────────────────────────
banner("4. log|det J| consistency with autograd")
# Build flows again (not zeros'd) to get a non-trivial Jacobian
nsf = NSF(a=[-4.0, -4.0], b=[4.0, 4.0], bins=8, transforms=4).to(device)
ncsf = NCSF(a=[-pi, -pi], b=[pi, pi], bins=8, transforms=4).to(device)
nsf_asym = NSF(a=[1.0, -2.0], b=[3.0, 2.0], bins=6, transforms=3).to(device)

for name, flow in [("NSF", nsf), ("NSF asym", nsf_asym), ("NCSF", ncsf)]:
    F = flow.t()
    a_, b_ = flow.a, flow.b
    d = a_.size(0)
    torch.manual_seed(42)
    x = a_ + (b_ - a_) * torch.rand(5, d)
    with torch.no_grad():
        y, ladj_fast = F.call_and_ladj(x)
    # autograd-computed exact log|det J| for each sample
    ladj_exact = torch.stack([
        exact_log_abs_det_jacobian(F, x[i:i + 1]) for i in range(x.size(0))
    ])
    assert_close(ladj_fast, ladj_exact, atol=1e-3, name=f"{name}  call_and_ladj ladj ≈ autograd slogdet")

# ─────────────────────────────────────────────────────────────────
banner("5. NSF preserves the box")
nsf = NSF(a=[1.0, -2.0, 0.0], b=[3.0, 2.0, 5.0], bins=6, transforms=4).to(device)
a_, b_ = nsf.a, nsf.b
x = a_ + (b_ - a_) * torch.rand(1000, 3)
with torch.no_grad():
    y = nsf.t()(x)
inside = ((y >= a_) & (y <= b_)).all(dim=-1)
n_outside = (~inside).sum().item()
print(f"  Particles outside [a, b]: {n_outside} / {x.size(0)}")
# The spline maps [-half, half] → [-half, half], plus the translations,
# so y must land in [a, b] for every x in [a, b]. Strictly so, up to FP.
if n_outside > 0:
    print("  FAIL: NSF should preserve the box exactly.")
    sys.exit(1)
print("  [OK ] all particles inside the box")

# ─────────────────────────────────────────────────────────────────
banner("6. NCSF box preservation + per-coord CircularShift correctness")
# Zuko's NCSF / our NCSF is NOT globally periodic in its full input —
# the autoregressive conditioner MLP isn't periodic in x_<i. What IS
# guaranteed: for x ∈ [a, b]^d, F(x) ∈ [a, b]^d (the per-coord
# CircularShift wraps any RQS output back into the box). Test that.
ncsf = NCSF(a=[-pi, -pi, -pi], b=[pi, pi, pi], bins=8, transforms=4).to(device)
torch.manual_seed(1)
period = ncsf.b - ncsf.a
x = ncsf.a + period * torch.rand(500, 3)
with torch.no_grad():
    y = ncsf.t()(x)
inside = ((y >= ncsf.a - 1e-5) & (y <= ncsf.b + 1e-5)).all(dim=-1)
n_outside = (~inside).sum().item()
print(f"  Particles outside [-pi, pi]^3: {n_outside} / {x.size(0)}")
if n_outside > 0:
    sys.exit(1)
print("  [OK ] all NCSF outputs inside the periodic box")

# CircularShiftTransform unit test with per-coord bound.
from zflows.core.transforms import CircularShiftTransform
bound = torch.tensor([1.0, 2.5, 0.5])
cs = CircularShiftTransform(bound=bound)
# Inputs spanning multiple periods on each coord
u = torch.tensor([
    [ 0.3,  0.0,  0.1],
    [ 1.7,  3.0,  0.7],   # past +bound on coord 0 / coord 2; coord 1 well inside
    [-1.3, -3.0, -0.8],   # past -bound on coord 0 / coord 2
    [ 5.0,  7.5,  4.5],   # many periods out
])
v = cs._call(u)
# All outputs must land in [-bound, bound]
inside = (v >= -bound - 1e-6) & (v <= bound + 1e-6)
assert inside.all(), f"CircularShift output outside per-coord box:\n{v}"
# Inputs that differ by an integer multiple of 2*bound on a given coord
# should map to the same point on that coord.
u1 = torch.tensor([[ 0.3, 0.4,  0.2]])
u2 = u1 + torch.tensor([[2.0, 5.0, 1.0]])  # one period each (2*bound)
v1 = cs._call(u1)
v2 = cs._call(u2)
assert_close(v2, v1, atol=1e-6, name="CircularShift  per-coord period invariance")

# ─────────────────────────────────────────────────────────────────
banner("7. AdditiveTransform-sandwich invariant")
# Sandwich the spline with two AdditiveTransform shifts. The shifts have
# log|det J| = 0, so the overall log|det J| should equal that of the
# inner ComposedTransform of MAFs alone.
nsf = NSF(a=[1.0, -2.0], b=[3.0, 2.0], bins=6, transforms=4).to(device)
F = nsf.t()
# F.transforms is (Additive, inner_composed, Additive)
inner_composed = F.transforms[1]
a_, b_, c_, hw = nsf.a, nsf.b, nsf.center, nsf.halfwidth
x = a_ + (b_ - a_) * torch.rand(50, 2)
u = x - c_   # what the inner spline sees
with torch.no_grad():
    _, ladj_full = F.call_and_ladj(x)
    _, ladj_inner = inner_composed.call_and_ladj(u)
assert_close(ladj_full, ladj_inner, atol=1e-6, name="NSF  ladj == ladj_inner (translation contributes 0)")

# ─────────────────────────────────────────────────────────────────
print()
print("All NSF / NCSF verification checks passed.")
