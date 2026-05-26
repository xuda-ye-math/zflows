# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Verification script for every zflows.Flow subclass.

Run from the repo root:  python -m tests._verify_flow

Consolidates the earlier per-flow scripts (_verify_NSF_NCSF, _verify_CNF_RealNVP,
_zeros, _CNF_interface) into one banner-separated harness. Sections:

   1. Construction + .to(device) cleanliness.
   2. Bijection round-trip:     F.inv(F(x)) ≈ x  for every flow.
   3. .zeros()         → identity: y ≈ x, ladj ≈ 0.
   4. log|det J| via   call_and_ladj matches autograd slogdet.
   5. NSF preserves   the box [a, b]^d.
   6. NCSF box preservation + per-coord CircularShift correctness.
   7. NSF AdditiveTransform-sandwich invariant
      (translations contribute 0 to log|det J|).
   8. NSF / NCSF randmask=False legacy alternating-ordering path.
   9. RealNVP forward + inverse log-determinants cancel.
  10. CNF Hutchinson (`exact=False`) is an unbiased ladj estimator.
  11. CNF dual interface:   cnf.t()  ≡  cnf._ffj()
      (forward, inverse, round-trip, gradients all agree).
  12. Backprop reaches every parameter (RealNVP + CNF).
  13. zflows.loss.loss_compile / loss_compile_beta sanity:
      (a) loss_compile output matches raw reverse_KL (default beta + baked beta);
      (b) loss_compile_beta with runtime beta does not trigger recompiles.
"""

import sys
from math import pi

import torch
import torch._dynamo as dynamo

from zflows.core.flows import LinearMixingTransform
from zflows.core.transforms import FreeFormJacobianTransform, CircularShiftTransform
from zflows.flow import NSF, NCSF, CNF, OTFlow, RealNVP, Flow, ComposedTransform
import zflows.loss
from zflows.loss import reverse_KL, OT_loss
from zflows.potential import Gaussian
from zflows.utils import suppress_warnings, set_cache_size_limit

# Silence Triton/Inductor/Dynamo/Python warnings. No cache-limit bump needed:
# this verification suite does not call torch.compile directly.
suppress_warnings()


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
    """For a (1, d) input x, slogdet of the autograd Jacobian."""
    def f(z):
        return F(z.unsqueeze(0)).squeeze(0)
    J = torch.autograd.functional.jacobian(f, x.squeeze(0))
    _, ladj = torch.linalg.slogdet(J)
    return ladj


torch.manual_seed(0)
device = "cpu"   # CPU for portability; device mobility is exercised in §1.

# ─────────────────────────────────────────────────────────────────
banner("1. Construction + .to(device)")
nsf = NSF(a=[-4.0, -4.0], b=[4.0, 4.0], bins=8, transforms=4).to(device)
print(f"  NSF d=2: {sum(p.numel() for p in nsf.parameters())} params")

nsf_asym = NSF(a=[1.0, -2.0, 0.0], b=[3.0, 2.0, 5.0], bins=6, transforms=3).to(device)
print(f"  NSF d=3 asymmetric: halfwidth={nsf_asym.halfwidth.tolist()}")

ncsf = NCSF(a=[-pi, -pi, -pi], b=[pi, pi, pi], bins=8, transforms=4).to(device)
print(f"  NCSF d=3: {sum(p.numel() for p in ncsf.parameters())} params")

realnvp = RealNVP(dimension=4, transforms=4, hidden_features=(32, 32)).to(device)
print(f"  RealNVP d=4: {sum(p.numel() for p in realnvp.parameters())} params, "
      f"{len(realnvp._layers)} layers")

cnf = CNF(dimension=2, frequency=3, hidden_features=(32, 32)).to(device)
print(f"  CNF d=2: {sum(p.numel() for p in cnf.parameters())} params")

cnf_hutch = CNF(dimension=2, frequency=3, hidden_features=(32, 32), exact=False).to(device)
print(f"  CNF d=2 (Hutchinson): {sum(p.numel() for p in cnf_hutch.parameters())} params")

otf = OTFlow(dimension=2, hidden=32, layer=3, nt=8).to(device)
print(f"  OTFlow d=2: {sum(p.numel() for p in otf.parameters())} params")

# ─────────────────────────────────────────────────────────────────
banner("2. Bijection round-trip")
for name, flow, a, b in [
    ("NSF symmetric",  nsf,      [-4.0, -4.0],        [4.0, 4.0]),
    ("NSF asymmetric", nsf_asym, [1.0, -2.0, 0.0],    [3.0, 2.0, 5.0]),
    ("NCSF",           ncsf,     [-pi, -pi, -pi],     [pi, pi, pi]),
]:
    F = flow.t()
    a_t, b_t = torch.tensor(a), torch.tensor(b)
    d = len(a)
    x = a_t + (b_t - a_t) * torch.rand(200, d)
    with torch.no_grad():
        y = F(x)
        x_back = F.inv(y)
    assert_close(x_back, x, atol=1e-4, name=f"{name}  F.inv(F(x)) ≈ x")

# RealNVP — closed-form inverse, very tight
realnvp.eval()
x = torch.randn(256, 4)
with torch.no_grad():
    F = realnvp.t()
    y = F(x)
    x_back = F.inv(y)
assert_close(x_back, x, atol=1e-5, name="RealNVP    F.inv(F(x)) ≈ x")

# CNF — ODE-based inverse, looser tol
cnf.eval()
x = torch.randn(64, 2) * 0.7
with torch.no_grad():
    F = cnf.t()
    y = F(x)
    x_back = F.inv(y)
assert_close(x_back, x, atol=5e-4, name="CNF        F.inv(F(x)) ≈ x")

# OTFlow — fixed-step RK4 inverse (backward integration). Round-trip error
# is pure RK4 discretization (4th order: it falls ~16x per doubling of nt),
# and an untrained xavier-initialised potential is stiff, so use a finer nt
# here than the nt=8 default. This confirms the inverse direction is correct,
# not just close.
otf_rt = OTFlow(dimension=2, hidden=32, layer=3, nt=32).to(device)
otf_rt.eval()
x = torch.randn(64, 2) * 0.7
with torch.no_grad():
    F = otf_rt.t()
    y = F(x)
    x_back = F.inv(y)
assert_close(x_back, x, atol=1e-3, name="OTFlow     F.inv(F(x)) ≈ x  (nt=32)")

# ─────────────────────────────────────────────────────────────────
banner("3. zeros() initialisation → identity bijection")
for name, builder, sampler, atol_y, atol_l in [
    ("NSF zeros",
     lambda: NSF(a=[-4.0, -4.0], b=[4.0, 4.0], bins=8, transforms=4),
     lambda f: f.a + (f.b - f.a) * torch.rand(64, f.a.size(0)),
     1e-5, 1e-5),
    ("NSF asym zeros",
     lambda: NSF(a=[1.0, -2.0, 0.0], b=[3.0, 2.0, 5.0], bins=6, transforms=3),
     lambda f: f.a + (f.b - f.a) * torch.rand(64, f.a.size(0)),
     1e-5, 1e-5),
    ("NCSF zeros",
     lambda: NCSF(a=[-pi, -pi, -pi], b=[pi, pi, pi], bins=8, transforms=4),
     lambda f: f.a + (f.b - f.a) * torch.rand(64, f.a.size(0)),
     1e-4, 1e-4),
    ("RealNVP zeros",
     lambda: RealNVP(dimension=4, transforms=4, hidden_features=(32, 32)),
     lambda _: torch.randn(64, 4),
     1e-5, 1e-5),
    ("CNF zeros",
     lambda: CNF(dimension=2, frequency=3, hidden_features=(32, 32)),
     lambda _: torch.randn(64, 2),
     1e-4, 1e-4),
    ("OTFlow zeros",
     lambda: OTFlow(dimension=2, hidden=32, layer=3, nt=8),
     lambda _: torch.randn(64, 2),
     1e-5, 1e-5),
]:
    flow = builder().to(device)
    flow.zeros()
    flow.eval()
    x = sampler(flow)
    with torch.no_grad():
        y, ladj = flow.t().call_and_ladj(x)
    assert_close(y, x, atol=atol_y, name=f"{name}  y ≈ x")
    assert_close(ladj, torch.zeros_like(ladj), atol=atol_l, name=f"{name}  ladj ≈ 0")

# ─────────────────────────────────────────────────────────────────
banner("4. log|det J| via call_and_ladj matches autograd slogdet")
# Rebuild non-trivial (non-zeros'd) flows so the Jacobian is informative.
torch.manual_seed(42)
nsf       = NSF(a=[-4.0, -4.0], b=[4.0, 4.0], bins=8, transforms=4).to(device)
nsf_asym  = NSF(a=[1.0, -2.0], b=[3.0, 2.0], bins=6, transforms=3).to(device)
ncsf      = NCSF(a=[-pi, -pi], b=[pi, pi], bins=8, transforms=4).to(device)
realnvp   = RealNVP(dimension=4, transforms=4, hidden_features=(32, 32)).to(device)
cnf       = CNF(dimension=2, frequency=3, hidden_features=(32, 32)).to(device)
# nt=32: the augmented-ODE ladj and the autograd slogdet of the discrete RK4
# map agree only as nt grows (both are 4th-order accurate to the continuous
# log-det); nt=8 leaves ~6e-2 discretization gap, nt=32 closes it to ~1e-4.
otf       = OTFlow(dimension=2, hidden=32, layer=3, nt=32).to(device)
realnvp.eval()
cnf.eval()
otf.eval()

for name, flow, sample_fn, atol in [
    ("NSF",      nsf,      lambda f: f.a + (f.b - f.a) * torch.rand(5, f.a.size(0)),  1e-3),
    ("NSF asym", nsf_asym, lambda f: f.a + (f.b - f.a) * torch.rand(5, f.a.size(0)),  1e-3),
    ("NCSF",     ncsf,     lambda f: f.a + (f.b - f.a) * torch.rand(5, f.a.size(0)),  1e-3),
    ("RealNVP",  realnvp,  lambda _: torch.randn(5, 4),                               1e-4),
    ("CNF",      cnf,      lambda _: torch.randn(5, 2) * 0.5,                         5e-3),
    ("OTFlow",   otf,      lambda _: torch.randn(5, 2) * 0.5,                         1e-3),
]:
    F = flow.t()
    x = sample_fn(flow)
    with torch.no_grad():
        _, ladj_fast = F.call_and_ladj(x)
    ladj_exact = torch.stack([
        exact_log_abs_det_jacobian(F, x[i:i + 1]) for i in range(x.size(0))
    ])
    assert_close(ladj_fast, ladj_exact, atol=atol, name=f"{name}  ladj ≈ autograd slogdet")

# ─────────────────────────────────────────────────────────────────
banner("5. NSF preserves the box")
nsf_box = NSF(a=[1.0, -2.0, 0.0], b=[3.0, 2.0, 5.0], bins=6, transforms=4).to(device)
a_, b_ = nsf_box.a, nsf_box.b
x = a_ + (b_ - a_) * torch.rand(1000, 3)
with torch.no_grad():
    y = nsf_box.t()(x)
inside = ((y >= a_) & (y <= b_)).all(dim=-1)
n_outside = (~inside).sum().item()
print(f"  Particles outside [a, b]: {n_outside} / {x.size(0)}")
if n_outside > 0:
    print("  FAIL: NSF should preserve the box exactly.")
    sys.exit(1)
print("  [OK ] all particles inside the box")

# ─────────────────────────────────────────────────────────────────
banner("6. NCSF box preservation + per-coord CircularShift correctness")
# NCSF / our NCSF is NOT globally periodic in its full input — the
# autoregressive conditioner MLP isn't periodic in x_<i. What IS
# guaranteed: for x ∈ [a, b]^d, F(x) ∈ [a, b]^d (the per-coord
# CircularShift wraps any RQS output back into the box).
ncsf_box = NCSF(a=[-pi, -pi, -pi], b=[pi, pi, pi], bins=8, transforms=4).to(device)
torch.manual_seed(1)
period = ncsf_box.b - ncsf_box.a
x = ncsf_box.a + period * torch.rand(500, 3)
with torch.no_grad():
    y = ncsf_box.t()(x)
inside = ((y >= ncsf_box.a - 1e-5) & (y <= ncsf_box.b + 1e-5)).all(dim=-1)
n_outside = (~inside).sum().item()
print(f"  Particles outside [-pi, pi]^3: {n_outside} / {x.size(0)}")
if n_outside > 0:
    sys.exit(1)
print("  [OK ] all NCSF outputs inside the periodic box")

# CircularShiftTransform unit test with per-coord bound.
bound = torch.tensor([1.0, 2.5, 0.5])
cs = CircularShiftTransform(bound=bound)
u = torch.tensor([
    [ 0.3,  0.0,  0.1],
    [ 1.7,  3.0,  0.7],
    [-1.3, -3.0, -0.8],
    [ 5.0,  7.5,  4.5],
])
v = cs._call(u)
inside = (v >= -bound - 1e-6) & (v <= bound + 1e-6)
assert inside.all(), f"CircularShift output outside per-coord box:\n{v}"
u1 = torch.tensor([[ 0.3, 0.4,  0.2]])
u2 = u1 + torch.tensor([[2.0, 5.0, 1.0]])  # one period each (2*bound)
v1 = cs._call(u1)
v2 = cs._call(u2)
assert_close(v2, v1, atol=1e-6, name="CircularShift  per-coord period invariance")

# ─────────────────────────────────────────────────────────────────
banner("7. AdditiveTransform-sandwich invariant (NSF)")
nsf_s = NSF(a=[1.0, -2.0], b=[3.0, 2.0], bins=6, transforms=4).to(device)
F = nsf_s.t()
inner_composed = F.transforms[1]   # F = (shift -c, inner, shift +c)
x = nsf_s.a + (nsf_s.b - nsf_s.a) * torch.rand(50, 2)
u = x - nsf_s.center               # what the inner spline sees
with torch.no_grad():
    _, ladj_full = F.call_and_ladj(x)
    _, ladj_inner = inner_composed.call_and_ladj(u)
assert_close(ladj_full, ladj_inner, atol=1e-6,
             name="NSF  ladj == ladj_inner  (translation contributes 0)")

# ─────────────────────────────────────────────────────────────────
banner("8. randmask=False alternating-ordering path")
# The default randmask=True is exercised by every earlier section.
# This block confirms the legacy alternating arange / arange.flip ordering
# still produces a valid bijection and an identity under .zeros().
for name, builder, a_, b_ in [
    ("NSF randmask=False",
     lambda: NSF(a=[-4.0, -4.0], b=[4.0, 4.0], bins=8, transforms=4,
                 randmask=False),
     [-4.0, -4.0], [4.0, 4.0]),
    ("NCSF randmask=False",
     lambda: NCSF(a=[-pi, -pi, -pi], b=[pi, pi, pi], bins=8, transforms=4,
                  randmask=False),
     [-pi, -pi, -pi], [pi, pi, pi]),
    ("RealNVP randmask=False",
     lambda: RealNVP(dimension=4, transforms=4, hidden_features=(32, 32),
                     randmask=False),
     None, None),
]:
    flow = builder().to(device)
    F = flow.t()
    if a_ is not None:
        a_t, b_t = torch.tensor(a_), torch.tensor(b_)
        d = len(a_)
        x = a_t + (b_t - a_t) * torch.rand(200, d)
    else:
        x = torch.randn(200, 4)
    with torch.no_grad():
        y = F(x)
        x_back = F.inv(y)
    assert_close(x_back, x, atol=1e-4, name=f"{name}  F.inv(F(x)) ≈ x")

    flow_z = builder().to(device)
    flow_z.zeros()
    flow_z.eval()
    F_z = flow_z.t()
    with torch.no_grad():
        y_z, ladj_z = F_z.call_and_ladj(x)
    assert_close(y_z, x, atol=1e-5, name=f"{name}  zeros() y ≈ x")
    assert_close(ladj_z, torch.zeros_like(ladj_z), atol=1e-5,
                 name=f"{name}  zeros() ladj ≈ 0")

# ─────────────────────────────────────────────────────────────────
banner("9. RealNVP forward + inverse ladj cancel")
realnvp.eval()
x = torch.randn(64, 4)
with torch.no_grad():
    F = realnvp.t()
    y, ladj_fwd = F.call_and_ladj(x)
    x_back, ladj_inv = F.inv.call_and_ladj(y)
assert_close(ladj_fwd + ladj_inv, torch.zeros_like(ladj_fwd),
             atol=1e-5, name="RealNVP  ladj_fwd + ladj_inv ≈ 0")
assert_close(x_back, x, atol=1e-5, name="RealNVP  inv(F(x)) ≈ x")

# ─────────────────────────────────────────────────────────────────
banner("10. CNF Hutchinson estimator is unbiased")
# Compare the exact ladj to the mean of M Hutchinson trials with the SAME
# drift (params copied). The estimator has O(1/sqrt(M)) MC noise; at M=32
# and d=2, |avg - exact| should be << 1.
cnf_exact = CNF(dimension=2, frequency=3, hidden_features=(32, 32), exact=True).to(device)
cnf_hutch = CNF(dimension=2, frequency=3, hidden_features=(32, 32), exact=False).to(device)
cnf_hutch.load_state_dict(cnf_exact.state_dict())
cnf_exact.eval()
cnf_hutch.eval()

torch.manual_seed(7)
x = torch.randn(4, 2) * 0.3
with torch.no_grad():
    _, ladj_ex = cnf_exact.t().call_and_ladj(x)
M = 32
ladj_h_avg = torch.zeros_like(ladj_ex)
for _ in range(M):
    with torch.no_grad():
        _, ladj_h = cnf_hutch.t().call_and_ladj(x)
    ladj_h_avg = ladj_h_avg + ladj_h / M
diff = (ladj_h_avg - ladj_ex).abs().max().item()
print(f"  exact ladj         = {ladj_ex.detach().cpu().tolist()}")
print(f"  Hutch  avg (M={M})  = {ladj_h_avg.detach().cpu().tolist()}")
print(f"  max |avg - exact|  = {diff:.3e}")
assert diff < 1e-1, f"Hutchinson estimator far from exact: {diff}"
print("  [OK ] Hutchinson estimator ≈ exact within MC noise")

# ─────────────────────────────────────────────────────────────────
banner("11. CNF dual interface  cnf.t()  ≡  cnf._ffj()")
# The zflows wrapper exists only to satisfy Flow.t() -> ComposedTransform.
# The inner FFJ transform does the actual work; both must agree.
cnf_i = CNF(dimension=2, frequency=3, hidden_features=(32, 32)).to(device)
T_z = cnf_i.t()         # zflows wrapper
T_n = cnf_i._ffj()      # native FFJ transform

assert isinstance(cnf_i, Flow)
assert isinstance(T_z, ComposedTransform)
assert isinstance(T_n, FreeFormJacobianTransform)
assert len(T_z.transforms) == 1
print(f"  isinstance(T_z, ComposedTransform)       = True")
print(f"  isinstance(T_n, FreeFormJacobianTransform) = True")
print(f"  len(T_z.transforms) (length-1 wrapper)   = {len(T_z.transforms)}")

cnf_i.eval()
x = torch.randn(8, 2, device=device)
with torch.no_grad():
    y_z, ladj_z = cnf_i.t().call_and_ladj(x)
    y_n, ladj_n = cnf_i._ffj().call_and_ladj(x)
assert_close(y_z,    y_n,    atol=1e-5, name="cnf.t() / cnf._ffj()  forward y agree")
assert_close(ladj_z, ladj_n, atol=1e-5, name="cnf.t() / cnf._ffj()  forward ladj agree")

with torch.no_grad():
    y = torch.randn(8, 2, device=device) * 0.5
    x_back_z = cnf_i.t().inv(y)
    x_back_n = cnf_i._ffj().inv(y)
assert_close(x_back_z, x_back_n, atol=1e-5,
             name="cnf.t() / cnf._ffj()  inverse agree")

# backprop through both interfaces, fresh state
cnf_i.train()
x = torch.randn(16, 2, device=device).clamp(-2, 2)

cnf_i.zero_grad()
y_z, ladj_z = cnf_i.t().call_and_ladj(x)
(y_z.pow(2).sum() - ladj_z.sum()).backward()
grads_z = [p.grad.detach().clone() for p in cnf_i.parameters() if p.grad is not None]

cnf_i.zero_grad()
y_n, ladj_n = cnf_i._ffj().call_and_ladj(x)
(y_n.pow(2).sum() - ladj_n.sum()).backward()
grads_n = [p.grad.detach().clone() for p in cnf_i.parameters() if p.grad is not None]

assert len(grads_z) == len(grads_n) > 0
err_grads = max((gz - gn).abs().max().item() for gz, gn in zip(grads_z, grads_n))
print(f"  #param tensors with non-None grad    = {len(grads_z)}")
print(f"  max |grad_t() - grad_ffj()|         = {err_grads:.3e}")
assert err_grads < 1e-4, f"gradients disagree: {err_grads}"
print("  [OK ] both interfaces reach identical gradients")

# ─────────────────────────────────────────────────────────────────
banner("12. Backprop reaches every parameter")
for name, flow, x_factory in [
    ("RealNVP", RealNVP(dimension=4, transforms=4, hidden_features=(32, 32)).to(device),
                lambda: torch.randn(16, 4)),
    ("CNF",     CNF(dimension=2, frequency=3, hidden_features=(32, 32)).to(device),
                lambda: torch.randn(16, 2) * 0.5),
    ("OTFlow",  OTFlow(dimension=2, hidden=32, layer=3, nt=8).to(device),
                lambda: torch.randn(16, 2) * 0.5),
]:
    flow.train()
    flow.zero_grad()
    x = x_factory()
    y, ladj = flow.t().call_and_ladj(x)
    (y.pow(2).sum() - ladj.sum()).backward()
    n_params = sum(1 for _ in flow.parameters())
    n_grads = sum(1 for p in flow.parameters() if p.grad is not None)
    print(f"  {name}: {n_grads}/{n_params} parameter tensors have non-None grad")
    if n_grads != n_params:
        print(f"  FAIL: {name} missing grads on {n_params - n_grads} params")
        sys.exit(1)
print("  [OK ] every parameter receives gradients")

# ─────────────────────────────────────────────────────────────────
banner("13. zflows.loss.loss_compile / loss_compile_beta sanity")
set_cache_size_limit(64) # headroom so any unintended recompile surfaces in the counter rather than getting silently throttled

torch.manual_seed(0)
nsf_lc = NSF(a=[-3.0, -3.0], b=[3.0, 3.0], bins=8, transforms=2, hidden_features=(32, 32)).to(device)
F_lc = nsf_lc.t()
u_lc = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0], device=device)
x_lc = torch.randn(64, 2, device=device)

# --- 13a. loss_compile: single-input fast path -------------------------
# The returned closure has signature `(x) -> scalar`; beta is either left
# at its default 1.0 or baked into the captured constants.
print()
print("  13a. loss_compile  — single-input fast path, no runtime beta")

# Default beta=1.0 path (beta absorbed into reverse_KL's default kwarg)
raw_default = zflows.loss.loss_compile(reverse_KL, u_lc, F_lc)
l_compiled_default = raw_default(x_lc).item()
l_raw_default = reverse_KL(x_lc, u_lc, F_lc).item()
print(f"  default beta=1.0:   compiled={l_compiled_default:.6f}, raw={l_raw_default:.6f}, "
      f"diff={abs(l_compiled_default - l_raw_default):.3e}")
if abs(l_compiled_default - l_raw_default) > 1e-3:
    print(f"  FAIL: loss_compile default-beta output drift")
    sys.exit(1)

# Baked-in beta path: pass beta as a captured constant.
for fixed_beta in (0.3, 0.7, 2.5):
    raw_fixed = zflows.loss.loss_compile(reverse_KL, u_lc, F_lc, fixed_beta)
    l_c = raw_fixed(x_lc).item()
    l_r = reverse_KL(x_lc, u_lc, F_lc, beta=fixed_beta).item()
    print(f"  baked  beta={fixed_beta}:  compiled={l_c:.6f}, raw={l_r:.6f}, "
          f"diff={abs(l_c - l_r):.3e}")
    if abs(l_c - l_r) > 1e-3:
        print(f"  FAIL: loss_compile baked-beta={fixed_beta} output drift")
        sys.exit(1)
print("  [OK ] loss_compile matches raw reverse_KL for default and baked-in beta")

# --- 13b. loss_compile_beta: runtime beta does NOT trigger recompiles ------
# The returned closure casts a Python `float` beta to a 0-d tensor before
# entering the compiled graph, so Dynamo treats beta as a dynamic input
# and a single artifact handles every value of beta. Sweeping a dozen
# distinct betas must NOT add any new graph traces.
print()
print("  13b. loss_compile_beta — runtime beta, single artifact across schedule")

loss_fn = zflows.loss.loss_compile_beta(reverse_KL, u_lc, F_lc)

# Warmup: pay the one-time compile cost on a representative beta.
_ = loss_fn(x_lc, 0.1).item()

# Sweep distinct betas and count newly-traced graphs.
betas = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
c0 = dynamo.utils.counters["stats"]["unique_graphs"]
losses = []
for b in betas:
    losses.append(loss_fn(x_lc, b).item())
delta = dynamo.utils.counters["stats"]["unique_graphs"] - c0
print(f"  unique_graphs delta across {len(betas)} betas: {delta}  (expect 0 — single compiled artifact)")
if delta != 0:
    print(f"  FAIL: Dynamo recompiled {delta} time(s); beta is leaking value-specialization")
    sys.exit(1)

# Numerical correctness: compiled loss must match raw reverse_KL at every beta.
for b, l_c in zip(betas, losses):
    l_r = reverse_KL(x_lc, u_lc, F_lc, beta=b).item()
    if abs(l_c - l_r) > 1e-3:
        print(f"  FAIL: beta={b}: compiled={l_c}, raw={l_r}, diff={abs(l_c - l_r):.3e}")
        sys.exit(1)
print(f"  [OK ] one compiled graph reused across {len(betas)} betas; outputs match raw reverse_KL")

# --- 13c. OTFlow under loss_compile / loss_compile_beta ---------------------
# OTFlow's closed-form trace runs inside an RK4-unrolled ODE. This checks
# the whole transform captures + compiles cleanly (no graph break on the
# Hessian-trace, no per-beta recompile, gradients still flow).
print()
print("  13c. loss_compile / loss_compile_beta on OTFlow (closed-form trace in RK4 ODE)")

torch.manual_seed(0)
otf_lc = OTFlow(dimension=2, hidden=32, layer=3, nt=6).to(device)
F_ot = otf_lc.t()

# 13c-i: compiled reverse_KL matches eager.
raw_ot = zflows.loss.loss_compile(reverse_KL, u_lc, F_ot)
l_c = raw_ot(x_lc).item()
l_r = reverse_KL(x_lc, u_lc, F_ot).item()
print(f"  loss_compile:  compiled={l_c:.6f}, eager={l_r:.6f}, diff={abs(l_c - l_r):.3e}")
if abs(l_c - l_r) > 1e-3:
    print("  FAIL: OTFlow loss_compile output drift")
    sys.exit(1)

# 13c-ii: loss_compile_beta sweep adds no new graphs (the recompile bug).
loss_ot = zflows.loss.loss_compile_beta(reverse_KL, u_lc, F_ot)
_ = loss_ot(x_lc, 0.1).item()  # warmup compile
c0 = dynamo.utils.counters["stats"]["unique_graphs"]
betas_ot = [0.2, 0.5, 1.0, 2.0, 3.0]
losses_ot = [loss_ot(x_lc, b).item() for b in betas_ot]
delta = dynamo.utils.counters["stats"]["unique_graphs"] - c0
print(f"  loss_compile_beta: unique_graphs delta across {len(betas_ot)} betas: {delta}  (expect 0)")
if delta != 0:
    print(f"  FAIL: OTFlow loss_compile_beta recompiled {delta} time(s)")
    sys.exit(1)
for b, l_cb in zip(betas_ot, losses_ot):
    if abs(l_cb - reverse_KL(x_lc, u_lc, F_ot, beta=b).item()) > 1e-3:
        print(f"  FAIL: OTFlow loss_compile_beta drift at beta={b}")
        sys.exit(1)

# 13c-iii: backprop through the compiled loss reaches every Phi parameter.
otf_lc.zero_grad()
raw_ot(x_lc).backward()
n_params = sum(1 for _ in otf_lc.parameters())
n_grads = sum(1 for p in otf_lc.parameters() if p.grad is not None)
print(f"  backprop through compiled loss: {n_grads}/{n_params} params have grad")
if n_grads != n_params:
    print(f"  FAIL: OTFlow compiled-loss backprop missed {n_params - n_grads} params")
    sys.exit(1)
print("  [OK ] OTFlow compiles (no graph break / no per-beta recompile) and backprops")

# ─────────────────────────────────────────────────────────────────
banner("14. RealNVP with linear mixing (rotation / lu)")

for kind in ("rotation", "lu"):
    print()
    print(f"  --- mixing={kind!r} ---")

    # 14a — round-trip
    torch.manual_seed(0)
    flow_mix = RealNVP(dimension=4, transforms=4, mixing=kind, hidden_features=(32, 32)).to(device)
    flow_mix.eval()
    x_mix = torch.randn(256, 4, device=device)
    with torch.no_grad():
        F_mix = flow_mix.t()
        y_mix = F_mix(x_mix)
        x_back = F_mix.inv(y_mix)
    assert_close(x_back, x_mix, atol=1e-5, name=f"14a  F.inv(F(x)) ≈ x   (mixing={kind})")

    # 14b — .zeros() → identity
    flow_mix.zeros()
    with torch.no_grad():
        F_mix = flow_mix.t()
        y_mix, ladj_mix = F_mix.call_and_ladj(x_mix)
    assert_close(y_mix, x_mix, atol=1e-5, name=f"14b  zeros(): y ≈ x    (mixing={kind})")
    assert_close(ladj_mix, torch.zeros_like(ladj_mix), atol=1e-5, name=f"14b  zeros(): ladj ≈ 0 (mixing={kind})")

    # 14c — call_and_ladj matches autograd slogdet on a fresh, non-identity flow
    torch.manual_seed(1)
    flow_mix = RealNVP(dimension=4, transforms=4, mixing=kind, hidden_features=(32, 32)).to(device)
    # nudge mixing weights so log|det| is non-trivial (especially for lu)
    for layer in flow_mix._layers:
        if isinstance(layer, LinearMixingTransform):
            with torch.no_grad():
                layer.weight.add_(0.1 * torch.randn_like(layer.weight))
    flow_mix.train()
    F_mix = flow_mix.t()
    x1 = torch.randn(1, 4, device=device)
    with torch.no_grad():
        _, ladj_mix = F_mix.call_and_ladj(x1)
    expected = exact_log_abs_det_jacobian(F_mix, x1)
    assert_close(ladj_mix.squeeze(), expected, atol=1e-4, name=f"14c  call_and_ladj ≈ slogdet (mixing={kind})")

    # 14d — every parameter receives a gradient
    flow_mix.zero_grad()
    x_b = torch.randn(16, 4, device=device)
    y_b, ladj_b = flow_mix.t().call_and_ladj(x_b)
    (y_b.pow(2).sum() - ladj_b.sum()).backward()
    n_params = sum(1 for _ in flow_mix.parameters())
    n_grads = sum(1 for p in flow_mix.parameters() if p.grad is not None)
    print(f"  14d  {n_grads}/{n_params} parameter tensors have non-None grad   (mixing={kind})")
    if n_grads != n_params:
        print(f"  FAIL: mixing={kind}: missing grads on {n_params - n_grads} params")
        sys.exit(1)

    # 14f — captured F stays consistent under parameter mutation.
    # This is the central capture-once contract relied on by
    # zflows.loss.loss_compile / loss_compile_beta. Mixing layers must reread
    # their weight on every forward; eager-cached R / L / U regress here.
    torch.manual_seed(99)
    flow_cap = RealNVP(dimension=4, transforms=4, mixing=kind, hidden_features=(32, 32)).to(device)
    F_captured = flow_cap.t()
    x_cap = torch.randn(16, 4, device=device)
    # Simulate one optimizer.step() of in-place parameter updates.
    with torch.no_grad():
        for p in flow_cap.parameters():
            p.add_(0.1 * torch.randn_like(p))
    with torch.no_grad():
        y_cap, ladj_cap = F_captured.call_and_ladj(x_cap)
        y_fresh, ladj_fresh = flow_cap.t().call_and_ladj(x_cap)
    assert_close(y_cap,    y_fresh,    atol=1e-6, name=f"14f  captured F sees param updates: y    (mixing={kind})")
    assert_close(ladj_cap, ladj_fresh, atol=1e-6, name=f"14f  captured F sees param updates: ladj (mixing={kind})")

# 14e — lu mixing: after one optimizer step weights diverge from I and ladj is non-zero
print()
print("  --- 14e  lu mixing contributes a non-trivial log|det| after one Adam step ---")
torch.manual_seed(42)
flow_lu = RealNVP(dimension=4, transforms=4, mixing="lu", hidden_features=(32, 32)).to(device)
mixing_layers = [layer for layer in flow_lu._layers if isinstance(layer, LinearMixingTransform)]
print(f"  built {len(mixing_layers)} lu mixing layers (expected 3 = transforms-1)")
opt = torch.optim.Adam(flow_lu.parameters(), lr=0.1)
x_b = torch.randn(64, 4, device=device)
y_b, ladj_b = flow_lu.t().call_and_ladj(x_b)
# per-sample reverse-KL-style loss: 0.5 * ||y||^2 - ladj
loss_b = (0.5 * y_b.pow(2).sum(-1) - ladj_b).mean()
opt.zero_grad(); loss_b.backward(); opt.step()
I_d = torch.eye(4, device=device)
max_drift = max((layer.weight - I_d).abs().max().item() for layer in mixing_layers)
x2 = torch.randn(1, 4, device=device)
with torch.no_grad():
    _, ladj_after = flow_lu.t().call_and_ladj(x2)
print(f"  max |LU.weight - I| after step = {max_drift:.3e}")
print(f"  total flow ladj on test point   = {ladj_after.item():.3e}")
if max_drift <= 1e-3:
    print(f"  FAIL: LU weights did not move from identity (drift={max_drift})")
    sys.exit(1)
if abs(ladj_after.item()) <= 1e-3:
    print(f"  FAIL: LU log|det| stayed near zero ({ladj_after.item()})")
    sys.exit(1)
print("  [OK ] lu mixing layers produce a non-trivial log|det| after training")

# 14h — LULinearTransform near-singular safety: with |diag(L)| driven well
# below the internal clamp floor (1e-12), call_and_ladj must still return
# a finite log|det| (large negative, not -inf) instead of blowing the loss
# to NaN. Healthy LU (== I) still produces ladj exactly 0.
print()
print("  --- 14h  LULinearTransform near-singular |diag(L)| ladj stays finite ---")
from zflows.core.transforms import LULinearTransform
x14h = torch.randn(8, 4, device=device)
# (1) healthy LU = I -> ladj == 0
t_h = LULinearTransform(torch.eye(4, device=device))
_, ladj_h = t_h.call_and_ladj(x14h)
assert ladj_h.abs().max().item() == 0.0, f"healthy LU ladj should be 0, got {ladj_h.max().item()}"
# (2) near-singular LU: diagonal entry set far below clamp floor
LU_bad = torch.eye(4, device=device).clone()
LU_bad[2, 2] = 1e-20
t_bad = LULinearTransform(LU_bad)
y_bad, ladj_bad = t_bad.call_and_ladj(x14h)
assert torch.isfinite(ladj_bad).all().item(), f"near-singular ladj should be finite, got {ladj_bad}"
assert torch.isfinite(y_bad).all().item(), f"near-singular y should be finite, got {y_bad}"
print(f"  healthy LU ladj:           {ladj_h.max().item():.4e}  (expect 0)")
print(f"  near-singular LU ladj:     {ladj_bad.min().item():.4e}  (large -ve, finite)")
print(f"  [OK ] LULinearTransform clamp keeps ladj / y finite when diag(L) -> 0")

# ─────────────────────────────────────────────────────────────────
banner("14g. OT_loss = reverse_KL + OT regularizers (OTFlow)")
# OT_loss integrates the 4-channel augmented ODE (x, ladj, transport, HJB).
# With alpha_C = alpha_R = 0 it must reduce exactly to reverse_KL on the same
# flow; the regularizers are non-negative and additively decoupled.
torch.manual_seed(7)
otf_g = OTFlow(dimension=2, hidden=32, layer=3, nt=8).to(device)
u_g = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0], device=device)
x_g = torch.randn(64, 2, device=device)

# decoupling: alpha_C = alpha_R = 0  ==  reverse_KL
l_ot0 = OT_loss(x_g, u_g, otf_g, beta=1.0, alpha_C=0.0, alpha_R=0.0)
l_rk = reverse_KL(x_g, u_g, otf_g.t())
assert_close(l_ot0, l_rk, atol=1e-5,
             name="14g  OT_loss(alpha_C=alpha_R=0) == reverse_KL")

# transport / HJB channels are non-negative integrals
F_g = otf_g.t().transforms[0]
_, _, C_g, R_g = F_g.call_full(x_g)
assert (C_g >= 0).all() and (R_g >= 0).all(), "OT cost channels must be >= 0"
print(f"  transport cost C: mean={C_g.mean().item():.4f}, min={C_g.min().item():.3e} (>=0)")
print(f"  HJB residual  R: mean={R_g.mean().item():.4f}, min={R_g.min().item():.3e} (>=0)")

# full loss is finite and the regularizers actually move it
l_ot_full = OT_loss(x_g, u_g, otf_g, beta=1.0, alpha_C=1.0, alpha_R=1.0)
assert torch.isfinite(l_ot_full), "OT_loss returned non-finite"
print(f"  OT_loss(full)={l_ot_full.item():.6f}, reverse_KL={l_rk.item():.6f}")
print("  [OK ] OT_loss decouples to reverse_KL and adds non-negative OT regularizers")

# ─────────────────────────────────────────────────────────────────
banner("15. Captured F sees param updates (NSF, CNF, OTFlow)")
# Same capture-once-then-mutate contract as §14f, but for NSF and CNF
# rather than RealNVP's mixing layers. Catches any regression where a
# subclass of Transform snapshots derived state in __init__ instead of
# re-reading the underlying nn.Parameter via attribute access.

# NSF — MaskedAutoregressiveTransform.meta(x) is the lazy hook.
torch.manual_seed(100)
flow_nsf = NSF(a=[-3.0]*4, b=[3.0]*4, bins=8, transforms=2, hidden_features=(32, 32))
F_nsf = flow_nsf.t()
x15 = torch.randn(16, 4) * 0.5  # keep inside the box
with torch.no_grad():
    for p in flow_nsf.parameters():
        p.add_(0.05 * torch.randn_like(p))
with torch.no_grad():
    y_cap, ladj_cap = F_nsf.call_and_ladj(x15)
    y_fresh, ladj_fresh = flow_nsf.t().call_and_ladj(x15)
assert_close(y_cap,    y_fresh,    atol=1e-6, name="15a  NSF captured F sees param updates: y")
assert_close(ladj_cap, ladj_fresh, atol=1e-6, name="15a  NSF captured F sees param updates: ladj")

# CNF — FreeFormJacobianTransform holds a reference to the velocity module
# whose parameters update in place; the ODE integrator should reread them.
torch.manual_seed(101)
flow_cnf = CNF(dimension=2, frequency=3, hidden_features=(16, 16))
F_cnf = flow_cnf.t()
x15c = torch.randn(8, 2) * 0.5
with torch.no_grad():
    for p in flow_cnf.parameters():
        p.add_(0.05 * torch.randn_like(p))
with torch.no_grad():
    y_cap, ladj_cap = F_cnf.call_and_ladj(x15c)
    y_fresh, ladj_fresh = flow_cnf.t().call_and_ladj(x15c)
assert_close(y_cap,    y_fresh,    atol=1e-4, name="15b  CNF captured F sees param updates: y")
assert_close(ladj_cap, ladj_fresh, atol=1e-4, name="15b  CNF captured F sees param updates: ladj")

# OTFlow — OTFlowTransform holds a reference to OTPhi; the RK4 drift reads
# its parameters via trHess on every step, so a captured transform must
# track in-place updates. A snapshot of any Phi-derived tensor in __init__
# would make y / ladj here stale.
torch.manual_seed(102)
flow_otf = OTFlow(dimension=3, hidden=16, layer=3, nt=8)
F_otf = flow_otf.t()
x15o = torch.randn(8, 3) * 0.5
with torch.no_grad():
    for p in flow_otf.parameters():
        p.add_(0.05 * torch.randn_like(p))
with torch.no_grad():
    y_cap, ladj_cap = F_otf.call_and_ladj(x15o)
    y_fresh, ladj_fresh = flow_otf.t().call_and_ladj(x15o)
assert_close(y_cap,    y_fresh,    atol=1e-6, name="15c  OTFlow captured F sees param updates: y")
assert_close(ladj_cap, ladj_fresh, atol=1e-6, name="15c  OTFlow captured F sees param updates: ladj")

# ─────────────────────────────────────────────────────────────────
print()
print("All flow verification checks passed.")
