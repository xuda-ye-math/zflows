# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Verification script for every zflows.Flow subclass.

Run from the repo root:  python -m tests._verify_flows

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
  13. zflows.loss.compile_raw / compile_beta sanity:
      (a) compile_raw output matches raw reverse_KL (default beta + baked beta);
      (b) compile_beta with runtime beta does not trigger recompiles.
"""

import sys
from math import pi

import torch
import torch._dynamo as dynamo

from zflows.core.transforms import FreeFormJacobianTransform, CircularShiftTransform
from zflows.flow import NSF, NCSF, CNF, RealNVP, Flow, ComposedTransform
import zflows.loss
from zflows.loss import reverse_KL
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
      f"{len(realnvp._coupling)} coupling layers")

cnf = CNF(dimension=2, frequency=3, hidden_features=(32, 32)).to(device)
print(f"  CNF d=2: {sum(p.numel() for p in cnf.parameters())} params")

cnf_hutch = CNF(dimension=2, frequency=3, hidden_features=(32, 32), exact=False).to(device)
print(f"  CNF d=2 (Hutchinson): {sum(p.numel() for p in cnf_hutch.parameters())} params")

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
realnvp.eval()
cnf.eval()

for name, flow, sample_fn, atol in [
    ("NSF",      nsf,      lambda f: f.a + (f.b - f.a) * torch.rand(5, f.a.size(0)),  1e-3),
    ("NSF asym", nsf_asym, lambda f: f.a + (f.b - f.a) * torch.rand(5, f.a.size(0)),  1e-3),
    ("NCSF",     ncsf,     lambda f: f.a + (f.b - f.a) * torch.rand(5, f.a.size(0)),  1e-3),
    ("RealNVP",  realnvp,  lambda _: torch.randn(5, 4),                               1e-4),
    ("CNF",      cnf,      lambda _: torch.randn(5, 2) * 0.5,                         5e-3),
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
banner("13. zflows.loss.compile_raw / compile_beta sanity")
set_cache_size_limit(64) # headroom so any unintended recompile surfaces in the counter rather than getting silently throttled

torch.manual_seed(0)
nsf_lc = NSF(a=[-3.0, -3.0], b=[3.0, 3.0], bins=8, transforms=2, hidden_features=(32, 32)).to(device)
F_lc = nsf_lc.t()
u_lc = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0], device=device)
x_lc = torch.randn(64, 2, device=device)

# --- 13a. compile_raw: single-input fast path -------------------------
# The returned closure has signature `(x) -> scalar`; beta is either left
# at its default 1.0 or baked into the captured constants.
print()
print("  13a. compile_raw  — single-input fast path, no runtime beta")

# Default beta=1.0 path (beta absorbed into reverse_KL's default kwarg)
raw_default = zflows.loss.compile_raw(reverse_KL, u_lc, F_lc)
l_compiled_default = raw_default(x_lc).item()
l_raw_default = reverse_KL(x_lc, u_lc, F_lc).item()
print(f"  default beta=1.0:   compiled={l_compiled_default:.6f}, raw={l_raw_default:.6f}, "
      f"diff={abs(l_compiled_default - l_raw_default):.3e}")
if abs(l_compiled_default - l_raw_default) > 1e-3:
    print(f"  FAIL: compile_raw default-beta output drift")
    sys.exit(1)

# Baked-in beta path: pass beta as a captured constant.
for fixed_beta in (0.3, 0.7, 2.5):
    raw_fixed = zflows.loss.compile_raw(reverse_KL, u_lc, F_lc, fixed_beta)
    l_c = raw_fixed(x_lc).item()
    l_r = reverse_KL(x_lc, u_lc, F_lc, beta=fixed_beta).item()
    print(f"  baked  beta={fixed_beta}:  compiled={l_c:.6f}, raw={l_r:.6f}, "
          f"diff={abs(l_c - l_r):.3e}")
    if abs(l_c - l_r) > 1e-3:
        print(f"  FAIL: compile_raw baked-beta={fixed_beta} output drift")
        sys.exit(1)
print("  [OK ] compile_raw matches raw reverse_KL for default and baked-in beta")

# --- 13b. compile_beta: runtime beta does NOT trigger recompiles ------
# The returned closure casts a Python `float` beta to a 0-d tensor before
# entering the compiled graph, so Dynamo treats beta as a dynamic input
# and a single artifact handles every value of beta. Sweeping a dozen
# distinct betas must NOT add any new graph traces.
print()
print("  13b. compile_beta — runtime beta, single artifact across schedule")

loss_fn = zflows.loss.compile_beta(reverse_KL, u_lc, F_lc)

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

# ─────────────────────────────────────────────────────────────────
print()
print("All flow verification checks passed.")
