# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Verification script for zflows.potential machinery.

Run from the repo root:  python -m tests._verify_potential

Consolidates _enable_eval and _linear_combination into one banner-separated
harness. Sections:

   A. Potential.enable_eval() opt-in fast path:
        - structural state (_eval_fn populated, idempotent, .eval() switch);
        - .eval(x) gating;
        - langevin(adjust=True) routes through _eval_fn;
        - fallback path with no enable_eval();
        - numerical equivalence (same seed -> same samples).

   B. Linear_Combination potential under Langevin sampling:
        - ulc(x) = c0 * U0(x) + c1 * U1(x)  is integrable, well-formed;
        - samples have sensible moments (per-axis mean / std finite);
        - scatter plot saved to tests/_verify_potential.png for visual
          verification;
        - B.3 / B.4: coeffs always normalise to list[float] across every
          input form (list, tuple, 1-d Tensor, requires_grad=True, None),
          and `set_coeffs` updates in place, returns self, rejects None
          and length mismatches.

   C. Gaussian.samples(N, beta=...):
        - default vs beta=1.0 is byte-identical;
        - empirical variance contracts as 1/beta (tempered N(mean, var/beta)).

   D. potential_from(fn) / potential_instance_from(fn):
        - potential_from returns a Potential subclass; instance forwards to fn;
        - enable_grad() + enable_eval() chain through;
        - .grad(x) matches autograd of fn(x).sum() w.r.t. x;
        - potential_instance_from returns a ready-to-use instance
          equivalent to potential_from(fn)().
"""

from pathlib import Path

import torch

from zflows.potential import (
    Gaussian,
    Linear_Combination,
    Potential,
    potential_from,
    potential_instance_from,
)
from zflows.utils import langevin, set_cache_size_limit, suppress_warnings

# Silence Triton/Inductor/Dynamo/Python warnings; cache headroom for the
# Gaussian + Linear_Combination .enable_grad / .enable_eval compiles.
suppress_warnings()
set_cache_size_limit(32)


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
HERE = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════
# A. Potential.enable_eval()
# ══════════════════════════════════════════════════════════════════
banner("A. Potential.enable_eval()  /  Potential.eval(x)")

ITERS = 5
STEP = 1e-2

# A.1 — structural state
section("A.1  enable_eval populates _eval_fn; eval() preserves nn.Module switch")
u = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)
print(f"  before enable_eval: _eval_fn is None = {u._eval_fn is None}")
assert u._eval_fn is None
u.enable_eval()
print(f"  after  enable_eval: _eval_fn is None = {u._eval_fn is None}")
assert u._eval_fn is not None
# idempotent: second call must not rebuild
fn_first = u._eval_fn
u.enable_eval()
print(f"  second enable_eval keeps same closure: {u._eval_fn is fn_first}")
assert u._eval_fn is fn_first
# .eval() with no argument: standard nn.Module switch (returns self, sets training=False)
u.train()
assert u.training is True
ret = u.eval()  # no x -> nn.Module.eval()
print(f"  u.eval() (no x) returns self: {ret is u}")
print(f"  u.eval() (no x) sets training=False: {u.training is False}")
assert ret is u and u.training is False
print("  [OK ] structural state correct")

# A.2 — u.eval(x) gating
section("A.2  u.eval(x) raises a clear RuntimeError without enable_eval()")
u2 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)
x = torch.randn(8, 2, device=device)
try:
    u2.eval(x)
    raised = False
except RuntimeError as e:
    raised = True
    print(f"  raised: {e}")
assert raised, "u.eval(x) must raise without enable_eval()"
print("  [OK ] rejected as expected")

# A.3 — langevin(adjust=True) routes through _eval_fn (counted = 2 * iters)
section("A.3  MALA routes U(x), U(y) through u._eval_fn  (count == 2 * iters)")
u3 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad().enable_eval()
# Wrap _eval_fn in a call-counter. We do NOT touch .forward, so the
# compiled .grad path is left intact (otherwise dynamo would see the
# Python-int counter inside the gradient closure and recompile every
# iteration, polluting the test).
class Counter:
    def __init__(self, fn): self.fn, self.n = fn, 0
    def __call__(self, x):
        self.n += 1
        return self.fn(x)
counter = Counter(u3._eval_fn)
u3._eval_fn = counter
x0 = torch.randn(64, 2, device=device)
_ = langevin(x0, potential=u3, step=STEP, iters=ITERS, adjust=True)
print(f"  _eval_fn calls during MALA: {counter.n}  (expected 2 * iters = {2 * ITERS})")
assert counter.n == 2 * ITERS, f"_eval_fn was hit {counter.n} times, expected {2*ITERS}"
print("  [OK ] _eval_fn count matches")

# A.4 — fallback path
section("A.4  MALA falls back to potential(x) when _eval_fn is None")
u4 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad()
print(f"  _eval_fn is None: {u4._eval_fn is None}")
assert u4._eval_fn is None
x0 = torch.randn(64, 2, device=device)
y = langevin(x0, potential=u4, step=STEP, iters=ITERS, adjust=True)
print(f"  langevin output shape: {tuple(y.shape)}")
assert y.shape == (64, 2)
print("  [OK ] fallback path runs without error")

# A.5 — numerical equivalence (fast path == fallback under same seed)
section("A.5  same seed -> identical output with vs. without enable_eval")
torch.manual_seed(42)
u_a = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad().enable_eval()
torch.manual_seed(42)
u_b = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad()
torch.manual_seed(7)
x0 = torch.randn(64, 2, device=device)
torch.manual_seed(123)
y_a = langevin(x0, potential=u_a, step=STEP, iters=ITERS, adjust=True)
torch.manual_seed(123)
y_b = langevin(x0, potential=u_b, step=STEP, iters=ITERS, adjust=True)
err = (y_a - y_b).abs().max().item()
print(f"  max |y_with_eval - y_without_eval| = {err:.3e}")
assert err < 1e-4, f"fast path disagrees with fallback: {err}"
print("  [OK ] fast path is a pure speed optimization")


# ══════════════════════════════════════════════════════════════════
# B. Linear_Combination potential
# ══════════════════════════════════════════════════════════════════
banner("B. Linear_Combination potential under Langevin sampling")

# U0(x) = (x1^2 + x2^2) / 2 — isotropic harmonic
class U0(Potential):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2)

# U1(x) = 2 * cos(x1) — cosine modulation on x1 only
class U1(Potential):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.cos(x[:, 0])

u0 = U0().to(device)
u1 = U1().to(device)

# ulc(x) = 1.0 * U0(x) + 1.2 * U1(x)
# = 0.5 * (x1^2 + x2^2) + 2.4 * cos(x1)
# x2 ~ standard normal; x1 modulated by cosine (still even, mean ≈ 0).
ulc = Linear_Combination([u0, u1], [1.0, 1.2])
ulc.enable_grad()

# Langevin chain
section("B.1  Run Langevin on ulc, collect snapshots, check moments")
N = 32                # independent trajectories
ITERS_LC = 100000     # total Langevin iterations
RECORD_EVERY = 100    # snapshot stride
STEP_LC = 1e-2

torch.manual_seed(0)
x = torch.randn(N, 2).to(device)

n_chunks = ITERS_LC // RECORD_EVERY
snapshots = []
for _ in range(n_chunks):
    x = langevin(x, potential=ulc, step=STEP_LC, iters=RECORD_EVERY)
    snapshots.append(x.detach())
samples = torch.cat(snapshots, dim=0)  # [n_chunks * N, 2]
print(f"  collected samples shape: {tuple(samples.shape)}")

# Sanity asserts: every sample finite; symmetry-preserving moments are
# small (both axes have an even potential so the means should be ≈ 0).
assert torch.isfinite(samples).all(), "Langevin on ulc produced NaN/Inf"
mean_x = samples.mean(0).tolist()
std_x  = samples.std(0).tolist()
print(f"  per-axis mean = {[f'{v:+.3f}' for v in mean_x]}   (target ~[0, 0])")
print(f"  per-axis std  = {[f'{v:.3f}' for v in std_x]}    (target ~[*,  1])")
assert abs(mean_x[0]) < 0.1, f"x1 mean off (even potential => ~0): {mean_x[0]}"
assert abs(mean_x[1]) < 0.1, f"x2 mean off: {mean_x[1]}"
assert abs(std_x[1] - 1.0) < 0.05, f"x2 std off (expect ~1): {std_x[1]}"

# Visual sanity: scatter plot
section("B.2  Save scatter plot for visual sanity")
import matplotlib.pyplot as plt
samples_np = samples.cpu().numpy()
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(samples_np[:, 0], samples_np[:, 1], s=1, alpha=0.3, color="darkblue")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_aspect("equal")
ax.set_title(r"Langevin samples of $U_{\mathrm{lc}} = c_0 U_0 + c_1 U_1$")
plt.tight_layout()
png = HERE / "_verify_potential.png"
plt.savefig(png, dpi=200)
plt.close(fig)
print(f"  scatter saved → {png}")

ulc.release()
u0.release()
u1.release()

# B.3 — Linear_Combination normalises every coeff input to list[float].
# Whatever the user passes (list, tuple, 1-d Tensor, requires_grad=True
# tensor, None), `self.coeffs` must come out as a plain Python list of
# floats — no buffers, no grad tracking, no device pinning. None defaults
# to uniform 1/N.
section("B.3  Linear_Combination coeffs always normalise to list[float]")
u_a = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0], device=device)
u_b = Gaussian(mean=[3.0, 3.0], variance=[1.0, 1.0], device=device)
# (1) plain tensor input -> list[float].
lc_t = Linear_Combination([u_a, u_b], torch.tensor([0.3, 0.7], device=device))
assert isinstance(lc_t.coeffs, list) and all(isinstance(c, float) for c in lc_t.coeffs), \
    f"tensor input should normalise to list[float]; got {type(lc_t.coeffs).__name__}"
assert abs(lc_t.coeffs[0] - 0.3) < 1e-6 and abs(lc_t.coeffs[1] - 0.7) < 1e-6
# (2) requires_grad=True tensor is accepted too — detached on the way in.
lc_g = Linear_Combination(
    [u_a, u_b], torch.tensor([0.3, 0.7], device=device, requires_grad=True)
)
assert isinstance(lc_g.coeffs, list) and all(isinstance(c, float) for c in lc_g.coeffs)
# (3) coeffs=None -> uniform 1/N.
lc_d = Linear_Combination([u_a, u_b])
assert lc_d.coeffs == [0.5, 0.5], f"default coeffs should be uniform 1/N, got {lc_d.coeffs}"
print(f"  tensor input:                  coeffs = {lc_t.coeffs}")
print(f"  requires_grad=True tensor:     coeffs = {lc_g.coeffs}")
print(f"  default (None) for N=2:        coeffs = {lc_d.coeffs}")
print("  [OK ] coeffs always normalised to list[float]")


# B.4 — Linear_Combination.set_coeffs replaces the weights in place across
# every accepted input form (list, tuple, 1-d Tensor, requires_grad=True
# tensor), keeps `self.coeffs` as a plain list[float], returns self for
# chaining, rejects None and length mismatches, and the next forward
# call reflects the new mix exactly.
section("B.4  Linear_Combination.set_coeffs in-place update")
u_a2 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0], device=device)
u_b2 = Gaussian(mean=[3.0, 3.0], variance=[1.0, 1.0], device=device)
lc = Linear_Combination([u_a2, u_b2])
assert lc.coeffs == [0.5, 0.5], f"init should be uniform 1/N, got {lc.coeffs}"

# (1) list / tuple / tensor inputs all land as list[float].
lc.set_coeffs([0.2, 0.8])
assert lc.coeffs == [0.2, 0.8]
lc.set_coeffs((0.4, 0.6))
assert lc.coeffs == [0.4, 0.6]
lc.set_coeffs(torch.tensor([0.1, 0.9], device=device))
assert isinstance(lc.coeffs, list) and all(isinstance(c, float) for c in lc.coeffs)
assert abs(lc.coeffs[0] - 0.1) < 1e-6 and abs(lc.coeffs[1] - 0.9) < 1e-6

# (2) requires_grad=True tensor is detached, never stored as a Tensor.
lc.set_coeffs(torch.tensor([0.3, 0.7], device=device, requires_grad=True))
assert isinstance(lc.coeffs, list)
assert not any(isinstance(c, torch.Tensor) for c in lc.coeffs)

# (3) returns self -> chainable.
ret = lc.set_coeffs([0.5, 0.5])
assert ret is lc, "set_coeffs should return self for chaining"

# (4) None is rejected.
raised_none = False
try:
    lc.set_coeffs(None)
except AssertionError as e:
    raised_none = True
    msg_none = str(e)
assert raised_none and "None" in msg_none, \
    f"set_coeffs(None) should raise mentioning None; got: {msg_none if raised_none else 'no error'}"

# (5) length mismatch is rejected.
raised_len = False
try:
    lc.set_coeffs([0.3, 0.3, 0.4])
except AssertionError as e:
    raised_len = True
    msg_len = str(e)
assert raised_len and "2" in msg_len and "3" in msg_len, \
    f"set_coeffs length-mismatch should mention sizes; got: {msg_len if raised_len else 'no error'}"

# (6) forward reflects the update — set_coeffs([1, 0]) collapses to U_a alone.
x_lc = torch.randn(64, 2, device=device)
lc.set_coeffs([1.0, 0.0])
out_collapsed = lc(x_lc)
out_ua = u_a2(x_lc)
err_collapsed = (out_collapsed - out_ua).abs().max().item()
assert err_collapsed == 0.0, f"forward after set_coeffs([1, 0]) should equal U_a; err = {err_collapsed}"

# (7) re-mixing is bit-identical to constructing a fresh Linear_Combination
#     with the same coeffs (i.e. set_coeffs leaves no stale state behind).
lc.set_coeffs([0.35, 0.65])
lc_fresh = Linear_Combination([u_a2, u_b2], [0.35, 0.65])
err_fresh = (lc(x_lc) - lc_fresh(x_lc)).abs().max().item()
assert err_fresh == 0.0, f"set_coeffs vs fresh ctor mismatch: {err_fresh}"

print(f"  list / tuple / tensor / grad tensor -> list[float]:  OK")
print(f"  returns self (chainable):                            OK")
print(f"  None rejected, length mismatch rejected:             OK")
print(f"  set_coeffs([1, 0]) collapses forward to U_a:         err = {err_collapsed:.3e}")
print(f"  set_coeffs == fresh ctor (no stale state):           err = {err_fresh:.3e}")
print("  [OK ] set_coeffs updates in place and forward reflects it")


# B.5 — set_coeffs invalidates `_grad_fn` / `_eval_fn` and respects
# `enable_grad` / `enable_eval`. Sequence:
#   1. enable_grad + enable_eval populate both compiled paths;
#   2. set_coeffs(..., enable_grad=False, enable_eval=False) -> both
#      fields are cleared, .grad(x) / .eval(x) raise until re-enabled;
#   3. set_coeffs(..., enable_grad=True)  -> _grad_fn rebuilt against
#      the new coefficients; _eval_fn stays None;
#   4. set_coeffs(..., enable_eval=True)  -> _eval_fn rebuilt against
#      the new coefficients; _grad_fn stays None;
#   5. set_coeffs(..., enable_grad=True, enable_eval=True) -> both
#      rebuilt; .grad(x) matches autograd, .eval(x) matches forward.
section("B.5  set_coeffs clears compiled fast paths; enable_grad / enable_eval rebuild them")
u_c = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0], device=device)
u_d = Gaussian(mean=[3.0, 3.0], variance=[1.0, 1.0], device=device)
lc_e = Linear_Combination([u_c, u_d], [0.2, 0.8])
lc_e.enable_grad().enable_eval()
assert lc_e._grad_fn is not None and lc_e._eval_fn is not None, \
    "enable_grad + enable_eval should populate both fast paths"

# (1) defaults -> both fields cleared, .grad / .eval raise.
lc_e.set_coeffs([0.7, 0.3])
assert lc_e._grad_fn is None, "set_coeffs should clear _grad_fn by default"
assert lc_e._eval_fn is None, "set_coeffs should clear _eval_fn by default"
x_e = torch.randn(8, 2, device=device)
raised_grad = False
try:
    lc_e.grad(x_e)
except RuntimeError:
    raised_grad = True
raised_eval = False
try:
    lc_e.eval(x_e)
except RuntimeError:
    raised_eval = True
assert raised_grad and raised_eval, \
    "after set_coeffs() with both flags False, .grad(x) and .eval(x) should raise"

# (2) enable_grad=True only -> _grad_fn rebuilt, _eval_fn stays None.
lc_e.set_coeffs([0.4, 0.6], enable_grad=True)
assert lc_e._grad_fn is not None, "set_coeffs(enable_grad=True) should rebuild _grad_fn"
assert lc_e._eval_fn is None, \
    "enable_grad=True alone must not rebuild _eval_fn"

# Gradient must reflect the NEW coefficients, not the stale [0.2, 0.8].
x_e = torch.randn(64, 2, device=device, requires_grad=True)
fresh_46 = Linear_Combination([u_c, u_d], [0.4, 0.6])
g_via_fast = lc_e.grad(x_e.detach()).clone()                            # compiled path
g_via_auto = torch.autograd.grad(fresh_46(x_e).sum(), x_e)[0].detach()  # reference
err_grad = (g_via_fast - g_via_auto).abs().max().item()
assert err_grad < 1e-5, f"compiled grad doesn't match new mix: err = {err_grad}"

# (3) enable_eval=True only -> _eval_fn rebuilt, _grad_fn dropped again.
lc_e.set_coeffs([0.1, 0.9], enable_eval=True)
assert lc_e._eval_fn is not None, "set_coeffs(enable_eval=True) should rebuild _eval_fn"
assert lc_e._grad_fn is None, \
    "enable_eval=True alone must not rebuild _grad_fn"
fresh_19 = Linear_Combination([u_c, u_d], [0.1, 0.9])
v_via_fast = lc_e.eval(x_e.detach())
v_via_fwd  = fresh_19(x_e.detach())
err_eval_only = (v_via_fast - v_via_fwd).abs().max().item()
assert err_eval_only < 1e-5, f"rebuilt .eval(x) drifts from new mix: {err_eval_only}"

# (4) both flags True -> both compiled paths populated with the new coeffs.
lc_e.set_coeffs([0.55, 0.45], enable_grad=True, enable_eval=True)
assert lc_e._grad_fn is not None and lc_e._eval_fn is not None, \
    "both flags True should rebuild both fast paths"
fresh_55 = Linear_Combination([u_c, u_d], [0.55, 0.45])
g_both = lc_e.grad(x_e.detach()).clone()
g_ref  = torch.autograd.grad(fresh_55(x_e).sum(), x_e)[0].detach()
err_grad_both = (g_both - g_ref).abs().max().item()
v_both = lc_e.eval(x_e.detach())
v_ref  = fresh_55(x_e.detach())
err_eval_both = (v_both - v_ref).abs().max().item()
assert err_grad_both < 1e-5 and err_eval_both < 1e-5, \
    f"both rebuilt; grad err = {err_grad_both}, eval err = {err_eval_both}"

print(f"  defaults (both False)  -> both cleared, .grad/.eval raise:  OK")
print(f"  enable_grad=True only  -> .grad rebuilt;  err = {err_grad:.3e}")
print(f"  enable_eval=True only  -> .eval rebuilt;  err = {err_eval_only:.3e}")
print(f"  both True              -> grad err = {err_grad_both:.3e}, eval err = {err_eval_both:.3e}")
print("  [OK ] set_coeffs invalidates fast paths; flags refresh them independently")


# ══════════════════════════════════════════════════════════════════
# C. Gaussian.samples(N, beta=...)
# ══════════════════════════════════════════════════════════════════
banner("C. Gaussian.samples — tempered draws from N(mean, var / beta)")

g_temp = Gaussian(mean=[0.0, 0.0, 0.0], variance=[1.0, 4.0, 0.25], device=device)

# C.1 — defaulted beta vs beta=1.0 must be byte-identical under the same seed
section("C.1  default vs beta=1.0 is byte-identical")
torch.manual_seed(101)
s_def = g_temp.samples(20000)
torch.manual_seed(101)
s_b1  = g_temp.samples(20000, beta=1.0)
print(f"  equal = {torch.equal(s_def, s_b1)}")
assert torch.equal(s_def, s_b1)
print("  [OK ] defaults match explicit beta=1.0")

# C.2 — empirical variance contracts as 1/beta on each axis
section("C.2  empirical per-axis variance scales as variance / beta")
torch.manual_seed(102)
target_var = torch.tensor([1.0, 4.0, 0.25], device=device)
for beta in (0.5, 1.0, 4.0):
    s = g_temp.samples(50000, beta=beta)
    emp_var = s.var(dim=0)
    expected = target_var / beta
    err = (emp_var - expected).abs().max().item()
    print(f"  beta = {beta:>3}:  var = {emp_var.tolist()}   (expect {expected.tolist()})   max err = {err:.3e}")
    assert err < 0.1, f"beta={beta}: empirical variance off by {err}"
print("  [OK ] tempered variance contracts as 1/beta")


# ══════════════════════════════════════════════════════════════════
# D. potential_from(fn) — wrap a callable as a Potential
# ══════════════════════════════════════════════════════════════════
banner("D. potential_from(fn) wraps a stateless callable")

def _U_from(x: torch.Tensor) -> torch.Tensor:
    # U(x) = 0.5 ||x||^2 + 2 * cos(x_1)
    x1 = x[:, 0]
    return 0.5 * (x ** 2).sum(-1) + 2 * torch.cos(x1)

# D.1 — potential_from returns a Potential subclass; instances forward to fn
section("D.1  potential_from(fn) returns a subclass; instance(x) ≡ fn(x)")
U_from = potential_from(_U_from)
assert isinstance(U_from, type) and issubclass(U_from, Potential), \
    "potential_from must return a Potential subclass (not an instance)"
u_fn = U_from().to(device)
x_fn = torch.randn(64, 3, device=device)
y_potential = u_fn(x_fn)
y_direct    = _U_from(x_fn)
diff = (y_potential - y_direct).abs().max().item()
print(f"  max |potential_from(fn)()(x) - fn(x)| = {diff:.3e}")
assert diff == 0.0, f"forward drift: {diff}"
print("  [OK ] subclass instance forwards to the underlying callable")

# D.2 — enable_eval(), enable_grad() chain through on an instance
section("D.2  enable_eval / enable_grad work on the wrapped Potential")
u_fn = U_from().to(device).enable_grad().enable_eval()
assert u_fn._grad_fn is not None and u_fn._eval_fn is not None, \
    "enable_grad / enable_eval didn't populate the cached fns"

v_eval = u_fn.eval(x_fn)
err_eval = (v_eval - y_direct).abs().max().item()
print(f"  max |u.eval(x) - fn(x)| = {err_eval:.3e}")
assert err_eval < 1e-4, f"eval drift: {err_eval}"

g_potential = u_fn.grad(x_fn)
# autograd reference: gradient of sum_i U(x_i) w.r.t. x
x_ref = x_fn.clone().detach().requires_grad_(True)
y_ref = _U_from(x_ref).sum()
(g_ref,) = torch.autograd.grad(y_ref, x_ref)
err_grad = (g_potential - g_ref).abs().max().item()
print(f"  max |u.grad(x) - autograd(fn)| = {err_grad:.3e}")
assert err_grad < 1e-4, f"grad drift: {err_grad}"
print("  [OK ] compiled grad / eval fast paths match autograd / fn(x)")

# D.3 — potential_instance_from returns a ready-to-use instance equivalent
#       to potential_from(fn)()
section("D.3  potential_instance_from(fn) returns an instance equivalent to potential_from(fn)()")
u_inst = potential_instance_from(_U_from).to(device)
assert isinstance(u_inst, Potential) and not isinstance(u_inst, type), \
    "potential_instance_from must return a Potential instance (not a class)"
y_inst = u_inst(x_fn)
diff_inst = (y_inst - y_direct).abs().max().item()
print(f"  max |potential_instance_from(fn)(x) - fn(x)| = {diff_inst:.3e}")
assert diff_inst == 0.0, f"potential_instance_from forward drift: {diff_inst}"
# Same class as potential_from(fn) (both build a fresh _FunctionPotential subclass)
assert isinstance(u_inst, Potential)
assert type(u_inst).__name__ == "_FunctionPotential"
print("  [OK ] potential_instance_from yields a Potential instance with matching forward")


# ─────────────────────────────────────────────────────────────────
print()
print("All potential verification checks passed.")
