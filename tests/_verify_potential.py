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

   B. linear_combination potential under Langevin sampling:
        - ulc(x) = c0 * U_source(x) + c1 * U_target(x)  is integrable, well-formed;
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

   D. potential_from(fn):
        - returns a ready-to-use Potential *instance* (not a class);
        - instance(x) forwards to fn(x);
        - enable_grad() + enable_eval() chain through;
        - .grad(x) matches autograd of fn(x).sum() w.r.t. x.
"""

from pathlib import Path

import torch

from zflows.potential import (
    Gaussian,
    linear_combination,
    Potential,
    potential_from,
)
from zflows.utils import langevin, set_cache_size_limit, suppress_warnings

# Silence Triton/Inductor/Dynamo/Python warnings; cache headroom for the
# Gaussian + linear_combination .enable_grad / .enable_eval compiles.
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
# B. linear_combination potential
# ══════════════════════════════════════════════════════════════════
banner("B. linear_combination potential under Langevin sampling")

# U_source(x) = (x1^2 + x2^2) / 2 — isotropic harmonic
def U_source_forward(x: torch.Tensor) -> torch.Tensor: # Tensor [N, d] -> Tensor [N]
    return 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2)

# U_target(x) = 2 * cos(x1) — cosine modulation on x1 only
def U_target_forward(x: torch.Tensor) -> torch.Tensor: # Tensor [N, d] -> Tensor [N]
    return 2 * torch.cos(x[:, 0])

u_source = potential_from(U_source_forward).to(device)
u_target = potential_from(U_target_forward).to(device)

# ulc(x) = 1.0 * U_source(x) + 1.2 * U_target(x)
# = 0.5 * (x1^2 + x2^2) + 2.4 * cos(x1)
# x2 ~ standard normal; x1 modulated by cosine (still even, mean ≈ 0).
ulc = linear_combination([u_source, u_target], [1.0, 1.2])
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
u_source.release()
u_target.release()

# B.3 — linear_combination normalises every coeff input to list[float].
# Whatever the user passes (list, tuple, 1-d Tensor, requires_grad=True
# tensor, None), `self.coeffs` must come out as a plain Python list of
# floats — no buffers, no grad tracking, no device pinning. None defaults
# to uniform 1/N.
section("B.3  linear_combination coeffs always normalise to list[float]")
u_a = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0], device=device)
u_b = Gaussian(mean=[3.0, 3.0], variance=[1.0, 1.0], device=device)
# (1) plain tensor input -> list[float].
lc_t = linear_combination([u_a, u_b], torch.tensor([0.3, 0.7], device=device))
assert isinstance(lc_t.coeffs, list) and all(isinstance(c, float) for c in lc_t.coeffs), \
    f"tensor input should normalise to list[float]; got {type(lc_t.coeffs).__name__}"
assert abs(lc_t.coeffs[0] - 0.3) < 1e-6 and abs(lc_t.coeffs[1] - 0.7) < 1e-6
# (2) requires_grad=True tensor is accepted too — detached on the way in.
lc_g = linear_combination(
    [u_a, u_b], torch.tensor([0.3, 0.7], device=device, requires_grad=True)
)
assert isinstance(lc_g.coeffs, list) and all(isinstance(c, float) for c in lc_g.coeffs)
# (3) coeffs=None -> uniform 1/N.
lc_d = linear_combination([u_a, u_b])
assert lc_d.coeffs == [0.5, 0.5], f"default coeffs should be uniform 1/N, got {lc_d.coeffs}"
print(f"  tensor input:                  coeffs = {lc_t.coeffs}")
print(f"  requires_grad=True tensor:     coeffs = {lc_g.coeffs}")
print(f"  default (None) for N=2:        coeffs = {lc_d.coeffs}")
print("  [OK ] coeffs always normalised to list[float]")


# B.4 — linear_combination.set_coeffs replaces the weights in place across
# every accepted input form (list, tuple, 1-d Tensor, requires_grad=True
# tensor), keeps `self.coeffs` as a plain list[float], returns self for
# chaining, rejects None and length mismatches, and the next forward
# call reflects the new mix exactly.
section("B.4  linear_combination.set_coeffs in-place update")
u_a2 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0], device=device)
u_b2 = Gaussian(mean=[3.0, 3.0], variance=[1.0, 1.0], device=device)
lc = linear_combination([u_a2, u_b2])
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

# (7) re-mixing is bit-identical to constructing a fresh linear_combination
#     with the same coeffs (i.e. set_coeffs leaves no stale state behind).
lc.set_coeffs([0.35, 0.65])
lc_fresh = linear_combination([u_a2, u_b2], [0.35, 0.65])
err_fresh = (lc(x_lc) - lc_fresh(x_lc)).abs().max().item()
assert err_fresh == 0.0, f"set_coeffs vs fresh ctor mismatch: {err_fresh}"

print(f"  list / tuple / tensor / grad tensor -> list[float]:  OK")
print(f"  returns self (chainable):                            OK")
print(f"  None rejected, length mismatch rejected:             OK")
print(f"  set_coeffs([1, 0]) collapses forward to U_a:         err = {err_collapsed:.3e}")
print(f"  set_coeffs == fresh ctor (no stale state):           err = {err_fresh:.3e}")
print("  [OK ] set_coeffs updates in place and forward reflects it")


# B.5 — `linear_combination` design:
#   - the combined `_grad_fn` / `_eval_fn` are linked at __init__ time
#     as Python closures over `self`, NOT at enable_grad / enable_eval
#     time (so they exist for the entire lifetime of the instance);
#   - the runtime "needs .enable_grad() first" gate moves from a
#     pre-flight check on linear_combination (its _grad_fn is always
#     non-None) to a child-level check inside the closure;
#   - `enable_grad` / `enable_eval` overrides ONLY propagate to the
#     children whose _grad_fn / _eval_fn is None, and do NOT touch
#     `self._grad_fn` / `self._eval_fn` (they were linked in __init__);
#   - `set_coeffs` is a pure coeff update — no recompile, no
#     invalidation; the closure picks up the new coeffs on next call.
section("B.5  linear_combination: __init__-linked closures, child-level runtime gate")

u_c = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0], device=device)
u_d = Gaussian(mean=[3.0, 3.0], variance=[1.0, 1.0], device=device)
lc_e = linear_combination([u_c, u_d], [0.2, 0.8])

# (1) Right after __init__: combined closures populated, children cold.
assert lc_e._grad_fn is not None, "_grad_fn must be linked in __init__"
assert lc_e._eval_fn is not None, "_eval_fn must be linked in __init__"
assert all(U._grad_fn is None and U._eval_fn is None for U in lc_e.potentials), \
    "children must NOT be enabled by linear_combination.__init__"

# (2) Calling .grad(x) / .eval(x) before children enabled raises from
# the child's gate (NOT from linear_combination's gate, which passes
# because _grad_fn is non-None). The error message must come from the
# child's enable_grad gate.
x_e = torch.randn(8, 2, device=device)
raised_grad = False
try:
    lc_e.grad(x_e)
except RuntimeError as e:
    raised_grad = True
    msg_grad = str(e)
assert raised_grad and "enable_grad" in msg_grad and "Gaussian" in msg_grad, (
    f"runtime gate must fire from the child (Gaussian.grad); got: "
    f"{msg_grad if raised_grad else 'no error'}"
)
raised_eval = False
try:
    lc_e.eval(x_e)
except RuntimeError as e:
    raised_eval = True
    msg_eval = str(e)
assert raised_eval and "enable_eval" in msg_eval and "Gaussian" in msg_eval, (
    f"runtime gate must fire from the child (Gaussian.eval); got: "
    f"{msg_eval if raised_eval else 'no error'}"
)

# (3) enable_grad / enable_eval propagate to children and DO NOT touch
# self._grad_fn / self._eval_fn.
grad_id_init = id(lc_e._grad_fn)
eval_id_init = id(lc_e._eval_fn)
lc_e.enable_grad().enable_eval()
assert id(lc_e._grad_fn) == grad_id_init, \
    "enable_grad must NOT touch self._grad_fn (linked in __init__)"
assert id(lc_e._eval_fn) == eval_id_init, \
    "enable_eval must NOT touch self._eval_fn (linked in __init__)"
assert all(U._grad_fn is not None for U in lc_e.potentials), \
    "enable_grad must propagate to every child"
assert all(U._eval_fn is not None for U in lc_e.potentials), \
    "enable_eval must propagate to every child"

# (4) Closure returns the correct weighted sum.
x_e = torch.randn(64, 2, device=device, requires_grad=True)
fresh_28 = linear_combination([u_c, u_d], [0.2, 0.8])
g_fast = lc_e.grad(x_e.detach()).clone()
g_ref  = torch.autograd.grad(fresh_28(x_e).sum(), x_e)[0].detach()
err_grad_init = (g_fast - g_ref).abs().max().item()
v_fast = lc_e.eval(x_e.detach())
v_ref  = fresh_28(x_e.detach())
err_eval_init = (v_fast - v_ref).abs().max().item()
assert err_grad_init < 1e-4 and err_eval_init < 1e-4

# (5) set_coeffs preserves every identity (combined + children) and the
# closure picks up the new mix on next call.
child_grad_ids = [id(U._grad_fn) for U in lc_e.potentials]
child_eval_ids = [id(U._eval_fn) for U in lc_e.potentials]
lc_e.set_coeffs([0.7, 0.3])
assert id(lc_e._grad_fn) == grad_id_init and id(lc_e._eval_fn) == eval_id_init
assert [id(U._grad_fn) for U in lc_e.potentials] == child_grad_ids
assert [id(U._eval_fn) for U in lc_e.potentials] == child_eval_ids

fresh_73 = linear_combination([u_c, u_d], [0.7, 0.3])
g_fast = lc_e.grad(x_e.detach()).clone()
g_ref  = torch.autograd.grad(fresh_73(x_e).sum(), x_e)[0].detach()
err_grad_setc = (g_fast - g_ref).abs().max().item()
v_fast = lc_e.eval(x_e.detach())
v_ref  = fresh_73(x_e.detach())
err_eval_setc = (v_fast - v_ref).abs().max().item()
assert err_grad_setc < 1e-4 and err_eval_setc < 1e-4, \
    f"closure didn't read new coeffs; grad err={err_grad_setc}, eval err={err_eval_setc}"

# (6) set_coeffs takes ONLY the coeffs argument (no enable_grad /
# enable_eval flags). Calling it with kwargs must raise.
raised_kwarg = False
try:
    lc_e.set_coeffs([0.4, 0.6], enable_grad=True)
except TypeError:
    raised_kwarg = True
assert raised_kwarg, "set_coeffs must reject enable_grad / enable_eval kwargs"

# Multiple set_coeffs calls still preserve every id() (combined +
# children) — the closures and the compiled children survive forever.
lc_e.set_coeffs([0.4, 0.6])
assert id(lc_e._grad_fn) == grad_id_init and id(lc_e._eval_fn) == eval_id_init
assert [id(U._grad_fn) for U in lc_e.potentials] == child_grad_ids
assert [id(U._eval_fn) for U in lc_e.potentials] == child_eval_ids

# (7) enable_grad / enable_eval propagate UNCONDITIONALLY to every
# child; idempotency at the child level (Potential.enable_grad early-
# returns if _grad_fn is non-None) means hot children keep their
# existing compiled artifacts.
u_hot = Gaussian(mean=[1.0, 1.0], variance=[1.0, 1.0], device=device)
u_hot.enable_grad().enable_eval()
u_cold = Gaussian(mean=[5.0, 5.0], variance=[1.0, 1.0], device=device)
u_hot_gid = id(u_hot._grad_fn)
u_hot_eid = id(u_hot._eval_fn)
assert u_cold._grad_fn is None and u_cold._eval_fn is None
lc_skip = linear_combination([u_hot, u_cold], [0.5, 0.5])
lc_skip.enable_grad().enable_eval()
assert id(u_hot._grad_fn) == u_hot_gid, \
    "hot child's _grad_fn must survive (child-level idempotency)"
assert id(u_hot._eval_fn) == u_hot_eid, \
    "hot child's _eval_fn must survive (child-level idempotency)"
assert u_cold._grad_fn is not None and u_cold._eval_fn is not None, \
    "cold child must be compiled when linear_combination enables grad/eval"

# (8) Auto-enabled-from-hot-children: if EVERY child has its compiled
# `.grad(x)` / `.eval(x)` populated BEFORE the linear_combination is
# constructed, then `lc.grad(x)` / `lc.eval(x)` work IMMEDIATELY — no
# `lc.enable_grad()` / `lc.enable_eval()` call is needed. The combined
# closure is linked at __init__ and the child-level runtime gate
# passes for every child, so the call succeeds on the first attempt.
u_pre1 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0], device=device).enable_grad().enable_eval()
u_pre2 = Gaussian(mean=[2.0, 2.0], variance=[1.0, 1.0], device=device).enable_grad().enable_eval()
lc_auto = linear_combination([u_pre1, u_pre2], [0.3, 0.7])
# Combined closures already populated by __init__.
assert lc_auto._grad_fn is not None and lc_auto._eval_fn is not None
# Children stay hot.
assert u_pre1._grad_fn is not None and u_pre2._grad_fn is not None
assert u_pre1._eval_fn is not None and u_pre2._eval_fn is not None

# Calling .grad(x) / .eval(x) on the freshly built lc must succeed
# WITHOUT any lc.enable_grad() / lc.enable_eval() call.
x_auto = torch.randn(32, 2, device=device, requires_grad=True)
fresh_37 = linear_combination([u_pre1, u_pre2], [0.3, 0.7])
g_auto_grad = lc_auto.grad(x_auto.detach()).clone()
g_auto_ref  = torch.autograd.grad(fresh_37(x_auto).sum(), x_auto)[0].detach()
err_grad_auto = (g_auto_grad - g_auto_ref).abs().max().item()
assert err_grad_auto < 1e-4, (
    f"auto-enabled linear_combination.grad(x) drifts from autograd: {err_grad_auto}"
)

v_auto_eval = lc_auto.eval(x_auto.detach())
v_auto_ref  = fresh_37(x_auto.detach())
err_eval_auto = (v_auto_eval - v_auto_ref).abs().max().item()
assert err_eval_auto < 1e-4, (
    f"auto-enabled linear_combination.eval(x) drifts from forward: {err_eval_auto}"
)

print(f"  __init__ links combined _grad_fn / _eval_fn (children cold):  OK")
print(f"  runtime gate fires at child level:  grad -> {msg_grad.split(' ')[0]}, eval -> {msg_eval.split(' ')[0]}")
print(f"  enable_grad / enable_eval propagate to children, don't touch self._*:  OK")
print(f"  closure correctness vs autograd / forward:  grad err = {err_grad_init:.3e}, eval err = {err_eval_init:.3e}")
print(f"  set_coeffs preserves all id()s; closure reads new coeffs:    grad err = {err_grad_setc:.3e}, eval err = {err_eval_setc:.3e}")
print(f"  set_coeffs rejects enable_grad / enable_eval kwargs:          OK")
print(f"  enable_grad / enable_eval propagate; hot children preserved:  OK")
print(f"  linear_combination of pre-enabled children works immediately:")
print(f"    .grad(x) err = {err_grad_auto:.3e}, .eval(x) err = {err_eval_auto:.3e}  (no lc.enable_*() needed)")
print("  [OK ] __init__-linked closures + child-level runtime gate working as specified")


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
# D. potential_from(fn) — wrap a callable as a Potential instance
# ══════════════════════════════════════════════════════════════════
banner("D. potential_from(fn) wraps a stateless callable into an instance")

def _myforward(x: torch.Tensor) -> torch.Tensor:
    # 0.5 ||x||^2 + 2 * cos(x_1)
    x1 = x[:, 0]
    return 0.5 * (x ** 2).sum(-1) + 2 * torch.cos(x1)

# D.1 — potential_from returns a ready-to-use Potential *instance* (no
# manual instantiation), and instance(x) forwards to fn(x).
section("D.1  potential_from(fn) returns an instance; instance(x) ≡ fn(x)")
u_fn = potential_from(_myforward).to(device)
assert isinstance(u_fn, Potential), \
    "potential_from must return a Potential instance (not a class)"
assert not isinstance(u_fn, type), \
    "potential_from must NOT return a class"
x_fn = torch.randn(64, 3, device=device)
y_potential = u_fn(x_fn)
y_direct    = _myforward(x_fn)
diff = (y_potential - y_direct).abs().max().item()
print(f"  max |potential_from(fn)(x) - fn(x)| = {diff:.3e}")
assert diff == 0.0, f"forward drift: {diff}"
print("  [OK ] potential_from yields a Potential instance with matching forward")

# D.2 — enable_eval(), enable_grad() chain through on the instance.
section("D.2  enable_eval / enable_grad work on the wrapped Potential instance")
u_fn = potential_from(_myforward).to(device).enable_grad().enable_eval()
assert u_fn._grad_fn is not None and u_fn._eval_fn is not None, \
    "enable_grad / enable_eval didn't populate the cached fns"

v_eval = u_fn.eval(x_fn)
err_eval = (v_eval - y_direct).abs().max().item()
print(f"  max |u.eval(x) - fn(x)| = {err_eval:.3e}")
assert err_eval < 1e-4, f"eval drift: {err_eval}"

g_potential = u_fn.grad(x_fn)
# autograd reference: gradient of sum_i U(x_i) w.r.t. x
x_ref = x_fn.clone().detach().requires_grad_(True)
y_ref = _myforward(x_ref).sum()
(g_ref,) = torch.autograd.grad(y_ref, x_ref)
err_grad = (g_potential - g_ref).abs().max().item()
print(f"  max |u.grad(x) - autograd(fn)| = {err_grad:.3e}")
assert err_grad < 1e-4, f"grad drift: {err_grad}"
print("  [OK ] compiled grad / eval fast paths match autograd / fn(x)")


# ─────────────────────────────────────────────────────────────────
print()
print("All potential verification checks passed.")
