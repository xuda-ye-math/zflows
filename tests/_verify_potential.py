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
          verification.

   C. Gaussian.samples(N, beta=...):
        - default vs beta=1.0 is byte-identical;
        - empirical variance contracts as 1/beta (tempered N(mean, var/beta)).

   D. Potential._from(fn):
        - wrapped callable matches fn(x) under forward;
        - enable_grad() + enable_eval() chain through;
        - .grad(x) matches autograd of fn(x).sum() w.r.t. x.
"""

from pathlib import Path

import torch

from zflows.potential import Gaussian, Linear_Combination, Potential
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

# B.3 — Linear_Combination rejects requires_grad=True tensor coeffs.
# Buffer-registered coeffs are hidden from `optimizer.parameters()`; if the
# user passes a tensor with grad tracking on, the optimizer would silently
# never see the weights. The constructor must catch this up front.
section("B.3  Linear_Combination guards against requires_grad=True coeffs")
u_a = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0], device=device)
u_b = Gaussian(mean=[3.0, 3.0], variance=[1.0, 1.0], device=device)
# (1) plain tensor with requires_grad=False is accepted (existing behavior).
lc_ok = Linear_Combination([u_a, u_b], torch.tensor([0.3, 0.7], device=device))
assert lc_ok.coeffs.requires_grad is False, "buffer coeffs should not track grad"
# (2) requires_grad=True tensor must raise.
raised = False
try:
    Linear_Combination([u_a, u_b], torch.tensor([0.3, 0.7], device=device, requires_grad=True))
except AssertionError as e:
    raised = True
    msg = str(e)
assert raised, "Linear_Combination should reject requires_grad=True coeffs"
assert "requires" in msg.lower() or "grad" in msg.lower(), \
    f"rejection message should mention grad / requires_grad, got: {msg}"
print(f"  plain tensor accepted:               coeffs.requires_grad = False")
print(f"  requires_grad=True tensor rejected:  {msg.splitlines()[0][:80]}...")
print("  [OK ] grad-tracking coeffs caught at construction")


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
# D. Potential._from(fn) — wrap a callable as a Potential
# ══════════════════════════════════════════════════════════════════
banner("D. Potential._from(fn) wraps a stateless callable")

def _U_from(x: torch.Tensor) -> torch.Tensor:
    # U(x) = 0.5 ||x||^2 + 2 * cos(x_1)
    x1 = x[:, 0]
    return 0.5 * (x ** 2).sum(-1) + 2 * torch.cos(x1)

# D.1 — _from returns a Potential subclass; instances forward to fn
section("D.1  _from(fn) returns a subclass; instance(x) ≡ fn(x)")
U_from = Potential._from(_U_from)
assert isinstance(U_from, type) and issubclass(U_from, Potential), \
    "_from must return a Potential subclass (not an instance)"
u_fn = U_from().to(device)
x_fn = torch.randn(64, 3, device=device)
y_potential = u_fn(x_fn)
y_direct    = _U_from(x_fn)
diff = (y_potential - y_direct).abs().max().item()
print(f"  max |Potential._from(fn)()(x) - fn(x)| = {diff:.3e}")
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


# ─────────────────────────────────────────────────────────────────
print()
print("All potential verification checks passed.")
