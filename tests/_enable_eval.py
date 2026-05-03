# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""
Sanity checks for the Potential.enable_eval() opt-in fast path:

    u.enable_eval()    # builds u._eval_fn = torch.compile(u.forward)
    u.eval(x)          # routes through u._eval_fn (compiled forward)
    u.eval()           # standard nn.Module: switch to eval mode, return self

The langevin() / rejuvenation() utility checks `potential._eval_fn` and
routes the MALA accept/reject energy evaluations through the compiled
fast path when it is available, falling back to plain `potential(x)`
when it is not.

We verify that:
  1. structural — _eval_fn is None before .enable_eval(), populated after;
     .enable_eval() is idempotent (no recompile on second call);
     .eval() with no argument preserves the nn.Module eval-mode switch.
  2. .eval(x) raises a clear RuntimeError when .enable_eval() has not been
     called.
  3. langevin(adjust=True) on a Potential with .enable_eval() actually
     routes U(x) and U(y) through _eval_fn (counted = 2 * iters).
  4. langevin(adjust=True) on a Potential WITHOUT .enable_eval() falls
     back to potential(x) and still produces the same output.
  5. numerical equivalence: same seed, same input -> identical samples
     whether or not .enable_eval() was called (fast path is a pure speed
     optimization with no statistical effect).
"""

import torch
from zflows import Gaussian, langevin

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

ITERS = 5
STEP  = 1e-2

# ----------------------------------------------------------------------
# Test 1: structural — enable_eval populates _eval_fn, eval() preserves
# nn.Module eval-mode switch
# ----------------------------------------------------------------------
print("=" * 60)
print("Test 1: structural state of enable_eval / eval()")
print("=" * 60)

u = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)
print(f"before enable_eval: _eval_fn is None = {u._eval_fn is None}")
assert u._eval_fn is None

u.enable_eval()
print(f"after  enable_eval: _eval_fn is None = {u._eval_fn is None}")
assert u._eval_fn is not None

# idempotent: second call must not rebuild
fn_first = u._eval_fn
u.enable_eval()
print(f"second enable_eval keeps same closure: {u._eval_fn is fn_first}")
assert u._eval_fn is fn_first

# .eval() with no argument: standard nn.Module switch (returns self, sets training=False)
u.train()
assert u.training is True
ret = u.eval() # no x -> nn.Module.eval()
print(f"u.eval() (no x) returns self: {ret is u}")
print(f"u.eval() (no x) sets training=False: {u.training is False}")
assert ret is u and u.training is False
print("PASSED")

# ----------------------------------------------------------------------
# Test 2: u.eval(x) raises before enable_eval() is called
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 2: u.eval(x) raises a clear RuntimeError without enable_eval()")
print("=" * 60)

u2 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)
x = torch.randn(8, 2, device=device)
try:
    u2.eval(x)
    raised = False
except RuntimeError as e:
    raised = True
    print(f"  raised: {e}")
assert raised, "u.eval(x) must raise without enable_eval()"
print("PASSED")

# ----------------------------------------------------------------------
# Test 3: langevin(adjust=True) routes through _eval_fn when available
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 3: MALA routes U(x), U(y) through u._eval_fn  (count == 2*iters)")
print("=" * 60)

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
print(f"_eval_fn calls during MALA: {counter.n}  (expected 2 * iters = {2 * ITERS})")
assert counter.n == 2 * ITERS, f"_eval_fn was hit {counter.n} times, expected {2*ITERS}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 4: fall-back path — langevin(adjust=True) on a Potential without
# enable_eval() still runs (calls potential(x) directly)
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 4: MALA falls back to potential(x) when _eval_fn is None")
print("=" * 60)

u4 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device).enable_grad()
print(f"_eval_fn is None: {u4._eval_fn is None}")
assert u4._eval_fn is None

x0 = torch.randn(64, 2, device=device)
y = langevin(x0, potential=u4, step=STEP, iters=ITERS, adjust=True)
print(f"langevin output shape: {tuple(y.shape)}")
assert y.shape == (64, 2)
print("PASSED")

# ----------------------------------------------------------------------
# Test 5: numerical equivalence — fast path == fall-back under same seed
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 5: same seed -> identical output with vs. without enable_eval")
print("=" * 60)

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
print(f"max |y_with_eval - y_without_eval| = {err:.3e}")
assert err < 1e-4, f"fast path disagrees with fall-back: {err}"
print("PASSED")

print()
print("All enable_eval interface checks passed.")
