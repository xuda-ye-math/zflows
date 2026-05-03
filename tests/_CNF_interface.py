# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""
Sanity checks for the CNF dual interface:

    zflows wrapper:   T_z = cnf.t()           -> ComposedTransform
    zuko native:      T_n = cnf._ffj()        -> FreeFormJacobianTransform

Both should be the same bijection on R^d, just packaged differently. The
zflows wrapper exists only to satisfy the Flow protocol's
`t() -> ComposedTransform` return-type contract; the inner FFJ transform
does all the actual work.

We verify that:
  1. the type-level guarantees hold (T_z is ComposedTransform; T_n is
     FreeFormJacobianTransform; both expose .inv and .call_and_ladj).
  2. forward `(y, ladj)` outputs of the two interfaces agree numerically.
  3. inverse outputs agree numerically.
  4. each interface round-trips: x -> y -> x' with negligible error.
  5. backprop through T_z reaches the same flow parameters as backprop
     through T_n.
"""

import torch
from zuko.transforms import ComposedTransform, FreeFormJacobianTransform
from zflows import CNF, Flow

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

cnf = CNF(dimension=2, frequency=3, hidden_features=(32, 32)).to(device)

# ----------------------------------------------------------------------
# Test 1: structural — types and Flow-protocol membership
# ----------------------------------------------------------------------
print("=" * 60)
print("Test 1: structural types of cnf.t() and cnf._ffj()")
print("=" * 60)

T_z = cnf.t()      # zflows wrapper
T_n = cnf._ffj()   # zuko native

print(f"isinstance(cnf, Flow)             = {isinstance(cnf, Flow)}")
print(f"isinstance(T_z, ComposedTransform) = {isinstance(T_z, ComposedTransform)}")
print(f"isinstance(T_n, FreeFormJacobianTransform) = {isinstance(T_n, FreeFormJacobianTransform)}")
print(f"len(T_z.transforms) (length-1 wrapper) = {len(T_z.transforms)}")
assert isinstance(cnf, Flow)
assert isinstance(T_z, ComposedTransform)
assert isinstance(T_n, FreeFormJacobianTransform)
assert len(T_z.transforms) == 1
print("PASSED")

# ----------------------------------------------------------------------
# Test 2: forward call_and_ladj agreement
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 2: forward (y, ladj) agreement")
print("=" * 60)

x = torch.randn(8, 2, device=device)

with torch.no_grad():
    cnf.eval() # disable training-mode `phi=parameters()` path so both forms see phi=()
    y_z, ladj_z = cnf.t().call_and_ladj(x)
    y_n, ladj_n = cnf._ffj().call_and_ladj(x)

err_y = (y_z - y_n).abs().max().item()
err_ladj = (ladj_z - ladj_n).abs().max().item()
print(f"max |y_zflows  - y_zuko|  = {err_y:.3e}")
print(f"max |ladj_zfl  - ladj_zk| = {err_ladj:.3e}")
assert err_y < 1e-5,    f"forward y disagrees:    {err_y}"
assert err_ladj < 1e-5, f"forward ladj disagrees: {err_ladj}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 3: inverse agreement
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 3: inverse agreement")
print("=" * 60)

with torch.no_grad():
    cnf.eval()
    y = torch.randn(8, 2, device=device) * 0.5
    x_back_z = cnf.t().inv(y)
    x_back_n = cnf._ffj().inv(y)

err_inv = (x_back_z - x_back_n).abs().max().item()
print(f"max |inv_zflows - inv_zuko| = {err_inv:.3e}")
assert err_inv < 1e-5, f"inverse disagrees: {err_inv}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 4: round-trip x -> y -> x' for both interfaces
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 4: round-trip x -> y -> inv(y)")
print("=" * 60)

with torch.no_grad():
    cnf.eval()
    x = torch.randn(8, 2, device=device)
    T_z = cnf.t()
    T_n = cnf._ffj()
    y_z = T_z(x)
    y_n = T_n(x)
    x_z_back = T_z.inv(y_z)
    x_n_back = T_n.inv(y_n)

err_rt_z = (x - x_z_back).abs().max().item()
err_rt_n = (x - x_n_back).abs().max().item()
print(f"zflows wrapper round-trip max |x - inv(F(x))| = {err_rt_z:.3e}")
print(f"zuko native    round-trip max |x - inv(F(x))| = {err_rt_n:.3e}")
assert err_rt_z < 1e-4, f"zflows round-trip too large: {err_rt_z}"
assert err_rt_n < 1e-4, f"zuko native round-trip too large: {err_rt_n}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 5: training compatibility — backprop reaches the same parameters
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 5: backprop through both interfaces hits the same params")
print("=" * 60)

cnf.train()
x = torch.randn(16, 2, device=device).clamp(-2, 2)

# backprop through the zflows wrapper
cnf.zero_grad()
y_z, ladj_z = cnf.t().call_and_ladj(x)
loss_z = (y_z.pow(2).sum() - ladj_z.sum())
loss_z.backward()
grads_z = [p.grad.detach().clone() for p in cnf.parameters() if p.grad is not None]

# backprop through the zuko native interface, fresh state
cnf.zero_grad()
y_n, ladj_n = cnf._ffj().call_and_ladj(x)
loss_n = (y_n.pow(2).sum() - ladj_n.sum())
loss_n.backward()
grads_n = [p.grad.detach().clone() for p in cnf.parameters() if p.grad is not None]

assert len(grads_z) == len(grads_n) > 0
err_grads = max((gz - gn).abs().max().item() for gz, gn in zip(grads_z, grads_n))
print(f"#parameter tensors with non-None grad: {len(grads_z)}")
print(f"max |grad_zflows - grad_zuko|         = {err_grads:.3e}")
assert err_grads < 1e-4, f"gradients disagree: {err_grads}"
print("PASSED")

print()
print("All CNF interface checks passed.")
