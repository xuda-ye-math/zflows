"""
Sanity checks for zflows.call_and_ladj_grad.

We compare its output against torch.autograd.grad applied row-by-row
(brute-force per-sample gradient) on small batches, and verify that the
returned ladj_grad is differentiable w.r.t. the flow's parameters when
create_graph=True.
"""

import math
import torch
from zflows import NSF, NCSF, call_and_ladj_grad

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# ----------------------------------------------------------------------
# Test 1: NSF, agreement with row-by-row autograd gradient
# ----------------------------------------------------------------------
print("=" * 60)
print("Test 1: NSF — agreement with row-by-row autograd")
print("=" * 60)

flow = NSF(a=[-4, -4], b=[4, 4], bins=8, transforms=4, hidden_features=(32, 32)).to(device)
N, d = 16, 2
x = torch.randn(N, d, device=device).clamp(-3.5, 3.5)

# call_and_ladj_grad path
y_fast, ladj_fast, ladj_grad_fast = call_and_ladj_grad(flow.t(), x, create_graph=False)

# row-by-row reference: grad of ladj_i w.r.t. x_i, computed independently per sample
ladj_grad_ref = torch.zeros_like(x)
for i in range(N):
    xi = x[i].detach().clone().requires_grad_(True)
    _, ladj_i = flow.t().call_and_ladj(xi.unsqueeze(0))
    (gi,) = torch.autograd.grad(ladj_i.sum(), xi)
    ladj_grad_ref[i] = gi

err = (ladj_grad_fast - ladj_grad_ref).abs().max().item()
print(f"max |fast - row-by-row| = {err:.3e}")
assert err < 1e-4, f"NSF ladj_grad disagrees with reference: {err}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 2: NCSF, agreement with row-by-row autograd gradient
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 2: NCSF — agreement with row-by-row autograd")
print("=" * 60)

ncsf = NCSF(
    a=[-math.pi, -math.pi, -math.pi],
    b=[ math.pi,  math.pi,  math.pi],
    bins=8, transforms=4, hidden_features=(32, 32),
).to(device)
N, d = 16, 3
x = (torch.rand(N, d, device=device) * 2 * math.pi) - math.pi # uniform in [-pi, pi]^3

_, _, ladj_grad_fast = call_and_ladj_grad(ncsf.t(), x, create_graph=False)

ladj_grad_ref = torch.zeros_like(x)
for i in range(N):
    xi = x[i].detach().clone().requires_grad_(True)
    _, ladj_i = ncsf.t().call_and_ladj(xi.unsqueeze(0))
    (gi,) = torch.autograd.grad(ladj_i.sum(), xi)
    ladj_grad_ref[i] = gi

err = (ladj_grad_fast - ladj_grad_ref).abs().max().item()
print(f"max |fast - row-by-row| = {err:.3e}")
assert err < 1e-4, f"NCSF ladj_grad disagrees with reference: {err}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 3: create_graph=True — ladj_grad must be differentiable w.r.t. flow params
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 3: create_graph=True — backprop through ladj_grad")
print("=" * 60)

flow = NSF(a=[-4, -4], b=[4, 4], bins=8, transforms=4, hidden_features=(32, 32)).to(device)
x = torch.randn(8, 2, device=device).clamp(-3.5, 3.5)

_, _, ladj_grad = call_and_ladj_grad(flow.t(), x, create_graph=True)
loss = ladj_grad.pow(2).sum() # any scalar that depends on ladj_grad
loss.backward() # should not raise

n_with_grad = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in flow.parameters())
n_total = sum(1 for _ in flow.parameters())
print(f"flow params receiving non-zero grad: {n_with_grad} / {n_total}")
assert n_with_grad > 0, "no flow parameter received gradient — create_graph path is broken"
print("PASSED")

# ----------------------------------------------------------------------
# Test 4: create_graph=False — ladj_grad must be detached from flow params
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 4: create_graph=False — ladj_grad has no parameter graph")
print("=" * 60)

_, _, ladj_grad_nograph = call_and_ladj_grad(flow.t(), x, create_graph=False)
print(f"ladj_grad.requires_grad = {ladj_grad_nograph.requires_grad}")
try:
    ladj_grad_nograph.pow(2).sum().backward()
    raised = False
except RuntimeError as e:
    raised = True
    print(f"backward() raised as expected: {type(e).__name__}")
assert raised or not ladj_grad_nograph.requires_grad, \
    "create_graph=False should not produce a backprop-able tensor"
print("PASSED")

print()
print("All tests passed.")
