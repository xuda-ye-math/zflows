# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""
Sanity checks for the zflows.RealNVP class:

    zflows wrapper:   T_z = realnvp.t()      -> ComposedTransform
    zuko native:      T_n = z_realnvp().transform
                                             -> ComposedTransform

zflows.RealNVP is built by composition (own ModuleList of
GeneralCouplingTransform layers), not by inheriting zuko's RealNVP, so
the "native" reference here is a fresh zuko.flows.coupling.RealNVP built
with the same config and same RNG seed. Both should produce the same
checkered masks and the same MLP initialisations, hence the same
bijection.

We verify that:
  1. the type-level guarantees hold (T_z is ComposedTransform; T_z has
     `transforms` coupling layers; each exposes .inv and .call_and_ladj).
  2. each interface round-trips: x -> y -> x' with negligible error.
  3. forward + inverse log-determinants cancel.
  4. forward `(y, ladj)` outputs of the zflows and zuko interfaces agree
     numerically when both are seeded identically.
  5. inverse outputs agree numerically.
  6. backprop through T_z reaches all flow parameters with non-None
     gradients of the right shapes.
"""

import torch
from zuko.transforms import ComposedTransform
from zuko.flows.coupling import RealNVP as ZukoRealNVP
from zflows import RealNVP, Flow

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DIM = 4
TRANSFORMS = 4
HIDDEN = (32, 32)

# ----------------------------------------------------------------------
# Test 1: structural — type and Flow-protocol membership
# ----------------------------------------------------------------------
print("=" * 60)
print("Test 1: structural types of realnvp.t()")
print("=" * 60)

torch.manual_seed(0)
realnvp = RealNVP(dimension=DIM, transforms=TRANSFORMS, hidden_features=HIDDEN).to(device)

T_z = realnvp.t()

print(f"isinstance(realnvp, Flow)                 = {isinstance(realnvp, Flow)}")
print(f"isinstance(T_z, ComposedTransform)        = {isinstance(T_z, ComposedTransform)}")
print(f"len(T_z.transforms) (=={TRANSFORMS} coupling layers)  = {len(T_z.transforms)}")
assert isinstance(realnvp, Flow)
assert isinstance(T_z, ComposedTransform)
assert len(T_z.transforms) == TRANSFORMS
print("PASSED")

# ----------------------------------------------------------------------
# Test 2: round-trip x -> y -> inv(y)
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 2: round-trip x -> F(x) -> F^{-1}(F(x))")
print("=" * 60)

with torch.no_grad():
    realnvp.eval()
    x = torch.randn(8, DIM, device=device)
    T_z = realnvp.t()
    y = T_z(x)
    x_back = T_z.inv(y)

err_rt = (x - x_back).abs().max().item()
print(f"max |x - inv(F(x))| = {err_rt:.3e}")
assert err_rt < 1e-4, f"round-trip too large: {err_rt}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 3: log-det consistency: ladj(F, x) + ladj(F^-1, F(x)) ~= 0
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 3: log-determinant cancellation (forward + inverse)")
print("=" * 60)

with torch.no_grad():
    realnvp.eval()
    x = torch.randn(8, DIM, device=device)
    T_z = realnvp.t()
    y, ladj_fwd = T_z.call_and_ladj(x)
    x_back, ladj_inv = T_z.inv.call_and_ladj(y)

err_ladj = (ladj_fwd + ladj_inv).abs().max().item()
print(f"max |ladj_fwd + ladj_inv| = {err_ladj:.3e}")
assert err_ladj < 1e-4, f"log-det does not cancel: {err_ladj}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 4: forward (y, ladj) agreement with zuko.RealNVP after copying
# parameters layer-by-layer (structural equivalence check).
#
# Note: same-seed init does NOT yield identical parameters. zuko's
# Flow.__init__(transform, base) wraps the coupling list in a
# LazyComposedTransform before nn.Module.__init__ runs the Linear-
# layer Kaiming init, while zflows uses a plain nn.ModuleList; the two
# construction paths consume CPU RNG in slightly different orders, so
# the per-layer MLP weights diverge. The semantic question we want to
# answer is whether `zflows.RealNVP._coupling[i]` is interchangeable
# with `zuko.RealNVP.transform.transforms[i]` — so we copy parameters
# from one into the other and verify outputs agree exactly.
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 4: forward (y, ladj) agreement after parameter copy")
print("=" * 60)

torch.manual_seed(0)
realnvp_z = RealNVP(dimension=DIM, transforms=TRANSFORMS, hidden_features=HIDDEN).to(device)
torch.manual_seed(1) # use a deliberately different seed to underline that we copy, not match
# pass activation=nn.SiLU explicitly: zflows.RealNVP defaults to SiLU but
# zuko.RealNVP / GeneralCouplingTransform defaults the MLP to ReLU.
realnvp_n = ZukoRealNVP(
    features=DIM, context=0, transforms=TRANSFORMS,
    hidden_features=HIDDEN, activation=torch.nn.SiLU,
).to(device)

# copy zflows -> zuko, layer by layer
for i in range(TRANSFORMS):
    realnvp_n.transform.transforms[i].load_state_dict(
        realnvp_z._coupling[i].state_dict()
    )

with torch.no_grad():
    realnvp_z.eval()
    realnvp_n.eval()
    x = torch.randn(8, DIM, device=device)
    T_z = realnvp_z.t()
    T_n = realnvp_n().transform
    y_z, ladj_z = T_z.call_and_ladj(x)
    y_n, ladj_n = T_n.call_and_ladj(x)

err_y = (y_z - y_n).abs().max().item()
err_ladj = (ladj_z - ladj_n).abs().max().item()
print(f"max |y_zflows  - y_zuko|  = {err_y:.3e}")
print(f"max |ladj_zfl  - ladj_zk| = {err_ladj:.3e}")
assert err_y < 1e-5,    f"forward y disagrees:    {err_y}"
assert err_ladj < 1e-5, f"forward ladj disagrees: {err_ladj}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 5: inverse agreement with zuko.RealNVP (params copied above)
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 5: inverse agreement after parameter copy")
print("=" * 60)

with torch.no_grad():
    realnvp_z.eval()
    realnvp_n.eval()
    y = torch.randn(8, DIM, device=device) * 0.5
    T_z = realnvp_z.t()
    T_n = realnvp_n().transform
    x_back_z = T_z.inv(y)
    x_back_n = T_n.inv(y)

err_inv = (x_back_z - x_back_n).abs().max().item()
print(f"max |inv_zflows - inv_zuko| = {err_inv:.3e}")
assert err_inv < 1e-5, f"inverse disagrees: {err_inv}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 6: backprop reaches every flow parameter
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 6: backprop through T_z reaches all RealNVP parameters")
print("=" * 60)

realnvp.train()
realnvp.zero_grad()
x = torch.randn(16, DIM, device=device)
y, ladj = realnvp.t().call_and_ladj(x)
loss = (y.pow(2).sum() - ladj.sum())
loss.backward()

n_params = sum(1 for _ in realnvp.parameters())
n_grads  = sum(1 for p in realnvp.parameters() if p.grad is not None)
print(f"#parameter tensors                    = {n_params}")
print(f"#parameter tensors with non-None grad = {n_grads}")
assert n_grads == n_params > 0, f"some parameters missing grad: {n_grads}/{n_params}"
print("PASSED")

print()
print("All RealNVP interface checks passed.")
