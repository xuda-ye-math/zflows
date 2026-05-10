# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""
Sanity checks for the .zeros() initialisers on every zflows.Flow subclass.

Each Flow class exposes a `.zeros()` method that initialises the network
to produce the identity bijection. After calling it, we verify:

    T(x)        == x
    ladj(T, x)  == 0

up to numerical tolerance, on inputs sampled inside the relevant domain:

    NSF     : [a, b]^d
    NCSF    : [a, b]^d (defaults [-pi, pi]^d)
    RealNVP : R^d
    CNF     : R^d (ODE-integrated, looser tolerance)
"""

import math
import torch
from zflows.flow import NSF, NCSF, RealNVP, CNF

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# ----------------------------------------------------------------------
# Test 1: NSF.zeros() => identity on [a, b]^d
# ----------------------------------------------------------------------
print("=" * 60)
print("Test 1: NSF.zeros() yields identity on [a, b]^d")
print("=" * 60)

a, b = [-1.0, -2.0], [1.0, 3.0]
nsf = NSF(a=a, b=b, bins=7, transforms=4, hidden_features=(64, 64)).to(device)
nsf.zeros()
nsf.eval()

with torch.no_grad():
    a_t = torch.tensor(a, device=device)
    b_t = torch.tensor(b, device=device)
    u = torch.rand(64, 2, device=device)
    x = a_t + (b_t - a_t) * u             # uniform in [a, b]^d
    T = nsf.t()
    y, ladj = T.call_and_ladj(x)

err_y    = (y - x).abs().max().item()
err_ladj = ladj.abs().max().item()
print(f"max |T(x) - x|  = {err_y:.3e}")
print(f"max |ladj|      = {err_ladj:.3e}")
assert err_y    < 1e-5, f"NSF.zeros() not identity: {err_y}"
assert err_ladj < 1e-5, f"NSF.zeros() ladj nonzero: {err_ladj}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 2: NCSF.zeros() => identity on [a, b]^d
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 2: NCSF.zeros() yields identity on [a, b]^d")
print("=" * 60)

a, b = [-math.pi, -math.pi], [math.pi, math.pi]
ncsf = NCSF(a=a, b=b, bins=7, transforms=4, hidden_features=(64, 64)).to(device)
ncsf.zeros()
ncsf.eval()

with torch.no_grad():
    a_t = torch.tensor(a, device=device)
    b_t = torch.tensor(b, device=device)
    u = torch.rand(64, 2, device=device)
    x = a_t + (b_t - a_t) * u
    T = ncsf.t()
    y, ladj = T.call_and_ladj(x)

err_y    = (y - x).abs().max().item()
err_ladj = ladj.abs().max().item()
print(f"max |T(x) - x|  = {err_y:.3e}")
print(f"max |ladj|      = {err_ladj:.3e}")
assert err_y    < 1e-4, f"NCSF.zeros() not identity: {err_y}"
assert err_ladj < 1e-4, f"NCSF.zeros() ladj nonzero: {err_ladj}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 3: RealNVP.zeros() => identity on R^d
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 3: RealNVP.zeros() yields identity on R^d")
print("=" * 60)

realnvp = RealNVP(dimension=4, transforms=4, hidden_features=(32, 32)).to(device)
realnvp.zeros()
realnvp.eval()

with torch.no_grad():
    x = torch.randn(64, 4, device=device)
    T = realnvp.t()
    y, ladj = T.call_and_ladj(x)

err_y    = (y - x).abs().max().item()
err_ladj = ladj.abs().max().item()
print(f"max |T(x) - x|  = {err_y:.3e}")
print(f"max |ladj|      = {err_ladj:.3e}")
assert err_y    < 1e-5, f"RealNVP.zeros() not identity: {err_y}"
assert err_ladj < 1e-5, f"RealNVP.zeros() ladj nonzero: {err_ladj}"
print("PASSED")

# ----------------------------------------------------------------------
# Test 4: CNF.zeros() => identity on R^d (ODE-integrated, looser tol)
# ----------------------------------------------------------------------
print()
print("=" * 60)
print("Test 4: CNF.zeros() yields identity on R^d")
print("=" * 60)

cnf = CNF(dimension=2, frequency=3, hidden_features=(32, 32)).to(device)
cnf.zeros()
cnf.eval()

with torch.no_grad():
    x = torch.randn(64, 2, device=device)
    T = cnf.t()
    y, ladj = T.call_and_ladj(x)

err_y    = (y - x).abs().max().item()
err_ladj = ladj.abs().max().item()
print(f"max |T(x) - x|  = {err_y:.3e}")
print(f"max |ladj|      = {err_ladj:.3e}")
assert err_y    < 1e-4, f"CNF.zeros() not identity: {err_y}"
assert err_ladj < 1e-4, f"CNF.zeros() ladj nonzero: {err_ladj}"
print("PASSED")

print()
print("All .zeros() identity checks passed.")
