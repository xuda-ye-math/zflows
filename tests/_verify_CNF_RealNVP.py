"""Verification script for CNF / RealNVP correctness.

Run from the repo root:  python -m tests._verify_CNF_RealNVP

Both flows live on R^d natively (no [a, b] box, no sandwich) — these
checks are simpler than NSF / NCSF's.

Checks:
  1. Construction.
  2. Forward / inverse bijection round-trip.
  3. zeros() initialisation gives the identity (y == x, ladj == 0).
  4. log|det J| from call_and_ladj matches torch.autograd slogdet.
  5. RealNVP closed-form ladj_fwd + ladj_inv == 0 (perfect bijection).
  6. CNF Hutchinson (`exact=False`) gives the same SCALE of ladj as
     the exact path (unbiased estimator).
  7. Backprop reaches all parameters.
"""

import sys

import torch

from zflows.flow import CNF, RealNVP


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
device = "cpu"

# ─────────────────────────────────────────────────────────────────
banner("1. Construction")
realnvp = RealNVP(dimension=4, transforms=4, hidden_features=(32, 32)).to(device)
print(f"  RealNVP d=4: {sum(p.numel() for p in realnvp.parameters())} params, "
      f"{len(realnvp._coupling)} coupling layers")

cnf = CNF(dimension=2, frequency=3, hidden_features=(32, 32)).to(device)
print(f"  CNF d=2: {sum(p.numel() for p in cnf.parameters())} params")

cnf_hutch = CNF(dimension=2, frequency=3, hidden_features=(32, 32), exact=False).to(device)
print(f"  CNF d=2 (Hutchinson): {sum(p.numel() for p in cnf_hutch.parameters())} params")

# ─────────────────────────────────────────────────────────────────
banner("2. Bijection round-trip")
realnvp.eval()
cnf.eval()

# RealNVP: closed-form inverse — machine precision expected
x = torch.randn(256, 4)
with torch.no_grad():
    F = realnvp.t()
    y = F(x)
    x_back = F.inv(y)
assert_close(x_back, x, atol=1e-5, name="RealNVP  F.inv(F(x)) ≈ x")

# CNF: ODE-based inverse — looser tol (~1e-4)
x = torch.randn(64, 2) * 0.7  # bounded init for ODE stability
with torch.no_grad():
    F = cnf.t()
    y = F(x)
    x_back = F.inv(y)
assert_close(x_back, x, atol=5e-4, name="CNF      F.inv(F(x)) ≈ x")

# ─────────────────────────────────────────────────────────────────
banner("3. zeros() initialisation → identity bijection")

realnvp_z = RealNVP(dimension=4, transforms=4, hidden_features=(32, 32)).to(device)
realnvp_z.zeros()
realnvp_z.eval()
x = torch.randn(64, 4)
with torch.no_grad():
    F = realnvp_z.t()
    y, ladj = F.call_and_ladj(x)
assert_close(y, x, atol=1e-5, name="RealNVP zeros  y ≈ x")
assert_close(ladj, torch.zeros_like(ladj), atol=1e-5, name="RealNVP zeros  ladj ≈ 0")

cnf_z = CNF(dimension=2, frequency=3, hidden_features=(32, 32)).to(device)
cnf_z.zeros()
cnf_z.eval()
x = torch.randn(64, 2)
with torch.no_grad():
    F = cnf_z.t()
    y, ladj = F.call_and_ladj(x)
assert_close(y, x, atol=1e-5, name="CNF zeros      y ≈ x")
assert_close(ladj, torch.zeros_like(ladj), atol=1e-5, name="CNF zeros      ladj ≈ 0")

# ─────────────────────────────────────────────────────────────────
banner("4. log|det J| via call_and_ladj matches autograd slogdet")

# RealNVP — exact closed-form, very tight tolerance
realnvp = RealNVP(dimension=4, transforms=4, hidden_features=(32, 32)).to(device)
realnvp.eval()
torch.manual_seed(42)
x = torch.randn(5, 4)
with torch.no_grad():
    F = realnvp.t()
    _, ladj_fast = F.call_and_ladj(x)
ladj_exact = torch.stack([
    exact_log_abs_det_jacobian(F, x[i:i + 1]) for i in range(x.size(0))
])
assert_close(ladj_fast, ladj_exact, atol=1e-4, name="RealNVP  ladj ≈ slogdet")

# CNF exact — looser tolerance because slogdet on Jacobian of an ODE
# solver picks up FP noise too
cnf = CNF(dimension=2, frequency=3, hidden_features=(32, 32)).to(device)
cnf.eval()
x = torch.randn(5, 2) * 0.5
with torch.no_grad():
    F = cnf.t()
    _, ladj_fast = F.call_and_ladj(x)
ladj_exact = torch.stack([
    exact_log_abs_det_jacobian(F, x[i:i + 1]) for i in range(x.size(0))
])
assert_close(ladj_fast, ladj_exact, atol=5e-3, name="CNF      ladj ≈ slogdet")

# ─────────────────────────────────────────────────────────────────
banner("5. RealNVP forward+inverse ladj cancel")
realnvp.eval()
x = torch.randn(64, 4)
with torch.no_grad():
    F = realnvp.t()
    y, ladj_fwd = F.call_and_ladj(x)
    x_back, ladj_inv = F.inv.call_and_ladj(y)
assert_close(
    ladj_fwd + ladj_inv, torch.zeros_like(ladj_fwd),
    atol=1e-5, name="RealNVP  ladj_fwd + ladj_inv ≈ 0",
)
assert_close(x_back, x, atol=1e-5, name="RealNVP  inv(F(x)) ≈ x")

# ─────────────────────────────────────────────────────────────────
banner("6. CNF Hutchinson estimator is unbiased")
# Compare the exact ladj to the mean of M Hutchinson trials. Both
# evaluate the SAME drift (same params), so the Hutchinson estimator
# averaged over many trials should converge to the exact value.

# Build matching exact / Hutchinson CNFs by copying params.
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
print(f"  exact ladj      = {ladj_ex.detach().cpu().tolist()}")
print(f"  Hutch avg (M={M}) = {ladj_h_avg.detach().cpu().tolist()}")
print(f"  max |avg - exact| = {diff:.3e}")
# Hutchinson MC has O(1/sqrt(M)) noise; this should be << 1 in absolute
# value at M=32 for d=2. Looser bound here.
assert diff < 1e-1, f"Hutchinson estimator far from exact: {diff}"
print("  [OK ] Hutchinson estimator ≈ exact within MC noise")

# ─────────────────────────────────────────────────────────────────
banner("7. Backprop reaches every parameter")
for name, flow in [("RealNVP", realnvp), ("CNF", cnf)]:
    flow.train()
    flow.zero_grad()
    if name == "RealNVP":
        x = torch.randn(16, 4)
    else:
        x = torch.randn(16, 2) * 0.5
    y, ladj = flow.t().call_and_ladj(x)
    loss = (y.pow(2).sum() - ladj.sum())
    loss.backward()
    n_params = sum(1 for _ in flow.parameters())
    n_grads = sum(1 for p in flow.parameters() if p.grad is not None)
    print(f"  {name}: {n_grads}/{n_params} parameter tensors have non-None grad")
    if n_grads != n_params:
        print(f"  FAIL: {name} missing grads on {n_params - n_grads} params")
        sys.exit(1)
print("  [OK ] every parameter receives gradients")

print()
print("All CNF / RealNVP verification checks passed.")
