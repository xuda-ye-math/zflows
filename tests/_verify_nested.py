# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Verification script for NESTED `linear_combination` potentials.

Run from the repo root:  python -m tests._verify_nested

`Linear_Combination`'s class docstring marks nesting (a `linear_combination`
instance used as a *child* of another) as untested / not recommended. This
harness actually exercises it, focused on the two compiled fast paths
(`.grad(x)` and `.eval(x)`), since the nesting concern is precisely the
reduce-overhead static-buffer aliasing of the children's compiled closures.

How nesting works: each `Linear_Combination` links combined `_grad_fn` /
`_eval_fn` closures at `__init__` that compute `sum_k coeffs[k] * U_k.grad(x)`
(resp. `.eval(x)`). When a child `U_k` is itself a `linear_combination`, its
`.grad`/`.eval` is its own combined closure, so the recursion composes; and the
`c * U.grad(x)` product allocates a fresh tensor on every term, decoupling it
from any reduce-overhead static buffer before the next child overwrites it.
`enable_grad()` / `enable_eval()` propagate to every child, so a single call on
the outermost combination cascades down to all leaves.

Sections:
   A. 2-level nesting: eval == forward == analytic; grad == autograd.
   B. one `enable_grad/eval` on the OUTER combination cascades to all leaves.
   C. 3-level deep nesting: eval + grad vs autograd.
   D. set_coeffs at inner / outer levels propagates (closures read coeffs fresh).
   E. tensor coeffs at a nested level still normalise + compute correctly.
   F. mode='default' and mode='reduce-overhead' agree numerically.
   G. gating: an un-enabled leaf inside a nest raises RuntimeError on .grad/.eval.
   H. downstream: MALA / HMC on a nested-LC Gaussian hits the analytic moments.
"""

import torch

from zflows.potential import Gaussian, Potential, linear_combination, potential_from
from zflows.utils import langevin, hamiltonian_monte_carlo, set_cache_size_limit, suppress_warnings

suppress_warnings()
set_cache_size_limit(128) # many leaf closures compile across the nested sections


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


def assert_close(actual, expected, atol, name):
    diff = (actual - expected).abs().max().item()
    ok = diff <= atol
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: max |diff| = {diff:.3e}  (atol={atol})")
    assert ok, f"{name} exceeded atol={atol} (diff={diff:.3e})"


def autograd_grad(u: Potential, x: torch.Tensor) -> torch.Tensor:
    """True gradient via autograd through the plain (recursive) forward."""
    xr = x.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad(u(xr).sum(), xr)
    return g.detach()


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print(f"device: {device}")

D = 3
ATOL = 1e-4

# Leaf potentials: two Gaussians + one non-Gaussian `potential_from`.
def make_leaves():
    u0 = Gaussian(mean=[0.0] * D, variance=[1.0] * D).to(device)
    u1 = Gaussian(mean=[1.5] * D, variance=[0.5] * D).to(device)
    u2 = potential_from(lambda x: 0.25 * (x ** 4).sum(dim=-1) + (2.0 * torch.cos(x[:, 0]))).to(device)
    u3 = Gaussian(mean=[-1.0] * D, variance=[2.0] * D).to(device)
    return u0, u1, u2, u3


# ══════════════════════════════════════════════════════════════════
# A. 2-level nesting: eval / grad correctness
# ══════════════════════════════════════════════════════════════════
banner("A. 2-level nested linear_combination: eval & grad vs reference")
u0, u1, u2, u3 = make_leaves()
lc_inner = linear_combination([u0, u1], [0.4, 0.6])           # 0.4*u0 + 0.6*u1
lc_outer = linear_combination([lc_inner, u2], [0.7, 0.3])     # 0.7*lc_inner + 0.3*u2
lc_outer.enable_grad().enable_eval()                          # cascades to u0, u1, u2

x = torch.randn(2000, D, device=device)

# A.1 — eval matches the plain recursive forward and the hand-written analytic
section("A.1  outer.eval(x) == outer(x) == 0.7*(0.4*u0+0.6*u1)+0.3*u2")
with torch.no_grad():
    analytic = 0.7 * (0.4 * u0(x) + 0.6 * u1(x)) + 0.3 * u2(x)
    assert_close(lc_outer.eval(x), lc_outer(x), atol=ATOL, name="A.1  eval == forward")
    assert_close(lc_outer.eval(x), analytic,    atol=ATOL, name="A.1  eval == analytic")

# A.2 — compiled grad matches autograd through the nested forward + analytic grad
section("A.2  outer.grad(x) == autograd == 0.7*(0.4*g0+0.6*g1)+0.3*g2")
with torch.no_grad():
    g_an = 0.7 * (0.4 * u0.grad(x) + 0.6 * u1.grad(x)) + 0.3 * u2.grad(x)
assert_close(lc_outer.grad(x), autograd_grad(lc_outer, x), atol=ATOL, name="A.2  grad == autograd")
assert_close(lc_outer.grad(x), g_an,                       atol=ATOL, name="A.2  grad == analytic")


# ══════════════════════════════════════════════════════════════════
# B. a single enable on the OUTER combination cascades to every leaf
# ══════════════════════════════════════════════════════════════════
banner("B. enable_grad/eval on the outer LC cascades to all nested leaves")
section("B.1  leaves start un-enabled; one outer enable hot-loads them all")
u0, u1, u2, u3 = make_leaves()
inner = linear_combination([u0, u1], [0.5, 0.5])
outer = linear_combination([inner, u2], [0.5, 0.5])
before = [leaf._grad_fn is None and leaf._eval_fn is None for leaf in (u0, u1, u2)]
print(f"  leaves un-enabled before: {before}")
assert all(before), "leaves should start with no compiled fast paths"
ret = outer.enable_grad().enable_eval()
assert ret is outer, "enable_* must return self for chaining"
after = [leaf._grad_fn is not None and leaf._eval_fn is not None for leaf in (u0, u1, u2)]
print(f"  leaves enabled after one outer.enable_grad().enable_eval(): {after}")
assert all(after), "a single outer enable must cascade to every nested leaf"
# and the fast paths actually work end-to-end
with torch.no_grad():
    assert_close(outer.eval(x), outer(x),                  atol=ATOL, name="B.1  cascaded eval works")
assert_close(outer.grad(x), autograd_grad(outer, x),       atol=ATOL, name="B.1  cascaded grad works")


# ══════════════════════════════════════════════════════════════════
# C. 3-level deep nesting
# ══════════════════════════════════════════════════════════════════
banner("C. 3-level deep nesting: eval & grad vs autograd")
u0, u1, u2, u3 = make_leaves()
lc_l1 = linear_combination([u0, u1], [0.3, 0.7])                 # level 1
lc_l2 = linear_combination([lc_l1, u2], [0.8, 0.2])             # level 2 (nests l1)
lc_l3 = linear_combination([lc_l2, u3], [0.6, 0.4])            # level 3 (nests l2)
lc_l3.enable_grad().enable_eval()                              # cascades l3 -> l2 -> l1 -> leaves
section("C.1  depth-3 eval == forward; grad == autograd")
with torch.no_grad():
    # effective leaf weights along the tree: u0:0.6*0.8*0.3, u1:0.6*0.8*0.7, u2:0.6*0.2, u3:0.4
    w0, w1, w2, w3 = 0.6 * 0.8 * 0.3, 0.6 * 0.8 * 0.7, 0.6 * 0.2, 0.4
    eff = w0 * u0(x) + w1 * u1(x) + w2 * u2(x) + w3 * u3(x)
    assert_close(lc_l3.eval(x), lc_l3(x), atol=ATOL, name="C.1  depth-3 eval == forward")
    assert_close(lc_l3.eval(x), eff,      atol=ATOL, name="C.1  depth-3 eval == effective-weight sum")
assert_close(lc_l3.grad(x), autograd_grad(lc_l3, x), atol=ATOL, name="C.1  depth-3 grad == autograd")


# ══════════════════════════════════════════════════════════════════
# D. set_coeffs propagates through the nest (closures read coeffs fresh)
# ══════════════════════════════════════════════════════════════════
banner("D. set_coeffs at inner / outer levels is picked up with no recompile")
u0, u1, u2, u3 = make_leaves()
inner = linear_combination([u0, u1], [0.5, 0.5])
outer = linear_combination([inner, u2], [0.5, 0.5])
outer.enable_grad().enable_eval()
section("D.1  retune inner AND outer; eval/grad reflect the new mix immediately")
inner.set_coeffs([0.2, 0.8])
outer.set_coeffs([0.9, 0.1])
with torch.no_grad():
    new_eval = 0.9 * (0.2 * u0(x) + 0.8 * u1(x)) + 0.1 * u2(x)
    new_grad = 0.9 * (0.2 * u0.grad(x) + 0.8 * u1.grad(x)) + 0.1 * u2.grad(x)
    assert_close(outer.eval(x), new_eval, atol=ATOL, name="D.1  eval tracks set_coeffs")
assert_close(outer.grad(x), new_grad,        atol=ATOL, name="D.1  grad tracks set_coeffs")
assert_close(outer.grad(x), autograd_grad(outer, x), atol=ATOL, name="D.1  grad == autograd after retune")


# ══════════════════════════════════════════════════════════════════
# E. tensor coeffs at a nested level
# ══════════════════════════════════════════════════════════════════
banner("E. tensor coeffs inside a nest normalise to list[float] + compute right")
u0, u1, u2, u3 = make_leaves()
inner = linear_combination([u0, u1], torch.tensor([0.35, 0.65]))      # 1-d tensor coeffs
outer = linear_combination([inner, u2], torch.tensor([0.55, 0.45]))
outer.enable_grad().enable_eval()
section("E.1  coeffs stored as list[float]; eval/grad correct")
assert isinstance(inner.coeffs, list) and all(isinstance(c, float) for c in inner.coeffs), \
    "tensor coeffs must normalise to list[float]"
assert isinstance(outer.coeffs, list) and all(isinstance(c, float) for c in outer.coeffs)
print(f"  inner.coeffs={inner.coeffs}  outer.coeffs={outer.coeffs}")
with torch.no_grad():
    ref = 0.55 * (0.35 * u0(x) + 0.65 * u1(x)) + 0.45 * u2(x)
    assert_close(outer.eval(x), ref, atol=ATOL, name="E.1  tensor-coeff nest eval")
assert_close(outer.grad(x), autograd_grad(outer, x), atol=ATOL, name="E.1  tensor-coeff nest grad")


# ══════════════════════════════════════════════════════════════════
# F. compile modes agree (the reduce-overhead static-buffer concern)
# ══════════════════════════════════════════════════════════════════
banner("F. default vs reduce-overhead nested grad/eval agree numerically")
u0, u1, u2, u3 = make_leaves()
inner_d = linear_combination([u0, u1], [0.4, 0.6])
outer_d = linear_combination([inner_d, u2], [0.7, 0.3])
outer_d.enable_grad(mode="default").enable_eval(mode="default")
v0, v1, v2, v3 = make_leaves()
inner_r = linear_combination([v0, v1], [0.4, 0.6])
outer_r = linear_combination([inner_r, v2], [0.7, 0.3])
outer_r.enable_grad(mode="reduce-overhead").enable_eval(mode="reduce-overhead")
section("F.1  same weights -> default and reduce-overhead match (no buffer aliasing)")
with torch.no_grad():
    assert_close(outer_r.eval(x), outer_d.eval(x), atol=ATOL, name="F.1  eval default == reduce")
assert_close(outer_r.grad(x), outer_d.grad(x),     atol=ATOL, name="F.1  grad default == reduce")


# ══════════════════════════════════════════════════════════════════
# G. gating: an un-enabled leaf inside a nest raises at runtime
# ══════════════════════════════════════════════════════════════════
banner("G. runtime gating: un-enabled nested leaf raises RuntimeError")
section("G.1  enable only some leaves; outer.grad/.eval hit the gate on the rest")
g0, g1, g2, g3 = make_leaves()
g0.enable_grad().enable_eval()           # leaf 0 enabled
g2.enable_grad().enable_eval()           # sibling enabled
# g1 deliberately left un-enabled
inner = linear_combination([g0, g1], [0.5, 0.5])   # combined _grad_fn/_eval_fn set, but g1 cold
outer = linear_combination([inner, g2], [0.5, 0.5])
for meth in ("grad", "eval"):
    raised = False
    try:
        getattr(outer, meth)(torch.randn(8, D, device=device))
    except RuntimeError as e:
        raised = True
        print(f"  outer.{meth}(x) with cold nested leaf raised: {str(e)[:70]}")
    assert raised, f"outer.{meth} must raise when a nested leaf lacks enable_{meth}()"
print("  [OK ] cold nested leaf is caught at the child-level runtime gate")


# ══════════════════════════════════════════════════════════════════
# H. downstream: MALA / HMC on a nested-LC Gaussian -> analytic moments
# ══════════════════════════════════════════════════════════════════
banner("H. samplers consume the nested compiled grad: hit the analytic Gaussian")
# A positive-weighted nest of diagonal Gaussians is itself a diagonal Gaussian
# potential U(x) = sum_i w_i * 0.5 * ((x - m_i)^2 / v_i), with precision
# A_jj = sum_i w_i / v_ij, mean mu_j = (sum_i w_i m_ij / v_ij) / A_jj, cov 1/A_jj.
m = torch.tensor([[0.0, 0.0], [2.0, -1.0], [-1.0, 3.0]], device=device)
v = torch.tensor([[1.0, 2.0], [0.5, 1.0], [2.0, 0.5]], device=device)
gA = Gaussian(mean=m[0].tolist(), variance=v[0].tolist()).to(device)
gB = Gaussian(mean=m[1].tolist(), variance=v[1].tolist()).to(device)
gC = Gaussian(mean=m[2].tolist(), variance=v[2].tolist()).to(device)
# nest: outer = 0.5*inner + 0.5*gC, inner = 0.6*gA + 0.4*gB  ->  w = [0.3, 0.2, 0.5]
inner = linear_combination([gA, gB], [0.6, 0.4])
u_nest = linear_combination([inner, gC], [0.5, 0.5]).enable_grad().enable_eval()
w = torch.tensor([0.3, 0.2, 0.5], device=device)
A = (w[:, None] / v).sum(0)                              # [2] precision diagonal
mu = (w[:, None] * m / v).sum(0) / A                     # [2] analytic mean
sd = (1.0 / A).sqrt()                                    # [2] analytic per-axis std
print(f"  analytic  mean = {[f'{c:+.3f}' for c in mu.tolist()]}  std = {[f'{c:.3f}' for c in sd.tolist()]}")

section("H.1  MALA on the nested-LC potential reproduces the analytic Gaussian")
torch.manual_seed(1)
# Start at the analytic Gaussian and verify the sampler keeps it stationary:
# a correct nested compiled grad => correct stationary distribution (a wrong
# grad would drift the mean / rescale the std away from the analytic values).
x0 = mu + sd * torch.randn(20000, 2, device=device)
xm = langevin(x0, u_nest, step=1e-2, iters=300, adjust=True, chunk=2)  # MALA needs grad + eval
print(f"  MALA      mean = {[f'{c:+.3f}' for c in xm.mean(0).tolist()]}  std = {[f'{c:.3f}' for c in xm.std(0).tolist()]}")
assert_close(xm.mean(0), mu, atol=0.05, name="H.1  MALA mean == analytic")
assert_close(xm.std(0),  sd, atol=0.05, name="H.1  MALA std  == analytic")

section("H.2  HMC on the nested-LC potential reproduces the analytic Gaussian")
torch.manual_seed(2)
xh = hamiltonian_monte_carlo(x0, u_nest, step=0.1, iters=10, burns=40, chunk=2)
print(f"  HMC       mean = {[f'{c:+.3f}' for c in xh.mean(0).tolist()]}  std = {[f'{c:.3f}' for c in xh.std(0).tolist()]}")
assert_close(xh.mean(0), mu, atol=0.05, name="H.2  HMC mean == analytic")
assert_close(xh.std(0),  sd, atol=0.05, name="H.2  HMC std  == analytic")


# ─────────────────────────────────────────────────────────────────
print()
print("All nested linear_combination verification checks passed.")
