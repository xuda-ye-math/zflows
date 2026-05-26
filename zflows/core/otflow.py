"""Optimal-transport flow (OT-Flow) machinery for zflows.

Ported from the reference implementation of

    Onken, Fung, Li, Ruthotto.
    "OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal
     Transport." AAAI 2021. https://arxiv.org/abs/2006.00104

(`OT-Flow/src/Phi.py`, `OT-Flow/src/OTFlowProblem.py`) and adapted to the
zflows `Transform` interface.

The defining idea: parameterise the ODE velocity field as the (negative)
gradient of a scalar potential, `v_theta(t, x) = -∇_x Φ_theta(t, x)`. Because
`Φ` has a specific antiderivative-of-tanh ResNet + low-rank quadratic
structure, the divergence `tr(∇_x v) = -tr(∇²_x Φ)` is available in *closed
form* (`OTPhi.trHess`) at `O(d·m)` cost per evaluation — no Hutchinson
estimator and no augmented O(d) Jacobian ODE as in FFJORD.

Public objects:
    - antideriv_tanh / deriv_tanh — the activation and its 2nd derivative
    - ResNN                       — the residual network N inside Φ
    - OTPhi                       — the scalar potential Φ_theta
    - OTFlowTransform             — Transform integrating dx/dt = -∇Φ via RK4
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Transform, constraints

from .numerics import rk4_fixed


__all__ = [
    "OTFlowTransform",
    "OTPhi",
    "ResNN",
    "antideriv_tanh",
    "deriv_tanh",
]


# ──────────────────────────────────────────────────────────────────────
# Activations — antiderivative of tanh and its second derivative
# ──────────────────────────────────────────────────────────────────────

def antideriv_tanh(x: Tensor) -> Tensor:
    """Antiderivative of tanh: ∫tanh = log cosh, written stably.

    `act(x) = |x| + log(1 + exp(-2|x|))` equals `log(cosh(x)) + log 2`,
    computed via the |x| factorisation to avoid overflow. Its first
    derivative is `tanh`, which is what makes the closed-form Hessian
    trace in `OTPhi.trHess` possible.
    """
    return torch.abs(x) + torch.log1p(torch.exp(-2.0 * torch.abs(x)))


def deriv_tanh(x: Tensor) -> Tensor:
    """Second derivative of `antideriv_tanh`, i.e. tanh'(x) = 1 - tanh(x)²."""
    return 1 - torch.tanh(x).pow(2)


# ──────────────────────────────────────────────────────────────────────
# ResNN — residual network N(s) inside the potential Φ
# ──────────────────────────────────────────────────────────────────────

class ResNN(nn.Module):
    """Residual network on space-time inputs `s = [x; t]` of width `dimension + 1`.

    Layout (matching the OT-Flow reference):
        u_0     = act(K_0 s + b_0)
        u_i     = u_{i-1} + h · act(K_i u_{i-1} + b_i),   i = 1 … layer-1
    with step `h = 1/(layer-1)` and `act = antideriv_tanh`. There are `layer`
    linear layers total: one opening `(dimension+1) → hidden` and `layer-1`
    square `hidden → hidden` layers.

    Arguments:
        dimension: spatial dimension d (network sees `dimension + 1` with
            time appended).
        hidden: hidden width of the ResNet.
        layer: number of ResNet layers (>= 2).
    """

    def __init__(self, dimension: int, hidden: int, layer: int = 2) -> None:
        super().__init__()
        assert layer >= 2, "layer must be an integer >= 2"
        self.dimension = dimension
        self.hidden = hidden
        self.layer = layer
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dimension + 1, hidden, bias=True))  # opening layer
        for _ in range(layer - 1):  # residual layers
            self.layers.append(nn.Linear(hidden, hidden, bias=True))
        self.act = antideriv_tanh
        self.h = 1.0 / (self.layer - 1)  # ResNet step size

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation N(s); x is `(nex, dimension+1)`, returns `(nex, hidden)`."""
        x = self.act(self.layers[0].forward(x))
        for i in range(1, self.layer):
            x = x + self.h * self.act(self.layers[i](x))
        return x


# ──────────────────────────────────────────────────────────────────────
# OTPhi — scalar potential Φ_theta with closed-form gradient + Hessian trace
# ──────────────────────────────────────────────────────────────────────

class OTPhi(nn.Module):
    r"""Scalar potential `Φ(s) = wᵀN(s) + ½ sᵀ(AᵀA) s + bᵀs + c`, `s = [x; t]`.

    The low-rank quadratic term uses `A ∈ ℝ^{r×(d+1)}` (so `AᵀA` is a
    rank-`r` symmetric PSD matrix); `b` is the weight of a single linear
    head. `trHess` returns both `∇_s Φ` and `tr(∇²_x Φ)` analytically — see
    Eq. (11) and Eq. (13) of the OT-Flow paper — using the fact that
    `act' = tanh` and `act'' = 1 - tanh²`.

    Note the reference's constant bias `c` is dropped: only `∇Φ`, `tr∇²Φ`,
    and `∂_tΦ` ever feed the flow, none of which depend on an additive
    constant, so that bias is non-identifiable and would carry a permanently
    zero gradient. Every remaining parameter is identifiable.

    Initialised so `Φ ≡ 0` would require zeroing every head; by default the
    ResNet and `A` are randomly initialised while `w = 1`. The flow's
    `.zeros()` zeros all heads to recover the identity bijection.

    Arguments:
        dimension: spatial dimension d (network sees `dimension + 1` with
            time appended).
        hidden: hidden width of the ResNet.
        layer: number of ResNet layers (>= 2).
        rank: rank of the quadratic term (clamped to `<= dimension + 1`).
    """

    def __init__(self, dimension: int, hidden: int, layer: int, rank: int = 10) -> None:
        super().__init__()
        self.dimension = dimension
        self.hidden = hidden
        self.layer = layer

        rank = min(rank, dimension + 1)  # rank cannot exceed the input dimension
        self.A = nn.Parameter(torch.empty(rank, dimension + 1))
        nn.init.xavier_uniform_(self.A)
        self.c = nn.Linear(dimension + 1, 1, bias=False)  # bᵀ[x;t]
        self.w = nn.Linear(hidden, 1, bias=False)
        self.N = ResNN(dimension, hidden, layer)

        # Match the reference initial values: w = 1, b = 0.
        nn.init.ones_(self.w.weight)
        nn.init.zeros_(self.c.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Φ(s) for `s = x` of shape `(nex, d+1)`; returns `(nex, 1)`."""
        symA = self.A.t() @ self.A  # AᵀA, shape (d+1, d+1)
        quad = 0.5 * torch.sum((x @ symA) * x, dim=1, keepdim=True)
        return self.w(self.N(x)) + quad + self.c(x)

    def trHess(self, x: Tensor, justGrad: bool = False):
        """Closed-form `∇_s Φ` and `tr(∇²_x Φ)` (trace over the spatial block).

        `x` is `(nex, d+1)` space-time input. Returns `∇_s Φ` of shape
        `(nex, d+1)` (the last column is `∂Φ/∂t`); when `justGrad=False`
        also returns `tr(∇²_x Φ)` of shape `(nex,)`, summing only the
        spatial `d×d` block of the Hessian. Recomputes the ResNet forward
        pass internally (it needs the per-layer pre-activations).
        """
        N = self.N
        m = N.layers[0].weight.shape[0]
        nex = x.shape[0]
        d = x.shape[1] - 1
        symA = self.A.t() @ self.A

        u = []                 # u_0 … u_{layer-1} from the forward pass
        z = N.layer * [None]   # z_0 … z_{layer-1} from the gradient backward pass

        # Forward pass through the ResNet, caching pre-/post-activations.
        opening = N.layers[0].forward(x)  # K_0 s + b_0
        u.append(N.act(opening))          # u_0
        feat = u[0]
        for i in range(1, N.layer):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        tanhopen = torch.tanh(opening)  # act'(K_0 s + b_0)

        # Backward pass accumulating the gradient z_i.
        for i in range(N.layer - 1, 0, -1):
            term = self.w.weight.t() if i == N.layer - 1 else z[i + 1]
            z[i] = term + N.h * torch.mm(
                N.layers[i].weight.t(),
                torch.tanh(N.layers[i].forward(u[i - 1])).t() * term,
            )
        z[0] = torch.mm(N.layers[0].weight.t(), tanhopen.t() * z[1])
        grad = z[0] + torch.mm(symA, x.t()) + self.c.weight.t()

        if justGrad:
            return grad.t()

        # ── trace of the Hessian (spatial block only) ──
        # t_0: contribution of the opening layer.
        Kopen = N.layers[0].weight[:, 0:d]   # drop the time column
        temp = deriv_tanh(opening.t()) * z[1]
        trH = torch.sum(
            temp.reshape(m, -1, nex) * Kopen.unsqueeze(2).pow(2), dim=(0, 1)
        )

        # ∇_s u_0ᵀ, propagated forward as Jac of shape (m, d, nex).
        temp = tanhopen.t()  # act'(K_0 s + b_0)
        Jac = Kopen.unsqueeze(2) * temp.unsqueeze(1)

        # t_i: contribution of each residual layer.
        for i in range(1, N.layer):
            KJ = torch.mm(N.layers[i].weight, Jac.reshape(m, -1)).reshape(m, -1, nex)
            term = self.w.weight.t() if i == N.layer - 1 else z[i + 1]
            temp = N.layers[i].forward(u[i - 1]).t()  # K_i u_{i-1} + b_i
            t_i = torch.sum(
                (deriv_tanh(temp) * term).reshape(m, -1, nex) * KJ.pow(2), dim=(0, 1)
            )
            trH = trH + N.h * t_i
            Jac = Jac + N.h * torch.tanh(temp).reshape(m, -1, nex) * KJ

        return grad.t(), trH + torch.trace(symA[0:d, 0:d])


# ──────────────────────────────────────────────────────────────────────
# OTFlowTransform — bijection integrating dx/dt = -∇Φ via fixed-step RK4
# ──────────────────────────────────────────────────────────────────────

class OTFlowTransform(Transform):
    r"""Continuous bijection `x ↦ y` flowing `dx/dt = -∇_x Φ(t, x)`.

    Holds a *reference* to its `OTPhi` module (never snapshots parameters),
    so a transform captured once still reflects later `optimizer.step()`
    updates — the same lazy-read contract as `FreeFormJacobianTransform`.
    The forward map and `log|det J|` come from a single fixed-step RK4
    integration of the augmented state; the closed-form `OTPhi.trHess`
    supplies the divergence, so no Hutchinson estimate or Jacobian ODE is
    needed.

    `_inverse` integrates the same drift backward in time. It is therefore
    RK4-approximate (not closed-form), and round-trip accuracy improves with
    `nt`.

    Arguments:
        phi: the `OTPhi` potential.
        dimension: spatial dimension d.
        t0:  trajectory start time.
        t1:  trajectory end time.
        nt:  number of fixed RK4 steps.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        phi: OTPhi,
        dimension: int,
        t0: float = 0.0,
        t1: float = 1.0,
        nt: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.phi = phi
        self.dimension = dimension
        self.t0 = t0
        self.t1 = t1
        self.nt = nt

    def _with_time(self, t: Tensor, x: Tensor) -> Tensor:
        """Append the scalar time `t` as a final column, giving `[x; t]`."""
        return torch.cat([x, t * x.new_ones(x.shape[0], 1)], dim=1)

    # ── pure-position drift (gradient only) for _call / _inverse ──
    def _drift(self, t: Tensor, x: Tensor) -> Tensor:
        grad = self.phi.trHess(self._with_time(t, x), justGrad=True)
        return -grad[:, : self.dimension]

    def _call(self, x: Tensor) -> Tensor:
        return rk4_fixed(self._drift, x, self.t0, self.t1, self.nt)

    def _inverse(self, y: Tensor) -> Tensor:
        return rk4_fixed(self._drift, y, self.t1, self.t0, self.nt)

    # ── augmented dynamics: [x, ℓ] for ladj, [x, ℓ, v, r] for OT costs ──
    def _f_ladj(self, t: Tensor, z: Tensor) -> Tensor:
        d = self.dimension
        grad, trH = self.phi.trHess(self._with_time(t, z[:, :d]))
        dx = -grad[:, :d]
        dl = -trH.unsqueeze(1)
        return torch.cat([dx, dl], dim=1)

    def _f_full(self, t: Tensor, z: Tensor) -> Tensor:
        d = self.dimension
        grad, trH = self.phi.trHess(self._with_time(t, z[:, :d]))
        dx = -grad[:, :d]
        dl = -trH.unsqueeze(1)
        dv = 0.5 * dx.pow(2).sum(dim=1, keepdim=True)          # ½|∇Φ|²
        dr = (dv - grad[:, -1:]).abs()                         # |½|∇Φ|² - ∂_tΦ|
        return torch.cat([dx, dl, dv, dr], dim=1)

    def call_and_ladj(self, x: Tensor) -> tuple[Tensor, Tensor]:
        d = self.dimension
        z0 = torch.cat([x, x.new_zeros(x.shape[0], 1)], dim=1)
        zT = rk4_fixed(self._f_ladj, z0, self.t0, self.t1, self.nt)
        return zT[:, :d], zT[:, d]

    def call_full(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward map plus OT diagnostics in one integration.

        Returns `(y, ladj, transport_cost, hjb_residual)` where
        `transport_cost = ∫ ½|∇Φ|² dt` and
        `hjb_residual   = ∫ |½|∇Φ|² - ∂_tΦ| dt` — the per-sample integrated
        OT regularisers consumed by `zflows.loss.OT_loss`.
        """
        d = self.dimension
        z0 = torch.cat([x, x.new_zeros(x.shape[0], 3)], dim=1)
        zT = rk4_fixed(self._f_full, z0, self.t0, self.t1, self.nt)
        return zT[:, :d], zT[:, d], zT[:, d + 1], zT[:, d + 2]

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _, ladj = self.call_and_ladj(x)
        return ladj
