# pyright: reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false

"""Unconditional normalizing flows for energy-based sampling.

Public API:
    Flow              — abstract base; subclasses implement .t() -> ComposedTransform
    NSF               — Neural Spline Flow on [a, b]^d (translation-sandwiched RQS)
    NCSF              — Neural Circular Spline Flow on [a, b]^d (periodic per coord)
    CNF               — Continuous Normalizing Flow on R^d (FFJORD)
    OTFlow            — Optimal-transport continuous flow on R^d (closed-form trace)
    RealNVP           — affine-coupling flow on R^d
    ComposedTransform — re-exported from .core.transforms

All flows assume context = 0, i.e. one fixed target. NSF and NCSF
parameterise their inner spline on per-coordinate
`[-halfwidth_i, halfwidth_i]` and wrap it with an additive
translation by ±center_i (no scaling), so the box-bound geometry
[a, b]^d is honoured without distorting the conditioner's dynamic
range.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .core.flows import (
    CircularRQSTransform,
    FFJTransform,
    GeneralCouplingTransform,
    LinearMixingTransform,
    MaskedAutoregressiveTransform,
    OTFlowLazy,
)
from .core.transforms import (
    AdditiveTransform,
    ComposedTransform,
    MonotonicRQSTransform,
)


class Flow(nn.Module, ABC):
    """Abstract base class for every normalizing flow in zflows.

    Subclasses inherit nn.Module machinery (.to(device), .parameters(),
    .state_dict(), .train()/.eval()) and must implement:

        def t(self) -> ComposedTransform: ...

    Canonical usage:

        F = flow.t()                          # ComposedTransform
        y, ladj = F.call_and_ladj(x)          # forward & log|det J|
        x_back  = F.inv(y)                    # inverse

    `flow.t()` is the only supported access path; do not invoke any
    internal `forward`/`__call__` directly.
    """
    @abstractmethod
    def t(self) -> ComposedTransform: ...


# ──────────────────────────────────────────────────────────────────────
# NSF — Neural Spline Flow on [a, b]^d
# ──────────────────────────────────────────────────────────────────────

class NSF(Flow):
    """Neural Spline Flow on [a_1, b_1] x ... x [a_d, b_d].

    The MAF-RQS conditioner runs on the per-coord centred box
    [-halfwidth_i, halfwidth_i]; t() sandwiches it with two
    AdditiveTransform shifts by ±center_i. No scaling — log|det J| is
    fully contributed by the inner spline.

    Arguments:
        a: lower corner of the box, shape (d,).
        b: upper corner of the box, shape (d,).
        bins: number of spline knots per coordinate; more bins give finer
            local detail at the cost of parameters and overfitting risk
            (recommend: 8-16 for smooth densities, up to 32 for sharper
            features).
        slope: minimum slope of each spline segment in the monotonic RQS
            transform. Acts as a floor on the derivative to keep the
            bijection strictly increasing and numerically stable
            (recommend: 1e-3 to 1e-2).
        transforms: number of stacked autoregressive layers. Too few
            underfits multimodal targets; too many hurts optimization
            (recommend: 4-6).
        randmask: per-layer feature ordering. True (default) draws a
            fresh torch.randperm(d) per layer — recommended at d >= 4
            because it breaks the bipartite symmetry that the alternating
            scheme imposes. False uses arange(d) / arange(d).flip(0)
            alternation (the prior behaviour). Reproducible from a global
            torch seed in either case.
        hidden_features: per-layer widths of the autoregressive conditioner
            MLP. A mild bottleneck works well (recommend: (64, 64) or
            (128, 64, 128); widen before deepening).
        activation: activation class (not instance) used inside the
            conditioner MLP (recommend: nn.SiLU or nn.GELU for smooth
            targets, nn.ReLU only when speed matters).
    """
    def __init__(
        self,
        a: Tensor | list[float],
        b: Tensor | list[float],
        bins: int = 8,
        slope: float = 1e-3,
        transforms: int = 4,
        randmask: bool = True,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        if not isinstance(a, Tensor):
            a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b, Tensor):
            b = torch.tensor(b, dtype=torch.float32)
        assert a.shape == b.shape and a.ndim == 1
        d = a.size(0)

        # Buffers move with .to(device) and are saved in state_dict.
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.register_buffer("center", (a + b) / 2)
        self.register_buffer("halfwidth", (b - a) / 2)
        self.slope = slope

        if randmask:
            orders_list = [torch.randperm(d) for _ in range(transforms)]
        else:
            orders_list = [
                torch.arange(d) if i % 2 == 0 else torch.arange(d).flip(0)
                for i in range(transforms)
            ]
        self._maf = nn.ModuleList([
            MaskedAutoregressiveTransform(
                features=d,
                univariate=self._univariate,
                shapes=[(bins,), (bins,), (bins - 1,)],
                order=orders_list[i],
                hidden_features=hidden_features,
                activation=activation,
            )
            for i in range(transforms)
        ])

    def _univariate(self, *phi: Tensor):
        # Resolve self.halfwidth lazily so .to(device) reassignment is
        # picked up by every subsequent .t() call.
        return MonotonicRQSTransform(*phi, bound=self.halfwidth, slope=self.slope)

    def t(self) -> ComposedTransform:
        """Bijection on [a, b]^d as a ComposedTransform.

        Supports .inv and .call_and_ladj(x) -> (y, log|det J|).
        """
        inner = ComposedTransform(*[m() for m in self._maf])
        return ComposedTransform(
            AdditiveTransform(shift=-self.center),
            inner,
            AdditiveTransform(shift= self.center),
        )

    def zeros(self) -> None:
        """Initialize the flow to the identity by zeroing the last layer
        of each conditioner MLP."""
        for m in self._maf:
            last = m.hyper[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)


# ──────────────────────────────────────────────────────────────────────
# NCSF — Neural Circular Spline Flow on [a, b]^d
# ──────────────────────────────────────────────────────────────────────

class NCSF(Flow):
    """Neural Circular Spline Flow on [a_1, b_1] x ... x [a_d, b_d],
    each coordinate periodic with its own period b_i - a_i.

    The MAF circular-RQS conditioner runs on the per-coord centred box
    [-halfwidth_i, halfwidth_i]; t() sandwiches it with AdditiveTransform
    shifts by ±center_i (no scaling). Default a = [-pi, ..., -pi],
    b = [pi, ..., pi] reproduces the original NCSF on the d-torus.

    Arguments:
        a: lower corner of the box, shape (d,) (typically -pi).
        b: upper corner of the box, shape (d,) (typically  pi).
        bins: number of spline knots per coordinate (recommend: 8-16).
        slope: minimum slope of each spline segment (recommend: 1e-3 to 1e-2).
        transforms: number of stacked autoregressive layers (recommend: 4-6).
        randmask: per-layer feature ordering. True (default) draws a
            fresh torch.randperm(d) per layer — the only legal expressivity
            lever on a torus (linear mixings break periodicity). False
            uses arange(d) / arange(d).flip(0) alternation (the prior
            behaviour). Reproducible from a global torch seed in either case.
        hidden_features: per-layer widths of the autoregressive conditioner
            MLP (recommend: (64, 64) or (128, 64, 128)).
        activation: activation class (not instance) used inside the
            conditioner MLP (recommend: nn.SiLU or nn.GELU).
    """
    def __init__(
        self,
        a: Tensor | list[float],
        b: Tensor | list[float],
        bins: int = 8,
        slope: float = 1e-3,
        transforms: int = 4,
        randmask: bool = True,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        if not isinstance(a, Tensor):
            a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b, Tensor):
            b = torch.tensor(b, dtype=torch.float32)
        assert a.shape == b.shape and a.ndim == 1
        d = a.size(0)

        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.register_buffer("center", (a + b) / 2)
        self.register_buffer("halfwidth", (b - a) / 2)
        self.slope = slope

        if randmask:
            orders_list = [torch.randperm(d) for _ in range(transforms)]
        else:
            orders_list = [
                torch.arange(d) if i % 2 == 0 else torch.arange(d).flip(0)
                for i in range(transforms)
            ]
        self._maf = nn.ModuleList([
            MaskedAutoregressiveTransform(
                features=d,
                univariate=self._univariate,
                shapes=[(bins,), (bins,), (bins - 1,)],
                order=orders_list[i],
                hidden_features=hidden_features,
                activation=activation,
            )
            for i in range(transforms)
        ])

    def _univariate(self, *phi: Tensor):
        return CircularRQSTransform(*phi, bound=self.halfwidth, slope=self.slope)

    def t(self) -> ComposedTransform:
        """Bijection on [a, b]^d as a ComposedTransform."""
        inner = ComposedTransform(*[m() for m in self._maf])
        return ComposedTransform(
            AdditiveTransform(shift=-self.center),
            inner,
            AdditiveTransform(shift= self.center),
        )

    def zeros(self) -> None:
        """Initialize the flow to the identity by zeroing the last layer
        of each conditioner MLP."""
        for m in self._maf:
            last = m.hyper[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)


# ──────────────────────────────────────────────────────────────────────
# CNF — Continuous Normalizing Flow on R^d (FFJORD)
# ──────────────────────────────────────────────────────────────────────

class CNF(Flow):
    """Continuous normalizing flow (CNF) with a free-form Jacobian (FFJORD).

    Acts as a bijection on R^d via an ODE drift learned by an MLP. The
    exact-log-det path is O(d) ODE evaluations per Jacobian; pass
    `exact=False` to switch to a Hutchinson stochastic estimate (faster,
    biased gradients during training).

    Integration is fixed-step RK4 (the only integrator in zflows): a
    deterministic flop budget and a torch.compile-friendly loop, unrolled
    under autograd so no adjoint bookkeeping is needed. Accuracy is set by
    `nt` rather than solver tolerances; round-trip and log-det error fall
    ~16x per doubling of `nt`.

    Arguments:
        dimension: number of features d.
        frequency: number of time-embedding frequencies in the ODE drift
            (recommend: 3-6).
        nt: number of fixed RK4 steps (recommend: 8-24; raise for tighter
            inverse round-trip / log-det accuracy at linear cost).
        exact: if True, evaluate log|det J| exactly via the augmented ODE.
            If False, use the Hutchinson trace estimator.
        hidden_features: ODE-MLP layer widths (recommend: (64, 64)).
        activation: ODE-MLP activation class (recommend: nn.SiLU).
    """
    def __init__(
        self,
        dimension: int,
        frequency: int = 3,
        nt: int = 16,
        exact: bool = True,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self._ffj = FFJTransform(
            dimension=dimension,
            frequency=frequency,
            nt=nt,
            exact=exact,
            hidden_features=hidden_features,
            activation=activation,
        )

    def t(self) -> ComposedTransform:
        """Bijection on R^d as a length-1 ComposedTransform."""
        return ComposedTransform(self._ffj())

    def zeros(self) -> None:
        """Drift = 0 → ODE flows trivially, identity bijection."""
        last = self._ffj.ode[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


# ──────────────────────────────────────────────────────────────────────
# OTFlow — optimal-transport continuous flow on R^d
# ──────────────────────────────────────────────────────────────────────

class OTFlow(Flow):
    """Optimal-transport continuous normalizing flow (OT-Flow).

    Reference:
        Onken, Fung, Li, Ruthotto. "OT-Flow: Fast and Accurate Continuous
        Normalizing Flows via Optimal Transport." AAAI 2021.
        https://arxiv.org/abs/2006.00104

    An upgraded `CNF`: a bijection on R^d defined by the ODE
    `dx/dt = -∇_x Φ_θ(t, x)` for a learnable scalar potential `Φ_θ`. Because
    `Φ_θ` has the OT-Flow antiderivative-of-tanh ResNet plus low-rank
    quadratic structure, the divergence `tr(∇_x v) = -tr(∇²_x Φ)` is computed
    in closed form (O(d·m)) — no Hutchinson estimator and no augmented O(d)
    Jacobian ODE as in FFJORD, which is where OT-Flow's speedup comes from.
    Integration uses a fixed-step RK4 scheme (deterministic flop budget,
    torch.compile-friendly).

    The forward map and `log|det J|` follow the standard `(y, ladj)` contract,
    so an `OTFlow` is a drop-in `Flow` for `reverse_KL` / `compile_raw` and the
    SMC utilities. The two extra optimal-transport diagnostics — the transport
    cost `∫½|∇Φ|² dt` and the HJB residual `∫|½|∇Φ|² - ∂_tΦ| dt` — are exposed
    through `zflows.loss.OT_loss`, which integrates all four channels in one
    pass; plain `reverse_KL` simply drops them.

    Arguments:
        dimension: number of features d.
        hidden: hidden width m of Φ's ResNet (recommend: 32-128).
        layer: number of ResNet layers nTh inside Φ; must be >= 2
            (recommend: 2-5).
        rank: rank of Φ's low-rank quadratic term, clamped to <= d + 1
            (recommend: min(10, d + 1)).
        nt: number of fixed RK4 steps. A well-regularised OT path needs few
            (recommend: 4-12); more steps tighten the inverse round-trip and
            log-det accuracy at linear cost.
        time_bound: (t0, t1) integration interval (default (0.0, 1.0)).

    Unlike the other flows, OTFlow takes no `activation` argument: the
    closed-form Hessian trace is derived specifically for the
    antiderivative-of-tanh activation, so it is not configurable.
    """
    def __init__(
        self,
        dimension: int,
        hidden: int = 64,
        layer: int = 3,
        rank: int = 10,
        nt: int = 8,
        time_bound: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self._ot = OTFlowLazy(
            dimension=dimension,
            hidden=hidden,
            layer=layer,
            rank=rank,
            nt=nt,
            time_bound=time_bound,
        )

    def t(self) -> ComposedTransform:
        """Bijection on R^d as a length-1 ComposedTransform."""
        return ComposedTransform(self._ot())

    def zeros(self) -> None:
        """Φ ≡ 0 → drift ∇Φ = 0 → identity bijection with ladj = 0.

        Zeros every head of the potential: the ResNet output weight `w`, the
        quadratic factor `A` (so AᵀA = 0), and the linear head `c`.
        """
        phi = self._ot.phi
        nn.init.zeros_(phi.w.weight)
        nn.init.zeros_(phi.A)
        nn.init.zeros_(phi.c.weight)


# ──────────────────────────────────────────────────────────────────────
# RealNVP — affine-coupling flow on R^d
# ──────────────────────────────────────────────────────────────────────

class RealNVP(Flow):
    """Affine-coupling normalizing flow (RealNVP, Dinh et al. 2016).

    N stacked coupling transforms with checkered (or random) feature
    masks; inverse and log|det J| are closed-form O(d). Optionally
    interleaves a learnable d x d linear mixing layer between every
    pair of consecutive couplings — the same idea as Glow's
    "invertible 1x1 convolution" on R^d.

    Arguments:
        dimension: number of features d.
        transforms: number of stacked coupling layers (recommend: 4-8).
        randmask: if True (default), draw a fresh randomised checkered
            mask per layer (better mixing at d >= 4). If False, use the
            canonical alternating-checkered RealNVP masks. Reproducible
            from a global torch seed in either case.
        mixing: one of None | "rotation" | "lu".
            If None (default), no mixing layer is inserted and the
            flow is a pure stack of coupling transforms. If "rotation",
            an orthogonal `R = exp(A - A^T)` map (log|det| ≡ 0) is
            inserted between every two consecutive couplings (i.e.
            `transforms - 1` mixing layers total). If "lu", a PLU map
            `L @ U` is inserted instead — its learnable diagonal of `L`
            provides a non-trivial log|det| that supplements the
            coupling layers. Mixing layers are initialised at identity.
        hidden_features: per-layer widths of the coupling-conditioner MLP
            (recommend: (64, 64) or (128, 128)).
        activation: conditioner MLP activation class (recommend: nn.SiLU).
    """
    def __init__(
        self,
        dimension: int,
        transforms: int = 4,
        randmask: bool = True,
        mixing: str | None = None,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList()
        for i in range(transforms):
            if randmask:
                mask = torch.randperm(dimension) % 2 == i % 2
            else:
                mask = torch.arange(dimension) % 2 == i % 2
            self._layers.append(
                GeneralCouplingTransform(
                    features=dimension,
                    mask=mask,
                    hidden_features=hidden_features,
                    activation=activation,
                )
            )
            # Insert a mixing layer between this coupling and the next
            # (skip after the last coupling — no successor to mix into).
            if mixing is not None and i < transforms - 1:
                self._layers.append(
                    LinearMixingTransform(features=dimension, kind=mixing)
                )

    def t(self) -> ComposedTransform:
        """Bijection on R^d as the composition of all coupling (and mixing) layers."""
        return ComposedTransform(*[layer() for layer in self._layers])

    def zeros(self) -> None:
        """Reset every layer to identity. `GeneralCouplingTransform` and
        `LinearMixingTransform` both expose a `.zeros()` method, so the
        same iteration handles couplings and mixing layers uniformly.
        """
        for layer in self._layers:
            layer.zeros()
