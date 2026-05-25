# pyright: reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false

"""Unconditional normalizing flows for energy-based sampling.

Public API:
    Flow              — abstract base; subclasses implement .t() -> ComposedTransform
    NSF               — Neural Spline Flow on [a, b]^d (translation-sandwiched RQS)
    NCSF              — Neural Circular Spline Flow on [a, b]^d (periodic per coord)
    CNF               — Continuous Normalizing Flow on R^d (FFJORD)
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
    MaskedAutoregressiveTransform,
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

    Arguments:
        dimension: number of features d.
        frequency: number of time-embedding frequencies in the ODE drift
            (recommend: 3-6).
        absolute_tolerance: absolute tolerance of the adaptive ODE solver
            (recommend: 1e-7 to 1e-5).
        relative_tolerance: relative tolerance (recommend: 1e-6 to 1e-4).
        exact: if True, evaluate log|det J| exactly via the augmented ODE.
            If False, use the Hutchinson trace estimator.
        hidden_features: ODE-MLP layer widths (recommend: (64, 64)).
        activation: ODE-MLP activation class (recommend: nn.SiLU).
    """
    def __init__(
        self,
        dimension: int,
        frequency: int = 3,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-5,
        exact: bool = True,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self._ffj = FFJTransform(
            features=dimension,
            freqs=frequency,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
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
# RealNVP — affine-coupling flow on R^d
# ──────────────────────────────────────────────────────────────────────

class RealNVP(Flow):
    """Affine-coupling normalizing flow (RealNVP, Dinh et al. 2016).

    N stacked coupling transforms with checkered (or random) feature
    masks; inverse and log|det J| are closed-form O(d).

    Arguments:
        dimension: number of features d.
        transforms: number of stacked coupling layers (recommend: 4-8).
        randmask: if True (default), draw a fresh randomised checkered
            mask per layer (better mixing at d >= 4). If False, use the
            canonical alternating-checkered RealNVP masks. Reproducible
            from a global torch seed in either case.
        hidden_features: per-layer widths of the coupling-conditioner MLP
            (recommend: (64, 64) or (128, 128)).
        activation: conditioner MLP activation class (recommend: nn.SiLU).
    """
    def __init__(
        self,
        dimension: int,
        transforms: int = 4,
        randmask: bool = True,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self._coupling = nn.ModuleList()
        for i in range(transforms):
            if randmask:
                mask = torch.randperm(dimension) % 2 == i % 2
            else:
                mask = torch.arange(dimension) % 2 == i % 2
            self._coupling.append(
                GeneralCouplingTransform(
                    features=dimension,
                    mask=mask,
                    hidden_features=hidden_features,
                    activation=activation,
                )
            )

    def t(self) -> ComposedTransform:
        """Bijection on R^d as the composition of all coupling transforms."""
        return ComposedTransform(*[c() for c in self._coupling])

    def zeros(self) -> None:
        """Zeroing the last conditioner layer → identity per coupling layer."""
        for c in self._coupling:
            last = c.hyper[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
