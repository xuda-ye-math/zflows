# pyright: reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false

from abc import ABC, abstractmethod
from functools import partial
import torch
from torch import Tensor, nn
from zuko.transforms import (
    MonotonicRQSTransform, AffineTransform, ComposedTransform,
)
from zuko.flows import MAF
from zuko.flows.spline import CircularRQSTransform
from zuko.flows.continuous import FFJTransform
from zuko.flows.coupling import GeneralCouplingTransform
from .potential import Potential

"""
All Flow classes assume context=0, i.e. the normalizing flow is always
unconditioned. The motivating use case is energy-based sampling, where
the target distribution is fixed and accuracy on that single target
matters more than the extra expressibility of a context-conditioned flow.
"""
class Flow(nn.Module, ABC):
    """
    Abstract base class for every normalizing flow in zflows.

    Subclasses inherit nn.Module machinery (.to(device), .parameters(),
    .state_dict(), .train()/.eval()) and must implement:

        def t(self) -> ComposedTransform: ...

    Built-in implementations are NSF and NCSF; future CNF / RealNVP /
    etc. classes should inherit from Flow and override t().

    Canonical usage:

        F = flow.t()                          # ComposedTransform
        y, ladj = F.call_and_ladj(x)          # forward & log|det J|
        x_back  = F.inv(y)                    # inverse

    Note: this is intentionally *different* from zuko's native interface
    `F = flow().transform`. zuko's pattern routes through the conditioner-
    aware `flow(context)` call, which is the right entry point for
    conditional flows but adds a vestigial pair of parens for unconditional
    ones. zflows commits to unconditional flows (see the module-level note
    above), so `flow.t()` is the only supported access path. Do NOT call
    `flow().transform` directly in zflows code — wrap any new Flow
    implementation behind a `t()` method instead, so user code stays
    uniform and the Flow contract holds.
    """
    @abstractmethod
    def t(self) -> ComposedTransform: ...

class NSF(Flow, MAF):
    """
    Neural Spline Flow whose transform is a bijection on the rectangle
    [a_1, b_1] x ... x [a_d, b_d].

    Internally an MAF-RQS runs on the symmetric box [-1, 1]^d;
    `t()` conjugates it by an affine to act on [a, b]^d:

        x in [a, b]^d
          --pre  affine: x -> (x - center) / half--> u in [-1, 1]^d
          --MAF-RQS--------------------------------> v in [-1, 1]^d
          --post affine: v -> center + half * v  --> y in [a, b]^d

    The pre/post affine Jacobians cancel, so the overall log|det J| equals
    that of the inner MAF-RQS.

    Arguments:
        a: lower corner of the box, shape (d,).
        b: upper corner of the box, shape (d,).
        bins: number of spline knots per coordinate; more bins give finer
            local detail at the cost of parameters and overfitting risk
            (recommend: 8-16 for smooth densities, up to 32 for sharper
            features).
        slope: minimum slope of each spline segment in the monotonic RQS
            transform. Acts as a floor on the derivative to keep the
            bijection strictly increasing and numerically stable; smaller
            values allow sharper density variations but risk ill-conditioned
            Jacobians, while larger values smooth the transform
            (recommend: 1e-3 to 1e-2).
        transforms: number of stacked autoregressive layers. Too few
            underfits multimodal targets; too many hurts optimization
            (recommend: 4-6).
        hidden_features: per-layer widths of the autoregressive conditioner
            MLP. A mild bottleneck works well (recommend: (64, 64) or
            (128, 64, 128); widen before deepening).
        activation: activation class (not instance) used inside the
            conditioner MLP (recommend: nn.SiLU or nn.GELU for smooth
            targets, nn.ReLU only when speed matters).
    """
    def __init__(
        self,
        a: Tensor | list[float], # (d,)  lower bounds
        b: Tensor | list[float], # (d,)  upper bounds
        bins: int = 8,
        slope: float = 1e-3,
        transforms: int = 4,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.SiLU, # pass the class, not an instance
    ):
        if not isinstance(a, Tensor):
            a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b, Tensor):
            b = torch.tensor(b, dtype=torch.float32)
        assert a.shape == b.shape and a.ndim == 1
        d = a.size(0)
        super().__init__(
            features=d, context=0,
            univariate=partial(MonotonicRQSTransform, bound=1.0, slope=slope),
            shapes=[(bins,), (bins,), (bins - 1,)],
            transforms=transforms,
            hidden_features=hidden_features,
            activation=activation,
        )
        # Buffers move with .to(device) and are saved in state_dict.
        self.register_buffer("a", a) # (d,)
        self.register_buffer("b", b) # (d,)
        self.register_buffer("center",    (a + b) / 2) # (d,)
        self.register_buffer("halfwidth", (b - a) / 2) # (d,)

    def t(self) -> ComposedTransform:
        """
        Bijection on [a, b]^d as a zuko ComposedTransform.
        Supports .inv and .call_and_ladj(x) -> (y, log|det J|).
        """
        inner = self().transform # ComposedTransform on [-1, 1]^d
        return ComposedTransform(
            AffineTransform(loc=-self.center / self.halfwidth, scale=1.0 / self.halfwidth),
            inner,
            AffineTransform(loc=self.center, scale=self.halfwidth),
        )

class NCSF(Flow, MAF):
    """
    Neural Circular Spline Flow whose transform is a bijection on the
    rectangle [a_1, b_1] x ... x [a_d, b_d], with each coordinate treated
    as a periodic angle (defaults to [-pi, pi]^d).

    Internally an MAF circular-RQS runs on the symmetric box [-pi, pi]^d;
    `t()` conjugates it by an affine to act on [a, b]^d:

        x in [a, b]^d
          --pre  affine: x -> (x - center) * pi / half --> u in [-pi, pi]^d
          --MAF circular-RQS-----------------------------> v in [-pi, pi]^d
          --post affine: v -> center + half * v / pi   --> y in [a, b]^d

    The pre/post affine Jacobians cancel, so the overall log|det J| equals
    that of the inner MAF circular-RQS.

    Arguments:
        a: lower corner of the box, shape (d,) (default: [-pi, ..., -pi]).
        b: upper corner of the box, shape (d,) (default: [ pi, ...,  pi]).
        bins: number of spline knots per coordinate; more bins give finer
            local detail at the cost of parameters and overfitting risk
            (recommend: 8-16 for smooth densities, up to 32 for sharper
            features).
        slope: minimum slope of each spline segment in the monotonic RQS
            transform. Acts as a floor on the derivative to keep the
            bijection strictly increasing and numerically stable; smaller
            values allow sharper density variations but risk ill-conditioned
            Jacobians, while larger values smooth the transform
            (recommend: 1e-3 to 1e-2).
        transforms: number of stacked autoregressive layers. Too few
            underfits multimodal targets; too many hurts optimization
            (recommend: 4-6).
        hidden_features: per-layer widths of the autoregressive conditioner
            MLP. A mild bottleneck works well (recommend: (64, 64) or
            (128, 64, 128); widen before deepening).
        activation: activation class (not instance) used inside the
            conditioner MLP (recommend: nn.SiLU or nn.GELU for smooth
            targets, nn.ReLU only when speed matters).
    """
    def __init__(
        self,
        a: Tensor | list[float], # (d,)  lower bounds (default -pi)
        b: Tensor | list[float], # (d,)  upper bounds (default  pi)
        bins: int = 8,
        slope: float = 1e-3,
        transforms: int = 4,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.SiLU, # pass the class, not an instance
    ):
        if not isinstance(a, Tensor):
            a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b, Tensor):
            b = torch.tensor(b, dtype=torch.float32)
        assert a.shape == b.shape and a.ndim == 1
        d = a.size(0)
        super().__init__(
            features=d, context=0,
            univariate=partial(CircularRQSTransform, slope=slope),
            shapes=[(bins,), (bins,), (bins - 1,)],
            transforms=transforms,
            hidden_features=hidden_features,
            activation=activation,
        )
        # Buffers move with .to(device) and are saved in state_dict.
        self.register_buffer("a", a) # (d,)
        self.register_buffer("b", b) # (d,)
        self.register_buffer("center",    (a + b) / 2) # (d,)
        self.register_buffer("halfwidth", (b - a) / 2) # (d,)

    def t(self) -> ComposedTransform:
        """
        Bijection on [a, b]^d as a zuko ComposedTransform.
        Supports .inv and .call_and_ladj(x) -> (y, log|det J|).
        """
        inner = self().transform # ComposedTransform on [-pi, pi]^d
        return ComposedTransform(
            AffineTransform(loc=-self.center * torch.pi / self.halfwidth, scale=torch.pi / self.halfwidth),
            inner,
            AffineTransform(loc=self.center, scale=self.halfwidth / torch.pi),
        )
    
class CNF(Flow):
    """
    Continuous normalizing flow (CNF) with a free-form Jacobian (FFJORD).

    Acts as a bijection on the full unbounded R^d via an ODE drift learned
    by an MLP. Unlike NSF / NCSF, no rectangular box is needed — CNFs live
    on R^d natively, so `t()` returns the inner transform without affine
    conjugation. The exact-log-det path is O(d) ODE evaluations per
    Jacobian; pass `exact=False` to switch to a Hutchinson stochastic
    estimate (faster, biased gradients during training).

    Arguments:
        dimension: number of features d.
        frequency: number of time-embedding frequencies in the ODE drift's
            input. The drift MLP sees the integration time t ∈ [0, 1]
            through 2 * frequency extra features (cos(k*pi*t), sin(k*pi*t)
            for k = 1, ..., frequency), letting it modulate the velocity
            field smoothly along t. Larger frequency gives a richer time
            profile (more flexibility, slightly larger first MLP layer);
            smaller frequency forces a near-constant drift (faster, less
            expressive) (recommend: 3-6).
        absolute_tolerance: absolute tolerance of the adaptive ODE solver
            (dopri5 from torchdiffeq under the hood). Each integration
            step is accepted when its local error is below
            absolute_tolerance + relative_tolerance * |solution|; smaller
            absolute_tolerance = tighter accuracy, more solver substeps,
            more compute. Use a tighter value if log|det J| disagrees
            noticeably between forward and inverse calls (round-trip
            error) (recommend: 1e-7 to 1e-5).
        relative_tolerance: relative tolerance of the same solver, scaling
            with the magnitude of the solution. Same speed/accuracy
            trade-off as absolute_tolerance; in practice
            absolute_tolerance dominates near the origin and
            relative_tolerance dominates for large |x| (recommend: 1e-6
            to 1e-4).
        exact: if True, evaluate log|det J| exactly via the augmented ODE
            (O(d) extra cost per step, deterministic gradients). If False,
            use the Hutchinson trace estimator (one extra ODE component,
            unbiased but stochastic gradients) — faster for large d but
            adds Monte-Carlo noise to training.
        hidden_features: ODE-MLP layer widths (recommend: (64, 64) or
            (128, 64, 128); widen before deepening).
        activation: ODE-MLP activation class, not instance (recommend:
            nn.SiLU or nn.GELU for smooth drifts; nn.ELU is the zuko
            default and works well too).
    """
    def __init__(
        self,
        dimension: int,
        frequency: int = 3,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-5,
        exact: bool = True,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.SiLU, # pass the class, not an instance
    ):
        super().__init__()
        self._ffj = FFJTransform(
            features=dimension,
            context=0,
            freqs=frequency,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
            exact=exact,
            hidden_features=hidden_features,
            activation=activation,
        )

    def t(self) -> ComposedTransform:
        """
        Bijection on R^d as a zuko ComposedTransform.
        Wraps the FFJ FreeFormJacobianTransform in a length-1
        ComposedTransform so the Flow contract `t() -> ComposedTransform`
        holds; .inv and .call_and_ladj delegate to the inner transform.
        """
        return ComposedTransform(self._ffj())

class RealNVP(Flow):
    """
    Affine-coupling normalizing flow (RealNVP, Dinh et al. 2016).

    Acts as a bijection on the full unbounded R^d via N stacked coupling
    transforms with checkered (alternating) feature masks. Each coupling
    layer splits coordinates into "active" and "passive" halves and
    applies an affine map  y_a = x_a * exp(s(x_p)) + t(x_p), where the
    scale s and shift t come from a shared MLP conditioned on x_p; the
    inverse and log-determinant are closed-form O(d).

    Like CNF, RealNVP lives on R^d natively — no rectangular box is
    needed and `t()` returns the composition of the coupling transforms
    without affine conjugation. Compared to NSF, each coupling layer is
    much weaker per parameter (affine vs. monotonic spline), so several
    stacked layers are needed; compared to CNF, every operation is
    closed-form and there is no ODE solver in the loop.

    Arguments:
        dimension: number of features d.
        transforms: number of stacked coupling layers. Each individual
            layer only deforms half the coordinates, so a few are needed
            to mix all features (recommend: 4-8; 4 is the zflows default,
            6-8 for harder targets in dimension >= 8).
        randmask: if True, use random per-layer masks instead of
            checkered (alternating) ones. Default False is the canonical
            RealNVP choice and works well in low dimension; set True for
            high d (>~50) to ensure all feature pairs eventually mix.
        hidden_features: per-layer widths of the coupling-conditioner
            MLP (recommend: (64, 64) or (128, 128); widen before
            deepening).
        activation: conditioner-MLP activation class, not instance
            (recommend: nn.SiLU or nn.GELU for smooth densities; the
            original paper used nn.ReLU, also fine).
    """
    def __init__(
        self,
        dimension: int,
        transforms: int = 4,
        randmask: bool = False,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: type[nn.Module] = nn.SiLU, # pass the class, not an instance
    ):
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
                    context=0,
                    mask=mask,
                    hidden_features=hidden_features,
                    activation=activation,
                )
            )

    def t(self) -> ComposedTransform:
        """
        Bijection on R^d as a zuko ComposedTransform — the composition
        of the affine coupling transforms. Supports .inv and
        .call_and_ladj(x) -> (y, log|det J|).
        """
        return ComposedTransform(*[c() for c in self._coupling])