# pyright: reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false

from functools import partial
import torch
from torch import Tensor, nn
from zuko.flows import MAF
from zuko.transforms import (
    MonotonicRQSTransform, CircularShiftTransform, AffineTransform, ComposedTransform,
)
from .potential import Potential

class NSF(MAF):
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

def CircularRQSTransform(*phi: Tensor, slope: float = 1e-3) -> ComposedTransform:
    r"""Creates a circular rational-quadratic spline (RQS) transformation."""

    return ComposedTransform(
        CircularShiftTransform(bound=torch.pi),
        MonotonicRQSTransform(*phi, bound=torch.pi, slope=slope),
    )

class NCSF(MAF):
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

# reverse KL divergence for energy-based normalizing flow
def reverse_KL(x: torch.Tensor, target: Potential, flow: NSF | NCSF):
    """
    The reverse KL divergence in energy-based normalizing flow.
    Estimates  E_{x ~ source}[ target(F(x)) - log|det J_F(x)| ],
    where F = flow.t() pushes source samples toward the target.
    Input:
        x:      Tensor [N, d]   samples drawn from the source distribution
        source: Potential       negative log-density of the source (up to const)
        target: Potential       negative log-density of the target (up to const)
        flow:   NSF | NCSF      normalizing flow providing F = flow.t()
    Output:
        loss: Tensor (scalar)   Monte Carlo estimate of the reverse KL
    """
    y, ladj = flow.t().call_and_ladj(x) # get y = F(x) and log_abs_det_jacobian
    return (target(y) - ladj).mean()

# forward KL divergence for data-driven normalizing flow
def forward_KL(y: torch.Tensor, source: Potential, flow: NSF | NCSF):
    """
    The forward KL divergence in data-driven normalizing flow.
    Estimates  E_{y ~ target}[ source(F^-1(y)) - log|det J_{F^-1}(y)| ],
    where F = flow.t() pushes source samples toward the target.
    Input:
        y:      Tensor [N, d]   samples drawn from the target distribution
        source: Potential       negative log-density of the source (up to const)
        flow:   NSF | NCSF      normalizing flow providing F = flow.t()
    Output:
        loss: Tensor (scalar)   Monte Carlo estimate of the forward KL
    """
    x, ladj = flow.t().inv.call_and_ladj(y) # x = F^-1(y), ladj = log|det J_{F^-1}(y)|
    return (source(x) - ladj).mean()