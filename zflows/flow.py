# pyright: reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false

from functools import partial
from typing import Protocol, runtime_checkable
import torch
from torch import Tensor, nn
from zuko.transforms import (
    MonotonicRQSTransform, AffineTransform, ComposedTransform,
)
from zuko.flows import MAF
from zuko.flows.spline import CircularRQSTransform
from .potential import Potential

"""
All Flow classes assume context=0, i.e. the normalizing flow is always
unconditioned. The motivating use case is energy-based sampling, where
the target distribution is fixed and accuracy on that single target
matters more than the extra expressibility of a context-conditioned flow.
"""
@runtime_checkable
class Flow(Protocol):
    """
    Structural interface for any normalizing flow used by zflows.

    A Flow is anything that exposes a `t()` method returning a zuko-style
    ComposedTransform supporting `.inv` and `.call_and_ladj(x) -> (y, log|det J|)`.
    Built-in implementations are NSF and NCSF; future CNF / RealNVP / etc.
    classes need only provide `.t()` to be plug-compatible — no shared base
    class is required.

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
    uniform and the Protocol contract holds.
    """
    def t(self) -> ComposedTransform: ...


def call_and_ladj_grad(
    flow_t: ComposedTransform,
    x: torch.Tensor,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a zuko ComposedTransform F = flow_t and return (y, ladj, ladj_grad):
        y         = F(x)             [N, d]
        ladj      = log|det J_F(x)|  [N]
        ladj_grad = d ladj / d x     [N, d]

    Typical use: pass flow.t() in:
        y, ladj, ladj_grad = _call_and_ladj_grad(flow.t(), x)

    Special-purpose helper: kept as a free function because it is a
    sample-aware convenience that requires `requires_grad` bookkeeping,
    not a transform-level operation.

    Argument:
        create_graph: if True (default) the returned ladj_grad is itself
            differentiable w.r.t. the flow's parameters; pass False for
            a ~30-50% faster, lower-memory call when only the value of
            ladj_grad is needed (e.g. diagnostics or no-backprop inference).

    Note: x must be an autograd leaf with requires_grad=True. If
    x.requires_grad is False, this function detaches and re-leafs it.
    """
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    y, ladj = flow_t.call_and_ladj(x)
    (ladj_grad,) = torch.autograd.grad(
        ladj.sum(), x, create_graph=create_graph,
    )
    return y, ladj, ladj_grad


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
def reverse_KL(x: torch.Tensor, target: Potential, flow: Flow):
    """
    The reverse KL divergence in energy-based normalizing flow.
    Estimates  E_{x ~ source}[ target(F(x)) - log|det J_F(x)| ],
    where F = flow.t() pushes source samples toward the target.
    Input:
        x:      Tensor [N, d]   samples drawn from the source distribution
        source: Potential       negative log-density of the source (up to const)
        target: Potential       negative log-density of the target (up to const)
        flow:   Flow            normalizing flow providing F = flow.t()
    Output:
        loss: Tensor (scalar)   Monte Carlo estimate of the reverse KL
    """
    y, ladj = flow.t().call_and_ladj(x) # get y = F(x) and log_abs_det_jacobian
    return (target(y) - ladj).mean()

# forward KL divergence for data-driven normalizing flow
def forward_KL(y: torch.Tensor, source: Potential, flow: Flow):
    """
    The forward KL divergence in data-driven normalizing flow.
    Estimates  E_{y ~ target}[ source(F^-1(y)) - log|det J_{F^-1}(y)| ],
    where F = flow.t() pushes source samples toward the target.
    Input:
        y:      Tensor [N, d]   samples drawn from the target distribution
        source: Potential       negative log-density of the source (up to const)
        flow:   Flow            normalizing flow providing F = flow.t()
    Output:
        loss: Tensor (scalar)   Monte Carlo estimate of the forward KL
    """
    x, ladj = flow.t().inv.call_and_ladj(y) # x = F^-1(y), ladj = log|det J_{F^-1}(y)|
    return (source(x) - ladj).mean()