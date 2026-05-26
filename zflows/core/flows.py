"""Unconditional lazy transformations used by zflows's public flows.

Ported from zuko/flows/{autoregressive,coupling,continuous,spline}.py
with all `context` / `c=None` plumbing removed. Each lazy transform is
an nn.Module whose `forward()` (no argument) returns a concrete
`Transform` from `.transforms`.

Public classes:
    - MaskedAutoregressiveTransform — backbone of NSF / NCSF
    - GeneralCouplingTransform      — backbone of RealNVP
    - FFJTransform                  — backbone of CNF

And one factory helper:
    - CircularRQSTransform(*phi, bound, slope) → Transform
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import partial
from math import ceil, prod, pi

import torch
import torch.nn as nn
from torch import BoolTensor, LongTensor, Size, Tensor
from torch.distributions import Transform

from .nn import MLP, MaskedMLP
from .numerics import broadcast, unpack
from .transforms import (
    AutoregressiveTransform,
    CircularShiftTransform,
    ComposedTransform,
    CouplingTransform,
    DependentTransform,
    FreeFormJacobianTransform,
    LULinearTransform,
    MonotonicAffineTransform,
    MonotonicRQSTransform,
    RotationTransform,
)


__all__ = [
    "CircularRQSTransform",
    "FFJTransform",
    "GeneralCouplingTransform",
    "LinearMixingTransform",
    "MaskedAutoregressiveTransform",
]


# ──────────────────────────────────────────────────────────────────────
# Masked autoregressive
# ──────────────────────────────────────────────────────────────────────

class MaskedAutoregressiveTransform(nn.Module):
    """Lazy unconditional masked autoregressive transformation.

    Compared to zuko's version, this drops `context`, `adjacency`, and
    relaxes `passes` to a single fully-autoregressive setting (passes ==
    features). The `univariate` argument is the constructor for the
    per-coordinate bijection used inside the autoregressive scheme.

    Arguments:
        features:        number of features d.
        univariate:      Callable(*phi) -> Transform. Default MonotonicAffineTransform.
        shapes:          per-parameter shape (excluding feature dim).
                         E.g. for RQS: [(K,), (K,), (K - 1,)].
        order:           feature ordering, shape (d,). Defaults to arange(d).
        hidden_features: MLP hidden widths.
        activation:      activation CLASS (not instance).
    """

    def __init__(
        self,
        features: int,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[Size] = ((), ()),
        order: LongTensor | None = None,
        hidden_features: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.univariate = univariate
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        if order is None:
            order = torch.arange(features)
        else:
            order = torch.as_tensor(order, dtype=torch.long)
        assert order.ndim == 1 and order.shape[0] == features

        # Fully autoregressive: each output i sees only inputs with strictly
        # smaller order index. self.order keeps the original ordering.
        self.register_buffer("order", order)
        self.passes = features

        adjacency = order[:, None] > order
        adjacency = torch.repeat_interleave(adjacency, repeats=self.total, dim=0)

        self.hyper = MaskedMLP(
            adjacency,
            hidden_features=hidden_features,
            activation=activation,
        )

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))
        order = self.order.tolist()
        if len(order) > 10:
            order = order[:5] + [...] + order[-5:]
            order = str(order).replace("Ellipsis", "...")
        return "\n".join([f"(base): {base}", f"(order): {order}"])

    def meta(self, x: Tensor) -> Transform:
        phi = self.hyper(x)
        phi = phi.unflatten(-1, (-1, self.total))
        phi = unpack(phi, self.shapes)
        # DependentTransform reinterprets the last dim as event-dim so
        # log|det J| sums over coordinates automatically.
        return DependentTransform(self.univariate(*phi), 1)

    def forward(self) -> Transform:
        return AutoregressiveTransform(self.meta, self.passes)


# ──────────────────────────────────────────────────────────────────────
# Coupling
# ──────────────────────────────────────────────────────────────────────

class GeneralCouplingTransform(nn.Module):
    """Lazy unconditional general coupling transformation.

    Compared to zuko's version, this drops `context`. The MLP input
    dim is `features_a` (the number of "kept" coordinates).
    """

    def __init__(
        self,
        features: int,
        mask: BoolTensor,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[Size] = ((), ()),
        hidden_features: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.univariate = univariate
        self.shapes = shapes
        self.total = sum(prod(s) for s in shapes)

        mask = torch.as_tensor(mask, dtype=torch.bool)
        assert mask.ndim == 1 and mask.shape[0] == features

        features_a = int(mask.sum().item())
        features_b = features - features_a
        assert features_a > 0 and features_b > 0

        self.register_buffer("mask", mask)
        self.hyper = MLP(
            features_a,
            features_b * self.total,
            hidden_features=hidden_features,
            activation=activation,
        )

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))
        mask = self.mask.int().tolist()
        if len(mask) > 10:
            mask = mask[:5] + [...] + mask[-5:]
            mask = str(mask).replace("Ellipsis", "...")
        return "\n".join([f"(base): {base}", f"(mask): {mask}"])

    def meta(self, x: Tensor) -> Transform:
        phi = self.hyper(x)
        phi = phi.unflatten(-1, (-1, self.total))
        phi = unpack(phi, self.shapes)
        return DependentTransform(self.univariate(*phi), 1)

    def forward(self) -> Transform:
        return CouplingTransform(self.meta, self.mask)

    def zeros(self) -> None:
        """Reset to identity by zeroing the last conditioner-MLP layer's
        weight and bias. With phi = 0 the univariate transform reduces
        to the identity on the masked-out coordinates.
        """
        last = self.hyper[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


# ──────────────────────────────────────────────────────────────────────
# FFJORD continuous flow
# ──────────────────────────────────────────────────────────────────────

class FFJTransform(nn.Module):
    """Lazy unconditional free-form-Jacobian transformation (FFJORD).

    Compared to zuko's version, drops `context` and the c-arg branch of
    the drift `f(t, x)`. Time is embedded via 2 * freqs cos/sin
    components, concatenated with x as the ODE-MLP input.
    """

    def __init__(
        self,
        features: int,
        freqs: int = 3,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        exact: bool = True,
        hidden_features: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()

        self.ode = MLP(
            features + 2 * freqs,
            features,
            hidden_features=hidden_features,
            activation=activation,
        )
        self.register_buffer("times", torch.tensor((0.0, 1.0)))
        self.register_buffer("freqs", torch.arange(1, freqs + 1) * pi)

        self.atol = atol
        self.rtol = rtol
        self.exact = exact

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        x = torch.cat(broadcast(t, x, ignore=1), dim=-1)
        return self.ode(x)

    def forward(self) -> Transform:
        # Pass parameters as `phi` only while training so the ODE adjoint
        # backprops through them; at eval time phi=() avoids the autograd
        # graph machinery.
        phi = tuple(self.parameters()) if self.training else ()
        return FreeFormJacobianTransform(
            f=self.f,
            t0=self.times[0],
            t1=self.times[1],
            phi=phi,
            atol=self.atol,
            rtol=self.rtol,
            exact=self.exact,
        )


# ──────────────────────────────────────────────────────────────────────
# CircularRQSTransform helper
# ──────────────────────────────────────────────────────────────────────

def CircularRQSTransform(
    *phi: Tensor,
    bound: Tensor | float = pi,
    slope: float = 1e-3,
) -> Transform:
    """Circular RQS bijection on per-coord [-bound, bound].

    Composes the modular wrap with a monotonic RQS sharing the same
    `bound`. With `bound` a (d,) tensor, each coordinate gets its own
    period 2 * bound_i.
    """
    return ComposedTransform(
        CircularShiftTransform(bound=bound),
        MonotonicRQSTransform(*phi, bound=bound, slope=slope),
    )


# ──────────────────────────────────────────────────────────────────────
# LinearMixingTransform — lazy 1x1 invertible "conv" for RealNVP / Glow
# ──────────────────────────────────────────────────────────────────────

class LinearMixingTransform(nn.Module):
    """Lazy unconditional linear mixing on R^d.

    The Glow-style 1x1 invertible "convolution" used to interleave
    cross-coordinate mixing between coupling layers. Two parametric
    families are supported:

      - kind="rotation": orthogonal map R = exp(A - A^T) of the
        skew-symmetric part of a free d x d matrix; log|det| ≡ 0.
      - kind="lu":       PLU-style map L @ U with L lower-triangular
        (diagonal carries the log|det|) and U strict-upper-triangular
        plus identity (unit diagonal).

    Initialised at identity (rotation: A = 0 -> R = exp(0) = I;
    lu: LU = I -> L = I, U = I) so that the layer contributes nothing
    at construction time. Training moves the matrix away from identity
    through normal gradients; calling .zeros() returns it to identity.
    """

    def __init__(self, features: int, kind: str = "rotation") -> None:
        super().__init__()
        assert kind in ("rotation", "lu"), f"unknown mixing kind {kind!r}"
        self.kind = kind
        self.features = features
        if kind == "rotation":
            self.weight = nn.Parameter(torch.zeros(features, features))
        else:  # "lu"
            self.weight = nn.Parameter(torch.eye(features))

    def extra_repr(self) -> str:
        return f"features={self.features}, kind={self.kind!r}"

    def forward(self) -> Transform:
        if self.kind == "rotation":
            return RotationTransform(self.weight)
        return LULinearTransform(self.weight)

    def zeros(self) -> None:
        """Reset to identity (A = 0 or LU = I)."""
        with torch.no_grad():
            if self.kind == "rotation":
                self.weight.zero_()
            else:
                self.weight.copy_(torch.eye(
                    self.features,
                    device=self.weight.device,
                    dtype=self.weight.dtype,
                ))
