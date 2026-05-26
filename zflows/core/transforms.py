"""Bijective transformations used by zflows flows.

Ported from `zuko/transforms.py`, restricted to the subset that any
zflows flow actually uses, plus two behavioural tweaks:

  - `MonotonicRQSTransform.bound` may be a per-coordinate `(d,)` tensor
    (or any tensor broadcastable to `widths.shape[:-1]`), not just a
    scalar. NSF / NCSF use this so spline knots span
    `[-halfwidth_i, halfwidth_i]` per coordinate without needing an
    affine scaling sandwich.
  - `CircularShiftTransform.bound` accepts the same per-coord tensor.

Side effect on import: `torch.distributions.transforms.Transform`
gains a `call_and_ladj(x) -> (y, log|det J|)` method, exactly as zuko
does. This is needed so that `Transform.inv` instances (which we don't
subclass) also pick it up.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from textwrap import indent
from typing import Any

import torch
import torch.nn.functional as F
from torch import BoolTensor, LongTensor, Size, Tensor
from torch.distributions import Transform, constraints
from torch.distributions.transforms import *  # noqa: F401,F403
from torch.distributions.utils import _sum_rightmost

from .numerics import bisection, broadcast, odeint


__all__ = [
    "AdditiveTransform",
    "AutoregressiveTransform",
    "CircularShiftTransform",
    "ComposedTransform",
    "CouplingTransform",
    "DependentTransform",
    "FreeFormJacobianTransform",
    "IdentityTransform",
    "LULinearTransform",
    "MonotonicAffineTransform",
    "MonotonicRQSTransform",
    "RotationTransform",
]


# ──────────────────────────────────────────────────────────────────────
# Patch Transform.call_and_ladj.
#
# Sub-classes that override `call_and_ladj` (e.g. MonotonicRQSTransform)
# get a fused y + ladj path; everything else gets a generic fallback
# that calls _call + log_abs_det_jacobian. We also rename the private
# _InverseTransform for prettier reprs, exactly as zuko does.
# ──────────────────────────────────────────────────────────────────────

def _call_and_ladj(self, x: Tensor) -> tuple[Tensor, Tensor]:
    y = self.__call__(x)
    ladj = self.log_abs_det_jacobian(x, y)
    return y, ladj


Transform.call_and_ladj = _call_and_ladj  # type: ignore[attr-defined]
torch.distributions.transforms._InverseTransform.__name__ = "Inverse"  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────
# Composed / Dependent / Identity / Additive  — small structural pieces
# ──────────────────────────────────────────────────────────────────────

class ComposedTransform(Transform):
    """Composition f_n ∘ ... ∘ f_0 with fused call_and_ladj.

    Equivalent to torch.distributions.transforms.ComposeTransform plus
    zuko's optimised log-det accumulator.
    """

    def __init__(self, *transforms: Transform, **kwargs) -> None:
        super().__init__(**kwargs)
        assert transforms, "'transforms' cannot be empty"

        event_dim = 0
        for t in reversed(transforms):
            event_dim = t.domain.event_dim + max(event_dim - t.codomain.event_dim, 0)
        self.domain_dim = event_dim

        for t in transforms:
            event_dim += t.codomain.event_dim - t.domain.event_dim
        self.codomain_dim = event_dim
        self.transforms = transforms

    def __repr__(self) -> str:
        lines = [f"({i}): {t}" for i, t in enumerate(self.transforms)]
        body = indent("\n".join(lines), "  ")
        return f"{self.__class__.__name__}(\n{body}\n)"

    @property
    def domain(self) -> constraints.Constraint:
        domain = self.transforms[0].domain
        reinterpreted = self.domain_dim - domain.event_dim
        if reinterpreted > 0:
            return constraints.independent(domain, reinterpreted)
        return domain

    @property
    def codomain(self) -> constraints.Constraint:
        codomain = self.transforms[-1].codomain
        reinterpreted = self.codomain_dim - codomain.event_dim
        if reinterpreted > 0:
            return constraints.independent(codomain, reinterpreted)
        return codomain

    @property
    def bijective(self) -> bool:
        return all(t.bijective for t in self.transforms)

    def _call(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    @property
    def inv(self) -> Transform:
        new = self.__new__(ComposedTransform)
        new.transforms = [t.inv for t in reversed(self.transforms)]
        new.domain_dim = self.codomain_dim
        new.codomain_dim = self.domain_dim
        Transform.__init__(new)
        return new

    def _inverse(self, y: Tensor) -> Tensor:
        for t in reversed(self.transforms):
            y = t.inv(y)
        return y

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Tensor) -> tuple[Tensor, Tensor]:
        event_dim = self.domain_dim
        acc = 0
        for t in self.transforms:
            x, ladj = t.call_and_ladj(x)
            acc = acc + _sum_rightmost(ladj, event_dim - t.domain.event_dim)
            event_dim += t.codomain.event_dim - t.domain.event_dim
        return x, acc

    def forward_shape(self, shape: Size) -> Size:
        for t in self.transforms:
            shape = t.forward_shape(shape)
        return shape

    def inverse_shape(self, shape: Size) -> Size:
        for t in reversed(self.transforms):
            shape = t.inverse_shape(shape)
        return shape


class DependentTransform(Transform):
    """Treat the last `reinterpreted` dims of base as dependent.

    Optimised counterpart to torch's IndependentTransform.
    """

    def __init__(self, base: Transform, reinterpreted: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.base = base
        self.reinterpreted = reinterpreted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base}, {self.reinterpreted})"

    @property
    def domain(self) -> constraints.Constraint:
        return constraints.independent(self.base.domain, self.reinterpreted)

    @property
    def codomain(self) -> constraints.Constraint:
        return constraints.independent(self.base.codomain, self.reinterpreted)

    @property
    def bijective(self) -> bool:
        return self.base.bijective

    def _call(self, x: Tensor) -> Tensor:
        return self.base(x)

    @property
    def inv(self) -> Transform:
        return DependentTransform(self.base.inv, self.reinterpreted)

    def _inverse(self, y: Tensor) -> Tensor:
        return self.base.inv(y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        ladj = self.base.log_abs_det_jacobian(x, y)
        return _sum_rightmost(ladj, self.reinterpreted)

    def call_and_ladj(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y, ladj = self.base.call_and_ladj(x)
        ladj = _sum_rightmost(ladj, self.reinterpreted)
        return y, ladj

    def forward_shape(self, shape: Size) -> Size:
        return self.base.forward_shape(shape)

    def inverse_shape(self, shape: Size) -> Size:
        return self.base.inverse_shape(shape)


class IdentityTransform(Transform):
    """f(x) = x."""

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, IdentityTransform)

    def _call(self, x: Tensor) -> Tensor:
        return x

    def _inverse(self, y: Tensor) -> Tensor:
        return y

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x)


class AdditiveTransform(Transform):
    """f(x) = x + shift.

    Used by NSF/NCSF to translate the box [a, b]^d to the per-coord
    centred box [-half, half]^d (and back). shift is typically a (d,)
    tensor; broadcasting with `x` of shape (..., d) is automatic.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, shift: Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shift = shift

    def _call(self, x: Tensor) -> Tensor:
        return x + self.shift

    def _inverse(self, y: Tensor) -> Tensor:
        return y - self.shift

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x)


# ──────────────────────────────────────────────────────────────────────
# MonotonicAffineTransform  — used as default univariate in MAF
# ──────────────────────────────────────────────────────────────────────

class MonotonicAffineTransform(Transform):
    """f(x) = exp(a) * x + b with a clamped to [log(slope), -log(slope)].

    Default univariate transform inside autoregressive / coupling flows.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        shift: Tensor,
        scale: Tensor,
        slope: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.shift = shift
        self.log_scale = scale / (1 + abs(scale / math.log(slope)))
        self.scale = self.log_scale.exp()

    def _call(self, x: Tensor) -> Tensor:
        return x * self.scale + self.shift

    def _inverse(self, y: Tensor) -> Tensor:
        return (y - self.shift) / self.scale

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.log_scale.expand(x.shape)


# ──────────────────────────────────────────────────────────────────────
# MonotonicRQSTransform  — knots on per-coord [-bound, bound]
# ──────────────────────────────────────────────────────────────────────

class MonotonicRQSTransform(Transform):
    """Monotonic rational-quadratic spline on per-coord [-bound, bound].

    Reference:
        Neural Spline Flows (Durkan et al., 2019) — https://arxiv.org/abs/1906.04032

    `bound` may be a scalar (zuko default) or a tensor of shape (..., d)
    broadcastable to widths.shape[:-1]. With a per-coord (d,) tensor,
    each coordinate's knots span [-bound_i, bound_i] independently —
    this is what NSF / NCSF use to avoid the affine scaling sandwich.

    Arguments:
        widths:      unconstrained bin widths,      shape (..., K)
        heights:     unconstrained bin heights,     shape (..., K)
        derivatives: unconstrained knot slopes,     shape (..., K - 1)
        bound:       (co)domain bound; scalar or shape (..., 1) / (..., d).
        slope:       lower-bound on every segment's slope (numeric stability).
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivatives: Tensor,
        bound: Tensor | float = 1.0,
        slope: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        widths = widths / (1 + abs(2 * widths / math.log(slope)))
        heights = heights / (1 + abs(2 * heights / math.log(slope)))
        derivatives = derivatives / (1 + abs(derivatives / math.log(slope)))

        widths = F.pad(F.softmax(widths, dim=-1), (1, 0), value=0)
        heights = F.pad(F.softmax(heights, dim=-1), (1, 0), value=0)
        derivatives = F.pad(derivatives, (1, 1), value=0)

        # Per-coord bound broadcast: append a knot-index dim to `bound`
        # so it lines up with widths/heights' last dim.
        B = bound[..., None] if torch.is_tensor(bound) else bound

        self.horizontal = B * (2 * torch.cumsum(widths, dim=-1) - 1)
        self.vertical = B * (2 * torch.cumsum(heights, dim=-1) - 1)
        self.derivatives = torch.exp(derivatives)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bins={self.bins})"

    @property
    def bins(self) -> int:
        return self.horizontal.shape[-1] - 1

    def bin(self, k: LongTensor) -> tuple[Tensor, ...]:
        mask = torch.logical_and(0 <= k, k < self.bins)

        k = k % self.bins
        k0_k1 = torch.stack((k, k + 1))

        k0_k1, hs, vs, ds = broadcast(
            k0_k1[..., None],
            self.horizontal,
            self.vertical,
            self.derivatives,
            ignore=1,
        )

        x0, x1 = hs.gather(-1, k0_k1).squeeze(dim=-1)
        y0, y1 = vs.gather(-1, k0_k1).squeeze(dim=-1)
        d0, d1 = ds.gather(-1, k0_k1).squeeze(dim=-1)

        s = (y1 - y0) / (x1 - x0)
        return mask, x0, x1, y0, y1, d0, d1, s

    @staticmethod
    def searchsorted(seq: Tensor, value: Tensor) -> LongTensor:
        return torch.sum(seq < value[..., None], dim=-1)

    def _call(self, x: Tensor) -> Tensor:
        k = self.searchsorted(self.horizontal, x) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)
        y = y0 + (y1 - y0) * (s * z**2 + d0 * z * (1 - z)) / (
            s + (d0 + d1 - 2 * s) * z * (1 - z)
        )
        return torch.where(mask, y, x)

    def _inverse(self, y: Tensor) -> Tensor:
        k = self.searchsorted(self.vertical, y) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        y_ = mask * (y - y0)
        a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2 * s)
        b = (y1 - y0) * d0 - y_ * (d0 + d1 - 2 * s)
        c = -s * y_

        z = 2 * c / (-b - (b**2 - 4 * a * c).sqrt())
        x = x0 + z * (x1 - x0)
        return torch.where(mask, x, y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Tensor) -> tuple[Tensor, Tensor]:
        k = self.searchsorted(self.horizontal, x) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)
        y = y0 + (y1 - y0) * (s * z**2 + d0 * z * (1 - z)) / (
            s + (d0 + d1 - 2 * s) * z * (1 - z)
        )

        jacobian = (
            s**2
            * (2 * s * z * (1 - z) + d0 * (1 - z) ** 2 + d1 * z**2)
            / (s + (d0 + d1 - 2 * s) * z * (1 - z)) ** 2
        )
        return torch.where(mask, y, x), mask * jacobian.log()


# ──────────────────────────────────────────────────────────────────────
# CircularShiftTransform  — wrap-around on per-coord [-bound, bound]
# ──────────────────────────────────────────────────────────────────────

class CircularShiftTransform(Transform):
    """Circular shift bijection on per-coord [-bound, bound].

    f(x) = ((x + bound) mod 2*bound) - bound

    `bound` may be a scalar (zuko default) or a tensor broadcastable
    with the last dim of x; with `bound: (d,)`, each coordinate wraps
    on its own period 2*bound_i.

    Note: domain/codomain are declared `constraints.real` (rather than
    `constraints.interval(-B, B)`) because per-coord intervals can't be
    expressed as a single `constraints.Constraint`. The bijection is
    still identity-modulo-period on each coordinate.
    """

    bijective = True

    def __init__(self, bound: Tensor | float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bound = bound
        self.domain = constraints.real
        self.codomain = constraints.real

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bound={self.bound})"

    def _call(self, x: Tensor) -> Tensor:
        return torch.remainder(x + self.bound, 2 * self.bound) - self.bound

    def _inverse(self, y: Tensor) -> Tensor:
        return torch.remainder(y + self.bound, 2 * self.bound) - self.bound

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x)


# ──────────────────────────────────────────────────────────────────────
# AutoregressiveTransform / CouplingTransform / FreeFormJacobianTransform
# ──────────────────────────────────────────────────────────────────────

class AutoregressiveTransform(Transform):
    """y_i = f(x_i | x_<i) — autoregressive scheme.

    `meta(x)` returns a univariate Transform whose parameters depend
    autoregressively on x via a masked MLP.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        meta: Callable[[Tensor], Transform],
        passes: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.meta = meta
        self.passes = passes

    def _call(self, x: Tensor) -> Tensor:
        return self.meta(x)(x)

    def _inverse(self, y: Tensor) -> Tensor:
        x = torch.zeros_like(y)
        for _ in range(self.passes):
            x = self.meta(x).inv(y)
        return x

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.meta(x).log_abs_det_jacobian(x, y)

    def call_and_ladj(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.meta(x).call_and_ladj(x)


class CouplingTransform(Transform):
    """y_a = x_a, y_b = f(x_b | x_a) — coupling scheme.

    mask: boolean vector; True = "kept" (x_a), False = "transformed" (x_b).
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        meta: Callable[[Tensor], Transform],
        mask: BoolTensor,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.meta = meta
        self.idx_a = mask.nonzero().squeeze(-1)
        self.idx_b = (~mask).nonzero().squeeze(-1)

    def split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return x[..., self.idx_a], x[..., self.idx_b]

    def merge(self, x_a: Tensor, x_b: Tensor, shape: Size) -> Tensor:
        x = x_a.new_empty(shape)
        x[..., self.idx_a] = x_a
        x[..., self.idx_b] = x_b
        return x

    def _call(self, x: Tensor) -> Tensor:
        x_a, x_b = self.split(x)
        y_b = self.meta(x_a)(x_b)
        return self.merge(x_a, y_b, x.shape)

    def _inverse(self, y: Tensor) -> Tensor:
        y_a, y_b = self.split(y)
        x_b = self.meta(y_a).inv(y_b)
        return self.merge(y_a, x_b, y.shape)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        x_a, x_b = self.split(x)
        _, y_b = self.split(y)
        return self.meta(x_a).log_abs_det_jacobian(x_b, y_b)

    def call_and_ladj(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_a, x_b = self.split(x)
        y_b, ladj = self.meta(x_a).call_and_ladj(x_b)
        y = self.merge(x_a, y_b, x.shape)
        return y, ladj


class FreeFormJacobianTransform(Transform):
    """FFJORD continuous-time bijection: dx/dt = f_phi(t, x).

    `exact=True`  → exact log|det J| via O(d) augmented ODE.
    `exact=False` → Hutchinson trace estimator (stochastic, biased grads).
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        f: Callable[[Tensor, Tensor], Tensor],
        t0: float | Tensor = 0.0,
        t1: float | Tensor = 1.0,
        phi: Iterable[Tensor] = (),
        atol: float = 1e-6,
        rtol: float = 1e-5,
        exact: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.f = f
        self.t0 = t0
        self.t1 = t1
        self.phi = tuple(filter(lambda p: p.requires_grad, phi))
        self.atol = atol
        self.rtol = rtol
        self.exact = exact
        self.trace_scale = 1e-2

    def _call(self, x: Tensor) -> Tensor:
        return odeint(self.f, x, self.t0, self.t1, self.phi, self.atol, self.rtol)

    @property
    def inv(self) -> Transform:
        return FreeFormJacobianTransform(
            f=self.f,
            t0=self.t1,
            t1=self.t0,
            phi=self.phi,
            atol=self.atol,
            rtol=self.rtol,
            exact=self.exact,
        )

    def _inverse(self, y: Tensor) -> Tensor:
        return odeint(self.f, y, self.t1, self.t0, self.phi, self.atol, self.rtol)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Tensor) -> tuple[Tensor, Tensor]:
        create_graph = torch.is_grad_enabled() and (x.requires_grad or bool(self.phi))

        if self.exact:
            I = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
            I = I.expand(*x.shape, -1).movedim(-1, 0)
        else:
            eps = torch.randn_like(x)

        def f_aug(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.clone().requires_grad_()
                dx = self.f(t, x)
            if self.exact:
                jacobian = torch.autograd.grad(
                    dx, x, I, create_graph=create_graph, is_grads_batched=True
                )[0]
                trace = torch.einsum("i...i", jacobian)
            else:
                epsjp = torch.autograd.grad(dx, x, eps, create_graph=create_graph)[0]
                trace = (epsjp * eps).sum(dim=-1)
            return dx, trace * self.trace_scale

        ladj = torch.zeros_like(x[..., 0])
        y, ladj = odeint(f_aug, (x, ladj), self.t0, self.t1, self.phi, self.atol, self.rtol)
        return y, ladj * (1 / self.trace_scale)


# ──────────────────────────────────────────────────────────────────────
# Linear mixing transforms on R^d (Glow-style 1x1 invertible "conv")
# ──────────────────────────────────────────────────────────────────────

class RotationTransform(Transform):
    r"""Rotation `f(x) = R x` with `R = exp(A - A^T)` orthogonal.

    Because `A - A^T` is skew-symmetric, the matrix exponential is
    orthogonal, so the transform is volume-preserving and its
    log-abs-determinant is identically zero.

    Arguments:
        A: square matrix `A`, with shape `(*, D, D)`.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, A: Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.R = torch.linalg.matrix_exp(A - A.mT)

    def _call(self, x: Tensor) -> Tensor:
        return torch.einsum("...ij,...j->...i", self.R, x)

    def _inverse(self, y: Tensor) -> Tensor:
        return torch.einsum("...ij,...i->...j", self.R, y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x[..., 0])


class LULinearTransform(Transform):
    r"""Linear map `f(x) = L U x` with LU decomposition.

    `L` is the lower-triangular part of the input matrix (diagonal
    included, providing `log|det|`); `U` is the strict upper-triangular
    part plus the identity (unit diagonal), so the forward is a single
    `L @ U @ x` and the log-abs-determinant is `sum(log|diag(L)|)`.

    Arguments:
        LU: matrix whose lower / upper triangular parts hold the non-zero
            elements of `L` and `U`, with shape `(*, D, D)`.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, LU: Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        I = torch.eye(LU.shape[-1], dtype=LU.dtype, device=LU.device)
        self.L = torch.tril(LU)
        self.U = torch.triu(LU, diagonal=1) + I

    def _call(self, x: Tensor) -> Tensor:
        return torch.einsum("...ij,...j->...i", self.L @ self.U, x)

    def _inverse(self, y: Tensor) -> Tensor:
        return torch.linalg.solve_triangular(
            self.U,
            torch.linalg.solve_triangular(
                self.L,
                y.unsqueeze(-1),
                upper=False,
                unitriangular=False,
            ),
            upper=True,
            unitriangular=True,
        ).squeeze(-1)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        diag = torch.diagonal(self.L, dim1=-1, dim2=-2)
        ladj = diag.abs().log().sum(dim=-1)
        return ladj.expand_as(x[..., 0])
