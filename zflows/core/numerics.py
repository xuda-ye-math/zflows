"""Pure tensor/autograd utilities used by zflows's transforms and flows.

Verbatim port of the relevant subset of `zuko/utils.py`:
    - Partial: nn.Module wrapper of functools.partial
    - bisection / Bisection: implicit-grad bisection root finder
    - broadcast: torch.broadcast_to over the leading dims
    - gauss_legendre / GaussLegendre: n-point quadrature on [a, b]
    - odeint / AdaptiveCheckpointAdjoint / dopri45 / NestedTensor:
        adjoint-checkpointed ODE solver
    - unpack: split a packed tensor along its last dim by shapes

This module is independent of distributions, flows, conditioning, etc.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from functools import cache
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.autograd.function import FunctionCtx, once_differentiable


__all__ = [
    "Partial",
    "bisection",
    "broadcast",
    "gauss_legendre",
    "odeint",
    "unpack",
]


# ──────────────────────────────────────────────────────────────────────
# Partial — nn.Module-aware functools.partial
# ──────────────────────────────────────────────────────────────────────

class Partial(nn.Module):
    """An nn.Module-aware version of functools.partial.

    Tensor args become parameters (or buffers, with buffer=True). Submodules
    in `f` are auto-registered. Forward returns f(*args, *extra, **kwargs,
    **extra).
    """

    def __init__(
        self,
        f: Callable,
        /,
        *args,
        buffer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.f = f

        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                if buffer:
                    self.register_buffer(f"_{i}", arg)
                else:
                    self.register_parameter(f"_{i}", nn.Parameter(arg))
            else:
                setattr(self, f"_{i}", arg)
        self._nargs = len(args)

        for key, arg in kwargs.items():
            if torch.is_tensor(arg):
                if buffer:
                    self.register_buffer(key, arg)
                else:
                    self.register_parameter(key, nn.Parameter(arg))
            else:
                setattr(self, key, arg)
        self._keys = list(kwargs.keys())

    @property
    def args(self) -> Sequence[Any]:
        return [getattr(self, f"_{i}") for i in range(self._nargs)]

    @property
    def kwargs(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._keys}

    def extra_repr(self) -> str:
        if isinstance(self.f, nn.Module):
            return ""
        return f"(f): {self.f}"

    def forward(self, *args, **kwargs) -> Any:
        return self.f(*self.args, *args, **self.kwargs, **kwargs)


# ──────────────────────────────────────────────────────────────────────
# Bisection — implicit-grad bisection root finder
# ──────────────────────────────────────────────────────────────────────

def bisection(
    f: Callable[[Tensor], Tensor],
    y: Tensor,
    a: float | Tensor,
    b: float | Tensor,
    n: int = 16,
    phi: Iterable[Tensor] = (),
) -> Tensor:
    """Bisection root finder for `f(x) = y` with implicit-grad backward."""

    a = torch.as_tensor(a).to(y)
    b = torch.as_tensor(b).to(y)
    return Bisection.apply(f, y, a, b, n, *phi)


class Bisection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        f: Callable[[Tensor], Tensor],
        y: Tensor,
        a: Tensor,
        b: Tensor,
        n: int,
        *phi: Tensor,
    ) -> Tensor:
        for _ in range(n):
            c = (a + b) / 2
            mask = f(c) < y
            a = torch.where(mask, c, a)
            b = torch.where(mask, b, c)
        x = (a + b) / 2

        ctx.f = f
        ctx.save_for_backward(x, *phi)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, grad_x: Tensor) -> tuple[Tensor, ...]:
        f = ctx.f
        x, *phi = ctx.saved_tensors

        with torch.enable_grad():
            x = x.detach().requires_grad_()
            y = f(x)

        jacobian = torch.autograd.grad(
            y, x, torch.ones_like(y), retain_graph=bool(phi)
        )[0]
        grad_y = grad_x / jacobian

        if phi:
            grad_phi = torch.autograd.grad(y, phi, -grad_y, retain_graph=True)
        else:
            grad_phi = ()
        return (None, grad_y, None, None, None, *grad_phi)


# ──────────────────────────────────────────────────────────────────────
# broadcast — torch.broadcast_to over the leading dims
# ──────────────────────────────────────────────────────────────────────

def broadcast(*tensors: Tensor, ignore: int | Sequence[int] = 0) -> list[Tensor]:
    """Broadcast tensors over leading dims; keep the last `ignore` dims intact."""

    if isinstance(ignore, int):
        ignore = [ignore] * len(tensors)

    dims = [t.dim() - i for t, i in zip(tensors, ignore, strict=True)]
    common = torch.broadcast_shapes(
        *(t.shape[:i] for t, i in zip(tensors, dims, strict=True))
    )

    return [
        torch.broadcast_to(t, common + t.shape[i:])
        for t, i in zip(tensors, dims, strict=True)
    ]


# ──────────────────────────────────────────────────────────────────────
# Gauss–Legendre — n-point quadrature on [a, b] with implicit-grad rule
# ──────────────────────────────────────────────────────────────────────

def gauss_legendre(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    n: int = 3,
    phi: Iterable[Tensor] = (),
) -> Tensor:
    """n-point Gauss-Legendre quadrature of f over [a, b]."""
    return GaussLegendre.apply(f, a, b, n, *phi)


class GaussLegendre(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        f: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        n: int,
        *phi: Tensor,
    ) -> Tensor:
        ctx.f, ctx.n = f, n
        ctx.save_for_backward(a, b, *phi)
        return GaussLegendre.quadrature(f, a, b, n)

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, grad_area: Tensor) -> tuple[Tensor, ...]:
        f, n = ctx.f, ctx.n
        a, b, *phi = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_a = -f(a) * grad_area
        else:
            grad_a = None

        if ctx.needs_input_grad[2]:
            grad_b = f(b) * grad_area
        else:
            grad_b = None

        if phi:
            with torch.enable_grad():
                area = GaussLegendre.quadrature(f, a, b, n)
            grad_phi = torch.autograd.grad(area, phi, grad_area, retain_graph=True)
        else:
            grad_phi = ()
        return (None, grad_a, grad_b, None, *grad_phi)

    @staticmethod
    @cache
    def nodes(n: int, **kwargs) -> tuple[Tensor, Tensor]:
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        kwargs.setdefault("dtype", torch.get_default_dtype())
        return (
            torch.as_tensor(nodes, **kwargs),
            torch.as_tensor(weights, **kwargs),
        )

    @staticmethod
    def quadrature(
        f: Callable[[Tensor], Tensor], a: Tensor, b: Tensor, n: int
    ) -> Tensor:
        nodes, weights = GaussLegendre.nodes(n, dtype=a.dtype, device=a.device)
        nodes = torch.lerp(a[..., None], b[..., None], nodes).movedim(-1, 0)
        return (b - a) * torch.tensordot(weights, f(nodes), dims=1)


# ──────────────────────────────────────────────────────────────────────
# ODE solver — Dormand-Prince adaptive RK with checkpointed adjoint
# ──────────────────────────────────────────────────────────────────────

def odeint(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor | Sequence[Tensor],
    t0: float | Tensor,
    t1: float | Tensor,
    phi: Iterable[Tensor] = (),
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Tensor | Sequence[Tensor]:
    """Integrate dx/dt = f(t, x) from t0 to t1 using adaptive Dormand-Prince
    with adjoint-checkpointed backprop."""

    settings = (atol, rtol, torch.is_grad_enabled())

    if torch.is_tensor(x):
        x0 = x
        g = f
    else:
        shapes = [y.shape for y in x]

        def pack(x_: Iterable[Tensor]) -> Tensor:
            return torch.cat([y.flatten() for y in x_])

        x0 = pack(x)
        g = lambda t, x_: pack(f(t, *unpack(x_, shapes)))

    t0 = torch.as_tensor(t0, dtype=x0.dtype, device=x0.device)
    t1 = torch.as_tensor(t1, dtype=x0.dtype, device=x0.device)
    assert not t0.shape and not t1.shape, "'t0' and 't1' must be scalars"

    x1 = AdaptiveCheckpointAdjoint.apply(settings, g, x0, t0, t1, *phi)
    return x1 if torch.is_tensor(x) else unpack(x1, shapes)


# fmt: off
def dopri45(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
    error: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """One step of the Dormand-Prince 4(5) Runge-Kutta method."""
    k1 = dt * f(t, x)
    k2 = dt * f(t + 1 / 5 * dt, x + 1 / 5 * k1)
    k3 = dt * f(t + 3 / 10 * dt, x + 3 / 40 * k1 + 9 / 40 * k2)
    k4 = dt * f(t + 4 / 5 * dt, x + 44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3)
    k5 = dt * f(
        t + 8 / 9 * dt,
        x + 19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4,
    )
    k6 = dt * f(
        t + dt,
        x
        + 9017 / 3168 * k1
        - 355 / 33 * k2
        + 46732 / 5247 * k3
        + 49 / 176 * k4
        - 5103 / 18656 * k5,
    )
    x_next = (
        x
        + 35 / 384 * k1
        + 500 / 1113 * k3
        + 125 / 192 * k4
        - 2187 / 6784 * k5
        + 11 / 84 * k6
    )
    if not error:
        return x_next
    k7 = dt * f(t + dt, x_next)
    x_star = (
        x
        + 5179 / 57600 * k1
        + 7571 / 16695 * k3
        + 393 / 640 * k4
        - 92097 / 339200 * k5
        + 187 / 2100 * k6
        + 1 / 40 * k7
    )
    return x_next, abs(x_next - x_star)
# fmt: on


class NestedTensor(tuple):
    """Tuple-of-tensors with elementwise +, -, scalar-*."""

    def __add__(self, other: NestedTensor) -> NestedTensor:
        return NestedTensor(x + y for x, y in zip(self, other, strict=True))

    def __sub__(self, other: NestedTensor) -> NestedTensor:
        return NestedTensor(x - y for x, y in zip(self, other, strict=True))

    def __rmul__(self, other: Tensor) -> NestedTensor:
        return NestedTensor(other * x for x in self)


class AdaptiveCheckpointAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        settings: tuple[float, float, bool],
        f: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        t0: Tensor,
        t1: Tensor,
        *phi: Tensor,
    ) -> Tensor:
        atol, rtol, grad_enabled = settings

        ctx.f = f
        ctx.save_for_backward(x, t0, t1, *phi)
        ctx.steps = []

        t, dt = t0, t1 - t0
        sign = torch.sign(dt)

        while sign * (t1 - t) > 0:
            dt = sign * torch.min(abs(dt), abs(t1 - t))

            while True:
                y, error = dopri45(f, x, t, dt, error=True)
                tolerance = atol + rtol * torch.max(abs(x), abs(y))
                error = torch.max(error / tolerance).clip(min=1e-9).item()

                if error < 1.0:
                    x, t = y, t + dt
                    if grad_enabled:
                        ctx.steps.append((x, t, dt))
                dt = dt * min(10.0, max(0.1, 0.9 / error ** (1 / 5)))
                if error < 1.0:
                    break
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, grad_x: Tensor) -> tuple[Tensor, ...]:
        f = ctx.f
        x0, t0, t1, *phi = ctx.saved_tensors
        x1, _, _ = ctx.steps[-1]

        if ctx.needs_input_grad[4]:
            grad_t1 = torch.sum(f(t1, x1) * grad_x)
        else:
            grad_t1 = None

        grad_phi = map(torch.zeros_like, phi)

        def g(t: Tensor, x_aug: NestedTensor) -> NestedTensor:
            x, grad_x_, *_ = x_aug
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                dx = f(t, x)
            grad_x_, *grad_phi_ = torch.autograd.grad(dx, (x, *phi), -grad_x_)
            return NestedTensor((dx, grad_x_, *grad_phi_))

        for x, t, dt in reversed(ctx.steps):
            x_aug = NestedTensor((x, grad_x, *grad_phi))
            _, grad_x, *grad_phi = dopri45(g, x_aug, t, -dt)

        if ctx.needs_input_grad[3]:
            grad_t0 = torch.sum(f(t0, x0) * grad_x)
        else:
            grad_t0 = None
        return (None, None, grad_x, grad_t0, grad_t1, *grad_phi)


# ──────────────────────────────────────────────────────────────────────
# unpack — split a packed tensor along its last dim by shapes
# ──────────────────────────────────────────────────────────────────────

def unpack(x: Tensor, shapes: Sequence[Size]) -> Sequence[Tensor]:
    """Inverse of `torch.cat([t.flatten() for t in tensors])` given shapes."""
    sizes = [math.prod(s) for s in shapes]
    x = x.split(sizes, -1)
    x = (y.unflatten(-1, (*s, 1)) for y, s in zip(x, shapes, strict=True))
    x = (y.squeeze(-1) for y in x)
    return tuple(x)
