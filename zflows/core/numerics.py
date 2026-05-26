"""Pure tensor/autograd utilities used by zflows's transforms and flows.

Subset ported from `zuko/utils.py` (with the adaptive Dormand-Prince ODE
solver replaced by a fixed-step RK4 integrator — the only integrator zflows
uses, for a deterministic flop budget and torch.compile friendliness):
    - Partial: nn.Module wrapper of functools.partial
    - bisection / Bisection: implicit-grad bisection root finder
    - broadcast: torch.broadcast_to over the leading dims
    - gauss_legendre / GaussLegendre: n-point quadrature on [a, b]
    - rk4_fixed: fixed-step RK4 ODE integrator (unrolled under autograd)
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
    "rk4_fixed",
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
# ODE solver — fixed-step RK4 (unrolled under autograd)
# ──────────────────────────────────────────────────────────────────────

def rk4_fixed(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t0: float | Tensor,
    t1: float | Tensor,
    nt: int,
) -> Tensor:
    """Integrate dx/dt = f(t, x) from t0 to t1 with `nt` fixed RK4 steps.

    The only integrator in zflows. It takes equal steps and integrates under
    the ambient autograd tape (no custom adjoint, no checkpointing), so
    gradients through the trajectory just unroll. Intended for short
    trajectories (nt ~ 4-24): a deterministic flop budget, and a
    torch.compile-friendly loop that traces once without specialising on a
    step count. Continuous flows (`CNF`/FFJORD, `OTFlow`) pack their augmented
    `(x, ladj, ...)` state into a single tensor and integrate it here.

    The sign of `(t1 - t0)` sets the direction: backward integration
    (t1 < t0) just makes the step `h` negative.
    """
    t0 = torch.as_tensor(t0, dtype=x.dtype, device=x.device)
    t1 = torch.as_tensor(t1, dtype=x.dtype, device=x.device)
    h = (t1 - t0) / nt
    t = t0
    for _ in range(nt):
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    return x


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
