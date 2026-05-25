"""Neural-network building blocks used by zflows flows.

Subset of `zuko/nn.py`:
    - Linear / MLP                    — used by RealNVP coupling MLP and CNF ODE MLP
    - MaskedLinear / MaskedMLP        — used by MAF conditioner

Unused (and dropped): MonotonicLinear, MonotonicMLP, TwoWayELU, Residual,
LayerNorm. The internal `linear()` helper survives because Linear.forward
needs to handle the (optional) stack dim.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import BoolTensor, Tensor


__all__ = ["Linear", "MaskedLinear", "MaskedMLP", "MLP"]


# ──────────────────────────────────────────────────────────────────────
# Linear — basic dense layer with optional stack dim
# ──────────────────────────────────────────────────────────────────────

def linear(x: Tensor, W: Tensor, b: Tensor | None = None) -> Tensor:
    """Forward of W x + b that handles both stacked and non-stacked W."""
    if W.dim() == 2:
        return F.linear(x, W, b)
    x = torch.einsum("...ij,...j->...i", W, x)
    return x if b is None else x + b


class Linear(nn.Module):
    """y = x W^T + b. With `stack=S`, builds S independent operators."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        stack: int | None = None,
    ) -> None:
        super().__init__()
        shape = () if stack is None else (stack,)
        self.weight = nn.Parameter(torch.empty(*shape, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(*shape, out_features)) if bias else None

        self.reset_parameters()
        self.in_features = in_features
        self.out_features = out_features

    def reset_parameters(self) -> None:
        bound = 1 / self.weight.shape[-1] ** 0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        if self.weight.dim() == 2:
            stack, fout, fin = (None, *self.weight.shape)
        else:
            stack, fout, fin = self.weight.shape
        bias = self.bias is not None
        if stack is None:
            return f"in_features={fin}, out_features={fout}, bias={bias}"
        return f"in_features={fin}, out_features={fout}, bias={bias}, stack={stack}"

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)


# ──────────────────────────────────────────────────────────────────────
# MLP — multi-layer perceptron (RealNVP coupling, CNF ODE drift)
# ──────────────────────────────────────────────────────────────────────

class MLP(nn.Sequential):
    """Multi-layer perceptron with configurable hidden widths/activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] | None = None,
        **kwargs,
    ) -> None:
        if activation is None:
            activation = nn.ReLU
        layers: list[nn.Module] = []
        for before, after in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
            strict=True,
        ):
            layers.extend([Linear(before, after, **kwargs), activation()])
        layers = layers[:-1]  # drop the trailing activation
        super().__init__(*layers)
        self.in_features = in_features
        self.out_features = out_features


# ──────────────────────────────────────────────────────────────────────
# MaskedLinear / MaskedMLP — autoregressive conditioner backbone (MAF)
# ──────────────────────────────────────────────────────────────────────

class MaskedLinear(nn.Linear):
    """Linear with a fixed boolean adjacency mask on the weight matrix."""

    def __init__(self, adjacency: BoolTensor, **kwargs) -> None:
        super().__init__(*reversed(adjacency.shape), **kwargs)
        self.register_buffer("mask", adjacency)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.mask * self.weight, self.bias)


class MaskedMLP(nn.Sequential):
    """Masked MLP whose Jacobian entries dy_i/dx_j are zero where A_ij = 0."""

    def __init__(
        self,
        adjacency: BoolTensor,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] | None = None,
    ) -> None:
        out_features, in_features = adjacency.shape
        if activation is None:
            activation = nn.ReLU

        # Merge outputs with identical dependency sets so the masked
        # linear layers can be smaller.
        adjacency, inverse = torch.unique(adjacency, dim=0, return_inverse=True)

        # P_ij = 1 iff A_ik = 1 ∀k with A_jk = 1 ("i depends on at most
        # what j depends on" — used to extend masks through hidden layers).
        precedence = (
            adjacency.double() @ adjacency.double().t() == adjacency.sum(dim=-1)
        )

        layers: list[nn.Module] = []
        indices: Tensor | None = None
        for i, features in enumerate((*hidden_features, out_features)):
            if i > 0:
                assert indices is not None
                mask = precedence[:, indices]
            else:
                mask = adjacency

            if (~mask).all():
                raise ValueError("The adjacency matrix leads to a null Jacobian.")

            if i < len(hidden_features):
                reachable = mask.sum(dim=-1).nonzero().squeeze(dim=-1)
                indices = reachable[torch.arange(features) % len(reachable)]
                mask = mask[indices]
            else:
                mask = mask[inverse]

            layers.append(MaskedLinear(adjacency=mask))
            layers.append(activation())
        layers.pop()  # drop trailing activation

        super().__init__(*layers)
        self.in_features = in_features
        self.out_features = out_features
