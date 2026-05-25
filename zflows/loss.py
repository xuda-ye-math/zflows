# pyright: reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false

from collections.abc import Callable

import torch
from .flow import ComposedTransform
from .potential import Potential

def source_KL_F(x: torch.Tensor, target: Potential, F: ComposedTransform):
    """
    KL loss using source samples to train F (the source -> target map).
    Estimates  E_{x ~ source}[ target(F(x)) - log|det J_F(x)| ],
    a Monte Carlo estimate of KL(F_# source || target) up to an additive const.
    Input:
        x:      Tensor [N, d]      samples drawn from the source distribution
        target: Potential          negative log-density of the target (up to const)
        F:      ComposedTransform  forward flow map (typically obtained as flow.t())
    Output:
        loss: Tensor (scalar)
    """
    y, ladj = F.call_and_ladj(x) # get y = F(x) and log_abs_det_jacobian
    return (target(y) - ladj).mean()

# alias: reverse KL divergence for energy-based normalizing flow
reverse_KL = source_KL_F

def source_KL_G(x: torch.Tensor, target: Potential, G: ComposedTransform):
    """
    KL loss using source samples to train G (the target -> source map).
    Estimates  E_{x ~ source}[ target(G^-1(x)) - log|det J_{G^-1}(x)| ],
    a Monte Carlo estimate of KL(source || G_# target) up to an additive const.
    Input:
        x:      Tensor [N, d]      samples drawn from the source distribution
        target: Potential          negative log-density of the target (up to const)
        G:      ComposedTransform  flow map target -> source (typically flow.t())
    Output:
        loss: Tensor (scalar)
    """
    y, ladj = G.inv.call_and_ladj(x) # y = G^-1(x), ladj = log|det J_{G^-1}(x)|
    return (target(y) - ladj).mean()

def target_KL_F(y: torch.Tensor, source: Potential, F: ComposedTransform):
    """
    KL loss using target samples to train F (the source -> target map).
    Estimates  E_{y ~ target}[ source(F^-1(y)) - log|det J_{F^-1}(y)| ],
    a Monte Carlo estimate of KL(target || F_# source) up to an additive const.
    Input:
        y:      Tensor [N, d]      samples drawn from the target distribution
        source: Potential          negative log-density of the source (up to const)
        F:      ComposedTransform  forward flow map (typically obtained as flow.t())
    Output:
        loss: Tensor (scalar)
    """
    x, ladj = F.inv.call_and_ladj(y) # x = F^-1(y), ladj = log|det J_{F^-1}(y)|
    return (source(x) - ladj).mean()

# alias: forward KL divergence for data-driven normalizing flow
forward_KL = target_KL_F

def target_KL_G(y: torch.Tensor, source: Potential, G: ComposedTransform):
    """
    KL loss using target samples to train G (the target -> source map).
    Estimates  E_{y ~ target}[ source(G(y)) - log|det J_G(y)| ],
    a Monte Carlo estimate of KL(G_# target || source) up to an additive const.
    Input:
        y:      Tensor [N, d]      samples drawn from the target distribution
        source: Potential          negative log-density of the source (up to const)
        G:      ComposedTransform  flow map target -> source (typically flow.t())
    Output:
        loss: Tensor (scalar)
    """
    x, ladj = G.call_and_ladj(y) # x = G(y), ladj = log|det J_G(y)|
    return (source(x) - ladj).mean()


def compile(
    loss_fn: Callable[[torch.Tensor, Potential, ComposedTransform], torch.Tensor],
    potential: Potential,
    transform: ComposedTransform,
    mode: str = 'default',
) -> Callable[[torch.Tensor], torch.Tensor]:
    """torch.compile any of the four KL losses, with the heavy-Python
    arguments baked into a closure.

    The four losses in this module all share the positional signature
    `(samples, potential, transform) -> scalar`, so a single helper
    covers them all. `potential` and `transform` are captured as
    closure constants — torch.compile sees them as static and does
    NOT re-guard on the fresh `ComposedTransform` that `flow.t()`
    builds on every call.

    Captured-once correctness:
        - flow.t() returns a `ComposedTransform` whose inner
          `AutoregressiveTransform` objects hold *lazy* `meta` callables
          that read flow parameters via attribute access on every
          forward pass.
        - `optimizer.step()` mutates those `nn.Parameter` tensors
          in-place, so the captured `transform` sees the post-step
          values without rebuilding.
        - This works whether or not you compile. Capturing the
          transform once is in fact a mild speedup even without
          torch.compile (saves the per-batch object construction).

    Caveats:
        - mode='default' is safe; mode='reduce-overhead' uses CUDA
          graphs (faster but needs static batch size and stable
          parameter memory; do not call `flow.zeros()` mid-training).
        - Build this AFTER any `potential.enable_grad()` /
          `enable_eval()` so the compiled graph wraps the fast paths.
        - Dynamo retraces on tensor-shape changes. Keep `BATCH` fixed.

    Arguments:
        loss_fn:   one of {source_KL_F, source_KL_G, target_KL_F,
                   target_KL_G} or the aliases reverse_KL / forward_KL.
        potential: the target (for source_KL_*) or source (for
                   target_KL_*) — whatever `loss_fn` expects as its
                   second positional argument.
        transform: the ComposedTransform to bake in. Typically `flow.t()`.
        mode:      torch.compile mode. Default 'default'.

    Returns:
        callable (x_batch: Tensor) -> scalar loss.

    Example:
        F = flow.t()
        loss = zflows.loss.compile(reverse_KL, u1, F)
        for x_batch in batches:
            l = loss(x_batch)
            optimizer.zero_grad(); l.backward(); optimizer.step()
    """
    @torch.compile(mode=mode)
    def compiled(x: torch.Tensor) -> torch.Tensor:
        return loss_fn(x, potential, transform)
    return compiled