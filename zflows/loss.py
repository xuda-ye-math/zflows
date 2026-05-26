# pyright: reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false

from collections.abc import Callable

import torch
from .flow import ComposedTransform
from .potential import Potential


# ──────────────────────────────────────────────────────────────────────
# KL loss estimators — Monte Carlo KL divergences on source / target
# ──────────────────────────────────────────────────────────────────────

def source_KL_F(x: torch.Tensor, target: Potential, F: ComposedTransform, beta: float = 1.0):
    """
    KL loss using source samples to train F (the source -> target map).
    Estimates  E_{x ~ source}[ beta * target(F(x)) - log|det J_F(x)| ],
    a Monte Carlo estimate of KL(F_# source || exp(-beta * target)) up to an additive const.
    Input:
        x:      Tensor [N, d]      samples drawn from the source distribution
        target: Potential          negative log-density of the target (up to const)
        F:      ComposedTransform  forward flow map (typically obtained as flow.t())
        beta:   float              inverse temperature scaling the target potential (default 1.0)
    Output:
        loss: Tensor (scalar)
    """
    y, ladj = F.call_and_ladj(x) # get y = F(x) and log_abs_det_jacobian
    return (beta * target(y) - ladj).mean()

# alias: reverse KL divergence for energy-based normalizing flow
reverse_KL = source_KL_F

def source_KL_G(x: torch.Tensor, target: Potential, G: ComposedTransform, beta: float = 1.0):
    """
    KL loss using source samples to train G (the target -> source map).
    Estimates  E_{x ~ source}[ beta * target(G^-1(x)) - log|det J_{G^-1}(x)| ],
    a Monte Carlo estimate of KL(source || G_# exp(-beta * target)) up to an additive const.
    Input:
        x:      Tensor [N, d]      samples drawn from the source distribution
        target: Potential          negative log-density of the target (up to const)
        G:      ComposedTransform  flow map target -> source (typically flow.t())
        beta:   float              inverse temperature scaling the target potential (default 1.0)
    Output:
        loss: Tensor (scalar)
    """
    y, ladj = G.inv.call_and_ladj(x) # y = G^-1(x), ladj = log|det J_{G^-1}(x)|
    return (beta * target(y) - ladj).mean()

def target_KL_F(y: torch.Tensor, source: Potential, F: ComposedTransform, beta: float = 1.0):
    """
    KL loss using target samples to train F (the source -> target map).
    Estimates  E_{y ~ target}[ beta * source(F^-1(y)) - log|det J_{F^-1}(y)| ],
    a Monte Carlo estimate of KL(target || F_# exp(-beta * source)) up to an additive const.
    Input:
        y:      Tensor [N, d]      samples drawn from the target distribution
        source: Potential          negative log-density of the source (up to const)
        F:      ComposedTransform  forward flow map (typically obtained as flow.t())
        beta:   float              inverse temperature scaling the source potential (default 1.0)
    Output:
        loss: Tensor (scalar)
    """
    x, ladj = F.inv.call_and_ladj(y) # x = F^-1(y), ladj = log|det J_{F^-1}(y)|
    return (beta * source(x) - ladj).mean()

# alias: forward KL divergence for data-driven normalizing flow
forward_KL = target_KL_F

def target_KL_G(y: torch.Tensor, source: Potential, G: ComposedTransform, beta: float = 1.0):
    """
    KL loss using target samples to train G (the target -> source map).
    Estimates  E_{y ~ target}[ beta * source(G(y)) - log|det J_G(y)| ],
    a Monte Carlo estimate of KL(G_# target || exp(-beta * source)) up to an additive const.
    Input:
        y:      Tensor [N, d]      samples drawn from the target distribution
        source: Potential          negative log-density of the source (up to const)
        G:      ComposedTransform  flow map target -> source (typically flow.t())
        beta:   float              inverse temperature scaling the source potential (default 1.0)
    Output:
        loss: Tensor (scalar)
    """
    x, ladj = G.call_and_ladj(y) # x = G(y), ladj = log|det J_G(y)|
    return (beta * source(x) - ladj).mean()


# ──────────────────────────────────────────────────────────────────────
# compile — torch.compile a loss with all its non-batch args captured
# ──────────────────────────────────────────────────────────────────────

def compile(
    loss_fn: Callable[..., torch.Tensor],
    *captured,
    mode: str = 'default',
) -> Callable[..., torch.Tensor]:
    """torch.compile any loss whose first argument is the per-batch tensor,
    last argument is the inverse temperature `beta`, and the rest are
    Python objects that should be baked into the closure.

    The four KL losses in this module all share the positional signature
    `(samples, potential, transform, beta) -> scalar`, so the canonical
    call is `compile(reverse_KL, potential, transform)` and the returned
    closure has signature `(x_batch, beta=1.0) -> scalar`. The helper is
    generic: any `loss_fn(x, *captured, beta) -> scalar` works.

    Why beta is a runtime argument:
        - Annealing / replica exchange / ESS-targeted schedules need to
          vary beta across training steps without rebuilding the loss.
        - To avoid a recompile per beta value (Dynamo specializes on
          Python `float`s by guarding on the value), beta is cast to a
          0-d tensor inside the wrapper before entering the compiled
          graph, so a single compiled artifact handles every beta.
        - Cost of the cast is ~1 microsecond per call — negligible
          against the multi-millisecond training step.
        - Passing beta directly as a 0-d tensor (e.g. one held on the
          device) skips the cast entirely.

    Why capture the rest as closure constants:
        - `torch.compile`'s Dynamo guards each cached specialization by
          input identity. The natural `loss_fn(x, target, flow.t())`
          pattern builds a fresh `ComposedTransform` Python object per
          call, so Dynamo either retraces every iteration or hits
          `BACKEND_MATCH` failures — either way the speedup vanishes.
        - Putting `target`, `transform`, etc. in the closure makes them
          invisible to the guard system; every call presents the same
          object identity and is a cache hit.

    Captured-once correctness for flow training:
        - `flow.t()` returns a `ComposedTransform` whose inner
          `AutoregressiveTransform` objects hold *lazy* `meta` callables
          that read flow parameters via attribute access on every
          forward pass.
        - `optimizer.step()` mutates those `nn.Parameter` tensors
          in-place, so the captured `transform` sees the post-step
          values without rebuilding.
        - Capturing the transform once is in fact a mild speedup even
          without torch.compile (saves the per-batch object construction).

    Caveats:
        - mode='default' is safe; mode='reduce-overhead' uses CUDA
          graphs (faster but needs static batch size and stable
          parameter memory; do not call `flow.zeros()` mid-training).
        - Build this AFTER any `potential.enable_grad()` /
          `enable_eval()` so the compiled graph wraps the fast paths.
        - Dynamo retraces on tensor-shape changes. Keep `BATCH` fixed.

    Arguments:
        loss_fn:   any callable with signature
                   `(x, *captured, beta) -> scalar`. The four KL losses
                   in this module all fit.
        *captured: the rest of `loss_fn`'s positional arguments BEFORE
                   beta, baked into the returned closure. For the KL
                   losses this is `(potential, transform)`.
        mode:      torch.compile mode (keyword-only). Default 'default'.

    Returns:
        callable (x_batch: Tensor, beta: float | Tensor = 1.0) -> scalar.

    Example (canonical KL case, fixed beta):
        F = flow.t()
        loss = zflows.loss.compile(reverse_KL, u1, F)
        for x_batch in batches:
            l = loss(x_batch)                       # beta = 1.0
            optimizer.zero_grad(); l.backward(); optimizer.step()

    Example (annealing schedule, no recompile):
        loss = zflows.loss.compile(reverse_KL, u1, F)
        for step, x_batch in enumerate(batches):
            beta = min(1.0, step / 1000)
            l = loss(x_batch, beta)
            optimizer.zero_grad(); l.backward(); optimizer.step()
    """
    @torch.compile(mode=mode)
    def compiled(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return loss_fn(x, *captured, beta)

    def wrapper(x: torch.Tensor, beta: float | torch.Tensor = 1.0) -> torch.Tensor:
        # Cast Python scalars to a 0-d tensor so Dynamo treats beta as a
        # dynamic input rather than specializing on its value. A 0-d
        # tensor passes through .item()-free arithmetic in the loss body.
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
        return compiled(x, beta)
    return wrapper