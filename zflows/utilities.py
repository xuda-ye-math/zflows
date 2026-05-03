# pyright: reportOperatorIssue=false

import torch
from .potential import Potential
from .flow import Flow

def compute_ESS(weights: torch.Tensor) -> torch.Tensor:
    """
    Compute the Effective Sample Size (ESS) of samples with given
    weights. The ESS lies in [0, 1] by Cauchy's inequality.
    Input:
        weights: Tensor [N]   (non-negative, not required to be normalized)
    Output:
        ESS: Tensor (scalar in [0, 1])
    """
    N = weights.shape[0]
    return weights.sum() ** 2 / (N * (weights ** 2).sum())

def compute_ESS_log(log_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute the Effective Sample Size (ESS) from log-weights, using
    logsumexp for numerical stability. The ESS lies in [0, 1] by
    Cauchy's inequality.

        log(ESS) = 2 * logsumexp(log_w) - log(N) - logsumexp(2 * log_w)

    Input:
        log_weights: Tensor [N]   unnormalized log-weights
    Output:
        ESS: Tensor (scalar in [0, 1])
    """
    N = log_weights.shape[0]
    log_num = 2 * torch.logsumexp(log_weights, dim=0)
    log_den = torch.logsumexp(2 * log_weights, dim=0) + torch.log(torch.tensor(N, dtype=log_weights.dtype, device=log_weights.device))
    return (log_num - log_den).exp()

def compute_CESS(source_weights: torch.Tensor, importance_weights: torch.Tensor):
    """
    Compute the Conditional Effective Sample Size (CESS) with given
    importance sampling weights applied on source distribution.
    The CESS lies in [0,1] by Cauchy's inequality.
    Input:
        source_weights:     Tensor [N]   (non-negative, not required to be normalized)
        importance_weights: Tensor [N]   (non-negative)
    Output:
        CESS: Tensor (scalar in [0, 1])
    """
    assert source_weights.shape == importance_weights.shape
    source_weights = source_weights / source_weights.sum()
    w1 = importance_weights * source_weights
    w2 = importance_weights * w1
    return w1.sum() ** 2 / w2.sum()

def compute_CESS_log(source_weights: torch.Tensor, log_importance_weights: torch.Tensor):
    """
    Compute the Conditional Effective Sample Size (CESS) where the
    importance weights are given in log-space (source_weights stays in
    linear space). Uses logsumexp for numerical stability.

        log(CESS) = 2 * logsumexp(log_s + log_iw) - logsumexp(log_s + 2 * log_iw)

    where log_s_i = log(source_weights_i / sum(source_weights)).

    Input:
        source_weights:         Tensor [N]   (non-negative, not required to be normalized)
        log_importance_weights: Tensor [N]   unnormalized log importance weights
    Output:
        CESS: Tensor (scalar in [0, 1])
    """
    assert source_weights.shape == log_importance_weights.shape
    log_s = source_weights.log() - torch.logsumexp(source_weights.log(), dim=0)
    log_w1 = log_s + log_importance_weights
    log_w2 = log_s + 2 * log_importance_weights
    return (2 * torch.logsumexp(log_w1, dim=0) - torch.logsumexp(log_w2, dim=0)).exp()

def importance_weights(samples: torch.Tensor, source: Potential, target: Potential, flow: Flow, chunk: int = 1) -> torch.Tensor:
    """
    Linear-space self-normalized importance weights for the proposal
    `nu = F_# source` against the target `mu_1 ~ exp(-target)`. Thin
    convenience wrapper around `importance_weights_log`: subtract the
    max log-weight for numerical stability, then exponentiate.

        w_i = exp(log_w_i - max_j log_w_j),   w in [0, 1].

    The omitted factor `exp(max log_w)` is a sample-dependent scalar
    that cancels in every *self-normalized* downstream use (ratios in
    compute_ESS, draws from compute_ESS_log / resample, MC averages of
    bounded test functions). Use this routine when the consumer expects
    plain non-negative weights (e.g. `resample(samples, weights)`); use
    `importance_weights_log` + `compute_ESS_log` / `compute_CESS_log`
    when log-space stability is required (very-low-overlap proposals,
    tail diagnostics).

    Input:
        samples: Tensor [N, d]   particles drawn from `source`
        source:  Potential       source (proposal-base) potential U_0
        target:  Potential       target potential U_1
        flow:    Flow            normalizing flow providing F = flow.t()
        chunk:   int             split `samples` along dim 0 into this many
                                 chunks and accumulate. Reduces peak GPU
                                 memory at the cost of wall time;
                                 statistically and numerically equivalent
                                 to chunk=1 (the per-sample log-weight
                                 only depends on its own (x, F(x))).
    Output:
        w: Tensor [N]   unnormalized importance weights in [0, 1].
    """
    log_w = importance_weights_log(samples, source, target, flow, chunk=chunk)
    return (log_w - log_w.max()).exp()

def importance_weights_log(samples: torch.Tensor, source: Potential, target: Potential, flow: Flow, chunk: int = 1) -> torch.Tensor:
    """
    Self-normalized importance-sampling log-weights for the proposal
    `nu = F_# source` against the target `mu_1 ~ exp(-target)`, where
    `F = flow.t()` is the trained bijection that pushes source samples
    toward the target.

    For x ~ source, y = F(x), the proposal density is
        log nu(y) = log source(x) - log|det J_F(x)|.
    The unnormalized log-importance-weight is therefore
        log w(y) = log mu_1(y) - log nu(y)
                 = -target(y) + source(x) + log|det J_F(x)|,
    using `Potential` energies U = -log mu (up to additive constants
    that cancel after self-normalization).

    Use the regular forward call (`source(x)`, `target(y)`); no
    `enable_eval()` opt-in is needed here since this routine is not on
    the per-iter MALA hot path.

    Input:
        samples: Tensor [N, d]   particles drawn from `source`
        source:  Potential       source (proposal-base) potential U_0
        target:  Potential       target potential U_1
        flow:    Flow            normalizing flow providing F = flow.t()
        chunk:   int             split `samples` along dim 0 into this many
                                 chunks and concatenate the per-chunk
                                 log-weights. Reduces peak GPU memory at
                                 the cost of wall time; statistically and
                                 numerically equivalent to chunk=1 (each
                                 sample's log-weight depends only on its
                                 own (x, F(x))).
    Output:
        log_w: Tensor [N]   unnormalized log importance weights, ready
                            to feed into compute_ESS_log / compute_CESS_log
                            or to exponentiate (after subtracting max).
    """
    out = []
    for x in torch.chunk(samples, chunk, dim=0):
        y, ladj = flow.t().call_and_ladj(x)
        out.append(-target(y) + source(x) + ladj)
    return torch.cat(out, dim=0)

def resample(samples: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Multinomial resampling from weighted distribution with replacement
    Input:
        samples: Tensor [N, d]
        weights: Tensor [N]   (non-negative, not required to be normalized)
    Output:
        resampled: Tensor [N, d]
    """
    N = samples.shape[0]
    probs = weights / weights.sum()
    idx = torch.multinomial(probs, N, replacement=True)
    return samples[idx]

def langevin(samples: torch.Tensor, potential: Potential, step: float = 1e-3, iters: int = 100, adjust: bool = False, chunk: int = 1) -> torch.Tensor:
    """
    Langevin dynamics targeting the distribution exp(-U(x)).

    Proposal (Euler-Maruyama on the overdamped Langevin SDE):
        y = x - step * grad U(x) + sqrt(2 * step) * xi,   xi ~ N(0, I_d).

    With adjust=False (default), every proposal is accepted; this is the
    unadjusted Langevin algorithm (ULA), which has an O(step) bias but
    needs only one gradient call per iteration. With adjust=True, each
    proposal is accepted via Metropolis-Hastings, giving the standard MALA
    scheme whose stationary distribution is *exactly* exp(-U) (unbiased)
    at the cost of ~2x runtime (two gradient calls per iteration).

    The MH acceptance probability is min(1, exp(log_alpha)) with
        log_alpha = -U(y) + U(x) + log q(x|y) - log q(y|x),
    where the proposal density is Gaussian:
        log q(z|w) = -||z - w + step * grad U(w)||^2 / (4 * step) + const.
    Both the energy difference *and* the asymmetric-proposal correction are
    needed; using only the energy term leaves a residual O(step) bias.

    Requires `potential.enable_grad()` to have been called so that
    `potential.grad(x)` is available; otherwise raises RuntimeError.
    For the MALA branch (`adjust=True`), if `potential.enable_eval()` has
    also been called, the U(y) / U(x) energy evaluations route through
    the compiled `potential.eval(x)` fast path; otherwise they fall back
    to the regular `potential(x)` call.

    Input:
        samples:   Tensor [N, d]   initial particles
        potential: Potential       target potential U; must support .grad(x)
        step:      float           Euler-Maruyama step size
        iters:     int             number of Langevin steps
        adjust:    bool            if True, run MALA (unbiased); if False, run ULA
        chunk:     int             split `samples` along dim 0 into this many
                                   chunks and run the trajectories sequentially.
                                   Reduces peak GPU memory at the cost of wall
                                   time. Statistically equivalent to chunk=1
                                   (each chunk uses its own independent noise);
                                   set higher only if you hit OOM on the
                                   whole batch.
    Output:
        samples: Tensor [N, d]   particles after `iters` Langevin updates
    """
    if potential._grad_fn is None:
        raise RuntimeError(
            f"langevin() requires gradients on the potential; "
            f"call {type(potential).__name__}.enable_grad() before passing it in."
        )
    # MALA accept/reject needs U(x), U(y); use the compiled fast path if
    # the user has opted in via .enable_eval(), else fall back to __call__.
    U = potential.eval if potential._eval_fn is not None else potential
    noise_scale = (2.0 * step) ** 0.5
    out = []
    for x in torch.chunk(samples, chunk, dim=0):
        for _ in range(iters):
            gx = potential.grad(x)
            y = x - step * gx + noise_scale * torch.randn_like(x)
            if adjust:
                # log q(z|w) = -||z - w + step * grad U(w)||^2 / (4 * step) + const
                # Consume gx (-> log_q_yx) BEFORE calling potential.grad(y)
                log_q_yx = -((y - x + step * gx) ** 2).sum(dim=-1) / (4.0 * step) # log q(y|x)
                gy = potential.grad(y)
                log_q_xy = -((x - y + step * gy) ** 2).sum(dim=-1) / (4.0 * step) # log q(x|y)
                log_alpha = -U(y) + U(x) + log_q_xy - log_q_yx # [N]
                accept = torch.rand_like(log_alpha).log() < log_alpha # [N] bool
                x = torch.where(accept.unsqueeze(-1), y, x)
            else:
                x = y
        out.append(x)
    return torch.cat(out, dim=0)

# alias: in SMC literature, Langevin steps are the standard "rejuvenation" move
rejuvenation = langevin
