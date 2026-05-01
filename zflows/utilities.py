import torch

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
