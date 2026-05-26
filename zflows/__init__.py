"""zflows — PyTorch normalizing flows for unconditional energy-based sampling.

Strongly inspired by `zuko` (https://github.com/probabilists/zuko).

Public surface:

    zflows.potential : Potential, Uniform, Gaussian, Gaussian_Mixture, Linear_Combination
    zflows.flow      : Flow, NSF, NCSF, CNF, RealNVP, ComposedTransform
    zflows.loss      : reverse_KL, forward_KL, compile_raw, compile_beta
    zflows.utils     : compute_ESS, compute_ESS_log, compute_CESS, compute_CESS_log,
                       importance_weights, importance_weights_log,
                       resample, langevin, rejuvenation, hmc, lbfgs, optimization,
                       check_compile_available, set_cache_size_limit, suppress_warnings

`ComposedTransform` lives in `zflows.core.transforms` and is re-exported
from `zflows.flow` so downstream code stays implementation-agnostic.

Internals (`zflows.core.*`) follow zuko's layout, with two deliberate
divergences:

    1. there is no `context` / conditional-on-c plumbing anywhere;
    2. `MonotonicRQSTransform` / `CircularShiftTransform` accept a
       per-coordinate `bound` tensor so spline knots can natively span
       `[-bound_i, bound_i]` without an affine scaling sandwich.
"""

from . import flow, loss, potential, utils

__all__ = ["flow", "loss", "potential", "utils"]
