"""zflows — PyTorch normalizing flows for unconditional energy-based sampling.

Strongly inspired by `zuko` (https://github.com/probabilists/zuko).

Public surface:

    zflows.potential : Potential, potential_from, Uniform, Gaussian, Gaussian_Mixture, Linear_Combination, linear_combination
    zflows.flow      : Flow, NSF, NCSF, CNF, OTFlow, RealNVP, ComposedTransform
    zflows.loss      : reverse_KL, forward_KL, OT_loss, loss_compile, loss_compile_beta
    zflows.utils     : compute_ESS, compute_ESS_log, compute_CESS, compute_CESS_log,
                       importance_weights{,_F,_G}, importance_weights_log{,_F,_G},
                       resample, langevin, rejuvenation, stochastic_heun, hamiltonian_monte_carlo, hmc, lbfgs, optimization,
                       annealed_importance_sampling{,_F,_G},
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
