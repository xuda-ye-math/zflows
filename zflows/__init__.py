"""This package is a convenient wrapper of zuko for implementing normalizing flow.

Available names:
    zflows.potential : Potential, Uniform, Gaussian, Gaussian_Mixture, Linear_Combination
    zflows.flow      : Flow, NSF, NCSF, CNF, RealNVP, ComposedTransform
    zflows.loss      : reverse_KL, forward_KL
    zflows.utils     : compute_ESS, compute_ESS_log, compute_CESS, compute_CESS_log,
                       importance_weights, importance_weights_log,
                       resample, langevin, rejuvenation, lbfgs, optimization

`ComposedTransform` is re-exported from zuko so downstream code can stay
zuko-agnostic: write `from zflows.flow import ComposedTransform` rather
than reaching into `zuko.transforms`.
"""
