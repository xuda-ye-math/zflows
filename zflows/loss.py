# pyright: reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false

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