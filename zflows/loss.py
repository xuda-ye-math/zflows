# pyright: reportOperatorIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false

import torch
from .potential import Potential
from .flow import Flow

# reverse KL divergence for energy-based normalizing flow
def reverse_KL(x: torch.Tensor, target: Potential, flow: Flow):
    """
    The reverse KL divergence in energy-based normalizing flow.
    Estimates  E_{x ~ source}[ target(F(x)) - log|det J_F(x)| ],
    where F = flow.t() pushes source samples toward the target.
    Input:
        x:      Tensor [N, d]   samples drawn from the source distribution
        source: Potential       negative log-density of the source (up to const)
        target: Potential       negative log-density of the target (up to const)
        flow:   Flow            normalizing flow providing F = flow.t()
    Output:
        loss: Tensor (scalar)   Monte Carlo estimate of the reverse KL
    """
    y, ladj = flow.t().call_and_ladj(x) # get y = F(x) and log_abs_det_jacobian
    return (target(y) - ladj).mean()

# forward KL divergence for data-driven normalizing flow
def forward_KL(y: torch.Tensor, source: Potential, flow: Flow):
    """
    The forward KL divergence in data-driven normalizing flow.
    Estimates  E_{y ~ target}[ source(F^-1(y)) - log|det J_{F^-1}(y)| ],
    where F = flow.t() pushes source samples toward the target.
    Input:
        y:      Tensor [N, d]   samples drawn from the target distribution
        source: Potential       negative log-density of the source (up to const)
        flow:   Flow            normalizing flow providing F = flow.t()
    Output:
        loss: Tensor (scalar)   Monte Carlo estimate of the forward KL
    """
    x, ladj = flow.t().inv.call_and_ladj(y) # x = F^-1(y), ladj = log|det J_{F^-1}(y)|
    return (source(x) - ladj).mean()