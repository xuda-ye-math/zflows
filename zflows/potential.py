# pyright: reportOperatorIssue=false, reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false, reportReturnType=false

import torch
from torch import nn

class Potential(nn.Module):
    """
    Generic Potential class. forward() computes the potential function.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: Tensor [N, d]
        Output:
            _: Tensor [N]
        """
        raise NotImplementedError

class Uniform(Potential):
    """
    Uniform distribution with constant potential.
    """
    def __init__(
        self,
        a: torch.Tensor | list[float],
        b: torch.Tensor | list[float],
        device: torch.device | str = "cpu",
    ):
        """
        Input:
            a:      Tensor [d] or list[float]   lower bounds of the rectangle
            b:      Tensor [d] or list[float]   upper bounds of the rectangle
            device: torch.device | str          device on which buffers live
        """
        super().__init__()
        a = torch.as_tensor(a, dtype=torch.float32, device=device)
        b = torch.as_tensor(b, dtype=torch.float32, device=device)
        assert a.shape == b.shape
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.d = a.shape[0]

    @property
    def device(self) -> torch.device:
        return self.a.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: Tensor [N, d]
        Output:
            _: Tensor [N]
        """
        return x.new_zeros(x.shape[0])

    def samples(self, N: int) -> torch.Tensor:
        """
        Generate N independent samples in the rectangle region [a, b]
        Output:
            x: Tensor [N, d]
        """
        u = torch.rand(N, self.d, device=self.device)
        return self.a + (self.b - self.a) * u

class Gaussian(Potential):
    """
    Diagonal Gaussian distribution. The potential is the negative
    log density (up to an additive constant):
        U(x) = 0.5 * sum_i (x_i - mean_i)^2 / variance_i
    """
    def __init__(
        self,
        mean: torch.Tensor | list[float],
        variance: torch.Tensor | list[float],
        device: torch.device | str = "cpu",
    ):
        """
        Input:
            mean:     Tensor [d] or list[float]   per-coordinate mean
            variance: Tensor [d] or list[float]   per-coordinate variance (positive)
            device:   torch.device | str         device on which buffers live
        """
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        variance = torch.as_tensor(variance, dtype=torch.float32, device=device)
        assert mean.shape == variance.shape
        self.register_buffer("mean", mean)
        self.register_buffer("variance", variance)
        self.d = mean.shape[0]

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: Tensor [N, d]
        Output:
            _: Tensor [N]
        """
        return 0.5 * ((x - self.mean) ** 2 / self.variance).sum(dim=-1)

    def samples(self, N: int) -> torch.Tensor:
        """
        Generate N independent samples from the diagonal Gaussian
        Output:
            x: Tensor [N, d]
        """
        z = torch.randn(N, self.d, device=self.device)
        return self.mean + self.variance.sqrt() * z
