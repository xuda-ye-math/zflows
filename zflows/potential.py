# pyright: reportOperatorIssue=false, reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false, reportReturnType=false, reportIncompatibleMethodOverride=false

import torch
from torch import nn

class Potential(nn.Module):
    """
    Generic Potential class. forward() computes the potential function.

    Two opt-in fast paths are exposed, both built once via torch.compile
    and cached on the instance:

      .enable_grad() -> .grad(x)    fast batched dU/dx via vmap(grad(.))
      .enable_eval() -> .eval(x)    fast batched U(x) via compile(forward)

    Calling .grad(x) before .enable_grad(), or .eval(x) before
    .enable_eval(), raises RuntimeError. The .eval() entry point preserves
    the standard nn.Module eval-mode switch when called with no argument:

        u = U1().to(device).enable_grad().enable_eval()
        g = u.grad(x)   # [N, d], no requires_grad on x
        v = u.eval(x)   # [N], faster than u(x) in MALA accept/reject loops
        u.eval()        # nn.Module: switch to eval mode (no x)

    The .eval(x) path is intended for inference-time hot loops (MALA
    accept/reject, importance sampling) where a torch.compile-fused U(x)
    avoids per-call autograd-graph construction. Do NOT call .eval(x) on
    a Potential whose value will be backpropagated through during
    training -- compile mode "reduce-overhead" captures static-shape
    CUDA graphs that are not differentiable in the normal sense.
    """
    _grad_fn = None # populated by enable_grad()
    _eval_fn = None # populated by enable_eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: Tensor [N, d]
        Output:
            _: Tensor [N]
        """
        raise NotImplementedError

    def enable_grad(self, mode: str = "reduce-overhead") -> "Potential":
        """
        Compile a fast .grad(x) using torch.func.grad + torch.compile, vmapped
        over the batch dim. Returns self so the call can be chained, e.g.
            u = Gaussian(...).to(device).enable_grad()
        Idempotent: calling twice does not recompile.

        Argument:
            mode: passed through to torch.compile. The default
                "reduce-overhead" captures a CUDA graph on the first .grad(x)
                call, giving the fastest steady-state throughput for fixed-
                shape inputs (e.g. uniform-batch Langevin loops), at the cost
                of a few MB of static GPU buffers per captured shape. Advanced
                users may prefer:
                  - "default":     no CUDA graph; lower VRAM, ~10-30% slower.
                                   Use when batch shape varies between calls
                                   or when GPU memory is tight.
                  - "max-autotune": longer first-call compilation in exchange
                                   for additional kernel-level autotuning.
        """
        if self._grad_fn is not None:
            return self
        single = lambda x: self.forward(x.unsqueeze(0)).squeeze(0) # [d] -> scalar
        self._grad_fn = torch.compile(
            torch.func.vmap(torch.func.grad(single)),
            mode=mode,
        )
        return self

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: Tensor [N, d]
        Output:
            grad U(x): Tensor [N, d]
        Raises RuntimeError if .enable_grad() has not been called.
        """
        if self._grad_fn is None:
            raise RuntimeError(
                f"{type(self).__name__}.grad() requires .enable_grad() first."
            )
        return self._grad_fn(x)

    def enable_eval(self, mode: str = "reduce-overhead") -> "Potential":
        """
        Compile a fast .eval(x) path via torch.compile of self.forward,
        intended for hot inference loops (e.g. MALA accept/reject and
        importance-sampling reweighting). Returns self so the call can
        be chained, e.g.
            u = Gaussian(...).to(device).enable_eval()
        Idempotent: calling twice does not recompile.

        Argument:
            mode: passed through to torch.compile. The default
                "reduce-overhead" captures a CUDA graph on the first
                .eval(x) call, giving the fastest steady-state throughput
                for fixed-shape inputs (uniform-batch MALA loops), at the
                cost of a few MB of static GPU buffers per captured shape.
                See .enable_grad for the "default" / "max-autotune"
                alternatives -- same semantics here.

        Note: this is a forward-only fast path; do not use the result of
        .eval(x) inside a training loss that you back-propagate through.
        Use the regular u(x) call for that.
        """
        if self._eval_fn is not None:
            return self
        self._eval_fn = torch.compile(self.forward, mode=mode)
        return self

    def eval(self, x: torch.Tensor | None = None):
        """
        Dual-purpose, dispatched on the argument:

          - .eval()      no argument -> standard nn.Module behaviour:
                         switch to eval mode, return self.
          - .eval(x)     evaluate U(x) via the compiled fast path. Raises
                         RuntimeError if .enable_eval() has not been called.

        Input (when x is provided):
            x: Tensor [N, d]
        Output (when x is provided):
            U(x): Tensor [N]
        """
        if x is None:
            return super().eval()
        if self._eval_fn is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval(x) requires .enable_eval() first."
            )
        return self._eval_fn(x)

    def release(self) -> None:
        """
        Drop this instance's compiled .grad / .eval closures, submodules,
        parameters, and buffers, then return cached GPU blocks to the CUDA
        driver via torch.cuda.empty_cache().

        This is *redundant* in normal use: when a Potential goes out of scope
        Python's refcount + garbage collector reclaim its memory automatically.
        Use .release() ONLY when out-of-memory (OOM) becomes a concrete
        problem, e.g. swapping many large Potentials in a long-running
        process where you've confirmed via nvidia-smi that VRAM is not being
        returned fast enough.

        Scope:
          - Instance-scoped: only this Potential's state is cleared. Child
            Potentials referenced elsewhere (e.g. the U0 / U1 of a
            Linear_Combination still bound to local variables) survive and
            keep their own ._grad_fn / ._eval_fn.
          - Process-global side effect: torch.cuda.empty_cache() flushes
            free blocks across all CUDA devices for the whole process. It
            does not invalidate any other instance's compiled artifacts; the
            global Inductor / torch.compile kernel cache is left intact.
        """
        import gc
        self._grad_fn = None
        self._eval_fn = None
        self._modules.clear()
        self._parameters.clear()
        self._buffers.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    
class Gaussian_Mixture(Potential):
    """
    Diagonal Gaussian mixture distribution with K components. The
    unnormalized density is
        mu(x) propto sum_k w_k * N(x | mean_k, diag(variance_k)),
    and the potential U(x) = -log mu(x) (up to an additive constant).
    """
    def __init__(
        self,
        weights: torch.Tensor | list[float],
        mean: torch.Tensor | list[list[float]],
        variance: torch.Tensor | list[list[float]],
        device: torch.device | str = "cpu",
    ):
        """
        Input:
            weights:  Tensor [K] or list[float]              mixture weights (non-negative, not required to be normalized)
            mean:     Tensor [K, d] or list[list[float]]     per-component, per-coordinate mean
            variance: Tensor [K, d] or list[list[float]]     per-component, per-coordinate variance (positive)
            device:   torch.device | str                    device on which buffers live
        """
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        variance = torch.as_tensor(variance, dtype=torch.float32, device=device)
        assert weights.ndim == 1 and mean.ndim == 2 and variance.ndim == 2
        assert mean.shape == variance.shape
        assert weights.shape[0] == mean.shape[0]
        log_weights = weights.log() - torch.logsumexp(weights.log(), dim=0) # normalized log-weights
        self.register_buffer("log_weights", log_weights)
        self.register_buffer("mean", mean)
        self.register_buffer("variance", variance)
        self.K = mean.shape[0]
        self.d = mean.shape[1]

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: Tensor [N, d]
        Output:
            U(x): Tensor [N]
        """
        diff = x.unsqueeze(1) - self.mean.unsqueeze(0) # [N, K, d]
        log_comp = -0.5 * (diff ** 2 / self.variance.unsqueeze(0)).sum(dim=-1) \
                   - 0.5 * self.variance.log().sum(dim=-1).unsqueeze(0) # [N, K]
        return -torch.logsumexp(self.log_weights.unsqueeze(0) + log_comp, dim=-1) # [N]

    def samples(self, N: int) -> torch.Tensor:
        """
        Generate N independent samples from the diagonal Gaussian mixture
        Output:
            x: Tensor [N, d]
        """
        idx = torch.multinomial(self.log_weights.exp(), N, replacement=True) # [N]
        z = torch.randn(N, self.d, device=self.device)
        return self.mean[idx] + self.variance[idx].sqrt() * z

class Linear_Combination(Potential):
    """
    Linear combination of two potentials:
        U(x) = c0 * U0(x) + c1 * U1(x).
    Useful for Boltzmann interpolations U_t = (1 - t) * U0 + t * U1, where
    c0, c1 are typically swapped between integrator steps.

    The two child potentials are stored by reference (auto-registered as
    submodules), so .to(device), .parameters(), and .state_dict() recurse
    through them. Mutating self.c0 / self.c1 in place takes effect on the
    next forward call.
    """
    def __init__(
        self,
        U0: "Potential",
        U1: "Potential",
        c0: float,
        c1: float | None = None,
    ):
        """
        Input:
            U0: Potential       first potential U0
            U1: Potential       second potential U1
            c0: float           coefficient of U0
            c1: float | None    coefficient of U1; if None, defaults to 1 - c0,
                                giving the convex interpolation
                                U = c0 * U0 + (1 - c0) * U1.
        """
        super().__init__()
        self.U0 = U0
        self.U1 = U1
        self.c0 = c0
        self.c1 = 1.0 - c0 if c1 is None else c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: Tensor [N, d]
        Output:
            U(x): Tensor [N]
        """
        return self.c0 * self.U0(x) + self.c1 * self.U1(x)
