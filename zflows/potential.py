# pyright: reportOperatorIssue=false, reportArgumentType=false, reportIndexIssue=false, reportCallIssue=false, reportReturnType=false, reportIncompatibleMethodOverride=false

import torch
from torch import nn


# ──────────────────────────────────────────────────────────────────────
# Potential — abstract base + compiled grad/eval fast paths
# ──────────────────────────────────────────────────────────────────────

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

    @classmethod
    def _from(cls, fn) -> "Potential":
        """Wrap a stateless callable `(x: Tensor) -> Tensor` as a Potential.

        Convenience for one-line definitions that would otherwise require
        a full subclass with `def __init__(self): super().__init__()` and
        `def forward(self, x): ...`. The wrapped Potential supports the
        full toolchain (`.to(device)`, `.enable_grad()`, `.enable_eval()`,
        `.parameters()`) — there just won't be any learnable parameters
        because `fn` is a plain function.

        (Method name is `_from`, not `from`, because `from` is a Python
        keyword and can't be a method name.)

        Example:
            def U1(x):
                return 0.5 * (x ** 2).sum(-1) + 2 * torch.cos(x[:, 0])

            U1 = Potential._from(U1)
            U1.enable_grad()
            g = U1.grad(x)        # [N, d]

        For potentials that carry state (physical constants, learnable
        sub-modules, …), subclass `Potential` directly instead.
        """
        class _FunctionPotential(cls):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return fn(x)
        return _FunctionPotential()

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


# ──────────────────────────────────────────────────────────────────────
# Concrete potentials — Uniform, Gaussian, Gaussian_Mixture
# ──────────────────────────────────────────────────────────────────────

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

    def samples(self, N: int, beta: float = 1.0) -> torch.Tensor:
        """
        Generate N independent samples from the tempered diagonal
        Gaussian mu_beta ~ exp(-beta * U(x)). Since
            U(x) = 0.5 * sum_i (x_i - mean_i)^2 / variance_i,
        the beta-tempered distribution is N(mean, variance / beta),
        i.e. the same mean with covariance scaled by 1/beta. Default
        beta=1.0 reproduces the original sampler exactly.
        Input:
            N:    int     number of samples
            beta: float   inverse temperature (default 1.0)
        Output:
            x: Tensor [N, d]
        """
        z = torch.randn(N, self.d, device=self.device)
        return self.mean + (self.variance / beta).sqrt() * z
    
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


# ──────────────────────────────────────────────────────────────────────
# Compositional — Linear_Combination of potentials (annealing bridges)
# ──────────────────────────────────────────────────────────────────────

class Linear_Combination(Potential):
    """
    Linear combination of N potentials:
        U(x) = sum_k c_k * U_k(x).
    Useful for Boltzmann interpolations U_t = (1 - t) * U_0 + t * U_1 (the
    common N = 2 case), but generalizes naturally to multi-rung bridges
    and convex mixtures of an arbitrary number of building-block energies.

    The child potentials are stored as an `nn.ModuleList`, so
    `.to(device)`, `.parameters()`, and `.state_dict()` recurse through
    them. Coefficients are stored on `self.coeffs` in one of two forms:

      - **Python list of floats** (the simplest case). Mutate
        `self.coeffs[k] = new_value` between iterations to retune one
        term. Immune to `.to(device)` — `float * Tensor` lifts to the
        tensor's device automatically.
      - **`torch.Tensor` of shape `[N]`**. Registered as an `nn.Module`
        buffer so `.to(device)` / `.cuda()` / `.float()` move it along
        with the potentials — that's the redundancy that keeps a stray
        `.to('cuda')` from leaving the coeffs on CPU and tripping a
        device-mismatch error on the next forward. Mutate in place via
        `self.coeffs[k] = new_value` or `self.coeffs.fill_(...)` —
        modern PyTorch refuses any reassignment `self.coeffs = ...`
        with a clear `TypeError`, so the buffer registration is safe.

    A device-resident tensor also avoids the Dynamo Python-float
    specialization that would otherwise re-trace the compiled graph on
    every coefficient change — handy for annealed schedules sharing a
    single compiled forward.
    """
    def __init__(
        self,
        potentials: list["Potential"] | tuple["Potential", ...],
        coeffs: list[float] | tuple[float, ...] | torch.Tensor,
    ):
        """
        Input:
            potentials: list/tuple of N Potential instances (N >= 1)
            coeffs:     list/tuple of N floats, or a 1-d Tensor of shape
                        [N], holding the matching coefficients.

        Both inputs must be non-empty and have the same length.
        """
        super().__init__()
        assert len(potentials) == len(coeffs), \
            f"potentials ({len(potentials)}) and coeffs ({len(coeffs)}) must have the same length"
        assert len(potentials) >= 1, "Linear_Combination needs at least one term"
        self.potentials = nn.ModuleList(potentials)
        if isinstance(coeffs, torch.Tensor):
            assert coeffs.ndim == 1, \
                f"Tensor coeffs must be 1-d, got shape {tuple(coeffs.shape)}"
            # Coeffs are mixture / interpolation weights, not learnable
            # parameters. Registering a requires_grad=True tensor as a
            # buffer would silently hide it from `optimizer.parameters()`
            # while still accumulating `.grad` — a confusing footgun.
            # Reject it explicitly; users who want a trainable mixture
            # should subclass Linear_Combination and register a real
            # nn.Parameter themselves.
            assert not coeffs.requires_grad, (
                "Linear_Combination coeffs are stored as a buffer and must not "
                "require gradients; pass `coeffs.detach()` or build the tensor "
                "without `requires_grad=True`."
            )
            # Register as a buffer so .to(device) / .cuda() / .float() etc.
            # move the coeffs in lock-step with the potentials' parameters
            # and buffers. Without this, a tensor stored as a plain attribute
            # would stay on its original device while the potentials move.
            self.register_buffer("coeffs", coeffs)
        else:
            self.coeffs = list(coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: Tensor [N, d]
        Output:
            U(x): Tensor [N]
        """
        out = self.coeffs[0] * self.potentials[0](x)
        for c, U in zip(self.coeffs[1:], self.potentials[1:]):
            out = out + c * U(x)
        return out
