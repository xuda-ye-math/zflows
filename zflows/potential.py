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
# Functional wrappers — turn a plain callable into a Potential subclass
# ──────────────────────────────────────────────────────────────────────

def potential_from(fn) -> Potential:
    """Wrap a stateless callable `(x: Tensor) -> Tensor` as a ready-to-use
    `Potential` *instance* — like writing the subclass by hand and
    instantiating it, in one line.

    Returns the instance directly (lowercase-factory convention; the
    name is lowercase because the return is an instance, not a class).
    The instance supports the full toolchain (`.to(device)`,
    `.enable_grad()`, `.enable_eval()`, `.parameters()`) — there just
    won't be any learnable parameters because `fn` is a plain function.

    Example:
        def myforward(x: torch.Tensor) -> torch.Tensor:
            # x: [N, d] -> [N]   (batched for efficiency)
            return 0.5 * (x ** 2).sum(-1) + 2 * torch.cos(x[:, 0])

        u = potential_from(myforward).to(device)   # instance, chainable

    For potentials that carry state (physical constants, learnable
    sub-modules, …), subclass `Potential` directly instead.
    """
    class _FunctionPotential(Potential):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return fn(x)
    return _FunctionPotential()


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
    them. Coefficients are stored on `self.coeffs` as a plain Python
    `list[float]` regardless of how they were passed in (list, tuple,
    1-d Tensor, or `None`). Mutate `self.coeffs[k] = new_value` between
    iterations to retune one term. Immune to `.to(device)` —
    `float * Tensor` lifts to the tensor's device automatically.

    The combined `_grad_fn` / `_eval_fn` are populated **at __init__
    time** as Python closures that compute `sum_k self.coeffs[k] *
    U_k.grad(x)` (and same for `.eval(x)`). They read `self.coeffs[k]`
    fresh on every call, so coefficient changes via `set_coeffs` are
    picked up with no recompile.

    `enable_grad` / `enable_eval` are **overridden** to only propagate
    the compile request to each child potential (skipping children
    whose own `_grad_fn` / `_eval_fn` is already non-None). They do NOT
    touch `self._grad_fn` / `self._eval_fn` — those were linked in
    `__init__` and stay valid for the lifetime of the instance. The
    "needs `.enable_grad()` first" gate therefore moves from a
    pre-flight check on `Linear_Combination` (its `_grad_fn` is always
    non-None) to a *runtime* check inside the closure: an un-enabled
    child raises `RuntimeError` the first time the closure calls
    `U.grad(x)` on it.

    CUDA-graph buffer-aliasing safety: with `mode='reduce-overhead'`,
    each child's `.grad(x)` returns a static buffer that the next
    compiled call overwrites. The Python expression `c * U.grad(x)` is
    `(Python float) * Tensor`, which allocates a fresh tensor on every
    call, so the result is decoupled from the static buffer before the
    next child's `.grad(x)` clobbers it. Same convention as the `fx =
    beta * potential.grad(x)` pattern in `langevin` / `hmc` — no
    `.clone()` is needed.

    If you prefer plain Python callables over `Potential` objects, you can
    skip this class entirely: define `lambda x: sum(c_k * U_k(x) for ...)`
    yourself and wrap it via `potential_from(fn)` (returns an instance).
    """
    def __init__(
        self,
        potentials: list["Potential"] | tuple["Potential", ...],
        coeffs: list[float] | tuple[float, ...] | torch.Tensor | None = None,
    ):
        """
        Input:
            potentials: list/tuple of N Potential instances (N >= 1)
            coeffs:     list/tuple of N floats, or a 1-d Tensor of shape
                        [N], holding the matching coefficients. If None
                        (the default), defaults to a uniform 1/N on each
                        potential, i.e. the plain average
                        U(x) = (1/N) * sum_k U_k(x).
                        Tensor inputs are detached and converted to a
                        plain `list[float]` before storage.

        `potentials` must be non-empty; when `coeffs` is provided
        explicitly, its length must match.
        """
        super().__init__()
        assert len(potentials) >= 1, "Linear_Combination needs at least one term"
        if coeffs is None:
            coeffs = [1.0 / len(potentials)] * len(potentials)
        elif isinstance(coeffs, torch.Tensor):
            assert coeffs.ndim == 1, \
                f"Tensor coeffs must be 1-d, got shape {tuple(coeffs.shape)}"
            coeffs = coeffs.detach().cpu().tolist()
        assert len(potentials) == len(coeffs), \
            f"potentials ({len(potentials)}) and coeffs ({len(coeffs)}) must have the same length"

        # Device-consistency check: scan every parameter and buffer
        # across all children and require a single device. Potentials
        # with no parameters or buffers (e.g. `potential_from(fn)`,
        # `Uniform`'s pure-Python `forward`) are device-agnostic and
        # contribute nothing to the set, so they're silently allowed.
        devices = set()
        for p in potentials:
            for t in p.parameters():
                devices.add(t.device)
            for t in p.buffers():
                devices.add(t.device)
        if len(devices) > 1:
            raise RuntimeError(
                f"Linear_Combination: child potentials live on different "
                f"devices: {sorted(map(str, devices))}. Call `.to(device)` "
                f"on each potential so they all sit on the same device "
                f"before composing them."
            )

        self.potentials = nn.ModuleList(potentials)
        self.coeffs = [float(c) for c in coeffs]

        # Link the combined fast paths NOW, at __init__ time, as Python
        # closures over `self`. They read `self.coeffs[k]` and route to
        # `self.potentials[k].grad(x)` (resp. `.eval(x)`) on every call,
        # so set_coeffs updates propagate without invalidating anything.
        # The "needs .enable_grad() first" gate becomes a runtime check
        # at the child level: an un-enabled child raises RuntimeError
        # the first time the closure invokes its `.grad(x)` / `.eval(x)`.
        def _grad_combined(x):
            g = self.coeffs[0] * self.potentials[0].grad(x)
            for c, U in zip(self.coeffs[1:], self.potentials[1:]):
                g = g + c * U.grad(x)
            return g

        def _eval_combined(x):
            v = self.coeffs[0] * self.potentials[0].eval(x)
            for c, U in zip(self.coeffs[1:], self.potentials[1:]):
                v = v + c * U.eval(x)
            return v

        self._grad_fn = _grad_combined
        self._eval_fn = _eval_combined

    def enable_grad(self, mode: str = "reduce-overhead") -> "Linear_Combination":
        """Override: propagate `enable_grad(mode)` to every child
        unconditionally. The child's own `Potential.enable_grad` is
        idempotent (early-returns if `_grad_fn` is non-None), so calling
        it on an already-hot child is a safe no-op. Does NOT touch
        `self._grad_fn` (linked in `__init__` to the combined closure).

        Returns self so the call can be chained.
        """
        for U in self.potentials:
            U.enable_grad(mode)
        return self

    def enable_eval(self, mode: str = "reduce-overhead") -> "Linear_Combination":
        """Override: propagate `enable_eval(mode)` to every child
        unconditionally. The child's own `Potential.enable_eval` is
        idempotent, so calling it on an already-hot child is a safe
        no-op. Does NOT touch `self._eval_fn` (linked in `__init__` to
        the combined closure).

        Returns self so the call can be chained.
        """
        for U in self.potentials:
            U.enable_eval(mode)
        return self

    def set_coeffs(
        self,
        coeffs: list[float] | tuple[float, ...] | torch.Tensor,
    ) -> "Linear_Combination":
        """Replace `self.coeffs` in place with a new set of weights.

        Pure coefficient update: the combined `_grad_fn` / `_eval_fn`
        closures linked at `__init__` read `self.coeffs[k]` fresh on
        every call, so the new coefficients take effect on the next
        `.grad(x)` / `.eval(x)` call automatically — no recompile, no
        invalidation, and no interaction with the child compiled
        artifacts. If you also need to enable the children's compiled
        fast paths, call `.enable_grad()` / `.enable_eval()` explicitly
        (typically once, at construction time).

        Useful for annealed bridges or schedule updates that keep the
        same `Linear_Combination` instance across rungs — just retune
        the mix and continue calling `.grad(x)` / `.eval(x)`.

        Input:
            coeffs: list/tuple of N floats, or a 1-d Tensor of shape [N];
                    None is not accepted (call the constructor or assign
                    `self.coeffs = [1.0 / N] * N` if you want uniform).
                    Length must match the number of potentials.
        Returns:
            self, so the call can be chained.
        """
        assert coeffs is not None, "set_coeffs(): coeffs must not be None"
        if isinstance(coeffs, torch.Tensor):
            assert coeffs.ndim == 1, \
                f"Tensor coeffs must be 1-d, got shape {tuple(coeffs.shape)}"
            coeffs = coeffs.detach().cpu().tolist()
        assert len(coeffs) == len(self.potentials), (
            f"set_coeffs(): expected {len(self.potentials)} coeffs to match "
            f"the stored potentials, got {len(coeffs)}"
        )
        self.coeffs = [float(c) for c in coeffs]
        return self

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


def linear_combination(
    potentials: list["Potential"] | tuple["Potential", ...],
    coeffs: list[float] | tuple[float, ...] | torch.Tensor | None = None,
) -> "Linear_Combination":
    """Factory: build and return a `Linear_Combination` instance.

    Lowercase alias for `Linear_Combination(potentials, coeffs)` — the
    project-wide convention is uppercase = class, lowercase = instance,
    and this factory returns *only* an instance. Use this in user code.
    Reach for `Linear_Combination` directly only when you need the class
    itself (e.g. `isinstance(u, Linear_Combination)` checks or to
    subclass).
    """
    return Linear_Combination(potentials, coeffs)


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
