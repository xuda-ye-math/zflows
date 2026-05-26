# pyright: reportOperatorIssue=false

import torch
from .flow import ComposedTransform
from .potential import Potential


# ──────────────────────────────────────────────────────────────────────
# ESS / CESS — effective sample size diagnostics
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Importance weights — log/linear-space SMC reweighting
# ──────────────────────────────────────────────────────────────────────

def importance_weights(samples: torch.Tensor, source: Potential, target: Potential, F: ComposedTransform, beta_source: float = 1.0, beta_target: float = 1.0, chunk: int = 1) -> torch.Tensor:
    """
    Linear-space self-normalized importance weights for the proposal
    `nu = F_# mu_0` against the target `mu_1`, where the two ends are
    the tempered Gibbs distributions
        mu_0(x) ~ exp(-beta_source * source(x)),
        mu_1(y) ~ exp(-beta_target * target(y)).
    Thin convenience wrapper around `importance_weights_log`: subtract
    the max log-weight for numerical stability, then exponentiate.

        w_i = exp(log_w_i - max_j log_w_j),   w in [0, 1].

    The omitted factor `exp(max log_w)` is a sample-dependent scalar
    that cancels in every *self-normalized* downstream use (ratios in
    compute_ESS, draws from compute_ESS_log / resample, MC averages of
    bounded test functions). Use this routine when the consumer expects
    plain non-negative weights (e.g. `resample(samples, weights)`); use
    `importance_weights_log` + `compute_ESS_log` / `compute_CESS_log`
    when log-space stability is required (very-low-overlap proposals,
    tail diagnostics).

    Input:
        samples:     Tensor [N, d]      particles drawn from `source`
        source:      Potential          source (proposal-base) potential U_0
        target:      Potential          target potential U_1
        F:           ComposedTransform  forward flow map (typically obtained as flow.t())
        beta_source: float              inverse temperature of the source
                                        distribution (default 1.0).
        beta_target: float              inverse temperature of the target
                                        distribution (default 1.0). Pair this
                                        with the same beta that was passed to
                                        the loss / Langevin / HMC so the
                                        weights match the tempered training
                                        objective.
        chunk:       int                split `samples` along dim 0 into this many
                                        chunks and accumulate. Reduces peak GPU
                                        memory at the cost of wall time;
                                        statistically and numerically equivalent
                                        to chunk=1 (the per-sample log-weight
                                        only depends on its own (x, F(x))).
    Output:
        w: Tensor [N]   unnormalized importance weights in [0, 1].
    """
    log_w = importance_weights_log(samples, source, target, F, beta_source=beta_source, beta_target=beta_target, chunk=chunk)
    return (log_w - log_w.max()).exp()

def importance_weights_log(samples: torch.Tensor, source: Potential, target: Potential, F: ComposedTransform, beta_source: float = 1.0, beta_target: float = 1.0, chunk: int = 1) -> torch.Tensor:
    """
    Self-normalized importance-sampling log-weights for the proposal
    `nu = F_# mu_0` against the target `mu_1`, where the source and
    target are the tempered Gibbs distributions
        mu_0(x) ~ exp(-beta_source * source(x)),
        mu_1(y) ~ exp(-beta_target * target(y)),
    and `F` is the trained bijection that pushes source samples
    toward the target.

    For x drawn from the (beta_source-tempered) source, y = F(x), the
    proposal density is
        log nu(y) = -beta_source * source(x) - log|det J_F(x)|.
    The unnormalized log-importance-weight is therefore
        log w(y) = log mu_1(y) - log nu(y)
                 = -beta_target * target(y) + beta_source * source(x)
                   + log|det J_F(x)|,
    using `Potential` energies U = -log mu (up to additive constants
    that cancel after self-normalization). Default
    beta_source = beta_target = 1.0 recovers the standard case.

    Use the regular forward call (`source(x)`, `target(y)`); no
    `enable_eval()` opt-in is needed here since this routine is not on
    the per-iter MALA hot path.

    Input:
        samples:     Tensor [N, d]      particles drawn from `source`
        source:      Potential          source (proposal-base) potential U_0
        target:      Potential          target potential U_1
        F:           ComposedTransform  forward flow map (typically obtained as flow.t())
        beta_source: float              inverse temperature of the source
                                        distribution (default 1.0).
        beta_target: float              inverse temperature of the target
                                        distribution (default 1.0). Pair this
                                        with the same beta that was passed to
                                        the loss / Langevin / HMC so the
                                        weights match the tempered training
                                        objective.
        chunk:       int                split `samples` along dim 0 into this many
                                        chunks and concatenate the per-chunk
                                        log-weights. Reduces peak GPU memory at
                                        the cost of wall time; statistically and
                                        numerically equivalent to chunk=1 (each
                                        sample's log-weight depends only on its
                                        own (x, F(x))).
    Output:
        log_w: Tensor [N]   unnormalized log importance weights, ready
                            to feed into compute_ESS_log / compute_CESS_log
                            or to exponentiate (after subtracting max).
    """
    out = []
    for x in torch.chunk(samples, chunk, dim=0):
        y, ladj = F.call_and_ladj(x)
        out.append(-beta_target * target(y) + beta_source * source(x) + ladj)
    return torch.cat(out, dim=0)


# ──────────────────────────────────────────────────────────────────────
# Resample — multinomial resampling with replacement
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# L-BFGS — batched mode finder / MAP refinement
# ──────────────────────────────────────────────────────────────────────

def lbfgs(samples: torch.Tensor, potential: Potential, step: float = 1.0, iters: int = 100, memory: int = 6, armijo: bool = False, chunk: int = 1) -> torch.Tensor:
    """
    Batched L-BFGS for mode-finding / MAP refinement on the target
    exp(-U(x)). Every particle in `samples` carries its own (s, y)
    history and they all step in lockstep through vectorised tensor
    ops, so N=2000 particles optimise as cheaply as one (modulo
    per-row arithmetic). Exposed in zflows.utils both as `lbfgs` and as
    the `optimization` alias (which is the default mode-finder).

    L-BFGS builds a rank-`memory` approximation of the inverse Hessian
    from the last `memory` gradient differences, giving superlinear
    convergence on smooth potentials. It typically reaches near-machine
    precision in a few iterations per effective curvature direction,
    well-conditioned or not -- contrast with Adam-style sign descent,
    which is O(init_err / step) just to reach the basin.

    Algorithm, per iteration:
      1. Two-loop recursion: combine the current gradient g_k with the
         stored pairs (s_i, y_i) = (x_{i+1} - x_i, g_{i+1} - g_i) to get
         the search direction d_k = -H_k^{-1} g_k, where
            H_0^k = gamma_k * I,
            gamma_k = (s_last^T y_last) / (y_last^T y_last)   (= 1 when
                                                                empty).
      2. Update x_{k+1} = x_k + alpha_k * d_k. Two step-size policies:
           - `armijo=False` (default): no line search, alpha_k = step
             always. Newton-style update under the BFGS approximation.
             Fast, but can overshoot in non-convex regions or on the
             very first iteration (empty history, d = -g can be huge).
           - `armijo=True`: per-particle masked Armijo backtracking.
             Start with alpha = step, halve until each particle's
             trial satisfies the Armijo sufficient-decrease condition
                U(x + alpha * d) <= U(x) + C1 * alpha * (d . g),
             where C1 = 1e-4. All particles run K_MAX = 6 trials in
             lockstep through batched U evaluations; a per-particle
             `done` mask freezes alpha as soon as Armijo is satisfied.
             Particles that never satisfy Armijo within K_MAX trials
             take the smallest tested step (very conservative fallback).
             Cost per L-BFGS iter: 1 batched .grad() + K_MAX batched
             .eval() calls. `armijo=True` therefore requires
             `potential.enable_eval()`.
      3. Evaluate g_{k+1} = grad U(x_{k+1}). Curvature pair (s, y) is
         appended; if more than `memory` pairs are stored, the oldest
         is dropped. Per particle, pairs that violate the BFGS
         curvature condition s^T y > 0 are kept in the history but with
         rho_i = 0, which makes the two-loop recursion ignore them.
         This preserves full batch vectorisation (no per-particle
         history-length divergence).

    The `potential.grad` and `potential.eval` fast paths are compiled
    with reduce-overhead, so their outputs are static buffers that get
    overwritten on the next call. We `.clone()` after each grad call so
    that the previous g survives long enough to form y = g_new - g_old,
    and similarly for the cached U(x) value carried across iterations
    when armijo=True. (Same hazard as `langevin`'s "consume gx before
    potential.grad(y)" comment.)

    Requires `potential.enable_grad()`; with `armijo=True`, also
    requires `potential.enable_eval()`. Either missing -> RuntimeError.

    Input:
        samples:   Tensor [N, d]   initial particles
        potential: Potential       target potential U; must support .grad(x)
                                   (and .eval(x) when armijo=True)
        step:      float           multiplier on the L-BFGS direction.
                                   With armijo=False: fixed per-iter step.
                                   With armijo=True: initial trial alpha
                                   for the backtracking line search.
                                   1.0 ~ pure Newton step; reduce
                                   (e.g. 0.5, 0.1) for stiff problems.
        iters:     int             number of L-BFGS iterations
        memory:    int             curvature pairs (s, y) kept per
                                   particle (Nocedal's `m`). Typical 3-20;
                                   larger = better Hessian approximation
                                   and more memory (N * d * 2 * memory floats).
        armijo:    bool            if True, enable masked Armijo
                                   backtracking line search. Adds K_MAX
                                   compiled forward passes per iteration
                                   but guarantees sufficient decrease of
                                   U at every accepted step. Use it when
                                   the line-search-free update is
                                   unstable (first-iter blow-up, very
                                   non-convex landscapes).
        chunk:     int             split `samples` along dim 0 into this
                                   many chunks and run sequentially.
                                   Reduces peak GPU memory at the cost of
                                   wall time; statistically equivalent to
                                   chunk=1 (each particle's history is
                                   independent, and there is no noise).
    Output:
        samples: Tensor [N, d]   particles after `iters` L-BFGS updates
    """
    if potential._grad_fn is None:
        raise RuntimeError(
            f"lbfgs() requires gradients on the potential; "
            f"call {type(potential).__name__}.enable_grad() before passing it in."
        )
    if armijo and potential._eval_fn is None:
        raise RuntimeError(
            f"lbfgs(armijo=True) needs the compiled forward fast path; "
            f"call {type(potential).__name__}.enable_eval() before passing it in."
        )
    C1, SHRINK, K_MAX = 1e-4, 0.5, 6
    out = []
    for x in torch.chunk(samples, chunk, dim=0):
        history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # clone(): potential.grad / potential.eval use torch.compile
        # reduce-overhead, so each call's output is a static buffer that
        # the next call overwrites. Clone anything we need across calls.
        g = potential.grad(x).clone()
        U_x = potential.eval(x).clone() if armijo else None
        for _ in range(iters):
            # Two-loop recursion: r approximates H_k^{-1} g
            q = g
            alphas = []
            for s_i, y_i, rho_i in reversed(history):
                a_i = rho_i * (s_i * q).sum(dim=-1) # [N]
                q = q - a_i.unsqueeze(-1) * y_i
                alphas.append(a_i)
            alphas.reverse() # chronological order
            if history:
                s_last, y_last, _ = history[-1]
                gamma = (s_last * y_last).sum(dim=-1) / (y_last ** 2).sum(dim=-1).clamp(min=1e-10) # [N]
            else:
                gamma = x.new_ones(x.shape[0])
            r = gamma.unsqueeze(-1) * q
            for (s_i, y_i, rho_i), a_i in zip(history, alphas):
                beta_i = rho_i * (y_i * r).sum(dim=-1)
                r = r + (a_i - beta_i).unsqueeze(-1) * s_i
            if armijo:
                # Masked Armijo backtracking: per-particle alpha, all
                # particles run K_MAX trials in lockstep.
                d = -r # search direction
                dg = (d * g).sum(dim=-1) # [N], < 0 for descent direction
                alpha = x.new_full((x.shape[0],), step)
                done = x.new_zeros(x.shape[0], dtype=torch.bool)
                for _ in range(K_MAX):
                    x_trial = x + alpha.unsqueeze(-1) * d
                    U_trial = potential.eval(x_trial)
                    ok = U_trial <= U_x + C1 * alpha * dg # [N] bool
                    done = done | ok
                    alpha = torch.where(done, alpha, alpha * SHRINK)
                # After the loop, x_trial = x + alpha_final * d encodes
                # "accepted alpha" for done particles (alpha was frozen
                # at acceptance) and "smallest fallback alpha" otherwise.
                x_new = x_trial
                U_x_new = U_trial.clone() # carry to next iter as U_x
            else:
                x_new = x - step * r
                U_x_new = None
            g_new = potential.grad(x_new).clone() # clone for same static-buffer reason
            # Store curvature pair; mask out particles that violate s^T y > 0
            s_new = x_new - x
            y_new = g_new - g
            ys = (s_new * y_new).sum(dim=-1) # [N]
            rho_new = torch.where(ys > 1e-10, 1.0 / ys, ys.new_zeros(ys.shape))
            history.append((s_new, y_new, rho_new))
            if len(history) > memory:
                history.pop(0)
            x, g = x_new, g_new
            if armijo:
                U_x = U_x_new
        out.append(x)
    return torch.cat(out, dim=0)


# ──────────────────────────────────────────────────────────────────────
# Langevin — overdamped Langevin / MALA / tamed variants
# ──────────────────────────────────────────────────────────────────────

def langevin(samples: torch.Tensor, potential: Potential, beta: float = 1.0, step: float = 1e-3, iters: int = 100, adjust: bool = False, taming: float = 0, chunk: int = 1) -> torch.Tensor:
    """
    Langevin dynamics targeting the tempered distribution exp(-beta * U(x)).

    Proposal (Euler-Maruyama on the overdamped Langevin SDE
    dtheta = -beta * grad U(theta) dt + sqrt(2) dB):
        y = x - step * beta * grad U(x) + sqrt(2 * step) * xi,   xi ~ N(0, I_d).

    When `taming > 0`, the raw drift beta * grad U(x) is replaced with the
    tamed effective force
        G(x) = beta * grad U(x) / (1 + taming * ||beta * grad U(x)||),
    so that ||taming * G(x)|| <= 1. This stabilizes ULA on targets whose
    |grad U| grows super-linearly (polynomial-tail energies), where plain
    ULA can explode on outlier particles, and reduces to standard ULA in
    the bulk (taming * ||beta * grad U|| << 1). Tamed drift is incompatible
    with adjust=True, since the MH correction below assumes the Gaussian
    proposal centred on x - step * beta * grad U(x).

    With adjust=False (default), every proposal is accepted; this is the
    unadjusted Langevin algorithm (ULA), which has an O(step) bias but
    needs only one gradient call per iteration. With adjust=True, each
    proposal is accepted via Metropolis-Hastings, giving the standard MALA
    scheme whose stationary distribution is *exactly* exp(-beta * U)
    (unbiased) at the cost of ~2x runtime (two gradient calls per iteration).

    The MH acceptance probability is min(1, exp(log_alpha)) with
        log_alpha = beta * (U(x) - U(y)) + log q(x|y) - log q(y|x),
    where the proposal density is Gaussian:
        log q(z|w) = -||z - w + step * beta * grad U(w)||^2 / (4 * step) + const.
    Both the energy difference *and* the asymmetric-proposal correction are
    needed; using only the energy term leaves a residual O(step) bias.

    Requires `potential.enable_grad()` to have been called so that
    `potential.grad(x)` is available; otherwise raises RuntimeError.
    For the MALA branch (`adjust=True`), if `potential.enable_eval()` has
    also been called, the U(y) / U(x) energy evaluations route through
    the compiled `potential.eval(x)` fast path; otherwise they fall back
    to the regular `potential(x)` call.

    Input:
        samples:   Tensor [N, d]   initial particles
        potential: Potential       target potential U; must support .grad(x)
        beta:      float           inverse temperature; the stationary
                                   distribution is exp(-beta * U). Default
                                   1.0 recovers the standard scheme.
        step:      float           Euler-Maruyama step size
        iters:     int             number of Langevin steps
        adjust:    bool            if True, run MALA (unbiased); if False, run ULA
        taming:    float           if > 0, use tamed drift
                                   beta * grad U(x) / (1 + taming * ||beta * grad U(x)||).
                                   Stabilizes ULA on super-linearly growing
                                   potentials. Not compatible with adjust=True.
        chunk:     int             split `samples` along dim 0 into this many
                                   chunks and run the trajectories sequentially.
                                   Reduces peak GPU memory at the cost of wall
                                   time. Statistically equivalent to chunk=1
                                   (each chunk uses its own independent noise);
                                   set higher only if you hit OOM on the
                                   whole batch.
    Output:
        samples: Tensor [N, d]   particles after `iters` Langevin updates
    """
    if potential._grad_fn is None:
        raise RuntimeError(
            f"langevin() requires gradients on the potential; "
            f"call {type(potential).__name__}.enable_grad() before passing it in."
        )
    if adjust and taming > 0:
        raise ValueError("langevin(): adjust=True and taming>0 are mutually exclusive.")
    # MALA accept/reject needs U(x), U(y); use the compiled fast path if
    # the user has opted in via .enable_eval(), else fall back to __call__.
    U = potential.eval if potential._eval_fn is not None else potential
    noise_scale = (2.0 * step) ** 0.5
    out = []
    for x in torch.chunk(samples, chunk, dim=0):
        for _ in range(iters):
            fx = beta * potential.grad(x) # effective force beta * grad U(x)
            drift = fx / (1 + taming * fx.norm(dim=-1, keepdim=True)) if taming > 0 else fx
            y = x - step * drift + noise_scale * torch.randn_like(x)
            if adjust:
                # log q(z|w) = -||z - w + step * beta * grad U(w)||^2 / (4 * step) + const
                # Consume fx (-> log_q_yx) BEFORE calling potential.grad(y)
                log_q_yx = -((y - x + step * fx) ** 2).sum(dim=-1) / (4.0 * step) # log q(y|x)
                fy = beta * potential.grad(y)
                log_q_xy = -((x - y + step * fy) ** 2).sum(dim=-1) / (4.0 * step) # log q(x|y)
                # Evaluate U(y) then U(x); the -U(y) allocates a fresh tensor
                # before U(x) overwrites the CUDA-Graphs eval output buffer.
                log_alpha = beta * (-U(y) + U(x)) + log_q_xy - log_q_yx # [N]
                accept = torch.rand_like(log_alpha).log() < log_alpha # [N] bool
                x = torch.where(accept.unsqueeze(-1), y, x)
            else:
                x = y
        out.append(x)
    return torch.cat(out, dim=0)


# ──────────────────────────────────────────────────────────────────────
# HMC — Hamiltonian Monte Carlo with leapfrog + MH gate
# ──────────────────────────────────────────────────────────────────────

def hmc(samples: torch.Tensor, potential: Potential, beta: float = 1.0, step: float = 1e-2, iters: int = 10, burns: int = 10, chunk: int = 1) -> torch.Tensor:
    """
    Hamiltonian Monte Carlo (HMC) targeting the tempered distribution
    exp(-beta * U(x)).

    Each "burn" performs a full HMC trajectory:
      1. Resample momentum p ~ N(0, I_d) -- a complete, "high-temperature"
         refresh that discards any correlation with the previous burn.
      2. Integrate the Hamiltonian flow of H(x, p) = beta * U(x) + 0.5 * ||p||^2
         for `iters` leapfrog steps of size `step`. Each kick uses the
         effective force beta * grad U(x).
      3. Metropolis-Hastings accept/reject on the trajectory endpoint:
            log_alpha = beta * (U(x0) - U(x_end))
                       + 0.5 * (||p0||^2 - ||p_end||^2),
            accept with probability min(1, exp(log_alpha)).
         Leapfrog is exactly volume-preserving, so no Jacobian term enters.

    Compared to MALA (`langevin(adjust=True)`), HMC pays
    `iters + 1` gradient calls per MH decision instead of 2, but the
    trajectory moves O(step * iters) per decision rather than O(sqrt(step)),
    giving much lower autocorrelation at fixed acceptance.

    "Safer" than ULA/MALA on stiff targets in two senses:
      - The MH correction makes the stationary distribution exactly exp(-U),
        with no O(step) bias to tune away.
      - Divergent trajectories (NaN / inf energies from too-large leapfrog
        steps in steep regions) produce non-finite log_alpha, which is
        clamped to -inf so the particle reverts to its pre-trajectory
        position. NaN coordinates never enter the returned tensor; the
        worst a bad trajectory can do is waste compute on that burn.

    GPU-parallel structure:
      - Every particle in a chunk runs exactly `iters` leapfrog steps in
        lockstep; per-particle decisions only happen at the MH accept mask
        (`torch.where`), so the inner loop stays on one CUDA graph.
      - Efficient leapfrog combines adjacent trailing/leading half-kicks,
        costing `iters + 1` compiled grad calls per trajectory instead of
        the naive 2 * iters.
      - `potential.grad` and `potential.eval` use torch.compile
        reduce-overhead, so their outputs are static buffers overwritten on
        the next call. The leapfrog consumes each gradient before the next
        .grad() call (no clone needed), but U_start must survive across the
        leapfrog AND across the U_end .eval() call, so we .clone() it once.

    Requires `potential.enable_grad()`. If `potential.enable_eval()` has
    also been called, the U(x_start) / U(x_end) energies route through the
    compiled fast path; otherwise they fall back to `potential(x)`.

    Input:
        samples:   Tensor [N, d]   initial particles
        potential: Potential       target potential U; must support .grad(x)
        beta:      float           inverse temperature; stationary
                                   distribution is exp(-beta * U). Default
                                   1.0 recovers the standard HMC scheme.
        step:      float           leapfrog step size epsilon. Tune so the
                                   MH acceptance rate is ~0.6-0.8 (the HMC
                                   sweet spot from Beskos et al. 2013).
        iters:     int             number of leapfrog steps per trajectory.
                                   Trajectory length L = step * iters; pick
                                   L on the scale of the target's largest
                                   correlation length. Larger trades grad
                                   calls for lower autocorrelation.
        burns:     int             number of momentum refreshes / MH
                                   trajectories. Each burn fully redraws
                                   p ~ N(0, I) (a "high-temperature" reset)
                                   and runs one MH accept/reject decision.
        chunk:     int             split `samples` along dim 0 into this
                                   many chunks and run sequentially.
                                   Reduces peak GPU memory at the cost of
                                   wall time; statistically equivalent to
                                   chunk=1 (each chunk uses its own
                                   independent momentum and accept noise).
    Output:
        samples: Tensor [N, d]   particles after `burns` HMC trajectories
    """
    if potential._grad_fn is None:
        raise RuntimeError(
            f"hmc() requires gradients on the potential; "
            f"call {type(potential).__name__}.enable_grad() before passing it in."
        )
    # MH accept/reject needs U(x_start), U(x_end); use the compiled fast
    # path if the user has opted in via .enable_eval(), else fall back to
    # __call__ (same convention as langevin).
    U = potential.eval if potential._eval_fn is not None else potential
    out = []
    for x in torch.chunk(samples, chunk, dim=0):
        for _ in range(burns):
            x_start = x
            p_start = torch.randn_like(x)
            # clone(): U_start must survive across the leapfrog AND across
            # the U_end .eval() call, which would otherwise overwrite the
            # static buffer under reduce-overhead.
            U_start = U(x_start).clone()
            K_start = 0.5 * (p_start ** 2).sum(dim=-1) # [N]

            # Efficient leapfrog: iters + 1 grad calls, combined half-kicks.
            # Effective force is beta * grad U; the `beta *` allocates a
            # fresh tensor each call, so g never aliases the CUDA-Graphs
            # buffer behind potential.grad.
            p = p_start
            if iters >= 1:
                g = beta * potential.grad(x)
                p = p - 0.5 * step * g
                for _ in range(iters - 1):
                    x = x + step * p
                    g = beta * potential.grad(x)
                    p = p - step * g
                x = x + step * p
                g = beta * potential.grad(x)
                p = p - 0.5 * step * g

            U_end = U(x) # [N]
            K_end = 0.5 * (p ** 2).sum(dim=-1) # [N]

            # MH accept/reject with NaN guard: divergent trajectories
            # produce non-finite log_alpha -> -inf -> reject -> revert to
            # x_start, so the returned tensor is always finite.
            log_alpha = beta * (U_start - U_end) + (K_start - K_end) # [N]
            log_alpha = torch.where(
                torch.isfinite(log_alpha),
                log_alpha,
                log_alpha.new_full((), float("-inf")),
            )
            accept = torch.rand_like(log_alpha).log() < log_alpha # [N] bool
            x = torch.where(accept.unsqueeze(-1), x, x_start)
        out.append(x)
    return torch.cat(out, dim=0)


# alias: L-BFGS is the default mode-finder / MAP-refinement routine in zflows
optimization = lbfgs

# alias: in SMC literature, Langevin steps are the standard "rejuvenation" move
rejuvenation = langevin


# ──────────────────────────────────────────────────────────────────────
# torch.compile environment helpers
# ──────────────────────────────────────────────────────────────────────

def check_compile_available() -> bool:
    """Diagnose whether the current environment can actually run torch.compile.

    Runs three checks, in order. The first two emit non-blocking warnings
    and do NOT affect the return value — they only flag known footguns
    (non-Linux OS, missing `nvcc`) so the user gets a useful pointer if
    the third check then fails. The third check is the authoritative one:
    it really compiles + runs a tiny function and the return value reflects
    only its success.

    Checks:
      1. **OS:** zflows has only been tested on Linux + NVIDIA GPU.
         `torch.compile` does not run on native Windows
         (https://github.com/pytorch/pytorch/issues/167062); on macOS the
         flow code imports but the compiled fast paths are untested.
         Warns if `platform.system() != 'Linux'`.
      2. **nvcc on $PATH:** `torch.compile`'s Triton / TorchInductor backend
         JIT-compiles a small CUDA helper on first use, which requires the
         CUDA Toolkit's C++ compiler `nvcc` — *not* just the CUDA runtime
         shipped with the PyTorch wheel. Warns if `shutil.which('nvcc')`
         returns None. Harmless if you only intend to compile on CPU.
      3. **Sanity test (authoritative):** actually `torch.compile` a small
         function and run one forward pass. On CUDA, uses
         `mode='reduce-overhead'` (the mode that `Potential.enable_grad` /
         `enable_eval` default to; `loss.compile_raw` / `compile_beta` default to `'default'`); on CPU, uses
         `mode='default'`. The return value is True iff this step succeeds.

    Prints a one-line PASS/WARN/FAIL summary for each step.

    Intended usage: **run interactively or from a standalone diagnostic
    script** — not from your main training code. The sanity test really
    invokes `torch.compile`, which costs compile time on every call and
    consumes a Dynamo cache slot. Use it once to validate a fresh
    environment, then remove the call.

    Returns:
        bool: True iff the sanity test succeeded. The OS / nvcc warnings
              do NOT influence this value — a Mac with `nvcc` missing but
              a working CPU `torch.compile` will still return True.
    """
    import platform
    import shutil
    import warnings

    import torch

    # (1) OS check — warn, do not gate
    sys_name = platform.system()
    if sys_name == "Linux":
        print(f"[OK ]   OS = {sys_name}")
    else:
        warnings.warn(
            f"torch.compile might not be supported on your system: {sys_name}. "
            f"zflows is tested only on Linux + NVIDIA GPU. On native Windows, "
            f"use WSL; macOS is untested."
        )
        print(f"[WARN] OS = {sys_name}  (torch.compile may not work)")

    # (2) nvcc check — warn, do not gate (CPU-only setups don't need it).
    # Try $PATH first, then fall back to the Ubuntu default install
    # locations that `cuda-toolkit` leaves off $PATH unless the user
    # explicitly added them: the symlinked /usr/local/cuda/bin/nvcc and,
    # for side-by-side installs, the versioned /usr/local/cuda-*/bin/nvcc.
    from pathlib import Path
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        candidates = [Path("/usr/local/cuda/bin/nvcc")]
        # Versioned installs (e.g. /usr/local/cuda-12.4/bin/nvcc); newest first.
        candidates += sorted(
            Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True
        )
        for c in candidates:
            if c.is_file():
                nvcc_path = str(c)
                break
    if nvcc_path is not None:
        print(f"[OK ]   nvcc = {nvcc_path}")
    else:
        warnings.warn(
            "nvcc not found on $PATH or in /usr/local/cuda/bin or "
            "/usr/local/cuda-*/bin. Install the full CUDA Toolkit from "
            "https://developer.nvidia.com/cuda-downloads. Harmless if you "
            "only intend to compile on CPU."
        )
        print("[WARN] nvcc not found")

    # (3) Sanity test — the authoritative check; gates the return value
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "reduce-overhead" if device == "cuda" else "default"

    @torch.compile(mode=mode)
    def _probe(x: torch.Tensor) -> torch.Tensor:
        return x.sin() + x.cos()

    try:
        x = torch.randn(8, device=device)
        y = _probe(x)
        if device == "cuda":
            torch.cuda.synchronize()
        _ = y.sum().item()
        print(f"[OK ]   sanity test passed (device={device}, mode={mode})")
        ok = True
    except Exception as e:
        print(f"[FAIL] sanity test (device={device}, mode={mode}): "
              f"{type(e).__name__}: {e}")
        ok = False

    print()
    print("Note: please run check_compile_available() interactively or in a "
          "standalone python script. Do not call it from your main training "
          "code — the sanity test really invokes torch.compile, which costs "
          "compile time on every call and consumes a Dynamo cache slot.")
    return ok


def set_cache_size_limit(limit: int = 8) -> None:
    """Set torch._dynamo's per-code-object compile-cache size limit.

    Dynamo caches compiled specializations per ``__code__`` object. The stock
    default is 8 — plenty for a normal training loop with one model and a
    fixed batch shape. Sweep / benchmark code that compiles many distinct
    closures sharing one code body can exceed the limit and silently fall
    back to eager from cell N+1 onward. Common triggers:

      - hyperparameter sweeps where `zflows.loss.compile_raw` /
        `compile_beta` are called once per cell (each call produces a
        fresh closure on the same code object, distinguished by the
        captured `transform` and `potential`);
      - annealed training that constructs many short-lived `Potential`
        instances of the same subclass, each calling `.enable_grad()`;
      - scripts that mix several distinct compiled functions.

    Bump the limit only when you actually observe eager fallback (set
    `torch._dynamo.config.suppress_errors = False` and watch for recompile
    warnings, or set `torch._logging.set_logs(recompiles=True)` to see
    them directly). Each cached spec costs a small amount of memory and a
    constant lookup overhead on every call, so don't raise it gratuitously.
    Needing > 100 usually indicates an uncontrolled retrace pattern that's
    better fixed at its source (mark dynamic shapes, share backends, or
    restructure to reuse compiled objects).

    Argument:
        limit: max number of specializations to cache per code object.
            Default 8 matches torch._dynamo's stock setting (so calling
            with no argument resets to the default).

    Typical usage at the top of a sweep script::

        from zflows.utils import set_cache_size_limit
        set_cache_size_limit(64)   # generous headroom for ~30 specs
    """
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = limit


def suppress_warnings() -> None:
    """Silence the various warning/log channels that PyTorch, Triton, and
    Inductor emit during a typical zflows training run (especially with
    `torch.compile` / `zflows.loss.compile_raw` / `compile_beta` in the loop).

    Covers four orthogonal layers of noise:
      1. **Python `warnings`** (e.g. inductor's TF32 hint, deprecation
         warnings from `torch.distributions`): filter to "ignore".
      2. **torch._logging channels** (recompiles, graph breaks): toggled off.
      3. **Triton autotune stderr** (per-kernel "AUTOTUNE addmm ..." banners
         from the Triton C-side autotuner): silenced via
         `TRITON_PRINT_AUTOTUNING=0`. Only takes effect for kernels that
         have not been autotuned yet — call this BEFORE the first
         `torch.compile` invocation.
      4. **Inductor compile-worker interleaving** (gcc/nvcc warnings,
         `_POSIX_C_SOURCE redefined`, etc.): serialized to one worker via
         `TORCHINDUCTOR_COMPILE_THREADS=1`. Same caveat as (3): set it
         before workers spawn (i.e. before the first compile call).

    The function is idempotent and safe to call multiple times. Real
    compile failures still raise — only the routine noise is muted.

    Typical usage at the top of a training script::

        from zflows.utils import suppress_warnings
        suppress_warnings()
        # ... rest of imports and training code
    """
    import os
    import warnings

    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

    warnings.filterwarnings("ignore")

    # Import lazily to avoid pulling torch._logging into module-load if the
    # user never calls this function.
    import torch._logging
    torch._logging.set_logs(recompiles=False, graph_breaks=False)