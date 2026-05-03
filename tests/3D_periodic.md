# 3D test: periodic target on the 3-torus, with rejuvenation

We train a Neural Circular Spline Flow (NCSF) $F$ on the 3-torus $\mathbb T^3 = [-\pi, \pi]^3$, then use the trained flow as a proposal for self-normalized importance sampling, *resample* to obtain equally-weighted particles, and finally *rejuvenate* them with overdamped Langevin dynamics that needs the gradient $\nabla U_1$.

- **Source:** $\mu_0 = \mathrm{Uniform}([-\pi, \pi]^3)$, with constant potential $U_0$.
- **Target:** a periodic von-Mises *ridge mixture* on $\mathbb T^3$,
$$
U_1(\theta) = -\log\Bigl[\,e^{\kappa\cos(\theta_1 - \theta_2)} + e^{\kappa\cos(\theta_2 - \theta_3)} + e^{\kappa\cos(\theta_3 - \theta_1)}\,\Bigr], \quad \kappa = 4.
$$
The density is genuinely periodic (only $\cos$ of differences appears), and concentrates on three intersecting *ridges* $\{\theta_i = \theta_j\} \subset \mathbb T^3$ — a target that an axis-aligned proposal cannot fit, but a circular spline flow can.
- **Flow:** $F$ is an NCSF; internally it composes circular shifts with monotonic rational-quadratic splines on the angular box $[-\pi, \pi]^3$, then conjugates by an affine to act on the user-specified box.

## Mathematical background

Both `Potential` subclasses return $U(x)$ for a batch of points $x \in [-\pi, \pi]^3$. By definition,
$$
\mu_0(x) \propto \exp(-U_0(x)), \qquad \mu_1(\theta) \propto \exp(-U_1(\theta)).
$$
As always with reverse KL training, only $U_1$ is needed — never the normalizing constant of $\mu_1$.

### NCSF and reverse-$\mathrm{KL}$ training

NCSF differs from NSF in that the inner univariate transform is a *circular* RQS spline: it composes a periodic shift with a monotonic spline on $[-\pi, \pi]$, so the resulting bijection respects the torus topology — densities are continuous (and $C^1$) when wrapped at $\pm\pi$. The reverse-$\mathrm{KL}$ loss is identical to the Euclidean case,
$$
\mathcal L_{\mathrm{reverse}}[F] = \mathbb E_{x \sim \mu_0}\bigl[\, U_1(F(x)) - \log |\det J_F(x)| \,\bigr],
$$
because `flow.t().call_and_ladj(x)` returns the same `(y, ladj)` interface.

### Importance sampling and $\mathrm{ESS}$

With the trained flow as proposal $\nu = F_\# \mu_0$, the unnormalized log-weights are
$$
\log w(\theta) = -U_1(\theta) + U_0(x) + \log |\det J_F(x)|, \qquad \theta = F(x), \; x \sim \mu_0,
$$
and the Effective Sample Size $\mathrm{ESS} = (\sum w_i)^2 / (N \sum w_i^2) \in [0, 1]$ tells us how concentrated those weights are. The one-call API is `importance_weights_log(samples=x, source=u0, target=u1, flow=flow)`, returning the log-weights ready for `compute_ESS_log` (or `(log_w - log_w.max()).exp()` if you need the linear weights).

### Resampling: from weighted to equally-weighted particles

IS produces $N$ samples $\{\theta_i\}$ with unequal weights $\{w_i\}$. Many downstream uses (plotting, MCMC kernels, particle filters) want *equally-weighted* samples. Multinomial resampling draws indices $\{j_i\}_{i=1}^{N}$ i.i.d. from the categorical $\mathrm{Cat}(w_1, \dots, w_N)$ and returns $\{\theta_{j_i}\}$. Asymptotically the resulting empirical measure has the same target distribution, but with a discrete *empirical artefact*: many particles collapse onto the same locations. That artefact is exactly what the rejuvenation step below is designed to dissolve.

`resample(samples, weights)` does this in one line.

### `enable_grad` and `enable_eval`: opt-in compiled fast paths for `Potential`

The rejuvenation step needs $\nabla U_1(\theta)$ for the proposal *and* $U_1(\theta)$ itself for the MALA accept/reject decision. The base `Potential` class does **not** build either fast path by default — many use cases (forward-only sampling, IS evaluation, KL training) only need a regular `u1(x)` call, and we don't want to pay the `torch.compile` startup cost or pull in `torch.func` for those. Two opt-ins are exposed:

```python
u1.enable_grad()       # compiles vmap(grad(u1)),  stored in u1._grad_fn
u1.enable_eval()       # compiles u1.forward,      stored in u1._eval_fn
g = u1.grad(theta)     # [N, d], works without theta.requires_grad=True
v = u1.eval(theta)     # [N], faster than u1(theta) on the MALA hot loop
```

Under the hood (in [`zflows/potential.py`](../zflows/potential.py)), `enable_grad` caches $\nabla U_1 = \texttt{torch.compile}(\texttt{vmap}(\texttt{grad}(U_1)))$, and `enable_eval` caches $\texttt{torch.compile}(U_1)$, so every subsequent `.grad(\theta)` / `.eval(\theta)` is a single fused kernel call with no Python-level autograd-graph construction. Both calls are **idempotent** (a second invocation does not recompile) and chainable. Inside the MALA branch of `langevin()`, if `_eval_fn` is populated the energy evaluations $U_1(y)$ and $U_1(x)$ in $\log\alpha$ route through it; otherwise they fall back to the regular `u1(\theta)` call. `u1.eval()` with no argument retains the standard `nn.Module` semantics (switch to eval mode, return `self`). The two `os.environ.setdefault(...)` lines in the script just suppress Triton's autotuning chatter and serialize Inductor's worker pool for cleaner output during the first compile.

### Rejuvenation: overdamped Langevin

`rejuvenation` is an alias for `langevin`. Both apply $T$ steps of Euler-Maruyama on the overdamped Langevin SDE
$$
\mathrm{d}\theta_t \;=\; -\nabla U_1(\theta_t)\,\mathrm{d}t + \sqrt{2}\,\mathrm{d}B_t,
$$
whose unique invariant distribution is exactly $\mu_1 \propto \exp(-U_1)$. The discrete update is
$$
\theta_{k+1} \;=\; \theta_k \;-\; h\,\nabla U_1(\theta_k) \;+\; \sqrt{2h}\,\xi_k, \qquad \xi_k \sim \mathcal N(0, I_d),
$$
with default step $h = 10^{-3}$ and $T = 100$ iterations. After resampling we have many duplicate particles; Langevin moves them according to the gradient flow of $U_1$ (drifting toward modes) plus stochastic diffusion, which **breaks the duplicates apart** and decorrelates the particle cloud. In SMC/particle-filter terminology this step is called *rejuvenation*.

Why gradient flow + noise: pure gradient descent ($-\nabla U_1$) is mode-seeking and would collapse particles onto local minima. The $\sqrt{2h}$ noise is exactly calibrated so that, in the continuous-time limit, the stationary density is $\mu_1$ rather than a delta at the mode. Discretization adds an $O(h)$ bias; for tighter targets one would compose Langevin with a Metropolis-Hastings accept/reject step (MALA), which the script enables via `adjust=True`.

## Implementation and execution

The full pipeline lives in [`3D_periodic.py`](3D_periodic.py). Run from the project root:

```bash
python -m tests.3D_periodic
```

Pointers into the script:

- imports & device setup: [`3D_periodic.py:1–12`](3D_periodic.py#L1-L12)
- source (uniform on the 3-torus) and target ($U_1$ ridge mixture): [`3D_periodic.py:14–32`](3D_periodic.py#L14-L32)
- NCSF init: [`3D_periodic.py:35–39`](3D_periodic.py#L35-L39)
- training parameters: [`3D_periodic.py:42–45`](3D_periodic.py#L42-L45)
- training loop (mini-batched reverse KL): [`3D_periodic.py:47–68`](3D_periodic.py#L47-L68)
- IS reweighting via `importance_weights_log` + $\mathrm{ESS}$: [`3D_periodic.py:73–79`](3D_periodic.py#L73-L79)
- resample weighted cloud → equal-weight cloud: [`3D_periodic.py:82–84`](3D_periodic.py#L82-L84)
- `enable_eval` + `enable_grad` + Langevin rejuvenation: [`3D_periodic.py:86–92`](3D_periodic.py#L86-L92)
- 3D scatter plot: [`3D_periodic.py:99–112`](3D_periodic.py#L99-L112)

<p align="center"><img src="3D_periodic.png" alt="3D periodic test" width="500px"></p>

## Recap of the pipeline

1. **Train.** Reverse KL fits an NCSF proposal $\nu = F_\# \mu_0$ to the target $\mu_1 \propto \exp(-U_1)$, using only $U_1$ (no target samples).
2. **Importance sampling.** Push $\mu_0$-samples through $F$, compute log-weights $\log w = -U_1(F(x)) + U_0(x) + \log|\det J_F|$, report $\mathrm{ESS}$ as a self-test.
3. **Resample.** Multinomial resampling $\{\theta_i, w_i\} \to \{\theta_{j_i}\}$ converts a weighted cloud into an equally-weighted cloud (with duplicate particles).
4. **Enable gradients (and forward).** `u1.enable_grad()` builds a `torch.compile`-compiled `vmap(grad(U_1))`; `u1.enable_eval()` additionally builds a compiled forward $U_1(\theta)$. Both are cached on the instance and reused on every subsequent call.
5. **Rejuvenate.** Overdamped Langevin (alias `rejuvenation`) with `adjust=True` runs MALA: the proposal uses `u1.grad`, and the accept/reject step uses `u1.eval` when `_eval_fn` is populated (else falls back to `u1(theta)`). The stationary density is exactly $\mu_1$.

The NCSF + IS + resample + Langevin pipeline is the basic building block of *flow-augmented SMC* on manifolds: the flow gives a good global proposal, IS provides unbiased correction, and Langevin gives local mixing using the geometry of $U_1$ itself.
