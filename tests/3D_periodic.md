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
and the Effective Sample Size $\mathrm{ESS} = (\sum w_i)^2 / (N \sum w_i^2) \in [0, 1]$ tells us how concentrated those weights are.

### Resampling: from weighted to equally-weighted particles

IS produces $N$ samples $\{\theta_i\}$ with unequal weights $\{w_i\}$. Many downstream uses (plotting, MCMC kernels, particle filters) want *equally-weighted* samples. Multinomial resampling draws indices $\{j_i\}_{i=1}^{N}$ i.i.d. from the categorical $\mathrm{Cat}(w_1, \dots, w_N)$ and returns $\{\theta_{j_i}\}$. Asymptotically the resulting empirical measure has the same target distribution, but with a discrete *empirical artefact*: many particles collapse onto the same locations. That artefact is exactly what the rejuvenation step below is designed to dissolve.

`resample(samples, weights)` does this in one line.

### `enable_grad`: opt-in compiled gradients for `Potential`

The rejuvenation step needs $\nabla U_1(\theta)$. The base `Potential` class does **not** build a gradient by default — many use cases (forward-only sampling, IS evaluation, KL training) only need $U_1$ itself, and we don't want to pay the `torch.compile` startup cost or pull in `torch.func` for those. The `.enable_grad()` method is the opt-in:

```python
u1.enable_grad()       # builds .grad once, returns self for chaining
g = u1.grad(theta)     # [N, d], works without theta.requires_grad=True
```

Under the hood (in [`zflows/potential.py`](../zflows/potential.py)), it caches
$$
\nabla U_1 \;=\; \texttt{torch.compile}\bigl(\texttt{vmap}(\texttt{grad}(U_1))\bigr)
$$
so every subsequent `.grad(theta)` is a single fused kernel call, with no Python-level autograd graph construction. The call is **idempotent** — calling `.enable_grad()` twice is a no-op — and it raises a clear `RuntimeError` if you call `.grad(...)` without first calling `.enable_grad()`. The two `os.environ.setdefault(...)` lines in the script just suppress Triton's autotuning chatter and serialize Inductor's worker pool for cleaner output during the first compile.

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

- imports & device setup: [`3D_periodic.py:1–11`](3D_periodic.py#L1-L11)
- source (uniform on the 3-torus) and target ($U_1$ ridge mixture): [`3D_periodic.py:13–31`](3D_periodic.py#L13-L31)
- NCSF init: [`3D_periodic.py:34–38`](3D_periodic.py#L34-L38)
- training parameters: [`3D_periodic.py:40–44`](3D_periodic.py#L40-L44)
- training loop (mini-batched reverse KL): [`3D_periodic.py:46–67`](3D_periodic.py#L46-L67)
- IS reweighting + $\mathrm{ESS}$: [`3D_periodic.py:70–81`](3D_periodic.py#L70-L81)
- resample weighted cloud → equal-weight cloud: [`3D_periodic.py:83–86`](3D_periodic.py#L83-L86)
- `enable_grad` + Langevin rejuvenation: [`3D_periodic.py:88–93`](3D_periodic.py#L88-L93)
- 3D scatter plot: [`3D_periodic.py:95–113`](3D_periodic.py#L95-L113)

<p align="center"><img src="3D_periodic.png" alt="3D periodic test" width="500px"></p>

## Recap of the pipeline

1. **Train.** Reverse KL fits an NCSF proposal $\nu = F_\# \mu_0$ to the target $\mu_1 \propto \exp(-U_1)$, using only $U_1$ (no target samples).
2. **Importance sampling.** Push $\mu_0$-samples through $F$, compute log-weights $\log w = -U_1(F(x)) + U_0(x) + \log|\det J_F|$, report $\mathrm{ESS}$ as a self-test.
3. **Resample.** Multinomial resampling $\{\theta_i, w_i\} \to \{\theta_{j_i}\}$ converts a weighted cloud into an equally-weighted cloud (with duplicate particles).
4. **Enable gradients.** `u1.enable_grad()` builds a `torch.compile`-compiled `vmap(grad(U_1))` once and caches it on the instance.
5. **Rejuvenate.** Overdamped Langevin (alias `rejuvenation`) breaks the duplicates apart and slightly corrects residual proposal bias by simulating an SDE whose stationary density is exactly $\mu_1$.

The NCSF + IS + resample + Langevin pipeline is the basic building block of *flow-augmented SMC* on manifolds: the flow gives a good global proposal, IS provides unbiased correction, and Langevin gives local mixing using the geometry of $U_1$ itself.
