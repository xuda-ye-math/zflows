<p align="center"><img src="logo.png" alt="zflows logo" width="240px"></p>

# zflows

A small convenience wrapper around [zuko](https://github.com/probabilists/zuko) for normalizing flows, with first-class support for energy-based sampling.

> **Status: experimental.** This project has only been tested on **Linux + NVIDIA GPU**. Some features (notably `Potential.enable_grad`, which relies on `torch.compile`) are **not supported on Windows** — see e.g. [pytorch/pytorch#167062](https://github.com/pytorch/pytorch/issues/167062). On Windows you can still use `NSF` / `NCSF` / `reverse_KL` / `forward_KL` / `resample`, but skip `enable_grad` and the Langevin `rejuvenation` step.
>
> This project was developed with [Claude Code](https://claude.com/claude-code).

## Features

**Flexible flow interfaces with documented hyperparameters.** Both `NSF` (Neural Spline Flow) and `NCSF` (Neural *Circular* Spline Flow, for periodic / angular features) expose a uniform constructor:

```python
NSF(a, b, bins=8, slope=1e-3, transforms=4, hidden_features=(64, 64), activation=nn.SiLU)
NCSF(a, b, bins=8, slope=1e-3, transforms=4, hidden_features=(64, 64), activation=nn.SiLU)
```

The user specifies the rectangular region `[a, b]` (any list / `Tensor`), the spline resolution and slope floor, the conditioner MLP width / depth, and the activation **class** (not instance). Every parameter has a docstring entry with a concrete recommended range and the failure mode it controls — see [`flow.py`](zflows/flow.py). NCSF treats each coordinate as periodic on its box and is the right choice for angles, phases, or any feature on a torus.

**Precompiled gradients on `Potential`.** Any subclass of `Potential` opts into a `torch.compile`-compiled `vmap(grad(U))` with a single call:

```python
u1 = U1().to(device).enable_grad()
g = u1.grad(theta) # [N, d], no .requires_grad_ on theta needed
```

The gradient closure is built once, cached on the instance, and reused every call — making heavy-load Langevin / MALA sampling fast (one fused kernel per step instead of an autograd graph rebuild). The call is idempotent and chainable; calling `.grad()` without `.enable_grad()` raises a clear `RuntimeError`.

**One-line KL losses.** `reverse_KL(x, target, flow)` and `forward_KL(y, source, flow)` are direct-call functions returning a scalar loss — drop them straight into a training loop, no boilerplate.

**SMC-style utilities.** `resample(samples, weights)` for multinomial resampling; `langevin(samples, potential, step, iters)` (alias `rejuvenation`) for overdamped Langevin updates; `compute_ESS`, `compute_ESS_log`, `compute_CESS`, `compute_CESS_log` for importance-sampling diagnostics, with log-space variants using `logsumexp` for numerical stability.

Together these compose into a complete *propose → reweight → resample → rejuvenate* pipeline with no glue code on the user side.

## Installation

`zflows` is pure Python; runtime dependencies ([`torch`](https://pytorch.org), [`zuko`](https://github.com/probabilists/zuko)) are resolved automatically by `pip`.

**1. Clone the repository.**

```bash
git clone https://github.com/xuda-ye-math/zflows.git
cd zflows
```

**2. Install in editable mode.** Local edits take effect immediately:

```bash
pip install -e .
```

**3. Verify the install.**

```bash
python -c "import zflows; print(zflows.__doc__)"
```

**Uninstall.**

```bash
pip uninstall zflows
```

## Mathematical background

<details>
<summary>click to expand; renders best in VS Code</summary>

Sampling problems on $\mathbb R^d$ (or on a torus) fall into two broad categories:

- **Energy-based sampling.** Given a confining potential $U_1(x)$, draw samples from the Boltzmann distribution $\mu_1 \propto \exp(-U_1)$.
- **Data-driven sampling.** Given empirical samples from a distribution $\mu_1$ with unknown density, generate further samples from $\mu_1$.

Both reduce in the normalizing-flow framework to the same recipe: pick a tractable source $\mu_0 \propto \exp(-U_0)$ and learn a diffeomorphism $F$ such that $F_{\#}\mu_0 \approx \mu_1$. The change-of-variable formula gives the pushforward density
$$
(F_{\#}\mu_0)(y) = \frac{\mu_0(x)}{|\det J_F(x)|}, \qquad y = F(x),
$$
where $J_F(x) \in \mathbb R^{d \times d}$ is the Jacobian of $F$ at $x$. The training objective is the $\mathrm{KL}$ divergence between $F_{\#}\mu_0$ and $\mu_1$.

For energy-based sampling we use the **reverse $\mathrm{KL}$**, which involves only the energy $U_1$ and not its normalizing constant:

$$
\begin{aligned}
\mathrm{KL}(F_{\#}\mu_0 \| \mu_1)
& = \int (F_{\#}\mu_0)(y) \log \frac{(F_{\#}\mu_0)(y)}{\mu_1(y)} \, \mathrm{d}y \\
& = \mathbb E_{x \sim \mu_0} [ U_1(F(x)) - U_0(x) - \log |\det J_F(x)| ] + \mathrm{const}.
\end{aligned}
$$

Dropping the (parameter-independent) constant yields the trainable loss

$$
\mathcal L_{\mathrm{reverse}}[F] = \mathbb E_{x \sim \mu_0} [ U_1(F(x)) - \log |\det J_F(x)| ].
$$

For data-driven sampling we use the **forward $\mathrm{KL}$**, obtained by exchanging the positions of $F_{\#}\mu_0$ and $\mu_1$ in the $\mathrm{KL}$ divergence:

$$
\begin{aligned}
\mathrm{KL}(\mu_1 \| F_{\#}\mu_0)
& = \int \mu_1(y) \log \frac{\mu_1(y)}{(F_{\#}\mu_0)(y)} \, \mathrm{d}y \\
& = \mathbb E_{y \sim \mu_1} [ U_0(F^{-1}(y)) + \log |\det J_F(F^{-1}(y))| ] + \mathrm{const}.
\end{aligned}
$$

which gives the trainable loss

$$
\mathcal L_{\mathrm{forward}}[F] = \mathbb E_{y \sim \mu_1} [ U_0(F^{-1}(y)) + \log |\det J_F(F^{-1}(y))| ].
$$

In both cases, once $F$ is trained, new samples from $\mu_1$ are generated by pushing fresh samples from $\mu_0$ through $F$.

</details>

## Numerical Experiment

Several end-to-end scripts are provided. Run from the project root:

<details open>
<summary><strong>1. Energy-based normalizing flow (reverse KL)</strong></summary>

[`tests/2D_reverse_KL.py`](tests/2D_reverse_KL.py) trains an `NSF` on a target specified only by an unnormalized energy $U_1(x) = \tfrac{1}{2}|x|^2 + 2\cos x_1$, then evaluates residual mismatch via importance sampling and $\mathrm{ESS}$.

```bash
python -m tests.2D_reverse_KL
```

<p align="center"><img src="tests/2D_reverse_KL.png" alt="reverse-KL test" width="600px"></p>

</details>

<details open>
<summary><strong>2. Data-driven normalizing flow (forward KL)</strong></summary>

[`tests/2D_forward_KL.py`](tests/2D_forward_KL.py) trains an `NSF` on samples from a 3-mode Gaussian mixture — only `u1.samples(N)` is ever called.

```bash
python -m tests.2D_forward_KL
```

<p align="center"><img src="tests/2D_forward_KL.png" alt="forward-KL test" width="600px"></p>

</details>

<details open>
<summary><strong>3. Periodic target with rejuvenation</strong></summary>

[`tests/3D_periodic.py`](tests/3D_periodic.py) trains an `NCSF` on a von-Mises ridge mixture on the 3-torus $[-\pi, \pi]^3$, then runs the full pipeline: importance sampling → resample → `enable_grad` → Langevin rejuvenation.

```bash
python -m tests.3D_periodic
```

<p align="center"><img src="tests/3D_periodic.png" alt="3D periodic test" width="400px"></p>

</details>

<details open>
<summary><strong>4. Annealed Boltzmann generator (4D, two repelling charges)</strong></summary>

[`tests/4D_Boltzmann_generator.py`](tests/4D_Boltzmann_generator.py) trains an `NSF` on the 4D target of two charges in $\mathbb R^2$ confined to a soft annulus and repelling via a regularized 3D Coulomb. A direct flow proposal would have $\mathrm{ESS} \approx 0$, so we anneal: build $M{=}12$ bridge potentials $U_k = (1-c_k)U_0 + c_k U_1$ via `Linear_Combination`, and at each rung run *resample → reverse-KL train → IS → resample → MALA rejuvenation* with the same flow warm-started across rungs. The figure shows the marginal annulus forming (top row) and the joint relative-angle distribution $\Delta\theta = \theta_2 - \theta_1$ on $S^1$ shifting from uniform at $k=0$ to peaked at $\pm\pi$ at $k=12$ — the antipodal Coulomb minimum.

```bash
python -m tests.4D_Boltzmann_generator
```

<p align="center"><img src="tests/4D_Boltzmann_generator.png" alt="4D Boltzmann generator" width="1000px"></p>

</details>
