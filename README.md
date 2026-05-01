# zflows

A small convenience wrapper around [zuko](https://github.com/probabilists/zuko) for normalizing flows, with first-class support for energy-based sampling.

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
g = u1.grad(theta)        # [N, d], no .requires_grad_ on theta needed
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

> **Note.** For best visual effects, open [`README.ipynb`](README.ipynb) in VS Code.

## Minimal examples

Three end-to-end scripts are provided. Run from the project root:

**Energy-based (reverse $\mathrm{KL}$).** [`tests/2D_reverse_KL.py`](tests/2D_reverse_KL.py) trains an `NSF` on a target specified only by an unnormalized energy $U_1(x) = \tfrac{1}{2}|x|^2 + 2\cos x_1$, then evaluates residual mismatch via importance sampling and $\mathrm{ESS}$.

```bash
python -m tests.2D_reverse_KL
```

![reverse-KL test](tests/2D_reverse_KL.png)

**Data-driven (forward $\mathrm{KL}$).** [`tests/2D_forward_KL.py`](tests/2D_forward_KL.py) trains an `NSF` on samples from a 3-mode Gaussian mixture — only `u1.samples(N)` is ever called.

```bash
python -m tests.2D_forward_KL
```

![forward-KL test](tests/2D_forward_KL.png)

**Periodic target with rejuvenation.** [`tests/3D_periodic.py`](tests/3D_periodic.py) trains an `NCSF` on a von-Mises ridge mixture on the 3-torus $[-\pi, \pi]^3$, then runs the full pipeline: importance sampling → resample → `enable_grad` → Langevin rejuvenation.

```bash
python -m tests.3D_periodic
```

![3D periodic test](tests/3D_periodic.png)
