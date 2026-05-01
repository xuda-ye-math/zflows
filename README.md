# zflows

A small convenience wrapper around [zuko](https://github.com/probabilists/zuko) for energy-based normalizing flows on user-specified rectangular regions.

> This project was developed with [Claude Code](https://claude.com/claude-code).

## Motivation

`zuko` ships excellent normalizing-flow primitives but does not, out of the box:

- restrict a flow to a user-specified box `[a_1, b_1] x ... x [a_d, b_d]` (its splines live on a fixed symmetric box);
- bundle the small bookkeeping needed to train a flow against an unnormalized target potential (source/target potentials, KL loss, importance-sampling diagnostics).

`zflows` fills that gap with a minimal, runnable wrapper.

## What's in the package

- `NSF` — a Neural Spline Flow whose transform is a bijection on `[a, b]^d`, with `a, b` accepting `Tensor` or `list[float]`.
- `Potential`, `Uniform`, `Gaussian` — `nn.Module` potentials returning `U(x)` and supporting `.samples(N)`.
- `compute_ESS`, `compute_ESS_log`, `compute_CESS`, `compute_CESS_log`, `resample` — importance-sampling diagnostics in linear and log space (the log-space variants use `logsumexp` for stability).

## Minimal example

See [`tests/test_2D_reverse_KL.ipynb`](tests/test_2D_reverse_KL.ipynb) (or the script [`tests/test_2D_reverse_KL.py`](tests/test_2D_reverse_KL.py)) for an end-to-end run that trains an NSF to push a 2D Gaussian source onto a target specified only by an unnormalized energy, then evaluates the result via importance sampling and ESS.

![reverse-KL test](tests/test_2D_reverse_KL.png)

## Installation

`zflows` is a pure-Python package; the only runtime dependencies are [`torch`](https://pytorch.org) and [`zuko`](https://github.com/probabilists/zuko), which `pip` will resolve automatically.

**1. Clone the repository.**

```bash
git clone https://github.com/xuda-ye-math/zflows.git
cd zflows
```

**2. Install in editable mode.** This registers the package with your active Python environment while leaving the source tree in place, so any local edits take effect immediately:

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
