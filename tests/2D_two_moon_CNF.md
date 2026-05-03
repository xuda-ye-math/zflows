# 2D test: data-driven training of a Continuous Normalizing Flow on two moons

We train a Continuous Normalizing Flow (CNF / FFJORD) by forward-KL on samples from the classic *two-moons* distribution. The point of this test is twofold: (i) exercise the `CNF` class on a target whose support is **topologically non-trivial** (two interlocking arcs, neither separable along an axis), and (ii) provide a side-by-side reference against the spline-based [`2D_forward_KL.py`](2D_forward_KL.py) test, so the CNF/NSF trade-off is concrete rather than abstract.

- **Source.** $\mu_0 = \mathcal N(0, I_2)$, the standard 2D Gaussian on $\mathbb R^2$. Unlike NSF, CNF lives on the full unbounded $\mathbb R^d$, so there is no rectangular box to specify.
- **Target.** Two interleaving half-circles in $\mathbb R^2$, sampled by drawing $t \sim \mathrm{Uniform}[0, \pi]$ and emitting $(\cos t - 0.5, \sin t - 0.25)$ (upper moon) or $(0.5 - \cos t, 0.25 - \sin t)$ (lower moon), with $\mathcal N(0, 0.1^2 I)$ Gaussian noise added per coordinate. The two moons are point-symmetric through the origin and live in roughly $[-1.5, 1.5]^2$.
- **Flow.** A `CNF` with `frequency=5` time-Fourier features and a `(128, 128, 128)` ODE-drift MLP; trained for 50 epochs of forward-KL with Adam at `lr=1e-2` and a cosine schedule to zero.

## Mathematical background

The forward-KL loss is identical to the spline case — the `Flow` interface hides which bijection sits underneath. With $F = \mathrm{flow}.t()$ the flow's bijection on $\mathbb R^2$,

$$
\mathcal L_{\mathrm{forward}}[F] = \mathbb E_{y \sim \mu_1} \bigl[\, U_0(F^{-1}(y)) + \log |\det J_F(F^{-1}(y))| \,\bigr],
$$

where $\mu_1$ is the empirical two-moons distribution and $U_0(x) = \tfrac{1}{2}|x|^2$ is the Gaussian source potential. Concretely the loop calls `forward_KL(y_batch, source=u0, flow=flow)`, which evaluates $F^{-1}$ and $\log|\det J_{F^{-1}}|$ in one ODE integration via `flow.t().inv.call_and_ladj(y_batch)`.

### What is different about a CNF

A CNF parametrizes $F$ as the time-1 flow of an ODE:

$$
\frac{\mathrm{d} x_t}{\mathrm{d} t} = v_\phi(x_t, t), \qquad x_0 = x, \quad F(x) = x_1, \qquad t \in [0, 1].
$$

The drift $v_\phi$ is an MLP whose extra inputs are $2 \cdot \mathtt{freqs}$ Fourier features in $t$ (`cos(k*pi*t), sin(k*pi*t)` for $k = 1, \dots, \mathtt{freqs}$). The log-determinant comes from the Jacobi formula along the ODE,

$$
\log |\det J_F(x)| = \int_0^1 \mathrm{tr}\bigl(\nabla_x v_\phi(x_t, t)\bigr)\, \mathrm{d}t,
$$

evaluated either *exactly* (an augmented ODE that costs $O(d)$ extra evaluations per step — `exact=True`, the default in `zflows.CNF`) or via the *Hutchinson trace estimator* (one extra ODE channel, unbiased but stochastic — `exact=False`). The integration uses an adaptive `dopri5` solver (Dormand–Prince 5(4)) under the hood with the user-specified `atol` / `rtol`.

The practical consequence is that *every* call to $F$ or $F^{-1}$ (forward sample, inverse sample, log-det) is one ODE integration with a data-dependent number of substeps — there is no closed-form alternative.

## Implementation and execution

The full pipeline lives in [`2D_two_moon_CNF.py`](2D_two_moon_CNF.py). Run from the project root:

```bash
python -m tests.2D_two_moon_CNF
```

Pointers into the script:

- imports & device setup: [`2D_two_moon_CNF.py:1-8`](2D_two_moon_CNF.py#L1-L8)
- source ($\mathcal N(0, I_2)$) and `two_moons_samples` helper: [`2D_two_moon_CNF.py:10-23`](2D_two_moon_CNF.py#L10-L23)
- CNF init: [`2D_two_moon_CNF.py:26`](2D_two_moon_CNF.py#L26)
- training parameters & cosine LR schedule: [`2D_two_moon_CNF.py:28-37`](2D_two_moon_CNF.py#L28-L37)
- training loop (mini-batched forward-KL): [`2D_two_moon_CNF.py:39-58`](2D_two_moon_CNF.py#L39-L58)
- plotting (source / pushforward / target): [`2D_two_moon_CNF.py:60-90`](2D_two_moon_CNF.py#L60-L90)

<p align="center"><img src="2D_two_moon_CNF.png" alt="2D two-moons CNF test" width="700px"></p>

## Analysis

**Pushforward fidelity.** The trained CNF cleanly reproduces the two-moons topology: the green pushforward $F_{\#} \mu_0$ in the middle panel matches the red ground-truth target on the right both in shape and in noise spread. The Gaussian source in the left panel has been bent into two interlocking arcs — a deformation that no axis-aligned bijection can achieve in a single transform, since it requires *opposite* curvature in different regions of the plane.

**Why this target is a CNF strength.** The two moons cannot be separated by axis-aligned splits: any horizontal or vertical line crosses both arcs. An NSF would need several stacked autoregressive transforms, each with permutation-invariant coupling, to reach the same fidelity, because each individual spline transform deforms the plane only along one coordinate at a time. The CNF, by contrast, learns a smooth velocity field $v_\phi(x, t)$ that bends *both* coordinates simultaneously and continuously — the integration time $t$ acts as an extra "depth" dimension that an autoregressive flow does not have.

**CNF vs. NSF — a concrete comparison.** Both flow classes solve the same forward-KL problem and expose the same `Flow` interface in `zflows`. The trade-off comes down to:

| | `NSF` (autoregressive splines) | `CNF` (FFJORD) |
|---|---|---|
| Forward $F(x)$ | one MLP pass per spline transform, **closed-form** | one adaptive-ODE integration (typically 20–200 dopri5 substeps) |
| Inverse $F^{-1}(y)$ | bisection or analytic per coord, **closed-form O(d)** | same ODE integrated in reverse |
| Log-det $\log\|\det J_F\|$ | sum of per-coord spline-derivative logs, **free** | augmented ODE channel (`exact=True`) or Hutchinson estimate |
| Domain | rectangular box $[a, b]^d$, soft-extends outside | full $\mathbb R^d$ natively |
| Smoothness of pushforward | $C^1$ at spline knots | $C^\infty$ in $x$ (smooth ODE flow) |
| Topologically twisted targets | needs many stacked transforms | naturally handled in one flow |
| Late-training wall-clock | constant per epoch | grows as drift sharpens (more solver substeps) |
| Importance-sampling sweep | one closed-form pass over $N$ particles | one ODE integration per particle batch — **50–500× slower** |
| MALA / Langevin rejuvenation | gradient autograds through MLPs | gradient routes through the adjoint ODE — costly |

**Implication for energy-based sampling.** Because reverse-KL training, importance sampling, and MALA rejuvenation each require many forward / inverse / log-det evaluations per step, the *closed-form* nature of NSF makes it the right default for the *propose → reweight → resample → rejuvenate* pipeline that the other tests in this folder ([`2D_reverse_KL.md`](2D_reverse_KL.md), [`3D_periodic.md`](3D_periodic.md), [`4D_Boltzmann_generator.md`](4D_Boltzmann_generator.md)) all use. CNFs become competitive when

1. the target's *topology* genuinely requires a smooth, non-axis-aligned deformation that splines fit only with many stacked transforms,
2. the dimension is high enough that the autoregressive conditioner becomes the bottleneck (typically $d \gtrsim 50$),
3. an equivariance constraint on the bijection is needed (e.g. $\mathrm{E}(n)$- or permutation-equivariant drifts are easy on a CNF, awkward on a spline), or
4. the training loss is flow-matching / score-matching, which is CNF-native and bypasses the log-det entirely.

For 2D–4D energy-based sampling on Cartesian or toroidal boxes, NSF / NCSF win on every axis that matters in a typical pipeline. The two-moons test exists precisely to mark the boundary: it is the *smallest* case where the CNF's smooth-deformation advantage shows up in the figure.
