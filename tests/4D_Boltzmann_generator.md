# 4D test: an annealed Boltzmann generator for two repelling charges

We sample from a 4-dimensional energy-based target via a *Boltzmann generator* — a normalizing flow trained against a sequence of bridge potentials that anneal from a tractable source to the physical target. This is the smallest non-trivial example that exhibits the hallmarks of real molecular Boltzmann-generator workloads: a continuous symmetry the flow has to discover, a hard repulsive barrier, and a particle-exchange symmetry.

- **Source.** $\mu_0 = \mathcal N(0, I_4)$, with potential $U_0(x) = \tfrac{1}{2}|x|^2$.
- **Target.** Two unit charges in $\mathbb R^2$, each soft-confined to a ring of radius $r_0=2$, with a regularized 3D Coulomb interaction:
$$
U_1(x_1, x_2) \;=\; a\bigl[\,(|x_1|^2 - r_0^2)^2 + (|x_2|^2 - r_0^2)^2\,\bigr] \;+\; \frac{q^2}{\sqrt{|x_1 - x_2|^2 + \varepsilon^2}}, \qquad x = (x_1, x_2) \in \mathbb R^4,
$$
with $a=1$, $q^2=4$, $r_0=2$, $\varepsilon=10^{-3}$. The regularization keeps gradients finite at $x_1 = x_2$ while letting the target behave like a true Coulomb away from collisions.
- **Symmetries.** $U_1$ is invariant under (i) rigid 2D rotations of $(x_1, x_2)$ and (ii) particle exchange $(x_1, x_2) \leftrightarrow (x_2, x_1)$. The flow has to learn both from data, since `NSF` on Cartesian $[a,b]^4$ has neither built in.
- **Why anneal.** Training reverse-KL directly from $\mu_0$ to $\mu_1$ is an out-of-distribution problem at initialization (the Gaussian and the repulsive ring barely overlap), so importance weights collapse and the flow gets stuck. The annealing schedule $U_k = (1 - c_k) U_0 + c_k U_1$ with $c_k = k/M$ keeps each consecutive pair $(\mu_{k-1}, \mu_k)$ close enough that incremental training works.

## Mathematical background

Both potentials are `Potential` subclasses returning $U(x)$ for a batch of points. The Coulomb term is the only piece that needs the regularizer; everything else is closed-form polynomial.

### The annealing pipeline

We build a ladder of $M+1$ rungs $c_0 = 0, c_1 = 1/M, \dots, c_M = 1$ and a sequence of bridge potentials
$$
U_k(x) = (1 - c_k)\,U_0(x) + c_k\,U_1(x), \qquad k = 0, 1, \dots, M,
$$
with $\mu_k \propto \exp(-U_k)$. Each rung's training step does five things, in order:

1. **Resample.** Draw a working set $x_{\mathrm{train}, k-1}$ of size $N_{\mathrm{train}}$ uniformly with replacement from the previous rung's particle cloud $x_{\mathrm{valid}, k-1} \sim \mu_{k-1}$.
2. **Train.** Update the *single* shared flow $F$ to minimize reverse-KL from $\mu_{k-1}$ to $\mu_k$, treating $x_{\mathrm{train}, k-1}$ as samples from the source. The $U_{k-1}(x)$ term is parameter-independent and drops out of the gradient, so `reverse_KL(x, target=U_k, flow=F)` is the correct loss as-is.
3. **Importance sample.** Push the full validation set $x_{\mathrm{valid}, k-1}$ through $F$ to get proposals $y$ and log-weights $\log w = -U_k(y) + U_{k-1}(x_{\mathrm{valid}, k-1}) + \log|\det J_F|$. Report the ESS as a self-test for this rung.
4. **Resample by weight.** Multinomial draw to convert the weighted cloud into an equally-weighted cloud $\tilde y$.
5. **Rejuvenate.** Run MALA (Metropolis-adjusted Langevin) against $U_k$ to break duplicate particles and remove residual proposal bias. Output $x_{\mathrm{valid}, k}$ for the next rung.

Only the validation cloud $x_{\mathrm{valid}, k}$ is carried across rungs; the optimizer state is re-used so the flow warm-starts each step.

### ESS along the ladder

At every rung we measure the Effective Sample Size of the flow's proposal,
$$
\mathrm{ESS}_k \;=\; \frac{\bigl(\sum_i w_i^{(k)}\bigr)^2}{N_{\mathrm{valid}}\,\sum_i \bigl(w_i^{(k)}\bigr)^2}, \qquad \log w^{(k)} = -U_k\!\bigl(F(x)\bigr) + U_{k-1}(x) + \log|\det J_F(x)|.
$$
This is the first-order diagnostic of how well $F$ moves $\mu_{k-1}$ to $\mu_k$. A pipeline that worked well at every rung gives $\mathrm{ESS}_k$ uniformly in the $0.4{-}0.9$ range. Catastrophic mode-mismatch (e.g. trying to jump from $\mu_0$ straight to $\mu_M$) collapses $\mathrm{ESS}$ to $\sim 0$, which is exactly what annealing is designed to avoid.

## Implementation and execution

The full pipeline lives in [`4D_Boltzmann_generator.py`](4D_Boltzmann_generator.py). Run from the project root:

```bash
python -m tests.4D_Boltzmann_generator
```

Pointers into the script:

- imports, env vars, device & seed: [`4D_Boltzmann_generator.py:1–17`](4D_Boltzmann_generator.py#L1-L17)
- source (4D Gaussian) and target ($U_1$ class): [`4D_Boltzmann_generator.py:19–48`](4D_Boltzmann_generator.py#L19-L48)
- `.pt` cache: training is skipped on subsequent runs by loading `4D_Boltzmann_generator.pt`: [`4D_Boltzmann_generator.py:50–57`](4D_Boltzmann_generator.py#L50-L57)
- training branch — flow init, validation cloud, ladder $c_k$: [`4D_Boltzmann_generator.py:58–86`](4D_Boltzmann_generator.py#L58-L86)
- per-rung loop (5 steps: resample → train → IS → resample → MALA): [`4D_Boltzmann_generator.py:92–136`](4D_Boltzmann_generator.py#L92-L136)
- saving the cache: [`4D_Boltzmann_generator.py:138–144`](4D_Boltzmann_generator.py#L138-L144)
- ESS history printout: [`4D_Boltzmann_generator.py:146–150`](4D_Boltzmann_generator.py#L146-L150)
- two-row visualization (Cartesian + polar): [`4D_Boltzmann_generator.py:152–218`](4D_Boltzmann_generator.py#L152-L218)

The first invocation runs the annealing ($M = 12$ rungs of training + IS + MALA) and saves a `.pt` cache containing `x_valid_history` (M+1 snapshots) and `ess_history` (M floats). Subsequent invocations load the cache and skip directly to the visualization.

### Visualizing the annealed cloud

The figure below plots the validation particles at $k = 0, 4, 8, 12$ in two rows.

- **Row 1 (Cartesian).** Both particles' $(x, y)$ positions are scattered, in different colours. The Gaussian blob at $k=0$ should collapse onto a ring of radius $r_0 = 2$ as $k$ grows — that's the *marginal* signature of confinement.
- **Row 2 (S$^1$ polar histogram).** The relative angle $\Delta\theta = \theta_2 - \theta_1 \in (-\pi, \pi]$, with $\theta_i = \mathrm{atan2}(y_i, x_i)$. This is the *joint* signal that the rotational symmetry of row 1 hides:
  - At $k=0$, particles are independent and $\Delta\theta$ is uniform on the circle. Equivalently, $\lvert\Delta\theta\rvert$ has mean $\pi/2 \approx 90^\circ$.
  - As $c_k \to 1$, Coulomb repulsion drives the particles antipodal, so $\Delta\theta$ concentrates near $\pm \pi \approx 180^\circ$ (a tall bar at the *left* of the polar plot).

<p align="center"><img src="4D_Boltzmann_generator.png" alt="4D Boltzmann generator" width="900px"></p>

## Analysis

**Marginal annulus formation (row 1).** The Gaussian source at $k=0$ has mass concentrated near the origin, mostly inside the eventual target ring. By $k=4$ the cloud has flattened: a partial ring at $r \approx r_0$ with substantial mass still in the interior. By $k=8$ the interior is mostly emptied and the ring is sharp; by $k=12$ both colour clouds densely fill the ring with thickness controlled by $a$ (here $\sim r_0 / \sqrt{2 a}$). The blue and red distributions are *visually identical*, which they should be — the joint $\mu_1$ is invariant under particle exchange, so the marginal of particle 1 equals that of particle 2 (rotation-invariant uniform on the ring).

**Joint angular structure (row 2).** This is where the actual physics shows up.

- At $k=0$ the polar histogram is essentially flat — particles are uncorrelated (i.i.d. 4D Gaussians factorize as two i.i.d. 2D Gaussians, whose angular difference is uniform on $S^1$). $|\Delta\theta|$ has mean $\pi/2$.
- The $k=4$ histogram is still close to flat. At this rung the bridge $U_4 = \tfrac{2}{3} U_0 + \tfrac{1}{3} U_1$ has very weak Coulomb compared to the still-strong harmonic source, so repulsion barely shifts the angular density.
- At $k=8$ the density becomes visibly anisotropic: the bars near $\Delta\theta = 0$ are slightly suppressed and a faint hump grows on the left. The bridge is now Coulomb-dominated.
- At $k=12$ (pure target) the suppression at $\Delta\theta \approx 0$ is clear and a tall peak sits at $\Delta\theta = \pm\pi$ (left side of the polar plot). The width of the peak is set by the temperature ratio $q^2 / (a r_0^4) = 4/16 = 1/4$ — repulsion is strong enough to make the antipodal configuration the *dominant* one but not strong enough to lock it. We're in the warm-Wigner-crystal regime, exactly where Boltzmann generators are most useful (slow MCMC, multimodal target, no analytic samples).

**ESS as a self-test.** The per-rung ESS values printed by the script (for the cached run, climbing from $\sim 0.74$ at $k=1$ to $\sim 0.97$ at $k=12$) quantify the visual story: each rung shifts the distribution by a small enough amount that the flow's pushforward is a usable proposal, even when the *cumulative* shift from $\mu_0$ to $\mu_M$ would have ESS $\approx 0$ as a single-shot proposal. That's the entire point of the annealed schedule.

**What this test demonstrates about `zflows`.** This is the smallest example that exercises every component end-to-end:

- `Linear_Combination` builds the bridge potentials with a single coefficient parameter that can be reused at every rung;
- `reverse_KL` is the per-rung loss without any modification (the source-energy term drops out automatically);
- `compute_ESS_log` is the per-rung diagnostic;
- `resample` converts weighted clouds to equal-weight ones for the next rung;
- `Potential.enable_grad()` provides the compiled gradient that `rejuvenation` (MALA via overdamped Langevin) needs to break duplicate particles and remove residual flow bias;
- `NSF` provides the spline bijection on the rectangular box, with one set of parameters re-used across all $M$ rungs (warm-start fine-tuning).

The full pipeline is the *propose → reweight → resample → rejuvenate* loop that drives every modern flow-augmented SMC sampler — packaged here in the smallest dimension where you can still see it work.
