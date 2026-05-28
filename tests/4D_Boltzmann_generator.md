# 4D test: an annealed Boltzmann generator for two repelling charges

We sample from a 4-dimensional energy-based target via a *Boltzmann generator* — a normalizing flow trained against a sequence of bridge potentials that anneal from a tractable source to the physical target. This is the smallest non-trivial example that exhibits the hallmarks of real molecular Boltzmann-generator workloads: a continuous symmetry the flow has to discover, a hard repulsive barrier, and a particle-exchange symmetry.

- **Source.** $\mu_0 = \mathcal N(0, I_4)$, with potential $U_0(x) = \frac{1}{2}|x|^2$.
- **Target.** Two unit charges in $\mathbb R^2$, each soft-confined to a ring of radius $r_0=2$, with a regularized 3D Coulomb interaction:
$$
U_1(x_1, x_2) \;=\; a\bigl[\,(|x_1|^2 - r_0^2)^2 + (|x_2|^2 - r_0^2)^2\,\bigr] \;+\; \frac{q^2}{\sqrt{|x_1 - x_2|^2 + \varepsilon^2}}, \qquad x = (x_1, x_2) \in \mathbb R^4,
$$
with $a=1$, $q^2=4$, $r_0=2$, $\varepsilon=10^{-3}$. The regularization keeps gradients finite at $x_1 = x_2$ while letting the target behave like a true Coulomb away from collisions.
- **Symmetries.** $U_1$ is invariant under (i) rigid 2D rotations of $(x_1, x_2)$ and (ii) particle exchange $(x_1, x_2) \leftrightarrow (x_2, x_1)$. The flow has to learn both from data, since `NSF` on Cartesian $[a,b]^4$ has neither built in.
- **Why anneal.** Training reverse KL directly from $\mu_0$ to $\mu_1$ is an out-of-distribution problem at initialization (the Gaussian and the repulsive ring barely overlap), so importance weights collapse and the flow gets stuck. The annealing schedule $U_k = (1 - c_k) U_0 + c_k U_1$ with $c_k = k/M$ keeps each consecutive pair $(\mu_{k-1}, \mu_k)$ close enough that incremental training works.

## Mathematical background

Both potentials are `Potential` subclasses returning $U(x)$ for a batch of points. The Coulomb term is the only piece that needs the regularizer; everything else is closed-form polynomial.

### The annealing pipeline

We build a ladder of $M+1$ rungs $c_0 = 0, c_1 = 1/M, \dots, c_M = 1$ and a sequence of bridge potentials
$$
U_k(x) = (1 - c_k)\,U_0(x) + c_k\,U_1(x), \qquad k = 0, 1, \dots, M,
$$
with $\mu_k \propto \exp(-U_k)$.

**Setup, once before the loop.** Two `Linear_Combination` bridge potentials are constructed up front and reused at every rung:

- `U_prev = Linear_Combination([u_target, u_source])` — denominator in the importance-weight, $U_{k-1}$;
- `U_curr = Linear_Combination([u_target, u_source])` — numerator in the importance-weight + reverse-KL training target + rejuvenation target, $U_k$.

The hoisted `loss_fn = zflows.loss.loss_compile(reverse_KL, U_curr, F)` then captures the bridge potential `U_curr` *by reference* — when its coefficients are mutated in step (0) below, the same compiled closure reads the new mix on the next forward (Dynamo guards on the Python float values in `U_curr.coeffs` and re-specialises on guard miss; `set_cache_size_limit(32)` gives enough room for all $M = 12$ rung specs). `F = flow.t()` is similarly captured **before the rung loop starts** — the lazy `Transform` machinery re-reads `flow.parameters()` via attribute access on every forward, so updates from `optimizer.step()` propagate without rebuilding `F`. Bridge interpolation between two different potentials is *not* the high-→low-temperature tempering that `loss_compile_beta` is designed for, so `loss_compile` (with default `mode='default'`, no CUDA graph) is the right helper here.

Each rung then does six things, in order (step 0 retunes the bridge potentials; steps 1–5 are the standard propose → reweight → resample → rejuvenate cycle):

0. **Retune the bridge potentials.** `U_prev.set_coeffs([c_p, 1 - c_p])` and `U_curr.set_coeffs([c_k, 1 - c_k], enable_grad=True, enable_eval=True)`. `set_coeffs` overwrites `self.coeffs` in place as a plain `list[float]` *and* drops the previous rung's compiled `.grad(x)` / `.eval(x)` fast paths so they cannot return stale-coefficient values; the two flags then rebuild both compiled paths against the new mix in the same call, so MALA in step (4) sees a freshly specialised gradient and energy. `U_prev` is only consumed by `importance_weights_log` through its plain forward, so its fast paths are deliberately left off.
1. **Resample.** Draw a working set $x_{\mathrm{train}, k-1}$ of size $N_{\mathrm{train}}$ uniformly with replacement from the previous rung's particle cloud $x_{\mathrm{valid}, k-1} \sim \mu_{k-1}$.
2. **Train.** Update the *single* shared flow to minimize reverse KL from $\mu_{k-1}$ to $\mu_k$, treating $x_{\mathrm{train}, k-1}$ as samples from the source. The $U_{k-1}(x)$ term is parameter-independent and drops out of the gradient, so reverse KL against $U_k$ alone is the correct loss — and since $U_k$ is literally `U_curr` (the linear combination we just retuned), the hoisted `loss_fn(x_batch)` is the entire training step. The first batch of the *first* rung pays a one-time `torch.compile` cost (Dynamo trace + Inductor lowering); the first batch of every subsequent rung pays only a partial retrace (the new `U_curr.coeffs` floats fail the cached guards) before falling back into the fused-kernel fast path. After the first batch of each rung, per-step training time drops sharply.
3. **Importance sample.** Push the full validation set $x_{\mathrm{valid}, k-1}$ through $F$ to get proposals $y$ and log-weights $\log w = -U_k(y) + U_{k-1}(x_{\mathrm{valid}, k-1}) + \log|\det J_F|$ via `importance_weights_log(samples, source=U_prev, target=U_curr, F=flow.t())`. Report the ESS as a self-test for this rung.
4. **Resample by weight.** Multinomial draw to convert the weighted cloud into an equally-weighted cloud $\tilde y$.
5. **Rejuvenate.** Run MALA (Metropolis-adjusted Langevin) against $U_k$ to break duplicate particles and remove residual proposal bias. The compiled fast paths needed here — `U_curr.grad(x)` for the gradient $\nabla U_k$ in the Langevin proposal, and `U_curr.eval(x)` for the energy $U_k$ in the accept/reject step — were already rebuilt at the top of the rung by `set_coeffs(enable_grad=True, enable_eval=True)`, so each MALA iteration is two fused kernel calls instead of two autograd-graph rebuilds. Output $x_{\mathrm{valid}, k}$ for the next rung.

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

- imports + `suppress_warnings` / `set_cache_size_limit(32)`: [`4D_Boltzmann_generator.py:1–18`](4D_Boltzmann_generator.py#L1-L18)
- source (4D Gaussian) and target ($U_1$ class): [`4D_Boltzmann_generator.py:21–55`](4D_Boltzmann_generator.py#L21-L55)
- `.pth` cache: training is skipped on subsequent runs by loading `4D_Boltzmann_generator.pth`: [`4D_Boltzmann_generator.py:57–63`](4D_Boltzmann_generator.py#L57-L63)
- training branch — flow init, validation cloud, ladder $c_k$: [`4D_Boltzmann_generator.py:65–93`](4D_Boltzmann_generator.py#L65-L93)
- captured `F = flow.t()` (reused across rungs): [`4D_Boltzmann_generator.py:95–100`](4D_Boltzmann_generator.py#L95-L100)
- `U_prev` / `U_curr` bridge potentials built once + hoisted `loss_fn = loss_compile(reverse_KL, U_curr, F)`: [`4D_Boltzmann_generator.py:102–120`](4D_Boltzmann_generator.py#L102-L120)
- per-rung loop (6 steps: `set_coeffs` retune → resample → train → IS → resample → MALA): [`4D_Boltzmann_generator.py:122–180`](4D_Boltzmann_generator.py#L122-L180)
- saving the cache: [`4D_Boltzmann_generator.py:182–188`](4D_Boltzmann_generator.py#L182-L188)
- ESS history printout: [`4D_Boltzmann_generator.py:190–194`](4D_Boltzmann_generator.py#L190-L194)
- two-row visualization (Cartesian + polar): [`4D_Boltzmann_generator.py:196–246`](4D_Boltzmann_generator.py#L196-L246)

The first invocation runs the annealing ($M = 12$ rungs of training + IS + MALA) and saves a `.pth` cache containing `x_valid_history` (M+1 snapshots) and `ess_history` (M floats). Subsequent invocations load the cache and skip directly to the visualization.

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
- The $k=4$ histogram is still close to flat. At this rung the bridge $U_4 = \frac{2}{3} U_0 + \frac{1}{3} U_1$ has very weak Coulomb compared to the still-strong harmonic source, so repulsion barely shifts the angular density.
- At $k=8$ the density becomes visibly anisotropic: the bars near $\Delta\theta = 0$ are slightly suppressed and a faint hump grows on the left. The bridge is now Coulomb-dominated.
- At $k=12$ (pure target) the suppression at $\Delta\theta \approx 0$ is clear and a tall peak sits at $\Delta\theta = \pm\pi$ (left side of the polar plot). The width of the peak is set by the temperature ratio $q^2 / (a r_0^4) = 4/16 = 1/4$ — repulsion is strong enough to make the antipodal configuration the *dominant* one but not strong enough to lock it. We're in the warm-Wigner-crystal regime, exactly where Boltzmann generators are most useful (slow MCMC, multimodal target, no analytic samples).

**ESS as a self-test.** The per-rung ESS values printed by the script quantify the visual story:

| $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ESS | 0.78 | 0.78 | 0.92 | 0.96 | 0.97 | 0.97 | 0.96 | 0.97 | 0.96 | 0.97 | 0.96 | 0.97 |

The first two rungs sit at $\sim 0.78$ — the flow is still discovering the rotational symmetry while the bridge potential is mostly the Gaussian source — then a single jump to $\sim 0.92$ at $k=3$ once the Coulomb term starts mattering, and a plateau in the $0.96$–$0.97$ band from $k=4$ onward. Each rung shifts the distribution by a small enough amount that the flow's pushforward is a usable proposal, even though the *cumulative* shift from $\mu_0$ to $\mu_M$ would have ESS $\approx 0$ as a single-shot proposal. That's the entire point of the annealed schedule.

**What this test demonstrates about `zflows`.** This is the smallest example that exercises every component end-to-end:

- `Linear_Combination` builds the two bridge potentials *once* outside the loop; `set_coeffs([c_k, 1 - c_k], enable_grad=True, enable_eval=True)` then retunes the mix in place each rung, dropping the previous rung's compiled fast paths and rebuilding them against the new coefficients in a single call — so the same Python object survives all $M = 12$ rungs without leaking a fresh compile artifact per step;
- the built-in `reverse_KL` is the per-rung loss applied directly to the combined potential `U_curr` (the source-energy term drops out of the gradient automatically), and `zflows.loss.loss_compile(reverse_KL, U_curr, F)` is **hoisted outside the loop** so a single closure handles every rung — Dynamo respecialises on `U_curr.coeffs` guard misses, which fits comfortably under the `set_cache_size_limit(32)` ceiling;
- `importance_weights_log` is the one-call IS reweighting against the just-trained flow, ready to feed into `compute_ESS_log` for the per-rung diagnostic;
- `resample` converts weighted clouds to equal-weight ones for the next rung;
- `Potential.enable_grad()` / `.enable_eval()` provide the compiled gradient and forward fast paths that MALA (`rejuvenation` with `adjust=True`) uses for the proposal and the accept/reject step respectively — invoked via `set_coeffs(enable_*=True)` at the top of each rung;
- `NSF` provides the spline bijection on the rectangular box, with one set of parameters re-used across all $M$ rungs (warm-start fine-tuning);
- `zflows.utils.suppress_warnings()` and `set_cache_size_limit(32)` silence the routine Triton / Inductor / Dynamo log noise and give Dynamo enough cache headroom for the hoisted loss closure ($M = 12$ guard specs) plus the per-rung `enable_grad` / `enable_eval` specialisations.

The full pipeline is the *propose → reweight → resample → rejuvenate* loop that drives every modern flow-augmented SMC sampler — packaged here in the smallest dimension where you can still see it work.
