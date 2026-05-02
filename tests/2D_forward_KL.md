# 2D test: forward-$\mathrm{KL}$ training of a normalizing flow

We train a Neural Spline Flow (NSF) $F$ in the **data-driven** setting: only samples from the target $\mu_1$ are given, and we minimize the **forward $\mathrm{KL}$** $\mathrm{KL}(\mu_1 \,\|\, F_{\#}\mu_0)$.

- **Source:** $\mu_0 = \mathcal N(0, I)$ on $\mathbb R^2$ (standard Gaussian).
- **Target:** $\mu_1$ is a 3-mode diagonal Gaussian mixture on $\mathbb R^2$ with modes at the vertices of an equilateral triangle of radius $2$ and per-component variance $0.15$.
- **Flow:** $F$ is an NSF with bijection on $[-4, 4]^2 \subset \mathbb R^2$.
- **Training data:** a single batch of $N$ samples drawn once from $\mu_1$ (no access to its potential at training time).

## Mathematical background

$\mu_0$ is a standard Gaussian, available both as a closed-form density and as a sampler. $\mu_1$ is a 3-mode mixture — we use it both as a sampler (training) and, later, as a visual ground truth.

$$
\mu_1(y) \propto \sum_{k=1}^{3} w_k \, \mathcal N\bigl(y \,\big|\, m_k, \mathrm{diag}(\sigma_k^2)\bigr), \qquad \{w_k\} = (1,1,1).
$$

Note that the *training loop never calls* `u1.forward(...)`; only `u1.samples(N)` is used, mimicking a true data-driven setup.

### Forward-$\mathrm{KL}$ objective

Let $\nu = F_{\#}\mu_0$ denote the pushforward density. Its density at $y$, by the change-of-variables formula, is
$$
\nu(y) = \frac{\mu_0(F^{-1}(y))}{|\det J_F(F^{-1}(y))|}, \qquad x = F^{-1}(y).
$$
The forward $\mathrm{KL}$ is the dual of the reverse one — we swap the order of $\mu_1$ and $\nu$:
$$
\begin{aligned}
\mathrm{KL}(\mu_1 \,\|\, \nu)
& = \int_{\mathbb R^2} \mu_1(y) \log \frac{\mu_1(y)}{\nu(y)} \, \mathrm{d}y \\
& = \mathbb E_{y \sim \mu_1} \Bigl[ U_0(F^{-1}(y)) + \log |\det J_F(F^{-1}(y))| \Bigr] + \mathrm{const}.
\end{aligned}
$$
Dropping the parameter-free constant gives the trainable loss
$$
\mathcal L_{\mathrm{forward}}[F] = \mathbb E_{y \sim \mu_1} \Bigl[ U_0(F^{-1}(y)) + \log |\det J_F(F^{-1}(y))| \Bigr].
$$
Equivalently, this is just the negative log-likelihood of the data under the flow density $\nu$. The package exposes it as `forward_KL(y, source, flow)`.

## Implementation and execution

The full pipeline lives in [`2D_forward_KL.py`](2D_forward_KL.py). Run from the project root:

```bash
python -m tests.2D_forward_KL
```

Pointers into the script:

- imports & device setup: [`2D_forward_KL.py:1–8`](2D_forward_KL.py#L1-L8)
- source (standard Gaussian) and target (3-mode mixture): [`2D_forward_KL.py:10–26`](2D_forward_KL.py#L10-L26)
- flow init: [`2D_forward_KL.py:29`](2D_forward_KL.py#L29)
- training parameters: [`2D_forward_KL.py:31–35`](2D_forward_KL.py#L31-L35)
- training data sampled from $\mu_1$, training loop (mini-batched forward-KL): [`2D_forward_KL.py:37–59`](2D_forward_KL.py#L37-L59)
- plotting (source / pushforward / target ground truth): [`2D_forward_KL.py:61–91`](2D_forward_KL.py#L61-L91)

<p align="center"><img src="2D_forward_KL.png" alt="forward-KL test" width="700px"></p>

## Analysis

The loss decreases monotonically from $\sim 1.7$ down to $\sim 0.4$, and the pushforward panel clearly recovers all three modes. A few qualitative observations:

- **Mode coverage.** Forward $\mathrm{KL}$ is *mass-covering* (zero-avoiding): $\mathrm{KL}(\mu_1 \,\|\, \nu)$ blows up on any region where $\mu_1 > 0$ but $\nu \to 0$. This biases the flow toward placing mass on every cluster of training data, so all three modes survive — no mode collapse, in contrast to what reverse $\mathrm{KL}$ can suffer from.
- **Mode width.** The pushforward modes are visibly *broader* than the ground-truth modes, and faint *bridges* connect them. This is the flip side of mass-covering: the flow extends mass into low-density gaps to keep $\nu$ strictly positive everywhere $\mu_1$ has support, even at the cost of putting mass where $\mu_1$ is essentially zero. With a continuous diffeomorphism $F$ acting on a connected base $\mu_0$, the pushforward $\nu$ is also connected, so it cannot exactly match a multi-modal target with disjoint support; the bridges are an unavoidable artifact of fitting a connected proposal to a disconnected target with forward $\mathrm{KL}$.
- **Implication.** If sharp modes matter (e.g.\ rare-event sampling), one would either train for longer / with more capacity, switch to reverse $\mathrm{KL}$ (energy-based, *mode-seeking*; see [`2D_reverse_KL.md`](2D_reverse_KL.md)), or use a multi-modal base distribution to break the connectedness constraint.
