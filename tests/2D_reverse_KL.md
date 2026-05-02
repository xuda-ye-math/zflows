# 2D test: reverse-$\mathrm{KL}$ training of a normalizing flow

We train a Neural Spline Flow (NSF) $F$ such that the pushforward of a 2D Gaussian source $\mu_0$ approximates a target $\mu_1$ on $\mathbb R^2$ specified only by an unnormalized energy $U_1$.

- **Source:** $\mu_0 = \mathcal N(0, I)$ on $\mathbb R^2$, with potential $U_0(x) = \tfrac{1}{2}|x|^2$.
- **Target:** $\mu_1 \propto \exp(-U_1)$ with $U_1(x) = \tfrac{1}{2}|x|^2 + 2\cos x_1$.
- **Flow:** $F$ is an NSF with bijection on the box $[-4, 4]^2 \subset \mathbb R^2$.

## Mathematical background

Both `Potential` subclasses return $U(x)$ for a batch of points $x \in \mathbb R^2$. By definition,
$$
\mu_0(x) \propto \exp(-U_0(x)), \qquad \mu_1(y) \propto \exp(-U_1(y)).
$$
We never need the normalizing constant of $\mu_1$, only $U_1$.

### Reverse-$\mathrm{KL}$ objective

Let $\nu = F_{\#} \mu_0$ be the pushforward density. With $y = F(x)$ and $x \sim \mu_0$, the change-of-variables formula gives
$$
\log \nu(y) = \log \mu_0(x) - \log |\det J_F(x)| = -U_0(x) - \log |\det J_F(x)|.
$$
The reverse $\mathrm{KL}$ divergence is
$$
\begin{aligned}
\mathrm{KL}(\nu \,\|\, \mu_1)
& = \int_{\mathbb R^2} \nu(y) \log \frac{\nu(y)}{\mu_1(y)} \, \mathrm{d}y \\
& = \mathbb E_{x \sim \mu_0} \Bigl[ \log \nu(F(x)) - \log \mu_1(F(x)) \Bigr] \\
& = \mathbb E_{x \sim \mu_0} \Bigl[ U_1(F(x)) - U_0(x) - \log |\det J_F(x)| \Bigr] + \mathrm{const}.
\end{aligned}
$$
Dropping the (parameter-independent) constant gives the trainable loss, estimated by Monte Carlo over a batch $\{x_i\}_{i=1}^{N} \sim \mu_0$:
$$
\mathcal L_{\mathrm{reverse}}[F] = \mathbb E_{x \sim \mu_0} \Bigl[ U_1(F(x)) - \log |\det J_F(x)| \Bigr].
$$
The package exposes this as `reverse_KL(x, target, flow)`.

### Importance sampling and $\mathrm{ESS}$

After training, $\nu = F_{\#} \mu_0$ is an approximation of $\mu_1$. Self-normalized importance sampling corrects the residual mismatch. With $y = F(x)$ and $x \sim \mu_0$, the unnormalized weight is
$$
w(y) = \frac{\mu_1(y)}{\nu(y)} \propto \frac{\exp(-U_1(y))}{\exp(-U_0(x) - \log |\det J_F(x)|)},
$$
with log-weight
$$
\log w = -U_1(y) + U_0(x) + \log |\det J_F(x)|.
$$
The Effective Sample Size measures how concentrated the weights are:
$$
\mathrm{ESS} = \frac{\bigl(\sum_{i=1}^{N} w_i\bigr)^2}{N \sum_{i=1}^{N} w_i^2} \in [0, 1].
$$
An $\mathrm{ESS}$ near $1$ means $\nu \approx \mu_1$; near $0$ means a few samples carry almost all the weight. We compute $\mathrm{ESS}$ in log-space (`compute_ESS_log`) using `logsumexp` for numerical stability.

## Implementation and execution

The full pipeline lives in [`2D_reverse_KL.py`](2D_reverse_KL.py). Run from the project root:

```bash
python -m tests.2D_reverse_KL
```

Pointers into the script:

- imports & device setup: [`2D_reverse_KL.py:1–8`](2D_reverse_KL.py#L1-L8)
- source and target potentials: [`2D_reverse_KL.py:10–22`](2D_reverse_KL.py#L10-L22)
- flow init: [`2D_reverse_KL.py:25`](2D_reverse_KL.py#L25)
- training parameters: [`2D_reverse_KL.py:27–31`](2D_reverse_KL.py#L27-L31)
- training loop (mini-batched reverse-KL): [`2D_reverse_KL.py:33–54`](2D_reverse_KL.py#L33-L54)
- IS reweighting + $\mathrm{ESS}$: [`2D_reverse_KL.py:57–68`](2D_reverse_KL.py#L57-L68)
- plotting: [`2D_reverse_KL.py:70–93`](2D_reverse_KL.py#L70-L93)

<p align="center"><img src="2D_reverse_KL.png" alt="reverse-KL test" width="700px"></p>

## Analysis

The training loss decreases monotonically from $\sim 2.0$ to $\sim 1.62$, and the post-training $\mathrm{ESS}$ lands near $0.99$ — a strong indicator that the pushforward $\nu = F_{\#}\mu_0$ matches the target $\mu_1$ closely. A few qualitative observations:

- **Mode shape captured.** The pushforward panel reproduces the characteristic "bow-tie" of $\mu_1 \propto \exp(-U_1)$: the cosine ridge term $2\cos x_1$ creates two density peaks near $x_1 = \pm \pi$ (where $\cos = -1$ minimizes the energy), separated by a low-density valley at $x_1 = 0$ (where $\cos = 1$ raises the energy). The flow has learned this anisotropic, slightly bimodal shape directly from the energy function $U_1$ — *no samples from $\mu_1$ were ever provided*.
- **Mode-seeking behavior.** Reverse $\mathrm{KL}$ is *zero-forcing*: $\mathrm{KL}(\nu \,\|\, \mu_1)$ blows up wherever $\nu > 0$ but $\mu_1 \to 0$, so the flow is penalized for placing mass outside the target's support. On a connected target like this one (the two density peaks are linked by a non-trivial valley, not disjoint support), this works well; on a target with truly disjoint modes, reverse $\mathrm{KL}$ is famously prone to *mode collapse* — the flow would lock onto one mode and ignore the others. Compare with the forward-$\mathrm{KL}$ writeup ([`2D_forward_KL.md`](2D_forward_KL.md)), which uses the dual loss and is *mass-covering* instead.
- **High $\mathrm{ESS}$ as a self-test.** $\mathrm{ESS} \approx 1$ means the importance weights $w_i \propto \mu_1(y_i)/\nu(y_i)$ are nearly uniform — the residual bias of the flow is small enough that no reweighting is needed in practice. If the flow were poorly trained, $\mathrm{ESS}$ would collapse toward $0$ (a few samples carrying all the mass), and the IS-weighted scatter on the right would degenerate into a few large dots.
