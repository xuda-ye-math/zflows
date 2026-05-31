Let $\mu_0$ be the source distribution and $\mu_1$ be the target distribution, and a flow $F$ is already trained to match $F_{\#} \mu_0\approx \mu_1$, where the density of $F_{\#} \mu_0$ is
$$
    (F_{\#} \mu_0)(y) = \frac{\mu_0(x)}{|\det J_F(x)|}, \quad x = F^{-1}(y)
$$ 
Therefore, the importance weights in terms of $x$ is exactly given by
$$
    w(y) = \frac{\mu_1(y)}{(F_{\#} \mu_0)(y)} =\exp( -U_1(y) + U_0(x) + \log|\det J_F(x)|).
$$
This is the usual importance weights computed via
```python
out = []
for x in torch.chunk(samples, chunk, dim=0):
    y, ladj = F.call_and_ladj(x)
    out.append(-beta_target * target(y) + beta_source * source(x) + ladj)
return torch.cat(out, dim=0)
```

The AIS aims to cobnstruct the intermediate distributions
$$
    \pi_k(y) = (\mu_1(y))^{\frac kM} ((F_{\#} \mu_0)(y))^{1-\frac kM},
$$
whose importance weights is exactly $\pi_k(y) / \pi_{k-1}(y) = (w(y))^{\frac1k}$.

The overall steps are:
(0) input: samples $x$ from $\mu_0$
    output: samples $y$ from $\mu_1$
(1) generate samples $y$ from $pi_0(y) = (F_{\#} \mu_0)(y)$ by simply pushforward
(2) for k=1,...,M:
    refresh the samples of x by x = F^{-1}(y)
    compute the importance weights:
        w(y) = \exp( -1/M*U_1(y) + 1/M*U_0(x) + 1/M*\log|\det J_F(x)|)
    obtain the weighted samples (y,w)
    resample and langevin in $mu_1$ to obtain fresh samples y

No linear combination is actually required here, the importance weights are directly computed with the raw potential functions.
Also, we note that the langevin rejuvenation is calculated at $mu_1$ directly, because computing the log-likelihood of $\pi_k$ is too expensive, and since the target distribution is de facto $\mu_1$, using $\mu_1$ will not introduce essential deviation.