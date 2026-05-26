# pyright: reportArgumentType=false, reportCallIssue=false

from pathlib import Path
import math
import torch

from zflows.flow import NCSF
from zflows.potential import Potential, Uniform
import zflows.loss
from zflows.loss import reverse_KL
from zflows.utils import (
    resample, compute_ESS_log, rejuvenation, importance_weights_log,
    set_cache_size_limit, suppress_warnings,
)

# Silence Triton autotune / Inductor / Dynamo / Python warnings, and give
# Dynamo enough cache headroom for the few compiled functions this script
# creates (one loss.compile + enable_grad + enable_eval).
suppress_warnings()
set_cache_size_limit(16)

HERE = Path(__file__).resolve().parent

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# source: U0(x) = const   (uniform on [-pi, pi]^3)
u0 = Uniform(a=[-math.pi, -math.pi, -math.pi], b=[math.pi, math.pi, math.pi]).to(device)

# target: von Mises ridge mixture on the 3-torus
# U1(x) = -log[ exp(k cos(x1-x2)) + exp(k cos(x2-x3)) + exp(k cos(x3-x1)) ]
class U1(Potential):
    def __init__(self, kappa: float = 4.0):
        super().__init__()
        self.kappa = kappa
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t1, t2, t3 = x[:, 0], x[:, 1], x[:, 2]
        logits = torch.stack([
            self.kappa * torch.cos(t1 - t2),
            self.kappa * torch.cos(t2 - t3),
            self.kappa * torch.cos(t3 - t1),
        ], dim=-1) # [N, 3]
        return -torch.logsumexp(logits, dim=-1) # [N]

u1 = U1(kappa=4.0).to(device)

# initialize Neural Circular Spline Flow (NCSF)
flow = NCSF(
    a=[-math.pi, -math.pi, -math.pi],
    b=[ math.pi,  math.pi,  math.pi],
    bins=8, transforms=4, hidden_features=(128, 128),
).to(device)

# training parameters
N: int = 20000 # number of samples
LR: float = 1e-3 # learning rate
BATCH: int = 2000 # batch size
EPOCH: int = 20 # number of epochs

x = u0.samples(N) # generate samples
optimizer = torch.optim.Adam(flow.parameters(), lr=LR)

# Compile the reverse-KL training step once. F = flow.t() is captured here
# but the lazy machinery re-reads flow's nn.Parameters by attribute access
# on every forward, so optimizer.step() updates flow correctly. beta is a
# runtime arg of loss_fn; pre-allocate it as a 0-d tensor so we don't pay
# a torch.as_tensor() allocation on every step.
F = flow.t()
loss_fn = zflows.loss.compile(reverse_KL, u1, F)
beta_t = torch.tensor(1.0, device=device)

for epoch in range(EPOCH):
    perm = torch.randperm(N, device=device)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, N, BATCH):
        idx = perm[start:start + BATCH]
        x_batch = x[idx]

        loss = loss_fn(x_batch, beta_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    print(f"epoch {epoch+1:>3}/{EPOCH}   KL_loss = {avg_loss:.6f}")

# importance sampling diagnostics + 3D scatter
import matplotlib.pyplot as plt

with torch.no_grad():
    x_plot = u0.samples(N) # fresh samples from source
    y_plot, _ = flow.t().call_and_ladj(x_plot) # pushforward F(x), used for resampling
    log_w = importance_weights_log(x_plot, source=u0, target=u1, F=flow.t(), chunk=2)

ess = compute_ESS_log(log_w)
print(f"ESS = {ess.item():.4f}")

# resample to N equally-weighted particles, then rejuvenate via Langevin
with torch.no_grad():
    w = (log_w - log_w.max()).exp() # normalize-stable weights
    y_resampled = resample(y_plot, w)

u1.enable_eval() # for faster MALA
u1.enable_grad() # opt-in: build .grad(x) only when needed
y_fresh = rejuvenation(y_resampled, potential=u1, adjust=True, chunk=4) # default step / iters

# subsample for a less crowded 3D scatter
N_PLOT = 10000
idx_plot = torch.randperm(y_fresh.shape[0])[:N_PLOT]
y_fresh_np = y_fresh[idx_plot].detach().cpu().numpy()

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(y_fresh_np[:, 0], y_fresh_np[:, 1], y_fresh_np[:, 2], s=0.6, alpha=0.5, color="darkgreen")
ax.set_xlim(-math.pi, math.pi)
ax.set_ylim(-math.pi, math.pi)
ax.set_zlim(-math.pi, math.pi)
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.set_zlabel(r"$\theta_3$")
ax.set_title(f"resample + rejuvenation (ESS = {ess.item():.3f})")

plt.tight_layout()
plt.savefig(HERE / "3D_periodic.png", dpi=300)
plt.show()
