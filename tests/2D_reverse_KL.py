from pathlib import Path
import torch
from zflows.flow import NSF
from zflows.potential import Gaussian, potential_from
from zflows.utils import compute_ESS_log, suppress_warnings
from zflows.loss import reverse_KL

suppress_warnings()

HERE = Path(__file__).resolve().parent

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# source: U_source(x) = (x1^2 + x2^2) / 2  (Gaussian prior)
u_source = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)

# target: U_target(x) = (x1^2 + x2^2) / 2 + 2 * cos(x1)
def U_target_forward(x: torch.Tensor) -> torch.Tensor: # Tensor [N, d] -> Tensor [N]
    x1 = x[:, 0]
    x2 = x[:, 1]
    return 0.5 * (x1 ** 2 + x2 ** 2) + 2 * torch.cos(x1)

u_target = potential_from(U_target_forward).to(device)

# initialize Neural Spline Flow (NSF)
flow = NSF(a=[-4, -4], b=[4, 4], bins=8, transforms=4, hidden_features=(64, 64)).to(device)

# training parameters
N: int = 10000 # number of samples
LR: float = 1e-3 # learning rate
BATCH: int = 1000 # batch size
EPOCH: int = 10 # number of epochs

x = u_source.samples(N) # generate samples
optimizer = torch.optim.Adam(flow.parameters(), lr=LR)

for epoch in range(EPOCH):
    perm = torch.randperm(N, device=device)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, N, BATCH):
        idx = perm[start:start + BATCH]
        x_batch = x[idx]

        loss = reverse_KL(x_batch, target=u_target, F=flow.t())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    print(f"epoch {epoch+1:>3}/{EPOCH}   KL_loss = {avg_loss:.6f}")

# plot the result
import matplotlib.pyplot as plt

with torch.no_grad():
    x_plot = u_source.samples(N) # fresh samples from source
    y_plot, ladj = flow.t().call_and_ladj(x_plot) # pushforward F(x)

    # importance sampling: target density ~ exp(-u_target(y)), proposal density q(y).
    # log q(y) = -u_source(x) - ladj, so log w = -u_target(y) + u_source(x) + ladj.
    log_w = -u_target(y_plot) + u_source(x_plot) + ladj

ess = compute_ESS_log(log_w)
print(f"ESS = {ess.item():.4f}")

x_np = x_plot.cpu().numpy()
y_np = y_plot.cpu().numpy()
w_np = (log_w - log_w.max()).exp().cpu().numpy() # for color scaling only

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].scatter(x_np[:, 0], x_np[:, 1], s=2, alpha=0.4, color="darkblue")
axes[0].set_title(r"source samples  $\mu_0$")

axes[1].scatter(y_np[:, 0], y_np[:, 1], s=2, alpha=0.4, color="darkgreen")
axes[1].set_title(r"pushforward  $F_\# \mu_0$")

w_norm = w_np / w_np.max() # in [0, 1] for size scaling
axes[2].scatter(y_np[:, 0], y_np[:, 1], s=8 * w_norm, alpha=0.4, color="darkred")
axes[2].set_title(f"IS-weighted samples (ESS = {ess.item():.3f})")

for ax in axes:
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(HERE / "2D_reverse_KL.png", dpi=150)
plt.show()
