from pathlib import Path
import torch
from zflows import CNF, Gaussian, forward_KL

HERE = Path(__file__).resolve().parent

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# source: standard 2D Gaussian on R^2 (CNF lives on R^d, no box needed)
u0 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)

# target: two-moons distribution, centered and rescaled to roughly fill [-2, 2]^2
def two_moons_samples(N: int, noise: float = 0.1) -> torch.Tensor:
    n1 = N // 2
    n2 = N - n1
    t1 = torch.rand(n1, device=device) * torch.pi
    t2 = torch.rand(n2, device=device) * torch.pi
    moon1 = torch.stack([torch.cos(t1) - 0.5, torch.sin(t1) - 0.25], dim=1)
    moon2 = torch.stack([0.5 - torch.cos(t2), 0.25 - torch.sin(t2)], dim=1)
    y = torch.cat([moon1, moon2], dim=0)
    y = y + noise * torch.randn_like(y)
    return y[torch.randperm(N, device=device)]

# initialize Continuous Normalizing Flow (CNF / FFJORD)
flow = CNF(dimension=2, frequency=5, hidden_features=(128, 128, 128)).to(device)

# training parameters
N: int = 10000 # number of samples
LR: float = 1e-2 # learning rate
BATCH: int = 1000 # batch size
EPOCH: int = 50 # number of epochs

# data-driven: training data are samples from the target distribution
y = two_moons_samples(N, noise=0.1)
optimizer = torch.optim.Adam(flow.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

for epoch in range(EPOCH):
    perm = torch.randperm(N, device=device)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, N, BATCH):
        idx = perm[start:start + BATCH]
        y_batch = y[idx]

        loss = forward_KL(y_batch, source=u0, flow=flow)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / n_batches
    print(f"epoch {epoch+1:>3}/{EPOCH}   KL_loss = {avg_loss:.6f}")

# plot the result
import matplotlib.pyplot as plt

with torch.no_grad():
    x_plot = u0.samples(N) # fresh samples from source
    y_pf, _ = flow.t().call_and_ladj(x_plot) # pushforward F(x)
    y_true = two_moons_samples(N, noise=0.1) # fresh samples from target (ground truth)

x_np = x_plot.cpu().numpy()
y_pf_np = y_pf.cpu().numpy()
y_true_np = y_true.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].scatter(x_np[:, 0], x_np[:, 1], s=2, alpha=0.4, color="darkblue")
axes[0].set_title(r"source samples  $\mu_0$")

axes[1].scatter(y_pf_np[:, 0], y_pf_np[:, 1], s=2, alpha=0.4, color="darkgreen")
axes[1].set_title(r"pushforward  $F_\# \mu_0$")

axes[2].scatter(y_true_np[:, 0], y_true_np[:, 1], s=2, alpha=0.4, color="darkred")
axes[2].set_title(r"target samples  $\mu_1$")

for ax in axes:
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(HERE / "2D_two_moon_CNF.png", dpi=300)
plt.show()
