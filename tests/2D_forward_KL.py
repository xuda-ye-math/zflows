from pathlib import Path
import torch
from zflows import NSF, Gaussian, Gaussian_Mixture, forward_KL

HERE = Path(__file__).resolve().parent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# source: standard Gaussian
u0 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)

# target: 3-mode diagonal Gaussian mixture on an equilateral triangle
u1 = Gaussian_Mixture(
    weights=[1.0, 1.0, 1.0],
    mean=[
        [ 2.0,  0.0],
        [-1.0,  1.732],
        [-1.0, -1.732],
    ],
    variance=[
        [0.15, 0.15],
        [0.15, 0.15],
        [0.15, 0.15],
    ],
).to(device)

# initialize Neural Spline Flow (NSF)
flow = NSF(a=[-4, -4], b=[4, 4], bins=8, transforms=4, hidden_features=(64, 64)).to(device)

# training parameters
N: int = 10000 # number of samples
LR: float = 1e-3 # learning rate
BATCH: int = 1000 # batch size
EPOCH: int = 20 # number of epochs

# data-driven: training data are samples from the target distribution
y = u1.samples(N)
optimizer = torch.optim.Adam(flow.parameters(), lr=LR)

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

    avg_loss = epoch_loss / n_batches
    print(f"epoch {epoch+1:>3}/{EPOCH}   KL_loss = {avg_loss:.6f}")

# plot the result
import matplotlib.pyplot as plt

with torch.no_grad():
    x_plot = u0.samples(N) # fresh samples from source
    y_pf, _ = flow.t().call_and_ladj(x_plot) # pushforward F(x)
    y_true = u1.samples(N) # fresh samples from target (ground truth)

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
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(HERE / "2D_forward_KL.png", dpi=150)
plt.show()
