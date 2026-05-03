# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

from pathlib import Path
from itertools import combinations
import torch
from zflows import RealNVP, Gaussian, Gaussian_Mixture, forward_KL

HERE = Path(__file__).resolve().parent

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# source: standard 2D Gaussian on R^2 (RealNVP lives natively on R^d, no box)
u0 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)

# target: 4-corner Gaussian mixture at (+/-2, +/-2)
u1 = Gaussian_Mixture(
    weights=[1.0, 1.0, 1.0, 1.0],
    mean=[
        [ 2.0,  2.0],
        [ 2.0, -2.0],
        [-2.0,  2.0],
        [-2.0, -2.0],
    ],
    variance=[
        [0.15, 0.15],
        [0.15, 0.15],
        [0.15, 0.15],
        [0.15, 0.15],
    ],
).to(device)

# initialize RealNVP with extra coupling layers (each affine layer is
# weaker than an NSF spline transform, so 4 modes need extra capacity)
flow = RealNVP(dimension=2, transforms=8, hidden_features=(64, 64)).to(device)

# training parameters
N: int = 10000 # number of samples
LR: float = 1e-3 # learning rate
BATCH: int = 1000 # batch size
EPOCH: int = 30 # number of epochs

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

# latent-space interpolation between the 4 mode centers
flow.eval()
F = flow.t()

anchors = torch.tensor([
    [ 2.0,  2.0],
    [ 2.0, -2.0],
    [-2.0,  2.0],
    [-2.0, -2.0],
], device=device)

with torch.no_grad():
    z_anchors = F.inv(anchors) # 4 latent anchor points
    rt = F(z_anchors) # round-trip: F(F^{-1}(x)) should equal x
    rt_err = (rt - anchors).abs().max().item()
print(f"\nround-trip error on anchors: max |x - F(F^-1(x))| = {rt_err:.3e}")
assert rt_err < 1e-4, f"inverse round-trip too large: {rt_err}"

# 6 mode pairs, 50 interpolation steps each
T_STEPS = 50
t = torch.linspace(0.0, 1.0, T_STEPS, device=device).unsqueeze(-1) # (T, 1)

pairs = list(combinations(range(4), 2)) # 6 pairs of (i, j)
z_paths = [] # straight lines in latent space
x_paths = [] # curved trajectories in data space
with torch.no_grad():
    for i, j in pairs:
        z_t = (1 - t) * z_anchors[i] + t * z_anchors[j] # (T, 2)
        x_t = F(z_t)
        z_paths.append(z_t.cpu().numpy())
        x_paths.append(x_t.cpu().numpy())

# also map all target samples back to latent for the latent-space scatter
with torch.no_grad():
    y_grid = u1.samples(N)
    z_grid = F.inv(y_grid)

# fresh source and pushforward for the (0, 1) panel
with torch.no_grad():
    x_src = u0.samples(N)
    y_pf, _ = F.call_and_ladj(x_src)
    y_true = u1.samples(N)

y_true_np = y_true.cpu().numpy()
y_pf_np   = y_pf.cpu().numpy()
z_grid_np = z_grid.cpu().numpy()
anchors_np  = anchors.cpu().numpy()
z_anchors_np = z_anchors.cpu().numpy()

# plot the result
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (0, 0) target samples
axes[0, 0].scatter(y_true_np[:, 0], y_true_np[:, 1], s=2, alpha=0.4, color="darkred")
axes[0, 0].set_title(r"target samples  $\mu_1$")

# (0, 1) pushforward
axes[0, 1].scatter(y_pf_np[:, 0], y_pf_np[:, 1], s=2, alpha=0.4, color="darkgreen")
axes[0, 1].set_title(r"pushforward  $F_\# \mu_0$")

for ax in (axes[0, 0], axes[0, 1]):
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")

# (1, 0) latent space: F^{-1}(target samples) + straight interpolation lines
axes[1, 0].scatter(z_grid_np[:, 0], z_grid_np[:, 1], s=2, alpha=0.3, color="gray")
colors = plt.cm.tab10(range(len(pairs))) # type: ignore
for path, color in zip(z_paths, colors):
    axes[1, 0].plot(path[:, 0], path[:, 1], color=color, linewidth=1.5)
axes[1, 0].scatter(z_anchors_np[:, 0], z_anchors_np[:, 1],
                   marker="*", s=200, color="black", zorder=5)
axes[1, 0].set_title(r"latent space:  $F^{-1}(\mu_1)$  +  straight lines")
axes[1, 0].set_xlim(-4, 4)
axes[1, 0].set_ylim(-4, 4)
axes[1, 0].set_aspect("equal")

# (1, 1) data space: faint target + decoded curves + anchor stars
axes[1, 1].scatter(y_true_np[:, 0], y_true_np[:, 1], s=2, alpha=0.15, color="gray")
for path, color in zip(x_paths, colors):
    axes[1, 1].plot(path[:, 0], path[:, 1], color=color, linewidth=1.5)
axes[1, 1].scatter(anchors_np[:, 0], anchors_np[:, 1],
                   marker="*", s=200, color="black", zorder=5)
axes[1, 1].set_title(r"data space:  $\mu_1$  +  decoded curves  $F(z_t)$")
axes[1, 1].set_xlim(-4, 4)
axes[1, 1].set_ylim(-4, 4)
axes[1, 1].set_aspect("equal")

plt.tight_layout()
plt.savefig(HERE / "2D_RealNVP_latent_interpolation.png", dpi=300)
plt.show()
