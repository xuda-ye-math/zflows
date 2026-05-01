import torch
from zflows import NSF, compute_ESS_log
from zflows.potential import Potential, Gaussian

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# source: U0(x) = (x1^2+x2^2)/2 (Gaussian prior)
u0 = Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0]).to(device)

# target: U1(x) = (x1^2+x2^2)/2 + 2*cos(x1)
class U1(Potential): # target potential
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:,0] # [N]
        x2 = x[:,1] # [N]
        return 0.5 * (x1 ** 2 + x2 ** 2) + 2 * torch.cos(x1)
    
u1 = U1().to(device) # to(device) can be removed here

# define KL loss function
def reverse_KL_loss(x: torch.Tensor, source: Potential, target: Potential, flow: NSF):
    """
    The reverse KL divergence in energy-driven normalizing flow.
    Estimates  E_{x ~ source}[ target(F(x)) - source(x) - log|det J_F(x)| ],
    where F = flow.t() pushes source samples toward the target.
    Input:
        x:      Tensor [N, d]   samples drawn from the source distribution
        source: Potential       negative log-density of the source (up to const)
        target: Potential       negative log-density of the target (up to const)
        flow:   NSF             normalizing flow providing F = flow.t()
    Output:
        loss: Tensor (scalar)   Monte Carlo estimate of the reverse KL
    """
    y, ladj = flow.t().call_and_ladj(x) # get y = F(x) and log_abs_det_jacobian
    return (target(y) - source(x) - ladj).mean()

# initialize Neural Spline Flow (NSF)
flow = NSF(a=[-4, -4], b=[4, 4], bins=8, transforms=4, hidden_features=(64, 64)).to(device)

# training parameters
N: int = 10000 # number of samples
LR: float = 1e-3 # learning rate
BATCH: int = 1000 # batch size
EPOCH: int = 10 # number of epochs

x = u0.samples(N) # generate samples
optimizer = torch.optim.Adam(flow.parameters(), lr=LR)

for epoch in range(EPOCH):
    perm = torch.randperm(N, device=device)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, N, BATCH):
        idx = perm[start:start + BATCH]
        x_batch = x[idx]

        loss = reverse_KL_loss(x_batch, source=u0, target=u1, flow=flow)

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
    y_plot, ladj = flow.t().call_and_ladj(x_plot) # pushforward F(x)

    # importance sampling: target density ~ exp(-u1(y)), proposal density q(y).
    # log q(y) = -u0(x) - ladj, so log w = -u1(y) + u0(x) + ladj.
    log_w = -u1(y_plot) + u0(x_plot) + ladj

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
plt.savefig("test_2D_reverse_KL.png", dpi=150)
plt.show()
