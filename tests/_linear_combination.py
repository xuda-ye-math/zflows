import torch
from zflows import Potential, Linear_Combination, langevin

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# U0(x) = (x1^2 + x2^2) / 2
class U0(Potential):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        return 0.5 * (x1 ** 2 + x2 ** 2)

# U1(x) = 2 * cos(x1)
class U1(Potential):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        return 2 * torch.cos(x1)

u0 = U0().to(device)
u1 = U1().to(device)

# create a linear combination of the potentials
ulc = Linear_Combination(u0, u1, 1.0, 1.2)

# enable gradient computation
import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1") # cleaner logs
ulc.enable_grad()

# set Langevin simulation parameters
N = 32 # simulate 32 independent trajectories
x = torch.randn(N, 2).to(device)
step = 1e-2
iterations = 100000
record_every = 100

# run Langevin in chunks of `record_every` steps; record state after each chunk
n_chunks = iterations // record_every
snapshots = []
for _ in range(n_chunks):
    x = langevin(x, potential=ulc, step=step, iters=record_every)
    snapshots.append(x.detach())

samples = torch.cat(snapshots, dim=0) # [n_chunks * N, 2]
print(f"collected samples: {samples.shape}")

# scatter plot of the collected samples
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
samples_np = samples.cpu().numpy()

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(samples_np[:, 0], samples_np[:, 1], s=1, alpha=0.3, color="darkblue")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_aspect("equal")
ax.set_title(r"Langevin samples of $U_{\mathrm{lc}} = c_0 U_0 + c_1 U_1$")

plt.tight_layout()
plt.savefig(HERE / "_linear_combination.png", dpi=300)
plt.show()

ulc.release()
u0.release()
u1.release()