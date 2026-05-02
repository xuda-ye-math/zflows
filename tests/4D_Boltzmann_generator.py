# pyright: reportArgumentType=false, reportCallIssue=false

from pathlib import Path
import os
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1") # cleaner logs

import torch
from zflows import (
    Potential, Gaussian, NSF, Linear_Combination,
    reverse_KL, compute_ESS_log, resample, rejuvenation,
)

HERE = Path(__file__).resolve().parent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# source: U_source(x) = (|r1|^2 + |r2|^2) / 2     (standard 4D Gaussian)
u_source = Gaussian(mean=[0.0] * 4, variance=[1.0] * 4).to(device)

# target: two charged particles in R^2, soft-annular trap + regularized 3D Coulomb.
# x = (x1, x2) in R^4, x_i in R^2
# U_target(x) = a * [(|x1|^2 - r0^2)^2 + (|x2|^2 - r0^2)^2]
#             + q2 / sqrt(|x1 - x2|^2 + eps^2)
class U_target(Potential):
    """
    Two particles on a soft annulus in R^2, repelling via a regularized
    3D Coulomb interaction. The regularization eps = 1e-3 is small enough
    that the target behaves like a true 3D Coulomb away from collisions,
    yet large enough that gradients stay finite for training.
    """
    def __init__(self, r0: float = 2.0, a: float = 1.0, q2: float = 4.0, eps: float = 1e-3):
        super().__init__()
        self.r0 = r0
        self.a = a
        self.q2 = q2
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, :2], x[:, 2:]
        sq1 = (x1 ** 2).sum(dim=-1) # [N]
        sq2 = (x2 ** 2).sum(dim=-1) # [N]
        conf = self.a * ((sq1 - self.r0 ** 2) ** 2 + (sq2 - self.r0 ** 2) ** 2)
        d2 = ((x1 - x2) ** 2).sum(dim=-1) # [N]
        coul = self.q2 / (d2 + self.eps ** 2).sqrt()
        return conf + coul

u_target = U_target(r0=2.0, a=1.0, q2=4.0).to(device) # eps = 1e-3 (default)

PT_PATH = HERE / "4D_Boltzmann_generator.pt"

if PT_PATH.exists():
    print(f"found existing {PT_PATH}, loading history (skipping training)")
    saved = torch.load(PT_PATH, weights_only=False)
    x_valid_history: list[torch.Tensor] = saved["x_valid_history"]
    ess_history: list[float] = saved["ess_history"]
    print(f"loaded {len(x_valid_history)} x_valid snapshots and {len(ess_history)} ESS values")
else:
    # initialize Neural Spline Flow (NSF) on the box [-3, 3]^4
    # - covers the bulk of the 4D Gaussian source (~3-sigma per coord)
    # - and the annular target (|x_i| concentrates near r0=2, fits in [-3, 3]^2 per particle)
    flow = NSF(
        a=[-3.0, -3.0, -3.0, -3.0],
        b=[ 3.0,  3.0,  3.0,  3.0],
        bins=12, transforms=4, hidden_features=(128, 128),
    ).to(device)

    # training parameters
    N_TRAIN: int = 40000  # number of training samples
    N_VALID: int = 120000 # number of validation samples
    LR: float = 1e-3 # learning rate
    BATCH: int = 2000 # batch size
    EPOCH: int = 40 # number of epochs

    # validation samples: at step k = 0 they are drawn from U_0 = U_source;
    # at the end of step k they are samples (approximately) from U_k.
    x_valid_prev = u_source.samples(N_VALID) # [N_VALID, 4]

    M = 12 # number of annealing steps; ladder has M+1 rungs c_0, c_1, ..., c_M
    c_list = torch.linspace(0, 1, M + 1) # c_k = k / M

    # snapshots of x_valid at each rung: index k holds samples (~) from mu_k
    x_valid_history: list[torch.Tensor] = [x_valid_prev.detach().cpu()]
    ess_history: list[float] = [] # ESS measured at the flow proposal in each of the M steps

    optimizer = torch.optim.Adam(flow.parameters(), lr=LR)

    # annealed Boltzmann generator: for each k = 1, ..., M, train the flow
    # using the previous and current bridges
    #     U_{k-1} = (1 - c_{k-1}) * U_source + c_{k-1} * U_target
    #     U_k     = (1 - c_k)     * U_source + c_k     * U_target
    for k in range(1, M + 1):
        U_prev = Linear_Combination(u_target, u_source, c_list[k - 1].item())
        U_curr = Linear_Combination(u_target, u_source, c_list[k    ].item())
        print(f"=== step {k}/{M}   c_{k-1} = {c_list[k-1].item():.3f}  ->  c_{k} = {c_list[k].item():.3f} ===")

        # (1) resample N_TRAIN training samples from x_valid_prev (~ mu_{k-1})
        idx = torch.randint(0, N_VALID, (N_TRAIN,), device=device)
        x_train_prev = x_valid_prev[idx]

        # (2) train the flow: reverse KL from x_train_prev (~mu_{k-1}) to U_curr (mu_k).
        # The U_{k-1}(x) term in the reverse-KL objective is parameter-independent,
        # so reverse_KL(x, target=U_curr, flow) gives the right loss directly.
        for epoch in range(EPOCH):
            perm = torch.randperm(N_TRAIN, device=device)
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, N_TRAIN, BATCH):
                x_batch = x_train_prev[perm[start:start + BATCH]]
                loss = reverse_KL(x_batch, target=U_curr, flow=flow)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 5 == 0 or epoch == EPOCH - 1:
                print(f"  epoch {epoch+1:>3}/{EPOCH}   KL_loss = {epoch_loss / n_batches:.6f}")

        # (3) push validation samples through the trained flow, IS-correct, resample
        with torch.no_grad():
            y_pf, ladj = flow.t().call_and_ladj(x_valid_prev) # F(x_valid_prev), log|det J|
            # log w = -U_k(y) + U_{k-1}(x) + log|det J_F(x)|
            log_w = -U_curr(y_pf) + U_prev(x_valid_prev) + ladj
            ess = compute_ESS_log(log_w)
            print(f"  ESS = {ess.item():.4f}")
            ess_history.append(ess.item())
            w = (log_w - log_w.max()).exp()
            y_resampled = resample(y_pf, w)

        # (4) rejuvenate against U_curr to break duplicates and decorrelate
        U_curr.enable_grad()
        x_valid_curr = rejuvenation(y_resampled, potential=U_curr, adjust=True, chunk=4)

        # (5) only x_valid_curr is carried into the next step;
        # also snapshot it (CPU copy) for the saved trajectory
        x_valid_prev = x_valid_curr
        x_valid_history.append(x_valid_curr.detach().cpu())

    # save x_valid at all M+1 rungs (indexed 0..M) and the per-step ESS;
    # flow weights are not stored.
    torch.save(
        {"x_valid_history": x_valid_history, "ess_history": ess_history},
        PT_PATH,
    )
    print(f"saved {len(x_valid_history)} x_valid snapshots and {len(ess_history)} ESS values to {PT_PATH}")

# Console: print ess_history per anneal step
print()
print("ESS history (per anneal step):")
for k, e in enumerate(ess_history, start=1):
    print(f"  step {k:>2}: ESS = {e:.4f}")

# Plot snapshots at steps k = 0, 4, 8, 12 in two rows.
#
# Row 1 (Cartesian): scatter BOTH particles' (x, y) positions in R^2
#   in different colors. Shows the *marginal* annulus formation: as
#   k grows, both clouds collapse from a Gaussian blob at the origin
#   onto a ring of radius r0 = 2. The dashed black circle marks r0.
#
# Row 2 (S^1 polar histogram): distribution of the relative angle
#   Δθ = θ_2 - θ_1 ∈ [-π, π], where θ_i = atan2(y_i, x_i).
#   This is the *joint* signal hidden by row 1's rotational symmetry:
#     k=0 (Gaussian source): θ_1, θ_2 are independent uniforms, so Δθ
#         is uniform on the circle — the polar histogram is flat,
#         |Δθ| has mean π/2 (= 90°);
#     k=M (target): Coulomb repulsion drives the particles antipodal,
#         so Δθ concentrates near ±π (= 180°) — visible as a tall bar
#         on the left side of the polar plot.
# ---------------------------------------------------------------
import math
import matplotlib.pyplot as plt

steps_to_plot = [0, 4, 8, 12]
M_total = len(x_valid_history) - 1
N_PLOT = 3000
N_BINS = 36 # angular bins, 10 degrees each

theta_grid = torch.linspace(0, 2 * math.pi, 200)
ring_x = (2.0 * torch.cos(theta_grid)).numpy()
ring_y = (2.0 * torch.sin(theta_grid)).numpy()

fig = plt.figure(figsize=(4 * len(steps_to_plot), 9))

for col, k in enumerate(steps_to_plot):
    x_k = x_valid_history[k]
    c_k = k / M_total

    # ---- Row 1: 2D scatter of both particles ----
    ax = fig.add_subplot(2, len(steps_to_plot), col + 1)
    idx = torch.randperm(x_k.shape[0])[:N_PLOT]
    pts = x_k[idx].numpy()
    ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.4, color="tab:blue", label="particle 1")
    ax.scatter(pts[:, 2], pts[:, 3], s=2, alpha=0.4, color="tab:red",  label="particle 2")
    ax.plot(ring_x, ring_y, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(f"step $k={k}$  ($c={c_k:.2f}$)")
    if col == 0:
        ax.legend(loc="upper right", fontsize=8, markerscale=3)

    # ---- Row 2: polar histogram of Δθ = θ_2 - θ_1 on S^1 ----
    ax_p = fig.add_subplot(2, len(steps_to_plot), len(steps_to_plot) + col + 1, projection="polar")
    theta1 = torch.atan2(x_k[:, 1], x_k[:, 0])
    theta2 = torch.atan2(x_k[:, 3], x_k[:, 2])
    delta = ((theta2 - theta1 + math.pi) % (2 * math.pi)) - math.pi # in (-π, π]
    counts = torch.histc(delta, bins=N_BINS, min=-math.pi, max=math.pi)
    width = 2 * math.pi / N_BINS
    bin_centers = torch.linspace(-math.pi + width / 2, math.pi - width / 2, N_BINS)
    ax_p.bar(bin_centers.numpy(), counts.numpy(), width=width,
             color="tab:purple", alpha=0.75, edgecolor="black", linewidth=0.3)
    ax_p.set_yticklabels([]) # de-clutter; only the angular shape matters
    ax_p.set_title(r"$\Delta\theta = \theta_2 - \theta_1$")

plt.tight_layout()
plt.savefig(HERE / "4D_Boltzmann_generator.png", dpi=150)
plt.show()