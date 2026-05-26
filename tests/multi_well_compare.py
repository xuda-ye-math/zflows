"""Multi-well (8-mode) reverse-KL benchmark: CNF vs OTFlow across dimension.

Both continuous flows are trained by the SAME objective (plain reverse KL, no
OT regularizers) against the same target, so this isolates the architectural
difference — CNF's free-form MLP velocity vs OTFlow's potential-gradient
velocity with a closed-form trace — and reports the effective sample size
(ESS) of the trained proposal under importance sampling.

Target potential (factorized, `dimension`-dependent):
    - the first 3 coordinates are symmetric DOUBLE WELLS,
    - the remaining coordinates are standard Gaussian (quadratic).
Three independent double wells give 2**3 = 8 modes; the Gaussian tail just
raises the dimension. The wells are deliberately shallow (a low barrier) so
mode-seeking reverse KL does not collapse onto a subset of the 8 modes.

Goal: a two-row table (CNF, OTFlow) of ESS for d = 4, 8, 12, 16, 20, 24.

Run from the repo root:  python -m tests.multi_well_compare
"""

import csv
import time
from pathlib import Path

import torch

from zflows.flow import CNF, OTFlow
from zflows.loss import reverse_KL
from zflows.potential import Gaussian, Potential
from zflows.utils import compute_ESS_log, suppress_warnings

suppress_warnings()

HERE = Path(__file__).resolve().parent
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── experiment configuration ──────────────────────────────────────────
DIMS = [4, 8, 12, 16, 20, 24]
N_WELL = 3            # number of double-well coords -> 2**3 = 8 modes
WELL_SEP = 1.5        # double-well minima at ±WELL_SEP
WELL_BARRIER = 1.5    # barrier height at the origin (shallow on purpose)

NT = 12               # fixed RK4 steps (same for both flows)
HIDDEN = 64           # ODE-net width (CNF MLP / OTFlow ResNet)
LAYER = 3             # OTFlow ResNet depth
STEPS = 500           # reverse-KL Adam steps
BATCH = 1024          # source samples per step
LR = 2e-3
N_ESS = 20000         # samples used to estimate ESS


# ── target potential ──────────────────────────────────────────────────

class Multi_Well(Potential):
    """First `n_well` coords are symmetric double wells, the rest Gaussian.

        U(x) = sum_{i<n_well} barrier * ((x_i / sep)^2 - 1)^2
             + sum_{i>=n_well} 0.5 * x_i^2

    Each double well has minima at ±sep (energy 0) and a barrier of height
    `barrier` at 0, so the joint has 2**n_well modes.
    """

    def __init__(self, dimension, n_well=3, sep=1.5, barrier=1.5):
        super().__init__()
        self.dimension = dimension
        self.n_well = n_well
        self.sep = sep
        self.barrier = barrier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dw = x[:, : self.n_well]
        gauss = x[:, self.n_well :]
        u_dw = (self.barrier * ((dw / self.sep) ** 2 - 1.0) ** 2).sum(dim=-1)
        u_gauss = 0.5 * (gauss ** 2).sum(dim=-1)
        return u_dw + u_gauss


# ── train one flow by reverse KL, then estimate ESS ────────────────────

def train_and_ess(flow, target, source, d):
    """Reverse-KL train `flow` against `target`, return (ESS, seconds)."""
    flow.zeros()  # warm-start at the identity map for a fair, stable start
    flow.train()
    optimizer = torch.optim.Adam(flow.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)

    t0 = time.perf_counter()
    for _ in range(STEPS):
        x = torch.randn(BATCH, d, device=device)   # source ~ N(0, I_d)
        loss = reverse_KL(x, target=target, F=flow.t())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    # ESS of the trained proposal via self-normalized importance sampling:
    # log w = log p_target(y) - log q(y),  q the pushforward of N(0, I).
    flow.eval()
    with torch.no_grad():
        x = torch.randn(N_ESS, d, device=device)
        y, ladj = flow.t().call_and_ladj(x)
        log_w = -target(y) + source(x) + ladj
        ess = compute_ESS_log(log_w).item()
    return ess, seconds


# ── sweep ───────────────────────────────────────────────────────────────

results = {"CNF": {}, "OTFlow": {}}
timings = {"CNF": {}, "OTFlow": {}}

for d in DIMS:
    target = Multi_Well(d, n_well=N_WELL, sep=WELL_SEP, barrier=WELL_BARRIER).to(device)
    source = Gaussian(mean=[0.0] * d, variance=[1.0] * d, device=device)

    builders = {
        "CNF": lambda: CNF(dimension=d, frequency=4, nt=NT, exact=True,
                           hidden_features=(HIDDEN, HIDDEN)),
        "OTFlow": lambda: OTFlow(dimension=d, hidden=HIDDEN, layer=LAYER,
                                 rank=min(10, d + 1), nt=NT),
    }
    for name, build in builders.items():
        torch.manual_seed(0)  # same source-sample stream per cell
        flow = build().to(device)
        ess, secs = train_and_ess(flow, target, source, d)
        results[name][d] = ess
        timings[name][d] = secs
        print(f"  d={d:>2}  {name:<6}  ESS={ess:.4f}  ({secs:.1f}s)")


# ── table ─────────────────────────────────────────────────────────────

def fmt_row(label, row):
    return f"{label:<8}" + "".join(f"{row[d]:>9.4f}" for d in DIMS)

print()
print(f"=== Multi-well ({2 ** N_WELL}-mode) reverse-KL ESS: CNF vs OTFlow ===")
print(f"(NT={NT}, hidden={HIDDEN}, steps={STEPS}, batch={BATCH}, "
      f"well sep={WELL_SEP}, barrier={WELL_BARRIER})")
print()
header = "flow    " + "".join(f"{'d='+str(d):>9}" for d in DIMS)
print(header)
print("-" * len(header))
print(fmt_row("CNF", results["CNF"]))
print(fmt_row("OTFlow", results["OTFlow"]))
print()
print("train seconds per cell:")
print(fmt_row("CNF", timings["CNF"]))
print(fmt_row("OTFlow", timings["OTFlow"]))

# ── write CSV (tidy long format: one row per flow × dimension) ─────────
csv_path = HERE / "multi_well_compare.csv"
with csv_path.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["flow", "dimension", "ess", "train_seconds"])
    for name in ("CNF", "OTFlow"):
        for d in DIMS:
            writer.writerow([name, d, f"{results[name][d]:.4f}", f"{timings[name][d]:.2f}"])
print()
print(f"csv saved → {csv_path}")
