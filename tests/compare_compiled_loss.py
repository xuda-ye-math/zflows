# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false

"""Benchmark `zflows.loss.compile_raw` against the raw `reverse_KL(x, target, flow.t())`
pattern across an NSF (d × hidden_features) grid.

What is measured:
    per-step wall-clock for the FULL training step
        loss = loss_fn(x_batch)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    in three modes:
        raw                   — fresh flow.t() per call, no compile
        compiled-default      — zflows.loss.compile_raw(..., mode='default')
        compiled-reduce-overhead — zflows.loss.compile_raw(..., mode='reduce-overhead')

torch.compile wraps only the forward; backward + Adam.step run in eager
regardless of mode. The reported numbers therefore include compile-eligible
forward AND eager backward/optimizer, which is what users experience in
real training loops.

CUDA-only. CPU mode would defeat the point of `reduce-overhead` (CUDA Graphs).
"""

from __future__ import annotations

import csv
import gc
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch._dynamo

from zflows.utils import set_cache_size_limit, suppress_warnings

# Silence Python warnings, Dynamo recompile/graph-break logs, Triton autotune
# stderr, and Inductor compile-worker interleaving in one call. Must run
# before the first torch.compile invocation for the env-var-based pieces
# (TRITON_PRINT_AUTOTUNING, TORCHINDUCTOR_COMPILE_THREADS) to take effect.
suppress_warnings()

# Dynamo's per-code-object cache evicts after 8 specializations by default.
# With 15 (d, hf) cells × 2 compiled modes = 30 specializations on the same
# inner closure code object, the default limit causes silent eager fallback
# from cell ~5 onward, masking the real compile speedup.
set_cache_size_limit(64)
# Surface real compile failures instead of silently falling back.
torch._dynamo.config.suppress_errors = False

import zflows.loss as zloss
from zflows.flow import NSF
from zflows.loss import reverse_KL
from zflows.potential import Gaussian


# ──────────────────────────────────────────────────────────────────────
# Configuration  — edit the grid at the top to sweep different axes
# ──────────────────────────────────────────────────────────────────────
DIMS: Sequence[int] = (2, 4, 8, 16, 32)
HIDDEN_FEATURES_GRID: Sequence[tuple[int, ...]] = ((64, 64), (128, 128), (256, 256))
BINS: int = 12
TRANSFORMS: int = 4
BATCH: int = 2000

WARMUP_STEPS: int = 30
TIMED_STEPS: int = 100

MODES: tuple[str, ...] = ("raw", "default", "reduce-overhead")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def build_loss_fn(
    mode: str,
    flow: NSF,
    potential: Gaussian,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable (x_batch) -> scalar loss for the requested mode.

    The compiled modes use `zflows.loss.compile_raw` — the single-input
    fast path with no wrapper / cast overhead — since the benchmark
    measures a fixed-beta workload.
    """
    if mode == "raw":
        def raw_loss(x: torch.Tensor) -> torch.Tensor:
            return reverse_KL(x, target=potential, F=flow.t())
        return raw_loss
    if mode == "default":
        return zloss.compile_raw(reverse_KL, potential, flow.t(), mode="default")
    if mode == "reduce-overhead":
        return zloss.compile_raw(reverse_KL, potential, flow.t(), mode="reduce-overhead")
    raise ValueError(f"unknown mode {mode!r}")


def time_full_step(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    x_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    n_warmup: int,
    n_timed: int,
) -> float:
    """Mean wall-clock ms per full training step (fwd + bwd + Adam.step)."""
    for _ in range(n_warmup):
        loss = loss_fn(x_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_timed):
        loss = loss_fn(x_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / n_timed


def sanity_check_compiled_matches_raw(
    d: int,
    hf: tuple[int, ...],
    potential: Gaussian,
    x: torch.Tensor,
    atol: float = 1e-3,
) -> None:
    """Build a fresh NSF, eval reverse_KL three ways under torch.no_grad, assert agreement."""
    torch.manual_seed(0)
    f = NSF(a=[-3.0] * d, b=[3.0] * d, bins=BINS, transforms=TRANSFORMS,
            hidden_features=hf).to(x.device)
    with torch.no_grad():
        l_raw = reverse_KL(x, target=potential, F=f.t()).item()
        l_def = zloss.compile_raw(reverse_KL, potential, f.t(), mode="default")(x).item()
        l_red = zloss.compile_raw(reverse_KL, potential, f.t(), mode="reduce-overhead")(x).item()
    if abs(l_def - l_raw) > atol or abs(l_red - l_raw) > atol:
        sys.exit(
            f"FAIL — compile-induced numerical drift at (d={d}, hf={hf}): "
            f"raw={l_raw:.6f}, default={l_def:.6f}, reduce-overhead={l_red:.6f}"
        )
    del f
    gc.collect()
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not torch.cuda.is_available():
        sys.exit(
            "compare_compiled_loss.py requires CUDA — `reduce-overhead` uses\n"
            "CUDA Graphs which are CUDA-only and the comparison loses its\n"
            "point on CPU."
        )
    device = "cuda"
    dev_name = torch.cuda.get_device_name(0)
    print(f"device: {device}  ({dev_name})")
    print(f"grid:   d ∈ {list(DIMS)}, hidden_features ∈ {list(HIDDEN_FEATURES_GRID)}")
    print(f"setup:  bins={BINS}, transforms={TRANSFORMS}, BATCH={BATCH}, "
          f"warmup={WARMUP_STEPS}, timed={TIMED_STEPS}")
    print()

    # Open CSV up-front so partial runs (e.g. timeout) still leave a usable file.
    csv_path = Path(__file__).resolve().parent / "compare_compiled_loss.csv"
    csv_fh = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow([
        "d", "hidden_features",
        "raw_ms", "default_ms", "reduce_ms",
        "speedup_default", "speedup_reduce",
    ])
    csv_fh.flush()

    bench_t0 = time.perf_counter()

    # rows[(d, hf)] = {'raw': ms, 'default': ms, 'reduce-overhead': ms}
    rows: dict[tuple[int, tuple[int, ...]], dict[str, float]] = {}

    for d in DIMS:
        potential = Gaussian(mean=[0.0] * d, variance=[1.0] * d).to(device)
        for hf in HIDDEN_FEATURES_GRID:
            torch.manual_seed(0)
            x_batch = torch.randn(BATCH, d, device=device)

            # one-shot correctness check across all three modes
            sanity_check_compiled_matches_raw(d, hf, potential, x_batch)

            cell: dict[str, float] = {}
            for mode in MODES:
                torch.manual_seed(0)
                flow = NSF(a=[-3.0] * d, b=[3.0] * d, bins=BINS,
                           transforms=TRANSFORMS, hidden_features=hf).to(device)
                opt = torch.optim.Adam(flow.parameters(), lr=1e-3)

                loss_fn = build_loss_fn(mode, flow, potential)

                ms = time_full_step(
                    loss_fn, x_batch, opt,
                    n_warmup=WARMUP_STEPS, n_timed=TIMED_STEPS,
                )
                cell[mode] = ms
                print(f"  d={d:>2}, hf={str(hf):<12}  mode={mode:<16}  "
                      f"{ms:>7.3f} ms/step")

                del flow, opt, loss_fn
                gc.collect()
                torch.cuda.empty_cache()

            rows[(d, hf)] = cell

            # flush this (d, hf) row to CSV immediately
            raw, dft, red = cell["raw"], cell["default"], cell["reduce-overhead"]
            csv_writer.writerow([
                d, "x".join(map(str, hf)),
                f"{raw:.3f}", f"{dft:.3f}", f"{red:.3f}",
                f"{raw / dft:.2f}", f"{raw / red:.2f}",
            ])
            csv_fh.flush()

    csv_fh.close()
    bench_t1 = time.perf_counter()

    # ──────────────────────────────────────────────────────────────
    # Final table
    # ──────────────────────────────────────────────────────────────
    print()
    print("=" * 96)
    print("Per-step training latency (full forward + backward + Adam.step),"
          " mean over", TIMED_STEPS, "steps")
    print("=" * 96)
    header = (
        f"{'d':>3} | {'hidden_features':<15} | "
        f"{'raw ms':>8} | {'default ms':>10} | {'reduce ms':>10} | "
        f"{'speedup default':>15} | {'speedup reduce':>14}"
    )
    print(header)
    print("-" * len(header))
    for d in DIMS:
        for hf in HIDDEN_FEATURES_GRID:
            cell = rows[(d, hf)]
            raw, dft, red = cell["raw"], cell["default"], cell["reduce-overhead"]
            sp_dft, sp_red = raw / dft, raw / red
            print(
                f"{d:>3} | {str(hf):<15} | "
                f"{raw:>8.3f} | {dft:>10.3f} | {red:>10.3f} | "
                f"{sp_dft:>15.2f} | {sp_red:>14.2f}"
            )

    print()
    print(f"total benchmark wall time: {(bench_t1 - bench_t0):.1f} s")
    print(f"csv saved → {csv_path}")


if __name__ == "__main__":
    main()
