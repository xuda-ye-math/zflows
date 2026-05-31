# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false

"""Benchmark the compiled forward / inverse fast paths (`flow.for_ladj` /
`flow.inv_ladj`) against the raw `flow.t().call_and_ladj(x)` /
`flow.t().inv.call_and_ladj(y)` pattern across an NSF (d × hidden_features) grid.

What is measured (pure map wall-clock, under torch.no_grad, no autograd) — each
map returns the fused `(points, log|det J|)`:
    forward + ladj:   flow.t().call_and_ladj(x)       vs   flow.for_ladj(x)
    inverse + ladj:   flow.t().inv.call_and_ladj(y)   vs   flow.inv_ladj(y)
in three modes each:
    raw                       — fresh flow.t() per call, no compile
    compiled-default          — flow.enable_for_ladj/​enable_inv_ladj(mode='default')
    compiled-reduce-overhead  — flow.enable_for_ladj/​enable_inv_ladj(mode='reduce-overhead')

The compiled paths capture `F = flow.t()` once and `torch.compile` it, so they
also save the per-call `flow.t()` reconstruction (and the eager per-op dispatch)
that the raw pattern pays. Empirically both directions speed up substantially,
and the speedup *grows with d* as eager dispatch overhead dominates the raw
pattern: on an RTX 5070 Ti the forward map sees ~3-19× and the inverse spline
bisection ~3-20× (largest at d=32, where raw inverse ~380 ms collapses to
~18-35 ms). The bisection emits a torch.compile graph-break warning but stays
numerically correct (see the sanity check) and still compiles to a large
speedup. Closed-form / ODE inverses (RealNVP, CNF, OTFlow) compile cleanly too;
swap the flow class below to compare.

CUDA-only. `reduce-overhead` captures CUDA Graphs (CUDA-only), and a serious
forward/inverse latency comparison must run on the GPU — CPU timings are
dominated by Python/dispatch overhead and miss the point. Exits on CPU.
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
# stderr, and Inductor compile-worker interleaving in one call. Must run before
# the first torch.compile invocation for the env-var-based pieces to take effect.
suppress_warnings()

# Dynamo's per-code-object cache evicts after 8 specializations by default.
# Each (d, hf) cell compiles for_ladj + inv_ladj in 2 modes on shared closure code
# objects; the default limit would cause silent eager fallback partway through
# the sweep and mask the real speedup.
set_cache_size_limit(64)
# Surface real compile failures instead of silently falling back.
torch._dynamo.config.suppress_errors = False

from zflows.flow import NSF


# ──────────────────────────────────────────────────────────────────────
# Configuration  — edit the grid at the top to sweep different axes
# ──────────────────────────────────────────────────────────────────────
DIMS: Sequence[int] = (2, 4, 8, 16, 32)
HIDDEN_FEATURES_GRID: Sequence[tuple[int, ...]] = ((64, 64), (128, 128), (256, 256))
BINS: int = 12
TRANSFORMS: int = 4
BATCH: int = 2000

WARMUP_STEPS: int = 20
TIMED_STEPS: int = 50

MODES: tuple[str, ...] = ("raw", "default", "reduce-overhead")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def make_flow(d: int, hf: tuple[int, ...], device: str) -> NSF:
    """Fresh NSF with deterministic weights (seeded) so every mode times the
    same mapping — the inverse cost depends on the learned bijection."""
    torch.manual_seed(0)
    return NSF(a=[-3.0] * d, b=[3.0] * d, bins=BINS, transforms=TRANSFORMS,
               hidden_features=hf).to(device)


def build_map_fns(
    mode: str,
    flow: NSF,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    """Return (forward_fn, inverse_fn), each a no-arg callable for the mode.

    `raw` rebuilds `flow.t()` per call (the realistic non-compiled pattern);
    the compiled modes enable the captured-once fast paths.
    """
    if mode == "raw":
        return (lambda: flow.t().call_and_ladj(x), lambda: flow.t().inv.call_and_ladj(y))
    flow.enable_for_ladj(mode=mode)
    flow.enable_inv_ladj(mode=mode)
    return (lambda: flow.for_ladj(x), lambda: flow.inv_ladj(y))


def time_map(fn: Callable[[], torch.Tensor], n_warmup: int, n_timed: int) -> float:
    """Mean wall-clock ms per map call, under torch.no_grad."""
    with torch.no_grad():
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_timed):
            fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / n_timed


def sanity_check_compiled_matches_raw(
    d: int,
    hf: tuple[int, ...],
    x: torch.Tensor,
    y: torch.Tensor,
    atol: float = 1e-3,
) -> None:
    """Assert for_ladj/inv_ladj (both compile modes) match
    flow.t().call_and_ladj(x) / flow.t().inv.call_and_ladj(y), points + ladj."""
    f = make_flow(d, hf, x.device.type)
    with torch.no_grad():
        y_raw, lj_raw = f.t().call_and_ladj(x)
        x_raw, ilj_raw = f.t().inv.call_and_ladj(y)
        for mode in ("default", "reduce-overhead"):
            g = make_flow(d, hf, x.device.type)
            g.enable_for_ladj(mode=mode); g.enable_inv_ladj(mode=mode)
            yc, ljc = g.for_ladj(x)
            if not (torch.allclose(yc, y_raw, atol=atol) and torch.allclose(ljc, lj_raw, atol=atol)):
                sys.exit(f"FAIL — for_ladj drift at (d={d}, hf={hf}, mode={mode})")
            xc, iljc = g.inv_ladj(y)
            if not (torch.allclose(xc, x_raw, atol=atol) and torch.allclose(iljc, ilj_raw, atol=atol)):
                sys.exit(f"FAIL — inv_ladj drift at (d={d}, hf={hf}, mode={mode})")
            del g
    del f
    gc.collect()
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not torch.cuda.is_available():
        sys.exit(
            "compare_compiled_inverse.py requires CUDA — `reduce-overhead` uses\n"
            "CUDA Graphs which are CUDA-only, and a serious forward/inverse\n"
            "latency benchmark must run on the GPU (CPU timings are dominated\n"
            "by Python/dispatch overhead and miss the point)."
        )
    device = "cuda"
    dev_name = torch.cuda.get_device_name(0)
    print(f"device: {device}  ({dev_name})")
    print(f"grid:   d ∈ {list(DIMS)}, hidden_features ∈ {list(HIDDEN_FEATURES_GRID)}")
    print(f"setup:  bins={BINS}, transforms={TRANSFORMS}, BATCH={BATCH}, "
          f"warmup={WARMUP_STEPS}, timed={TIMED_STEPS}")
    print()

    # Open CSV up-front so partial runs (e.g. timeout) still leave a usable file.
    csv_path = Path(__file__).resolve().parent / "compare_compiled_inverse.csv"
    csv_fh = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow([
        "d", "hidden_features",
        "fwd_raw_ms", "fwd_default_ms", "fwd_reduce_ms",
        "fwd_speedup_default", "fwd_speedup_reduce",
        "inv_raw_ms", "inv_default_ms", "inv_reduce_ms",
        "inv_speedup_default", "inv_speedup_reduce",
    ])
    csv_fh.flush()

    bench_t0 = time.perf_counter()

    # rows[(d, hf)] = {'fwd': {mode: ms}, 'inv': {mode: ms}}
    rows: dict[tuple[int, tuple[int, ...]], dict[str, dict[str, float]]] = {}

    for d in DIMS:
        for hf in HIDDEN_FEATURES_GRID:
            torch.manual_seed(0)
            x = torch.randn(BATCH, d, device=device)            # source-space points
            y = torch.randn(BATCH, d, device=device) * 1.5      # codomain points to invert

            # one-shot correctness check across all compiled modes
            sanity_check_compiled_matches_raw(d, hf, x, y)

            fwd: dict[str, float] = {}
            inv: dict[str, float] = {}
            for mode in MODES:
                flow = make_flow(d, hf, device)
                fwd_fn, inv_fn = build_map_fns(mode, flow, x, y)
                fwd[mode] = time_map(fwd_fn, WARMUP_STEPS, TIMED_STEPS)
                inv[mode] = time_map(inv_fn, WARMUP_STEPS, TIMED_STEPS)
                print(f"  d={d:>2}, hf={str(hf):<12}  mode={mode:<16}  "
                      f"fwd {fwd[mode]:>8.3f} ms   inv {inv[mode]:>8.3f} ms")

                del flow, fwd_fn, inv_fn
                gc.collect()
                torch.cuda.empty_cache()

            rows[(d, hf)] = {"fwd": fwd, "inv": inv}

            # flush this (d, hf) row to CSV immediately
            fr, fd, fx = fwd["raw"], fwd["default"], fwd["reduce-overhead"]
            ir, idf, ix = inv["raw"], inv["default"], inv["reduce-overhead"]
            csv_writer.writerow([
                d, "x".join(map(str, hf)),
                f"{fr:.3f}", f"{fd:.3f}", f"{fx:.3f}",
                f"{fr / fd:.2f}", f"{fr / fx:.2f}",
                f"{ir:.3f}", f"{idf:.3f}", f"{ix:.3f}",
                f"{ir / idf:.2f}", f"{ir / ix:.2f}",
            ])
            csv_fh.flush()

    csv_fh.close()
    bench_t1 = time.perf_counter()

    # ──────────────────────────────────────────────────────────────
    # Final big table
    # ──────────────────────────────────────────────────────────────
    print()
    print("=" * 124)
    print("Per-call map latency (mean over", TIMED_STEPS, "calls, no_grad).  "
          "fwd = flow.t().call_and_ladj(x) vs flow.for_ladj(x);  "
          "inv = flow.t().inv.call_and_ladj(y) vs flow.inv_ladj(y).  ↑ = raw / compiled (×).")
    print("=" * 124)
    header = (
        f"{'d':>3} | {'hidden_feat':<11} | "
        f"{'fwd raw':>8} {'fwd def':>8} {'fwd red':>8} {'↑def':>6} {'↑red':>6} | "
        f"{'inv raw':>8} {'inv def':>8} {'inv red':>8} {'↑def':>6} {'↑red':>6}"
    )
    print(header)
    print("-" * len(header))
    for d in DIMS:
        for hf in HIDDEN_FEATURES_GRID:
            fwd, inv = rows[(d, hf)]["fwd"], rows[(d, hf)]["inv"]
            fr, fd, fx = fwd["raw"], fwd["default"], fwd["reduce-overhead"]
            ir, idf, ix = inv["raw"], inv["default"], inv["reduce-overhead"]
            print(
                f"{d:>3} | {str(hf):<11} | "
                f"{fr:>8.3f} {fd:>8.3f} {fx:>8.3f} {fr / fd:>6.2f} {fr / fx:>6.2f} | "
                f"{ir:>8.3f} {idf:>8.3f} {ix:>8.3f} {ir / idf:>6.2f} {ir / ix:>6.2f}"
            )

    print()
    print(f"total benchmark wall time: {(bench_t1 - bench_t0):.1f} s")
    print(f"csv saved → {csv_path}")


if __name__ == "__main__":
    main()
