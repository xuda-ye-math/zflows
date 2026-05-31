# Benchmark: `flow.for_ladj` / `flow.inv_ladj` vs. raw `flow.t().call_and_ladj`

A **performance** test (sibling of [`compare_compiled_loss.md`](compare_compiled_loss.md)). It measures how much faster the forward and inverse maps of an `NSF` — each fused with its `log|det J|` — get when you compile them via the optional `Flow.enable_for_ladj()` / `Flow.enable_inv_ladj()` fast paths, instead of calling the obvious

```python
y,      ladj     = flow.t().call_and_ladj(x)        # forward + Jacobian
x_back, ladj_inv = flow.t().inv.call_and_ladj(y)    # inverse + Jacobian
```

on every step. Numbers are the **pure map** wall-clock under `torch.no_grad()` — no autograd, no optimizer — so they isolate the map cost itself.

## Why this test exists

The natural way to push points through a trained flow is `flow.t().call_and_ladj(x)` / `flow.t().inv.call_and_ladj(y)`. That works, but each call reconstructs a fresh `ComposedTransform`, walks the lazy `AutoregressiveTransform` list in Python, and launches dozens of tiny CUDA kernels — so at small `d` it is **host-side-overhead-bound**, not GPU-compute-bound. The inverse is worse: for spline flows it is a per-coordinate bisection, i.e. many sequential tiny ops.

`Flow.enable_for_ladj(mode=...)` / `Flow.enable_inv_ladj(mode=...)` (in [`zflows/flow.py`](../zflows/flow.py)) are the fix, mirroring `Potential.enable_grad`: each captures `F = self.t()` and `torch.compile`s it, returning `flow.for_ladj(x) == flow.t().call_and_ladj(x)` → `(y, ladj)` and `flow.inv_ladj(y) == flow.t().inv.call_and_ladj(y)` → `(x_pre, ladj_inv)`. Both hand back the **fused `(points, log|det J|)`** — `inv_ladj`'s ladj is the inverse map's, i.e. `−log|det J_F|` at the pre-image. This benchmark answers: *how much do you get for that one-line opt-in?*

## ⚠️ Do you need to re-enable after changing the flow?

The one thing to internalize: `enable_for_ladj` / `enable_inv_ladj` bind to `F = self.t()` at enable time, and `F` re-reads the flow's parameter **tensors** on every call. So the answer splits cleanly:

- **NO — in-place parameter updates are reflected automatically.** `optimizer.step()`, `load_state_dict(...)`, and `zeros()` all mutate the existing parameter tensors *in place* (same objects, same memory). The captured `F` reads those same tensors, so `for_ladj` / `inv_ladj` already return the updated map with **no re-enable**. Verified in `_verify_flow.py` §16e (still match `flow.t()` after an `Adam.step()`).

- **YES — if the parameters become *different* tensors, re-enable to refresh.** A `.to(device)` / `.to(dtype)` move, swapping a submodule (`flow.layer = NewLayer(...)`), or rebuilding the flow allocates **new** parameter tensors; the previously compiled artifact is now stale (under `mode='reduce-overhead'` it even points at freed CUDA-graph memory). Just call `enable_*` again:

  ```python
  flow.to(torch.float64)                    # parameters reallocated
  flow.enable_for_ladj().enable_inv_ladj()  # recompile against the new flow.t()
  ```

  This works **because `enable_for_ladj` / `enable_inv_ladj` are deliberately NOT idempotent** — each call rebuilds `F = self.t()` and recompiles a fresh `_for_ladj_fn` / `_inv_ladj_fn`. (An earlier version early-returned if already enabled; that guard was removed precisely so re-calling refreshes.) Verified in `_verify_flow.py` §16f (`.to(float64)` + re-enable stays correct) and §16b (re-call yields a fresh compiled fn).

**Rule of thumb:** enable once after the flow's structure/device/dtype is set; re-enable only when you *reallocated* parameters (not for ordinary training, which updates in place). Each (re-)enable pays the one-time torch.compile cost.

(Footnote: directly reassigning `p.data = some_new_tensor` of *identical* shape/dtype/device is a degenerate case `torch.compile` may freeze and not pick up even on re-enable — but no normal workflow does that; real changes either update in place or change tensor metadata.)

## The core mechanism

```python
def enable_inv_ladj(self, mode="reduce-overhead"):
    F = self.t()  # rebuilt each call so re-enabling refreshes the capture
    self._inv_ladj_fn = torch.compile(lambda x: F.inv.call_and_ladj(x), mode=mode)
    return self

def inv_ladj(self, x):  # -> (x_pre, ladj_inv)
    if self._inv_ladj_fn is None:
        raise RuntimeError("...requires .enable_inv_ladj() first.")
    return self._inv_ladj_fn(x)
```

`enable_for_ladj` is identical with `F.call_and_ladj(x)` instead of `F.inv.call_and_ladj(x)`. Both return the fused `(points, log|det J|)`. The compiled path additionally saves the per-call `flow.t()` reconstruction the raw pattern pays.

## Methodology

Grid fixed at the top of [`compare_compiled_inverse.py`](compare_compiled_inverse.py):

<div align="center">

| param        | value                                                                                  |
|--------------|----------------------------------------------------------------------------------------|
| flow         | `NSF` with `bins=12, transforms=4, randmask=True`                                       |
| sweep        | `dimension` ∈ {2, 4, 8, 16, 32} × `hidden_features` ∈ {(64,64), (128,128), (256,256)}  |
| batch size   | 2000                                                                                   |
| timed calls  | 50, after 20 warmup calls to absorb compile + early-iteration retracing                |
| modes        | `raw` (fresh `flow.t()` per call), `compiled-default`, `compiled-reduce-overhead`      |

</div>

For each of the 15 `(d, hf)` cells, the forward and inverse maps are timed in all three modes on a *seeded* `NSF` (identical weights per mode, so every mode times the same bijection). A per-cell sanity check asserts `for_ladj`/`inv_ladj` match `flow.t().call_and_ladj` / `flow.t().inv.call_and_ladj` — both the points **and** the log|det J| — to `1e-3` in both compile modes before any timing; if compile introduces drift, the run aborts. `suppress_warnings()` + `set_cache_size_limit(64)` keep the Dynamo/Triton noise down and prevent silent eager fallback across the 15 × 2 compiled closures.

**CUDA-only.** `reduce-overhead` uses CUDA Graphs, and a serious map-latency comparison must run on the GPU (CPU timings are dispatch-overhead noise). The script `sys.exit`s on CPU.

Run from the project root:

```bash
python -m tests.compare_compiled_inverse
```

The CSV ([`compare_compiled_inverse.csv`](compare_compiled_inverse.csv)) is written one row per `(d, hf)` cell, so a `Ctrl-C` mid-run still leaves the finished cells.

## Results (RTX 5070 Ti, fp32)

Per-call latency, mean over 50 calls; each call returns `(points, log|det J|)`. `↑` = raw / compiled. Reproduced from the committed [`compare_compiled_inverse.csv`](compare_compiled_inverse.csv):

<div align="center">

| $d$ | `hidden_features` | fwd raw | fwd def | fwd red | fwd ↑def | fwd ↑red | inv raw | inv def | inv red | inv ↑def | inv ↑red |
|----:|:------------------|--------:|--------:|--------:|---------:|---------:|--------:|--------:|--------:|---------:|---------:|
|   2 | (64, 64)          |   1.573 |   0.328 |   0.235 |     4.80 |     6.69 |   4.291 |   0.685 |   0.582 |     6.26 |     7.37 |
|   2 | (128, 128)        |   1.602 |   0.309 |   0.308 |     5.18 |     5.20 |   4.260 |   0.780 |   0.792 |     5.46 |     5.38 |
|   2 | (256, 256)        |   1.579 |   0.419 |   0.415 |     3.77 |     3.80 |   4.282 |   1.184 |   1.131 |     3.62 |     3.79 |
|   4 | (64, 64)          |   1.625 |   0.342 |   0.247 |     4.75 |     6.59 |   7.117 |   1.130 |   1.100 |     6.30 |     6.47 |
|   4 | (128, 128)        |   1.611 |   0.331 |   0.329 |     4.87 |     4.89 |   7.059 |   1.457 |   1.439 |     4.84 |     4.90 |
|   4 | (256, 256)        |   1.610 |   0.509 |   0.504 |     3.16 |     3.19 |   7.270 |   2.420 |   2.347 |     3.00 |     3.10 |
|   8 | (64, 64)          |   3.845 |   0.333 |   0.331 |    11.54 |    11.63 |  32.991 |   2.695 |   2.605 |    12.24 |    12.67 |
|   8 | (128, 128)        |   3.975 |   0.484 |   0.435 |     8.22 |     9.14 |  34.049 |   3.715 |   3.569 |     9.17 |     9.54 |
|   8 | (256, 256)        |   4.163 |   0.598 |   0.599 |     6.97 |     6.95 |  35.522 |   5.132 |   5.113 |     6.92 |     6.95 |
|  16 | (64, 64)          |   6.523 |   0.428 |   0.427 |    15.25 |    15.29 | 107.037 |   6.884 |   6.679 |    15.55 |    16.03 |
|  16 | (128, 128)        |   6.561 |   0.513 |   0.512 |    12.79 |    12.81 | 108.207 |   8.285 |   8.035 |    13.06 |    13.47 |
|  16 | (256, 256)        |   6.803 |   0.756 |   0.759 |     9.00 |     8.96 | 111.941 |  12.130 |  11.859 |     9.23 |     9.44 |
|  32 | (64, 64)          |  11.847 |   0.600 |   0.607 |    19.73 |    19.53 | 384.586 |  19.143 |  18.701 |    20.09 |    20.57 |
|  32 | (128, 128)        |  12.053 |   0.771 |   0.772 |    15.62 |    15.61 | 390.656 |  24.547 |  24.225 |    15.91 |    16.13 |
|  32 | (256, 256)        |  12.404 |   1.116 |   1.113 |    11.11 |    11.15 | 402.105 |  35.689 |  35.321 |    11.27 |    11.38 |

</div>

Things to notice:

- **The speedup grows with `d`.** At `d=2` the raw maps are already cheap (~1.6 ms fwd, ~4.3 ms inv) so compiling buys ~4–7×. At `d=32` the raw **inverse** costs **~385–402 ms** — the bisection's sequential per-coordinate ops murder eager dispatch — and compiling collapses it to ~19–36 ms, a **~11–21×** win. The forward shows the same trend (~20× at `d=32, (64,64)`). Classic host-overhead-bound signature: the bigger the op count, the more there is to fuse away.

- **The inverse benefits at least as much as the forward.** Despite the spline bisection emitting a `torch.compile` graph-break warning (it stays numerically correct — the sanity check enforces `1e-3` on both points and ladj), the inverse speedup (3.0–20.6×) tracks or exceeds the forward (3.2–19.7×), because the eager inverse had far more per-op overhead to remove.

- **`reduce-overhead` ≈ `default` here.** Unlike the loss benchmark (where CUDA Graphs gave a further 1.5–3×), the two compile modes are within noise for these pure maps. The forward/inverse graphs are simpler (no backward, no optimizer), so the kernel-launch savings CUDA Graphs add are already captured by the fused `default` graph; the residual launch count is small.

- **The gain shrinks as the MLP widens.** Within a fixed `d`, going (64,64) → (256,256) lowers the ratio (e.g. `d=8`: 12.7× → 6.9× inverse) because real GPU compute becomes a larger fraction of the call, leaving less Python overhead to amortize.

**Takeaway.** When you repeatedly map a fixed-shape batch through a *finalized* flow — e.g. the `G⁻¹` source-pushforward in [`utils.annealed_importance_sampling_G`](../zflows/utils.py), or any inference/inversion loop — `flow.enable_for_ladj()` / `flow.enable_inv_ladj()` give a real **~3–21× per-call speedup**, largest exactly where the eager map hurts most (high `d`, the inverse). Just heed the re-enable rules above: enable *after* the flow is structurally final, re-enable after a reallocation.

## Caveats

- **Re-enable after reallocation** — see the warning section. In-place updates are tracked automatically; `.to()` / dtype / structural changes need a fresh `enable_*` call (which recompiles, since these are not idempotent).
- **Compile cost is real.** Each new compiled closure pays a one-time trace + Inductor lowering (+ Triton autotune); the benchmark uses 20 warmup calls to absorb it. The full 15-cell sweep took ~15 min wall-clock, dominated by compilation.
- **Static batch shape assumed.** Both modes specialize on input shape; changing the batch size forces a retrace.
- **`reduce-overhead` requires CUDA** (CUDA Graphs). The script refuses to run on CPU.
- **Spline-inverse graph-break.** `NSF`/`NCSF` inverses graph-break under `torch.compile` (correct, but more graph than a single fused kernel); closed-form/ODE inverses (`RealNVP`, `CNF`, `OTFlow`) compile cleanly — swap the flow class in the script to compare.
- **Single-GPU numbers.** Absolute ms move across GPUs, but the *ratios* (Python-overhead removal) are GPU-independent.
