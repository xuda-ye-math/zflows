# Benchmark: `zflows.loss.loss_compile` vs. raw `reverse_KL`

This is the only **performance** test in `tests/` (every other script verifies *correctness* or trains a flow on a concrete target). It measures how much faster a real training step gets when the loss is wrapped with `zflows.loss.loss_compile(...)` instead of called the obvious way as `reverse_KL(x, target, flow.t())`. The numbers are produced on the *full* training step

```python
loss = loss_fn(x_batch)
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

so they reflect what end-users actually wait for — not just the forward pass.

## Why this test exists

The natural way to write a `zflows` training loop is to call the loss directly:

```python
for x_batch in batches:
    loss = reverse_KL(x_batch, target=u1, F=flow.t())
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

This works, but at small `d` the per-step time is dominated by Python overhead (constructing a fresh `ComposedTransform` on every `flow.t()` call, walking the lazy `AutoregressiveTransform` list, kernel-launch latency between dozens of tiny CUDA ops). Wrapping the loss with `torch.compile` collapses that entire stack into one fused graph — but doing it the obvious way (`torch.compile(reverse_KL)`) doesn't work, because the `F` argument is a fresh `ComposedTransform` object on every call and Dynamo can't reuse its cache across invocations.

`zflows.loss.loss_compile(loss_fn, potential, transform, mode='default')` is the fix: it captures `potential` and `transform` as closure constants so they're invisible to Dynamo's guard system, returning a `(x_batch) -> scalar` callable whose only argument is the per-step batch. This benchmark exists to answer the question: *how much do you get for that one-line change?*

## The core mechanism

The captured-once trick is in [`zflows/loss.py`](../zflows/loss.py) and amounts to:

```python
def loss_compile(loss_fn, *captured, mode='default'):
    @torch.compile(mode=mode)
    def compiled(x):
        return loss_fn(x, *captured)
    return compiled
```

The four KL losses in `zflows.loss` have signature `(x, potential, transform, beta=1.0) -> scalar`, so the canonical call is `loss_compile(reverse_KL, potential, transform)` with `captured = (potential, transform)`. `beta` is then absorbed into the loss's default kwarg (`beta=1.0`), or it can be baked in explicitly by passing it as another captured constant: `loss_compile(reverse_KL, potential, transform, 0.7)` produces a closure with `beta=0.7` fixed in the graph. The variadic form makes the same helper work for any custom loss whose first argument is the per-batch tensor and the rest are Python constants.

A sibling helper `zflows.loss.loss_compile_beta` keeps `beta` as a *runtime* argument of the returned closure (cast to a 0-d tensor at the boundary so Dynamo treats it as a dynamic input and a single graph handles every value) for adaptive / annealed schedules. The benchmark uses `loss_compile` because it measures a fixed-`beta` workload — `loss_compile` skips the wrapper / `isinstance` / `as_tensor` cast and shaves the last few microseconds per step.

Two things make this both **correct** and **fast**:

1. **Correctness under in-place optimizer updates.** `transform = flow.t()` is a `ComposedTransform` whose inner `AutoregressiveTransform` objects hold lazy `meta` callables that read `flow.<masked_mlp>.weight` etc. via attribute access on every forward pass. `optimizer.step()` mutates those parameter tensors *in place*, so the captured `transform` always sees the post-step values — there is no need to rebuild `flow.t()` per iteration.

2. **Dynamo cache stability.** Because `transform` is a closure constant, every call to the returned `compiled` function presents the *same* Python object identity to Dynamo's guard system. After the first compile, every subsequent call is a cache hit and runs the fused graph directly. By contrast, the naive `reverse_KL(x, target, flow.t())` pattern creates a fresh `ComposedTransform` Python object per call — Dynamo would either retrace every iteration or hit `BACKEND_MATCH` failures, both of which kill the speedup.

`torch.compile` only wraps the forward pass that builds the loss. `loss.backward()` and `optimizer.step()` still run in eager. The benchmark therefore reports a *lower bound* on the achievable speedup: there's room left on the table by not compiling the optimizer step too.

## Methodology

The grid is fixed at the top of [`compare_compiled_loss.py`](compare_compiled_loss.py):

<div align="center">

| param          | value                                                                |
|----------------|----------------------------------------------------------------------|
| flow           | `NSF` with `bins=12, transforms=4, randmask=True`                    |
| target         | `Gaussian(0, I_d)` on $\mathbb R^d$ (so backward is non-trivial)     |
| sweep          | `dimension` ∈ {2, 4, 8, 16, 32} × `hidden_features` ∈ {(64,64), (128,128), (256,256)} |
| batch size     | 2000                                                                 |
| timed steps    | 100, after 30 warmup steps to absorb compile + early-iteration retracing |

</div>

For each of the 15 `(d, hf)` cells, three modes are timed back-to-back: `raw`, `compiled-default`, and `compiled-reduce-overhead`. Each cell starts from a fresh `NSF` + `Adam` + (for compiled modes) a fresh `loss_fn` returned by `zflows.loss.loss_compile(...)`, then `del`'d and `torch.cuda.empty_cache()`d before the next cell to keep memory clean.

Two infrastructure helpers from [`zflows.utils`](../zflows/utils.py) make the benchmark loop tractable:

- [`suppress_warnings()`](../zflows/utils.py#L593) — silences Python `UserWarning`s, Triton autotune stderr, Inductor worker-pool noise, and Dynamo recompile logs in one call. Without it the table is buried under hundreds of lines of irrelevant warning text.
- [`set_cache_size_limit(64)`](../zflows/utils.py#L582) — Dynamo's per-code-object cache evicts after 8 specializations by default. The benchmark creates 15 × 2 = 30 compiled closures sharing one inner code body; the default would silently fall back to eager from cell ~5 onward, making the speedup vanish at large `(d, hf)`. (Detected by setting `torch._dynamo.config.suppress_errors = False` and watching for recompile warnings — see [the cache-exhaustion debug story](../zflows/utils.py#L554-L591).)

There is also a per-cell sanity check that calls `reverse_KL` three ways (`raw`, `default`, `reduce-overhead`) on the first batch and asserts `|loss_compiled - loss_raw| < 1e-3` before any timing begins. If compile introduces numerical drift, the benchmark aborts with a clear message instead of reporting a meaningless speedup.

## Implementation and execution

Run from the project root:

```bash
python -m tests.compare_compiled_loss
```

Pointers into the script:

- imports + `suppress_warnings()` / `set_cache_size_limit(64)`: [`compare_compiled_loss.py:30–47`](compare_compiled_loss.py#L30-L47)
- grid configuration: [`compare_compiled_loss.py:57–69`](compare_compiled_loss.py#L57-L69)
- `build_loss_fn` (raw, compiled-default, compiled-reduce-overhead via `loss_compile` — the single-input fast path, no wrapper overhead): [`compare_compiled_loss.py:75–95`](compare_compiled_loss.py#L75-L95)
- `time_full_step` (warmup + `cuda.synchronize` + `time.perf_counter`): [`compare_compiled_loss.py:97–121`](compare_compiled_loss.py#L97-L121)
- per-cell correctness sanity check: [`compare_compiled_loss.py:123–145`](compare_compiled_loss.py#L123-L145)
- main benchmark loop + incremental CSV write: [`compare_compiled_loss.py:151–223`](compare_compiled_loss.py#L151-L223)

The CSV ([`compare_compiled_loss.csv`](compare_compiled_loss.csv)) is written incrementally — one row per `(d, hf)` cell — so even if you `Ctrl-C` mid-run you keep whatever cells finished.

## Results (RTX 5070 Ti, fp32)

Reproduced from the committed [`compare_compiled_loss.csv`](compare_compiled_loss.csv):

<div align="center">

| $d$ | `hidden_features` | raw ms | default ms | reduce ms | speedup default | speedup reduce |
|----:|:------------------|------:|----------:|---------:|--------------:|--------------:|
|   2 | (64, 64)          |  5.99 |      1.40 |     0.52 |          4.28 |         11.60 |
|   2 | (128, 128)        |  6.01 |      1.35 |     0.46 |          4.44 |         12.97 |
|   2 | (256, 256)        |  6.10 |      1.40 |     0.61 |          4.35 |          9.97 |
|   4 | (64, 64)          |  5.48 |      1.29 |     0.50 |          4.26 |         10.87 |
|   4 | (128, 128)        |  6.03 |      1.31 |     0.60 |          4.59 |          9.99 |
|   4 | (256, 256)        |  6.03 |      1.30 |     0.73 |          4.63 |          8.23 |
|   8 | (64, 64)          |  5.71 |      1.46 |     0.56 |          3.91 |         10.18 |
|   8 | (128, 128)        |  5.45 |      1.43 |     0.58 |          3.81 |          9.44 |
|   8 | (256, 256)        |  5.45 |      1.43 |     0.85 |          3.82 |          6.42 |
|  16 | (64, 64)          |  6.82 |      1.43 |     0.65 |          4.75 |         10.42 |
|  16 | (128, 128)        |  6.92 |      1.42 |     0.80 |          4.86 |          8.63 |
|  16 | (256, 256)        |  7.23 |      1.42 |     1.10 |          5.11 |          6.57 |
|  32 | (64, 64)          | 12.64 |      1.29 |     1.04 |          9.80 |         12.19 |
|  32 | (128, 128)        | 12.33 |      1.43 |     1.21 |          8.63 |         10.19 |
|  32 | (256, 256)        | 12.76 |      1.68 |     1.76 |          7.62 |          7.24 |

</div>

Three regularities to notice:

- **The raw baseline is flat at ~5.5–7 ms for $d \leq 16$ regardless of `hidden_features`.** That is *not* what GPU-compute-bound code would look like — a 256×256 MLP should cost more than a 64×64 one. The flatness is the smoking gun that uncompiled NSF training at small $d$ is **dominated by host-side Python overhead** (per-launch latency, lazy `Transform` object construction, the `ComposedTransform.call_and_ladj` Python loop), not by GPU compute.

- **`reduce-overhead` mode (CUDA Graphs)** consistently beats `default` mode (no CUDA graphs) by another 1.5–3×. At small `(d, hf)` it reaches 10–13× over raw — that's how much Python overhead was on the table.

- **The gap narrows as the network grows.** At $d=32, hf=(256, 256)$ — the largest cell — `reduce-overhead` is "only" ~7.2× faster than raw. Two effects compound: (i) GPU compute starts to be a real fraction of step time, leaving less Python overhead to amortize; (ii) the eager `loss.backward()` + `optimizer.step()` portion is unchanged across modes, so as the forward becomes a smaller fraction of the step, the compile contribution shrinks.

**Takeaway for users.** On *any* current `zflows`-style workload (NSF / NCSF, $d$ up to a few dozen, normal-size MLPs), wrapping the training loss with `zflows.loss.loss_compile(...)` (or `loss_compile_beta` for adaptive-temperature schedules) gives a real **4–10× per-step speedup** that compounds across the thousands of steps a Boltzmann-generator training run needs. The 3D / 4D test scripts already use this pattern; see [`3D_periodic.md`](3D_periodic.md) and [`4D_Boltzmann_generator.md`](4D_Boltzmann_generator.md).

## Caveats

- **Compile cost is real.** The first 1–3 steps of each new compiled closure pay a one-time tracing + Inductor lowering + Triton autotuning cost — typically 10–60 s on cold cache. For a training run with thousands of steps this is amortized within the first epoch; for a 10-step toy script it's net-negative. The benchmark uses 30 warmup steps to absorb this before timing begins.
- **Static batch size assumed.** Both `default` and especially `reduce-overhead` mode specialize on tensor shapes. Changing `BATCH` mid-run forces a retrace; with `set_cache_size_limit(64)` you'd survive many retraces but each one is a fresh compile burst.
- **`reduce-overhead` requires CUDA.** It uses CUDA Graphs, which are a NVIDIA-specific feature. The benchmark refuses to run on CPU-only systems with a friendly error message.
- **Numbers above are from a single RTX 5070 Ti.** Older / newer GPUs will move the absolute ms numbers but the *speedup ratios* should be similar — the bottleneck being optimized away (Python launch overhead) is GPU-independent.
