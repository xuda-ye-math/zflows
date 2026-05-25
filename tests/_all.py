# pyright: reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false

"""Run every `tests/*.py` script as `python -m tests.<name>`, in its own
subprocess so each test is fully independent (no leaked state, no
shared Dynamo cache pollution).

Invocation::

    python -m tests._all

Behaviour:
  - Discovers every `tests/*.py` file except `__init__.py` and this script.
  - Runs the verification scripts (`_verify_*`) FIRST since they're cheap
    sanity checks and surface bugs before any of the longer demo / training
    scripts spend wall time.
  - Sets MPLBACKEND=Agg in the subprocess env so matplotlib `plt.show()`
    calls don't try to open a display.
  - Streams each subprocess's stdout / stderr live (no buffering).
  - Does NOT abort on the first failure. The final summary lists which
    scripts passed and which failed, and the runner exits with a non-zero
    status if any failed.

This is the Python sibling of `test_all.sh` — same behaviour, no shell
dependency, slightly nicer ordering. Either can be used to validate a
full sweep of the repo.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def discover_scripts() -> list[str]:
    """Return the stems of every test script to run, in stable order.

    Order:
        1. `_verify_*` scripts alphabetically  (fast sanity checks first)
        2. Everything else alphabetically       (demo + training scripts)
    Excludes `__init__.py` and this `_all.py` itself.
    """
    skip = {"__init__", "_all"}
    stems = sorted(p.stem for p in HERE.glob("*.py") if p.stem not in skip)
    verify = [s for s in stems if s.startswith("_verify_")]
    other = [s for s in stems if not s.startswith("_verify_")]
    return verify + other


def run_one(name: str) -> tuple[bool, float]:
    """Run `python -m tests.<name>` and return (passed, seconds)."""
    module = f"tests.{name}"
    print()
    print("═" * 72)
    print(f"  python -m {module}")
    print("═" * 72)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")  # no display required for tests with plt.show()
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
        env=env,
    )
    dt = time.perf_counter() - t0
    return result.returncode == 0, dt


def main() -> int:
    scripts = discover_scripts()
    if not scripts:
        print("No tests/*.py scripts found.")
        return 1

    print(f"Discovered {len(scripts)} scripts: {scripts}")

    results: list[tuple[str, bool, float]] = []
    t_start = time.perf_counter()
    for name in scripts:
        passed, dt = run_one(name)
        results.append((name, passed, dt))
    t_total = time.perf_counter() - t_start

    n_pass = sum(1 for _, p, _ in results if p)
    n_fail = sum(1 for _, p, _ in results if not p)
    print()
    print("═" * 72)
    print(f"Summary: {n_pass} passed, {n_fail} failed   (total {t_total:.1f} s)")
    print("═" * 72)
    for name, passed, dt in results:
        tag = "PASS" if passed else "FAIL"
        print(f"  {tag}  {dt:>7.1f} s   {name}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
