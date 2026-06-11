#!/usr/bin/env python3
"""Run every chapter example and regenerate outputs/.

CI runs this and fails on any diff against the committed canon, so a
printed number in the book can never silently drift from the code
that claims to produce it.
"""

import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent

EXAMPLES = [
    "ch01_linear_algebra.py",
    "ch02_probability.py",
    "ch03_calculus.py",
    "ch04_information.py",
    "ch05_gradient_descent.py",
    "ch06_decompositions.py",
    "ch07_spaces.py",
    "ch08_numerics.py",
    "capstone_pca.py",
]


def main() -> int:
    failures = []
    for name in EXAMPLES:
        script = EXAMPLES_DIR / name
        if not script.exists():
            print(f"  [skip] {name} (not present)")
            continue
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True
        )
        status = "ok" if result.returncode == 0 else "FAIL"
        print(f"  [{status}] {name}")
        if result.returncode != 0:
            print(result.stderr[-2000:])
            failures.append(name)
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("PASSED: all outputs regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
