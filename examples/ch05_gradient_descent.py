#!/usr/bin/env python3
"""Chapter 5 worked example: gradient descent on f(x, y) = x^2 + 4y^2.

Reproduces the book's step-by-step walkthrough (start (2, 1),
learning rate 0.1) and the optimizer comparison, writing the canonical
output to outputs/ch05_gradient_descent.txt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from mathpowersai.optimizers import compare_optimizers, gradient_descent

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def f(p):
    return p[0] ** 2 + 4 * p[1] ** 2


def grad_f(p):
    return np.array([2 * p[0], 8 * p[1]])


def main() -> None:
    lines = []
    path = gradient_descent(grad_f, np.array([2.0, 1.0]), lr=0.1,
                            n_steps=10)

    lines.append("Gradient descent on f(x, y) = x^2 + 4y^2")
    lines.append(f"Start: (2.0, 1.0), lr = 0.1, f = {f(path[0]):.2f}")
    lines.append(
        f"Step 1: ({path[1][0]:.4g}, {path[1][1]:.4g}), "
        f"f = {f(path[1]):.2f}"
    )
    lines.append(
        f"After 10 iterations: ({path[-1][0]:.2f}, "
        f"{path[-1][1]:.1e}), f = {f(path[-1]):.3f}"
    )
    lines.append("")
    lines.append("Optimizer comparison (per-optimizer default lr, "
                 "50 steps):")
    results = compare_optimizers(grad_f, np.array([2.0, 1.0]),
                                 n_steps=50)
    for name, p in results.items():
        lines.append(f"  {name:18s} final f = {f(p[-1]):.6f}")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "ch05_gradient_descent.txt").write_text(text)


if __name__ == "__main__":
    main()
