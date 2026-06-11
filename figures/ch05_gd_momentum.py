#!/usr/bin/env python3
"""Figure 5.6 — GD vs momentum on the ill-conditioned quadratic.

Generates the book figure FROM the companion package's optimizers, so
the plotted paths are exactly what the shipped code produces (the old
TikZ figure hardcoded a momentum path the algorithm does not take).

f(x, y) = x^2 + 10 y^2, kappa = 10, start (1, 1), lr = 0.08.
Momentum uses beta = 0.5: enough to damp the cross-valley oscillation
and accelerate along the gentle direction without overshooting.

Writes: Book/publish/images/figures/ch05_gd_momentum.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mathpowersai.optimizers import gradient_descent, momentum

OUT = (
    Path(__file__).resolve().parents[3]
    / "images" / "figures" / "ch05_gd_momentum.pdf"
)

PRIMARYRED = "#C0392B"
PRIMARYGREEN = "#1E8449"
PRIMARYBLUE = "#2C5F8A"


def f(p):
    return p[0] ** 2 + 10 * p[1] ** 2


def grad_f(p):
    return np.array([2 * p[0], 20 * p[1]])


def main() -> None:
    x0 = np.array([1.0, 1.0])
    gd_path = np.array(gradient_descent(grad_f, x0, lr=0.08, n_steps=14))
    mom_path = np.array(momentum(grad_f, x0, lr=0.08, beta=0.5, n_steps=14))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Elongated contours of the ill-conditioned bowl.
    gx = np.linspace(-1.25, 1.25, 300)
    gy = np.linspace(-1.15, 1.15, 300)
    X, Y = np.meshgrid(gx, gy)
    Z = X**2 + 10 * Y**2
    levels = [0.05, 0.2, 0.5, 1, 2, 4, 7, 11]
    ax.contour(X, Y, Z, levels=levels, colors=PRIMARYBLUE,
               linewidths=0.6, alpha=0.45)
    ax.contourf(X, Y, Z, levels=levels, cmap="Blues", alpha=0.18)

    # Momentum first (z=3), GD on top (z=4): the two paths share their
    # first segment (momentum starts with v = 0), and GD's zigzag is
    # the figure's protagonist, so it must stay visible from Start.
    ax.plot(mom_path[:, 0], mom_path[:, 1], "-s", color=PRIMARYGREEN,
            linewidth=1.8, markersize=4.5,
            label=r"Momentum ($\eta = 0.08$, $\beta = 0.5$)", zorder=3)
    ax.plot(gd_path[:, 0], gd_path[:, 1], "-o", color=PRIMARYRED,
            linewidth=1.8, markersize=4.5, label=r"GD ($\eta = 0.08$)",
            zorder=4)

    ax.plot(1, 1, "o", color="black", markersize=7, zorder=5)
    ax.annotate("Start", (1, 1), textcoords="offset points",
                xytext=(8, 4), fontsize=10, fontweight="bold")
    ax.plot(0, 0, "*", color="black", markersize=14, zorder=5)
    ax.annotate("Min", (0, 0), textcoords="offset points",
                xytext=(-34, -16), fontsize=10, fontweight="bold")

    ax.set_xlabel("$x$  (gentle curvature, $\\lambda = 2$)", fontsize=11)
    ax.set_ylabel("$y$  (steep curvature, $\\lambda = 20$)", fontsize=11)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.15, 1.15)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95,
              edgecolor="0.7")
    ax.set_aspect("equal")
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"GD final point:       {gd_path[-1]},  f = {f(gd_path[-1]):.4f}")
    print(f"Momentum final point: {mom_path[-1]},  f = {f(mom_path[-1]):.4f}")


if __name__ == "__main__":
    main()
