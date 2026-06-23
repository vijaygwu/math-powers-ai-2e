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
LIGHTBLUE = "#D8E7F4"


def f(p):
    return p[0] ** 2 + 10 * p[1] ** 2


def grad_f(p):
    return np.array([2 * p[0], 20 * p[1]])


def draw_path(ax, path, *, color, marker, zorder, linewidth=2.1):
    """Draw a readable optimizer path without over-marking the converged tail."""
    ax.plot(path[:, 0], path[:, 1], "-", color=color, linewidth=linewidth,
            solid_capstyle="round", zorder=zorder)
    marker_idx = np.unique(np.r_[1, np.arange(4, len(path) - 1, 3), len(path) - 1])
    ax.scatter(path[marker_idx, 0], path[marker_idx, 1], s=22, marker=marker,
               facecolor=color, edgecolor="white", linewidth=0.5, zorder=zorder + 1)


def add_path_arrows(ax, path, *, color, indices, zorder):
    """Place small arrowheads on selected path segments to make time direction clear."""
    for i in indices:
        if i + 1 >= len(path):
            continue
        ax.annotate(
            "",
            xy=path[i + 1],
            xytext=path[i],
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.8,
                shrinkA=3,
                shrinkB=3,
                mutation_scale=12,
            ),
            zorder=zorder,
        )


def main() -> None:
    x0 = np.array([1.0, 1.0])
    gd_path = np.array(gradient_descent(grad_f, x0, lr=0.08, n_steps=14))
    mom_path = np.array(momentum(grad_f, x0, lr=0.08, beta=0.5, n_steps=14))

    fig, ax = plt.subplots(figsize=(7.2, 4.85))

    # Elongated contours of the ill-conditioned bowl.
    gx = np.linspace(-1.25, 1.25, 300)
    gy = np.linspace(-1.15, 1.15, 300)
    X, Y = np.meshgrid(gx, gy)
    Z = X**2 + 10 * Y**2
    levels = [0.05, 0.2, 0.5, 1, 2, 4, 7, 11]
    ax.contour(X, Y, Z, levels=levels, colors=PRIMARYBLUE,
               linewidths=0.55, alpha=0.32)
    ax.contourf(X, Y, Z, levels=levels, colors=[LIGHTBLUE] * (len(levels) - 1),
                alpha=0.24)

    # Momentum first, GD on top: the two paths share their first segment, and
    # GD's zigzag needs to remain visible from Start.
    draw_path(ax, mom_path, color=PRIMARYGREEN, marker="s", zorder=3,
              linewidth=1.9)
    draw_path(ax, gd_path, color=PRIMARYRED, marker="o", zorder=4,
              linewidth=2.4)
    add_path_arrows(ax, gd_path, color=PRIMARYRED, indices=[0, 1, 2, 4, 7],
                    zorder=6)
    add_path_arrows(ax, mom_path, color=PRIMARYGREEN, indices=[0, 2, 4, 7],
                    zorder=5)

    ax.plot(1, 1, "o", color="black", markersize=7, zorder=5)
    ax.annotate("Start", (1, 1), textcoords="offset points",
                xytext=(-16, 6), ha="right", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                          alpha=0.85))
    ax.plot(0, 0, "*", color="black", markersize=14, zorder=5)
    ax.annotate("Min", (0, 0), textcoords="offset points",
                xytext=(-50, -23), fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                          alpha=0.85))

    ax.annotate("momentum damps\noscillation", xy=(0.42, 0.32),
                xytext=(-0.48, 0.88),
                arrowprops=dict(arrowstyle="->", color=PRIMARYGREEN, lw=1.0),
                color=PRIMARYGREEN, fontsize=8, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.82))
    ax.annotate(r"GD zigzags ($\eta=0.08$)", xy=(0.7056, 0.36),
                xytext=(0.94, 0.72),
                arrowprops=dict(arrowstyle="->", color=PRIMARYRED, lw=1.0),
                color=PRIMARYRED, fontsize=8, fontweight="bold",
                ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.72))

    ax.set_xlabel("$x$: gentle direction ($\\lambda = 2$)", fontsize=11)
    ax.set_ylabel("$y$: steep direction ($\\lambda = 20$)", fontsize=11)
    ax.set_xlim(-1.25, 1.30)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"GD final point:       {gd_path[-1]},  f = {f(gd_path[-1]):.4f}")
    print(f"Momentum final point: {mom_path[-1]},  f = {f(mom_path[-1]):.4f}")


if __name__ == "__main__":
    main()
