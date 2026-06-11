"""Regenerate Figure 8.5 (digit_reconstructions.pdf) from real MNIST.

Mirrors the capstone chapter's setup exactly: fetch_openml('mnist_784'),
seed 42, 10,000-sample subset, PCA via SVD on the centered data, then a
single digit reconstructed at 1/10/50/100/200 principal components beside
the original. Panel titles only — the LaTeX \\caption carries the prose.

Writes:
    Book/publish/images/pca_reconstructions/digit_reconstructions.pdf
    Book/publish/images/pca_reconstructions/digit_reconstructions.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

OUT_DIR = Path(__file__).resolve().parents[3] / "images" / "pca_reconstructions"

COMPONENTS_TO_SHOW = [1, 10, 50, 100, 200]
SAMPLE_INDEX = 0  # first digit of the seeded subset, as in the chapter


def main():
    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float64)
    y = mnist.target.astype(int)

    n_samples = 10000
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]

    data_mean = X_subset.mean(axis=0)
    X_centered = X_subset - data_mean

    # Principal components via SVD of the centered data matrix.
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    eigenvectors = Vt.T  # columns are principal components

    x = X_subset[SAMPLE_INDEX]
    x_centered = X_centered[SAMPLE_INDEX]
    digit = y_subset[SAMPLE_INDEX]
    print(f"Reconstructing sample {SAMPLE_INDEX} (digit {digit})")

    n_panels = len(COMPONENTS_TO_SHOW) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(2.1 * n_panels, 2.4))

    for ax, n_comp in zip(axes[:-1], COMPONENTS_TO_SHOW):
        V_k = eigenvectors[:, :n_comp]
        x_recon = (x_centered @ V_k) @ V_k.T + data_mean
        ax.imshow(x_recon.reshape(28, 28), cmap="gray")
        ax.set_title(f"{n_comp} PC" + ("s" if n_comp > 1 else ""), fontsize=12)
        ax.axis("off")

    axes[-1].imshow(x.reshape(28, 28), cmap="gray")
    axes[-1].set_title("Original", fontsize=12)
    axes[-1].axis("off")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"digit_reconstructions.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
