"""
Capstone demo: PCA from scratch.

Companion script for the Part I capstone of "The Math That
Powers AI" (2nd ed.).

Default mode runs on a seeded synthetic dataset (low-rank plus
noise Gaussian, np.random.default_rng(42)) so it is fast and
deterministic, and writes outputs/capstone_pca.txt.

With --mnist, attempts the book's full MNIST run via sklearn's
fetch_openml (downloads ~55MB on first run).

When scikit-learn is available, the from-scratch PCA is also
validated against sklearn.decomposition.PCA.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mathpowersai.pca import (  # noqa: E402
    PCA,
    compare_methods,
    pca_via_eigendecomposition,
    pca_via_svd,
    variance_threshold_components,
)

THRESHOLDS = [0.5, 0.8, 0.9, 0.95, 0.99]


def make_synthetic_data(rng, n_samples=500, n_features=30,
                        rank=5, noise_scale=0.1):
    """
    Seeded low-rank-plus-noise Gaussian dataset.

    X = Z W + noise + offset, where Z is (n_samples, rank) and
    W is (rank, n_features), so the signal lives in a rank-`rank`
    subspace of the `n_features`-dimensional space.
    """
    Z = rng.normal(size=(n_samples, rank))
    W = rng.normal(size=(rank, n_features))
    noise = noise_scale * rng.normal(
        size=(n_samples, n_features)
    )
    offset = rng.normal(size=n_features)
    return Z @ W + noise + offset


def load_mnist():
    """Load MNIST via sklearn's fetch_openml (book's setup)."""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        print(
            "scikit-learn is required for the MNIST run.\n"
            "Install it with:  pip install scikit-learn\n"
            "Or run without --mnist for the synthetic demo."
        )
        sys.exit(1)

    print("Loading MNIST dataset (may download ~55MB)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float64)

    # Use a subset for faster computation, as in the book
    n_samples = 10000
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], n_samples, replace=False)
    return X[idx]


def validate_against_sklearn(X, n_components, lines):
    """
    Validate the from-scratch PCA against sklearn PCA when
    sklearn is available: allclose on explained_variance_ratio_
    and on components up to sign.
    """
    try:
        from sklearn.decomposition import PCA as SklearnPCA
    except ImportError:
        lines.append(
            "sklearn not installed; skipping validation."
        )
        return

    ours = PCA(n_components=n_components).fit(X)
    theirs = SklearnPCA(n_components=n_components).fit(X)

    ratio_match = np.allclose(
        ours.explained_variance_ratio_,
        theirs.explained_variance_ratio_,
        rtol=1e-6, atol=1e-10,
    )
    lines.append(
        f"explained_variance_ratio_ matches sklearn: "
        f"{ratio_match}"
    )

    comp_matches = 0
    for i in range(n_components):
        a = ours.components_[i]
        b = theirs.components_[i]
        if (np.allclose(a, b, rtol=1e-5, atol=1e-8)
                or np.allclose(a, -b, rtol=1e-5, atol=1e-8)):
            comp_matches += 1
    lines.append(
        f"components_ match sklearn (up to sign): "
        f"{comp_matches}/{n_components}"
    )


def run_demo(X, n_components, label):
    """Run the capstone analysis; return report lines."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"PCA from Scratch: {label}")
    lines.append("=" * 60)
    lines.append(f"Data shape: {X.shape}")

    # Eigendecomposition vs SVD on the centered data
    X_centered = X - np.mean(X, axis=0)
    vals_eig, vecs_eig = pca_via_eigendecomposition(X_centered)
    vals_svd, vecs_svd = pca_via_svd(X_centered)
    cmp = compare_methods(
        vals_eig, vecs_eig, vals_svd, vecs_svd, verbose=False
    )
    lines.append("")
    lines.append("Eigendecomposition vs SVD:")
    lines.append(
        f"  Eigenvalues match: {cmp['eigenvalues_match']}"
    )
    lines.append(
        f"  Eigenvector matches (first {cmp['n_compared']}): "
        f"{cmp['eigenvector_matches']}/{cmp['n_compared']}"
    )
    lines.append(
        f"  Max eigenvalue difference: "
        f"{cmp['max_eigenvalue_diff']:.2e}"
    )

    # Fit / transform / reconstruct
    pca = PCA(n_components=n_components)
    X_proj = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_proj)
    mse = float(np.mean((X - X_recon) ** 2))
    captured = float(np.sum(pca.explained_variance_ratio_))
    lines.append("")
    lines.append(
        f"Reduced {X.shape[1]} -> {n_components} dims; variance "
        f"explained: {captured * 100:.1f}%"
    )
    lines.append(f"Reconstruction MSE ({n_components} PCs): "
                 f"{mse:.6f}")

    # The chapter's threshold sweep
    sweep = variance_threshold_components(X, THRESHOLDS)
    lines.append("")
    lines.append("Components needed for variance thresholds:")
    for t in THRESHOLDS:
        lines.append(f"  {t * 100:5.1f}%: {sweep[t]:3d} "
                     f"components")

    # Validation against sklearn (when available)
    lines.append("")
    validate_against_sklearn(X, n_components, lines)
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PCA-from-scratch capstone demo."
    )
    parser.add_argument(
        "--mnist", action="store_true",
        help="run the book's full MNIST demo (downloads data "
             "via sklearn fetch_openml)",
    )
    args = parser.parse_args(argv)

    if args.mnist:
        X = load_mnist()
        lines = run_demo(X, n_components=50, label="MNIST")
    else:
        rng = np.random.default_rng(42)
        X = make_synthetic_data(rng)
        lines = run_demo(
            X, n_components=5,
            label="Synthetic (seeded, low-rank + noise)",
        )

    report = "\n".join(lines) + "\n"
    print(report)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "capstone_pca.txt"
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
