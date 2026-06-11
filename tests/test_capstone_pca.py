"""
Pytest properties for the capstone PCA implementation.

Covers the chapter's exercise (covariance eigenvalues are
non-negative) and the core invariants of the from-scratch PCA,
plus a parity check against sklearn when it is importable.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mathpowersai.pca import (  # noqa: E402
    PCA,
    compare_methods,
    pca_via_eigendecomposition,
    pca_via_svd,
    variance_threshold_components,
)


def make_data(rng, n_samples=200, n_features=20, rank=4,
              noise_scale=0.05):
    """Seeded low-rank-plus-noise Gaussian dataset."""
    Z = rng.normal(size=(n_samples, rank))
    W = rng.normal(size=(rank, n_features))
    noise = noise_scale * rng.normal(
        size=(n_samples, n_features)
    )
    offset = rng.normal(size=n_features)
    return Z @ W + noise + offset


@pytest.fixture
def X():
    rng = np.random.default_rng(42)
    return make_data(rng)


def test_covariance_eigenvalues_nonnegative(X):
    """The book's exercise: eigenvalues of Sigma =
    (1/(n-1)) X^T X are non-negative (Sigma is PSD)."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    sigma = (Xc.T @ Xc) / (n - 1)
    eigenvalues = np.linalg.eigvalsh(sigma)
    assert np.all(eigenvalues >= -1e-10)


def test_explained_variance_ratio_properties(X):
    """Ratios sum to <= 1 and are non-increasing."""
    pca = PCA().fit(X)
    ratio = pca.explained_variance_ratio_
    assert np.all(ratio >= 0)
    assert ratio.sum() <= 1.0 + 1e-12
    assert np.all(np.diff(ratio) <= 1e-12)

    # Truncated fit keeps a prefix, so the same properties hold
    pca_k = PCA(n_components=5).fit(X)
    ratio_k = pca_k.explained_variance_ratio_
    assert ratio_k.sum() <= 1.0 + 1e-12
    assert np.all(np.diff(ratio_k) <= 1e-12)


def test_full_rank_roundtrip_reconstruction(X):
    """inverse_transform(transform(X)) with all components
    reconstructs X to 1e-8."""
    pca = PCA().fit(X)
    X_rec = pca.inverse_transform(pca.transform(X))
    assert np.allclose(X_rec, X, atol=1e-8)


def test_truncated_reconstruction_error_equals_discarded(X):
    """With k components, the (Bessel-normalized) squared
    reconstruction error equals the sum of the discarded
    eigenvalues: ||X - X_k||_F^2 / (n-1) = sum_{i>k} lambda_i."""
    n = X.shape[0]
    full = PCA().fit(X)
    for k in (1, 3, 7):
        pca = PCA(n_components=k).fit(X)
        X_rec = pca.inverse_transform(pca.transform(X))
        err = np.sum((X - X_rec) ** 2) / (n - 1)
        discarded = np.sum(full.explained_variance_[k:])
        assert np.isclose(err, discarded, rtol=1e-8,
                          atol=1e-10)


def test_eigendecomposition_matches_svd(X):
    """Both routes give the same eigenvalues/eigenvectors."""
    Xc = X - X.mean(axis=0)
    vals_eig, vecs_eig = pca_via_eigendecomposition(Xc)
    vals_svd, vecs_svd = pca_via_svd(Xc)
    result = compare_methods(
        vals_eig, vecs_eig, vals_svd, vecs_svd, verbose=False
    )
    assert result["eigenvalues_match"]
    assert (result["eigenvector_matches"]
            == result["n_compared"])


def test_variance_threshold_components(X):
    """Sweep returns valid, monotone component counts."""
    thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    sweep = variance_threshold_components(X, thresholds)
    counts = [sweep[t] for t in thresholds]
    assert all(c >= 1 for c in counts)
    assert counts == sorted(counts)

    # Each count actually reaches its threshold
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for t, c in sweep.items():
        assert cumvar[c - 1] >= t - 1e-12

    with pytest.raises(ValueError):
        variance_threshold_components(X, [0.0])
    with pytest.raises(ValueError):
        variance_threshold_components(X, [1.5])


def test_invalid_n_components_raises(X):
    """ValueError on invalid n_components."""
    for bad in (0, -1, X.shape[1] + 1, 2.5):
        with pytest.raises(ValueError):
            PCA(n_components=bad).fit(X)


def test_matches_sklearn_explained_variance_ratio(X):
    """Parity with sklearn PCA when sklearn is importable."""
    sk = pytest.importorskip("sklearn.decomposition")
    k = 5
    ours = PCA(n_components=k).fit(X)
    theirs = sk.PCA(n_components=k).fit(X)
    assert np.allclose(
        ours.explained_variance_ratio_,
        theirs.explained_variance_ratio_,
        rtol=1e-6, atol=1e-10,
    )
    # Components match up to sign
    for i in range(k):
        a = ours.components_[i]
        b = theirs.components_[i]
        assert (np.allclose(a, b, rtol=1e-5, atol=1e-8)
                or np.allclose(a, -b, rtol=1e-5, atol=1e-8))
