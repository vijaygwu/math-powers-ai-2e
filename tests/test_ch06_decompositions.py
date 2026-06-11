"""The book's Chapter 6 claims, as executable properties."""

import numpy as np
import pytest

from mathpowersai.decompositions import (
    eigendecomposition,
    gram_schmidt_qr,
    low_rank_error,
    lu_decomposition,
    lu_solve,
    pca_svd,
    random_rank_k_factorization,
    rank_k_storage,
    svd,
    symmetric_eigendecomposition,
    truncated_svd,
)


def random_symmetric(n, rng):
    M = rng.standard_normal((n, n))
    return (M + M.T) / 2.0


# --- Spectral theorem: symmetric => real eigenvalues ---------


def test_symmetric_eigenvalues_are_real():
    """Spectral theorem: symmetric A has real eigenvalues."""
    rng = np.random.default_rng(0)
    for n in (2, 5, 10):
        A = random_symmetric(n, rng)
        eigenvalues, Q = symmetric_eigendecomposition(A)
        # eigh returns a real dtype by construction.
        assert np.isrealobj(eigenvalues)
        # Q is orthogonal and A = Q Lambda Q^T.
        assert np.allclose(Q.T @ Q, np.eye(n), atol=1e-10)
        recon = Q @ np.diag(eigenvalues) @ Q.T
        assert np.allclose(recon, A, atol=1e-10)


def test_symmetric_eigh_matches_general_eig():
    """eig on a symmetric matrix has ~zero imaginary parts."""
    rng = np.random.default_rng(1)
    A = random_symmetric(6, rng)
    general, _ = eigendecomposition(A)
    assert np.allclose(general.imag, 0.0, atol=1e-10)
    sym, _ = symmetric_eigendecomposition(A)
    assert np.allclose(np.sort(general.real), np.sort(sym))


def test_book_eigenvalue_example():
    """Example 6.1: [[4, 1], [2, 3]] has eigenvalues 5 and 2."""
    eigenvalues, _ = eigendecomposition([[4.0, 1.0],
                                         [2.0, 3.0]])
    assert np.allclose(sorted(eigenvalues.real), [2.0, 5.0])


def test_symmetric_eigendecomposition_rejects_nonsymmetric():
    with pytest.raises(ValueError):
        symmetric_eigendecomposition([[1.0, 2.0], [0.0, 1.0]])


# --- SVD reconstruction A = U S V^T --------------------------


def test_svd_reconstruction_to_1e10():
    """A == U S V^T to within 1e-10."""
    rng = np.random.default_rng(2)
    for shape in ((7, 4), (4, 7), (5, 5)):
        A = rng.standard_normal(shape)
        U, S, Vt = svd(A)
        recon = U @ np.diag(S) @ Vt
        assert np.max(np.abs(recon - A)) < 1e-10


def test_svd_book_example_singular_values():
    """[[3, 2], [2, 3], [2, -2]] has sigma = (5, 3)."""
    A = np.array([[3.0, 2.0], [2.0, 3.0], [2.0, -2.0]])
    _, S, _ = svd(A)
    assert np.allclose(S, [5.0, 3.0])


def test_singular_values_are_sqrt_eigvals_of_AtA():
    """sigma_i = sqrt(lambda_i(A^T A))."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((8, 5))
    _, S, _ = svd(A)
    lam = np.sort(np.linalg.eigvalsh(A.T @ A))[::-1]
    assert np.allclose(S, np.sqrt(np.clip(lam, 0.0, None)))


# --- Eckart-Young: truncated SVD error formula ----------------


def test_truncated_svd_error_equals_discarded_sigmas():
    """||A - A_k||_F = sqrt(sum_(i>k) sigma_i^2)."""
    rng = np.random.default_rng(4)
    A = rng.standard_normal((10, 6))
    _, S, _ = svd(A)
    for k in range(1, 7):
        err = low_rank_error(A, k)
        predicted = np.sqrt(np.sum(S[k:] ** 2))
        assert abs(err - predicted) < 1e-10


def test_truncated_svd_has_rank_k():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((9, 7))
    for k in (1, 3, 5):
        A_k = truncated_svd(A, k)
        assert A_k.shape == A.shape
        assert np.linalg.matrix_rank(A_k, tol=1e-8) == k


def test_truncated_svd_beats_random_rank_k():
    """Numerical Eckart-Young: rank-k SVD beats 20 seeded
    random rank-k factorizations in Frobenius error."""
    rng = np.random.default_rng(6)
    A = rng.standard_normal((12, 9))
    k = 3
    best = low_rank_error(A, k)
    for _ in range(20):
        Bk = random_rank_k_factorization(12, 9, k, rng)
        rand_err = np.linalg.norm(A - Bk, "fro")
        assert best <= rand_err + 1e-12


def test_truncated_svd_invalid_k_raises():
    A = np.zeros((4, 3))
    for bad_k in (0, -1, 4):  # k > min(m, n) = 3, k < 1
        with pytest.raises(ValueError):
            truncated_svd(np.eye(4)[:, :3], bad_k)
        with pytest.raises(ValueError):
            low_rank_error(A + np.eye(4, 3), bad_k)


def test_rank_k_storage_book_numbers():
    """1000x1000, k = 50 -> 100,050 values (~10:1 ratio)."""
    assert rank_k_storage(1000, 1000, 50) == 100_050
    with pytest.raises(ValueError):
        rank_k_storage(1000, 1000, 1001)


# --- LU and QR demos ------------------------------------------


def test_lu_book_example():
    """Example 6.4: A = [[2,1,1],[4,3,3],[8,7,9]] = L U."""
    A = np.array([[2.0, 1.0, 1.0],
                  [4.0, 3.0, 3.0],
                  [8.0, 7.0, 9.0]])
    L, U = lu_decomposition(A)
    L_book = np.array([[1.0, 0.0, 0.0],
                       [2.0, 1.0, 0.0],
                       [4.0, 3.0, 1.0]])
    U_book = np.array([[2.0, 1.0, 1.0],
                       [0.0, 1.0, 1.0],
                       [0.0, 0.0, 2.0]])
    assert np.allclose(L, L_book)
    assert np.allclose(U, U_book)
    assert np.allclose(L @ U, A)


def test_lu_solve_matches_numpy():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((5, 5)) + 5.0 * np.eye(5)
    b = rng.standard_normal(5)
    x = lu_solve(A, b)
    assert np.allclose(x, np.linalg.solve(A, b))


def test_lu_rejects_nonsquare():
    with pytest.raises(ValueError):
        lu_decomposition(np.ones((2, 3)))


def test_gram_schmidt_qr_properties():
    rng = np.random.default_rng(8)
    A = rng.standard_normal((7, 4))
    Q, R = gram_schmidt_qr(A)
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-10)
    assert np.allclose(R, np.triu(R))
    assert np.allclose(Q @ R, A, atol=1e-10)


def test_gram_schmidt_rejects_dependent_columns():
    A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    with pytest.raises(ValueError):
        gram_schmidt_qr(A)


# --- PCA via SVD (chapter listing) ----------------------------


def test_pca_svd_projection_and_variance():
    rng = np.random.default_rng(9)
    latent = rng.standard_normal((100, 2)) * [8.0, 3.0]
    X = latent @ rng.standard_normal((2, 10))
    X += 0.01 * rng.standard_normal((100, 10))
    Z, k, ratio = pca_svd(X, variance=0.95)
    assert Z.shape == (100, k)
    assert k <= 2 + 1  # essentially 2-dimensional data
    assert ratio[:k].sum() >= 0.95
    # Z equals X_centered @ V_k (listing's stated equivalence).
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z_direct = Xc @ Vt[:k].T
    assert np.allclose(np.abs(Z), np.abs(Z_direct), atol=1e-8)


def test_pca_svd_invalid_variance_raises():
    with pytest.raises(ValueError):
        pca_svd(np.eye(3), variance=0.0)
    with pytest.raises(ValueError):
        pca_svd(np.eye(3), variance=1.5)
