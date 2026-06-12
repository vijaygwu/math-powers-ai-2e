"""Matrix decompositions from Chapter 6 (Matrix Decomposition).

Importable versions of the chapter's code: eigendecomposition
(with the spectral-theorem path for symmetric matrices via
``np.linalg.eigh``), the SVD ``A = U Sigma V^T``, the truncated
SVD and its Eckart-Young approximation error, LU and Gram-Schmidt
QR demos, and PCA via SVD as printed in the chapter's
``pythoncode`` listing (Method 2, numpy only).

The chapter mentions power iteration only as a forward pointer to
the numerical-methods chapter and prints no implementation, so
none is included here.
"""

import numpy as np


def _as_matrix(A):
    """Validate that ``A`` is a 2-D array and return it as float."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix")
    return A


def _require_square(A):
    A = _as_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"A must be square, got shape {A.shape}"
        )
    return A


def eigendecomposition(A):
    """Eigendecomposition A = V Lambda V^{-1} of a square matrix.

    Solves A v = lambda v for all eigenpairs.  If A has n linearly
    independent eigenvectors, then A = V Lambda V^{-1} where V has
    the eigenvectors as columns and
    Lambda = diag(lambda_1, ..., lambda_n).

    Parameters
    ----------
    A : array_like, shape (n, n)
        Square matrix.

    Returns
    -------
    eigenvalues : np.ndarray, shape (n,)
        Eigenvalues (possibly complex for non-symmetric A).
    eigenvectors : np.ndarray, shape (n, n)
        Matrix V whose columns are the eigenvectors.

    Raises
    ------
    ValueError
        If A is not a square 2-D matrix.
    """
    A = _require_square(A)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors


def symmetric_eigendecomposition(A, tol=1e-10):
    """Spectral decomposition A = Q Lambda Q^T of a symmetric A.

    Spectral Theorem: if A is symmetric (A = A^T), all eigenvalues
    of A are real, A has n orthonormal eigenvectors, and
    A = Q Lambda Q^T with Q orthogonal (Q^T Q = I) and Lambda
    diagonal with real entries.  Uses ``np.linalg.eigh``, which
    guarantees real eigenvalues for symmetric input.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Symmetric matrix.
    tol : float
        Tolerance for the symmetry check ``|A - A^T| <= tol``.

    Returns
    -------
    eigenvalues : np.ndarray, shape (n,)
        Real eigenvalues in ascending order.
    Q : np.ndarray, shape (n, n)
        Orthogonal matrix of eigenvectors (columns).

    Raises
    ------
    ValueError
        If A is not square or not symmetric within ``tol``.
    """
    A = _require_square(A)
    if not np.allclose(A, A.T, atol=tol):
        raise ValueError("A must be symmetric (A = A^T)")
    eigenvalues, Q = np.linalg.eigh(A)
    return eigenvalues, Q


def svd(A, full_matrices=False):
    """Singular Value Decomposition A = U Sigma V^T.

    Every matrix A in R^{m x n} can be factored as
    A = U Sigma V^T, where U and V are orthogonal and Sigma is
    diagonal with non-negative entries
    sigma_1 >= sigma_2 >= ... >= sigma_r > 0 (the singular
    values), with r = rank(A).

    Parameters
    ----------
    A : array_like, shape (m, n)
        Any matrix.
    full_matrices : bool
        If False (default), return the economy SVD
        A = U_r Sigma_r V_r^T.

    Returns
    -------
    U : np.ndarray
        Left singular vectors (columns).
    S : np.ndarray, shape (min(m, n),)
        Singular values in descending order.
    Vt : np.ndarray
        V^T; rows are the right singular vectors.
    """
    A = _as_matrix(A)
    U, S, Vt = np.linalg.svd(A, full_matrices=full_matrices)
    return U, S, Vt


def truncated_svd(A, k):
    """Best rank-k approximation A_k = U_k Sigma_k V_k^T.

    By the Eckart-Young-Mirsky theorem, the best rank-k
    approximation to A in both the Frobenius norm and the
    spectral norm is

        A_k = sum_{i=1}^{k} sigma_i u_i v_i^T
            = U_k Sigma_k V_k^T

    obtained by keeping only the k largest singular values.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Any matrix.
    k : int
        Target rank, 1 <= k <= min(m, n).

    Returns
    -------
    A_k : np.ndarray, shape (m, n)
        The rank-k approximation U_k Sigma_k V_k^T.

    Raises
    ------
    ValueError
        If k is not an integer in [1, min(m, n)].
    """
    A = _as_matrix(A)
    k = int(k)
    if k < 1 or k > min(A.shape):
        raise ValueError(
            f"k must satisfy 1 <= k <= min(m, n) = "
            f"{min(A.shape)}, got k = {k}"
        )
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return A_k


def low_rank_error(A, k):
    """Frobenius error of the best rank-k approximation of A.

    Eckart-Young-Mirsky theorem (as stated in the chapter): let
    A = U Sigma V^T be the SVD of A in R^{m x n} with singular
    values sigma_1 >= sigma_2 >= ... >= sigma_r > 0.  The best
    rank-k approximation to A in both the Frobenius norm and the
    spectral norm is A_k = U_k Sigma_k V_k^T, with approximation
    error

        ||A - A_k||_F = sqrt(sum_{i=k+1}^{r} sigma_i^2)
        ||A - A_k||_2 = sigma_{k+1}

    This function returns the Frobenius-norm error, computed
    directly as ||A - A_k||_F with A_k = truncated_svd(A, k).

    Parameters
    ----------
    A : array_like, shape (m, n)
        Any matrix.
    k : int
        Target rank, 1 <= k <= min(m, n).

    Returns
    -------
    float
        ||A - A_k||_F.

    Raises
    ------
    ValueError
        If k is not an integer in [1, min(m, n)].
    """
    A = _as_matrix(A)
    A_k = truncated_svd(A, k)
    return float(np.linalg.norm(A - A_k, "fro"))


def rank_k_storage(m, n, k):
    """Storage (number of values) for a rank-k factorization.

    From the chapter's image-compression example: a rank-k
    approximation A_k = U_k Sigma_k V_k^T requires storing
    U_k (m k values), Sigma_k (k values), and V_k (n k values),
    for a total of k (m + n + 1) values, versus m n for the full
    matrix.

    Parameters
    ----------
    m, n : int
        Matrix dimensions.
    k : int
        Rank, 1 <= k <= min(m, n).

    Returns
    -------
    int
        Total storage k * (m + n + 1).

    Raises
    ------
    ValueError
        If the dimensions are not positive or k is out of range.
    """
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1:
        raise ValueError("m and n must be positive")
    if k < 1 or k > min(m, n):
        raise ValueError(
            f"k must satisfy 1 <= k <= min(m, n) = "
            f"{min(m, n)}, got k = {k}"
        )
    return k * (m + n + 1)


def random_rank_k_factorization(m, n, k, rng):
    """A random rank-k matrix B C with B (m x k) and C (k x n).

    Used to check the Eckart-Young theorem numerically: any such
    random rank-k factorization should approximate a target matrix
    no better than the truncated SVD does.

    Parameters
    ----------
    m, n : int
        Matrix dimensions.
    k : int
        Rank, 1 <= k <= min(m, n).
    rng : np.random.Generator
        Random number generator (e.g. ``np.random.default_rng``).

    Returns
    -------
    np.ndarray, shape (m, n)
        The product B @ C with standard-normal entries in B and C.

    Raises
    ------
    ValueError
        If the dimensions are not positive or k is out of range.
    """
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1:
        raise ValueError("m and n must be positive")
    if k < 1 or k > min(m, n):
        raise ValueError(
            f"k must satisfy 1 <= k <= min(m, n) = "
            f"{min(m, n)}, got k = {k}"
        )
    B = rng.standard_normal((m, k))
    C = rng.standard_normal((k, n))
    return B @ C


def lu_decomposition(A):
    """LU decomposition A = L U without pivoting (Doolittle).

    Expresses a square matrix as A = L U where L is lower
    triangular with ones on the diagonal and U is upper
    triangular.  L stores the multipliers used during Gaussian
    elimination and U is the resulting upper triangular form.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Square matrix admitting an LU decomposition without row
        exchanges.

    Returns
    -------
    L : np.ndarray, shape (n, n)
        Unit lower triangular matrix.
    U : np.ndarray, shape (n, n)
        Upper triangular matrix.

    Raises
    ------
    ValueError
        If A is not square, or a zero pivot is encountered (the
        matrix then requires the PLU decomposition P A = L U).
    """
    A = _require_square(A)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for j in range(n - 1):
        if abs(U[j, j]) < 1e-12:
            raise ValueError(
                "zero pivot encountered; use PLU (P A = L U)"
            )
        for i in range(j + 1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] = U[i, j:] - L[i, j] * U[j, j:]
            U[i, j] = 0.0
    return L, U


def lu_solve(A, b):
    """Solve A x = b via LU decomposition (Algorithm 6.1).

    Step 1: compute A = L U (O(n^3), once).
    Step 2: solve L y = b by forward substitution (O(n^2)).
    Step 3: solve U x = y by backward substitution (O(n^2)).

    Parameters
    ----------
    A : array_like, shape (n, n)
        Square coefficient matrix.
    b : array_like, shape (n,)
        Right-hand side.

    Returns
    -------
    np.ndarray, shape (n,)
        Solution x.

    Raises
    ------
    ValueError
        If A is not square, b has the wrong length, or A is
        singular / needs pivoting.
    """
    A = _require_square(A)
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[0]
    if b.shape[0] != n:
        raise ValueError(
            f"b must have length {n}, got {b.shape[0]}"
        )
    L, U = lu_decomposition(A)
    if abs(U[n - 1, n - 1]) < 1e-12:
        raise ValueError("A is singular")
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def gram_schmidt_qr(A):
    """Thin QR decomposition A = Q R via Gram-Schmidt.

    Implements the chapter's Gram-Schmidt QR algorithm: for each
    column a_j, subtract its projections R_ij = q_i^T a_j onto the
    previous orthonormal columns, then normalize with
    R_jj = ||v_j|| and q_j = v_j / R_jj.  Q has orthonormal
    columns (Q^T Q = I) and R is upper triangular.

    Parameters
    ----------
    A : array_like, shape (m, n) with m >= n
        Matrix with linearly independent columns.

    Returns
    -------
    Q : np.ndarray, shape (m, n)
        Matrix with orthonormal columns.
    R : np.ndarray, shape (n, n)
        Upper triangular matrix.

    Raises
    ------
    ValueError
        If m < n or the columns are linearly dependent.
    """
    A = _as_matrix(A)
    m, n = A.shape
    if m < n:
        raise ValueError(
            f"A must have m >= n, got shape {A.shape}"
        )
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-12:
            raise ValueError(
                "columns of A are linearly dependent"
            )
        Q[:, j] = v / R[j, j]
    return Q, R


def pca_svd(X, variance=0.95):
    """PCA via SVD, following the chapter's code listing.

    Reproduces "Method 2: Using SVD directly" from the chapter's
    ``pythoncode`` listing (numpy only):

        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        variance_ratio = (S**2) / (S**2).sum()
        k = np.argmax(np.cumsum(variance_ratio) >= 0.95) + 1
        Z_svd = U[:, :k] * S[:k]  # Equivalent to X @ V[:, :k]

    The proportion of variance explained by the first k principal
    components is sum_{i<=k} sigma_i^2 / sum_i sigma_i^2.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Data matrix (rows are samples).
    variance : float
        Fraction of variance to retain, in (0, 1].

    Returns
    -------
    Z_svd : np.ndarray, shape (n_samples, k)
        Projected data U_k Sigma_k (equivalent to
        X_centered @ V[:, :k]).
    k : int
        Number of components retained.
    variance_ratio : np.ndarray, shape (min(m, n),)
        Variance explained by each component.

    Raises
    ------
    ValueError
        If X is not 2-D or variance is not in (0, 1].
    """
    X = _as_matrix(X)
    if not 0.0 < variance <= 1.0:
        raise ValueError("variance must be in (0, 1]")
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    variance_ratio = (S**2) / (S**2).sum()
    cumulative = np.cumsum(variance_ratio)
    # Guard against floating-point sums slightly below 1.0: without
    # this, variance=1.0 finds no True entry and argmax silently
    # returns 0 (i.e. k=1). Same guard as pca.components_for_variance.
    cumulative[-1] = max(cumulative[-1], 1.0)
    k = int(np.argmax(cumulative >= variance) + 1)
    Z_svd = U[:, :k] * S[:k]  # Equivalent to X @ V[:, :k]
    return Z_svd, k, variance_ratio
