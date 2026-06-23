"""
PCA from scratch -- companion code for the Part I capstone of
"The Math That Powers AI" (2nd ed.).

Implements Principal Component Analysis using only NumPy. The
implementation follows the book's conventions exactly:

  * Sample covariance with Bessel's correction:
        Sigma = (1/(n-1)) X^T X        (X centered)
  * Eigenvalues from singular values:
        lambda_i = sigma_i^2 / (n-1)
  * SVD of the centered data matrix is used for numerical
    stability (it avoids forming X^T X, which squares the
    condition number).

This module is NumPy-only: no scikit-learn imports.
"""

import numpy as np


class PCA:
    """
    Principal Component Analysis implementation from scratch.

    This implementation uses SVD for numerical stability and
    supports both fitting and transforming data, as well as
    inverse transforms.

    Parameters:
    --------
    n_components : int or None
        Number of components to keep. If None, keep all
        components.

    Attributes:
    --------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space (rows are eigenvectors)
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each component,
        lambda_i = sigma_i^2 / (n-1)
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each component
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from training data
    n_components_ : int
        The actual number of components used
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_components_ = None

    def fit(self, X):
        """
        Fit the PCA model with training data X.

        The principal axes are the right singular vectors of the
        centered data matrix; the explained variances are
        lambda_i = sigma_i^2 / (n-1), the eigenvalues of the
        sample covariance Sigma = (1/(n-1)) X^T X.

        Parameters:
        --------
        X : ndarray of shape (n_samples, n_features)
            Training data

        Returns:
        ------
        self : object
            Returns the instance itself

        Raises:
        ------
        ValueError
            If n_components is not None and is not an integer in
            the range 1 <= n_components <= min(n_samples,
            n_features).
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)

        # Validate n_components
        if self.n_components is not None:
            k = self.n_components
            if (not isinstance(k, (int, np.integer))
                    or isinstance(k, bool)
                    or k < 1 or k > max_components):
                raise ValueError(
                    "n_components must be an int with "
                    "1 <= n_components <= "
                    f"min(n_samples, n_features)={max_components}; "
                    f"got {k!r}"
                )

        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute SVD
        U, singular_values, Vt = np.linalg.svd(
            X_centered, full_matrices=False
        )

        # Determine number of components
        if self.n_components is None:
            self.n_components_ = max_components
        else:
            self.n_components_ = self.n_components

        # Store components (as rows, like sklearn)
        self.components_ = Vt[:self.n_components_]

        # Compute explained variance:
        # lambda_i = sigma_i^2 / (n-1)  (Bessel's correction)
        self.explained_variance_ = (
            (singular_values ** 2) / (n_samples - 1)
        )
        total_variance = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
        )

        # Truncate to n_components
        self.explained_variance_ = (
            self.explained_variance_[:self.n_components_]
        )
        self.explained_variance_ratio_ = (
            self.explained_variance_ratio_[:self.n_components_]
        )

        return self

    def _check_fitted(self):
        """Raise a clear error if fit() has not been called yet.

        Without this guard the arithmetic below dereferences
        mean_/components_ while they are still None, producing a
        confusing "unsupported operand type(s) for -: float and
        NoneType" TypeError instead of an actionable message.
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError(
                "This PCA instance is not fitted yet. Call fit() "
                "(or fit_transform()) before using transform(), "
                "inverse_transform(), or get_covariance()."
            )

    # -- Transform methods --

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        Parameters:
        --------
        X : ndarray of shape (n_samples, n_features)
            Data to transform

        Returns:
        ------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data

        Raises:
        ------
        ValueError
            If called before fit().
        """
        self._check_fitted()
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """
        Fit the model and apply dimensionality reduction to X.

        Parameters:
        --------
        X : ndarray of shape (n_samples, n_features)
            Training data

        Returns:
        ------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space.

        Parameters:
        --------
        X_transformed : ndarray of shape (n_samples, n_components)
            Data in PCA space

        Returns:
        ------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Reconstructed data in original space

        Raises:
        ------
        ValueError
            If called before fit().
        """
        self._check_fitted()
        return X_transformed @ self.components_ + self.mean_

    # -- Utility methods --

    def get_covariance(self):
        """
        Compute the covariance matrix from the stored components.

        Returns:
        ------
        cov : ndarray of shape (n_features, n_features)
            Estimated covariance matrix

        Raises:
        ------
        ValueError
            If called before fit().
        """
        self._check_fitted()
        return (
            self.components_.T
            @ np.diag(self.explained_variance_)
            @ self.components_
        )


def pca_via_eigendecomposition(X_centered):
    """
    Compute PCA using eigendecomposition of the covariance matrix
    Sigma = (1/(n-1)) X^T X.

    Parameters:
    --------
    X_centered : ndarray of shape (n_samples, n_features)
        Centered data matrix

    Returns:
    ------
    eigenvalues : ndarray of shape (n_features,)
        Eigenvalues in descending order
    eigenvectors : ndarray of shape (n_features, n_features)
        Eigenvectors as columns, corresponding to eigenvalues
    """
    # Step 1: Compute covariance matrix
    n = X_centered.shape[0]
    cov = (X_centered.T @ X_centered) / (n - 1)

    # Step 2: Compute eigendecomposition
    # np.linalg.eigh is optimized for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Step 3: Sort by eigenvalue (descending order)
    # eigh returns them in ascending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def pca_via_svd(X_centered):
    """
    Compute PCA using SVD of the data matrix.

    Parameters:
    --------
    X_centered : ndarray of shape (n_samples, n_features)
        Centered data matrix

    Returns:
    ------
    eigenvalues : ndarray
        Eigenvalues (computed from singular values via
        lambda_i = sigma_i^2 / (n-1))
    eigenvectors : ndarray
        Principal component directions (V from SVD), as columns
    """
    n = X_centered.shape[0]

    # Compute SVD: X = U @ Sigma @ V^T
    # full_matrices=False gives us the "economy" SVD
    U, singular_values, Vt = np.linalg.svd(
        X_centered, full_matrices=False
    )

    # Principal component directions are rows of V^T
    # (columns of V); already in descending singular-value order
    eigenvectors = Vt.T

    # Bessel's correction: divide by (n-1) for unbiased sample
    # covariance. Eigenvalues relate to singular values:
    # lambda_i = sigma_i^2 / (n-1)
    eigenvalues = (singular_values ** 2) / (n - 1)

    return eigenvalues, eigenvectors


def compare_methods(eig_vals1, eig_vecs1, eig_vals2, eig_vecs2,
                    verbose=True):
    """
    Compare eigendecomposition and SVD results.

    Two eigenvectors are equivalent if v1 = v2 or v1 = -v2 (the
    sign of an eigenvector is arbitrary).

    Parameters:
    --------
    eig_vals1, eig_vals2 : ndarray
        Eigenvalues from each method, descending order. They are
        compared over the leading min(len(vals1), len(vals2))
        entries.
    eig_vecs1, eig_vecs2 : ndarray
        Eigenvectors as columns from each method.
    verbose : bool
        If True (default), print the comparison like the book.

    Returns:
    ------
    result : dict
        Keys: 'eigenvalues_match' (bool), 'n_compared' (int),
        'eigenvector_matches' (int),
        'max_relative_eigenvalue_diff' (float; max |diff| over
        the leading eigenvalue, so machine-precision agreement
        reads as ~1e-15 regardless of the data's scale).
    """
    m = min(len(eig_vals1), len(eig_vals2))
    vals1 = np.asarray(eig_vals1)[:m]
    vals2 = np.asarray(eig_vals2)[:m]

    # Compare eigenvalues
    eigenvalue_match = bool(
        np.allclose(vals1, vals2, rtol=1e-10, atol=1e-12)
    )

    # Compare eigenvectors (up to sign)
    n_components = min(
        50, eig_vecs1.shape[1], eig_vecs2.shape[1]
    )
    matches = 0
    for i in range(n_components):
        v1 = eig_vecs1[:, i]
        v2 = eig_vecs2[:, i]
        if (np.allclose(v1, v2, rtol=1e-8, atol=1e-8)
                or np.allclose(v1, -v2, rtol=1e-8, atol=1e-8)):
            matches += 1

    # Max eigenvalue difference, relative to the leading
    # eigenvalue: an absolute diff is scale-dependent (raw-pixel
    # data has eigenvalues ~1e5-1e7), but the relative diff sits
    # at machine epsilon when the two methods agree.
    scale = max(float(np.max(np.abs(vals1))), 1e-300)
    max_rel_diff = float(np.max(np.abs(vals1 - vals2)) / scale)

    if verbose:
        print(f"Eigenvalues match: {eigenvalue_match}")
        print(
            f"Eigenvector matches (first {n_components}): "
            f"{matches}/{n_components}"
        )
        print("Max relative eigenvalue difference: "
              f"{max_rel_diff:.1e}")

    return {
        "eigenvalues_match": eigenvalue_match,
        "n_compared": n_components,
        "eigenvector_matches": matches,
        "max_relative_eigenvalue_diff": max_rel_diff,
    }


def variance_threshold_components(X, thresholds):
    """
    Components needed per cumulative-variance threshold (the
    chapter's output-box sweep).

    Fits a full PCA on X and, for each threshold t, finds the
    smallest k such that the first k components explain at least
    a fraction t of the total variance:

        sum_{i<=k} lambda_i / sum_i lambda_i >= t

    Parameters:
    --------
    X : ndarray of shape (n_samples, n_features)
        Input data (will be centered internally).
    thresholds : sequence of float
        Cumulative-variance thresholds in (0, 1], e.g.
        [0.5, 0.8, 0.9, 0.95, 0.99].

    Returns:
    ------
    result : dict
        Maps each threshold to the number of components (int)
        needed to reach it.

    Raises:
    ------
    ValueError
        If any threshold is outside (0, 1].
    """
    thresholds = list(thresholds)
    for t in thresholds:
        if not (0.0 < t <= 1.0):
            raise ValueError(
                f"thresholds must lie in (0, 1]; got {t!r}"
            )

    pca = PCA().fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    # Guard against floating-point sums slightly below 1.0
    cumulative_variance[-1] = max(cumulative_variance[-1], 1.0)

    result = {}
    for t in thresholds:
        n_comp = int(np.argmax(cumulative_variance >= t) + 1)
        result[t] = n_comp
    return result
