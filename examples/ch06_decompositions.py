"""Chapter 6 examples: Matrix Decomposition.

Reproduces the chapter's printed outputs -- eigenvalue and SVD
worked examples, rank-k storage savings, low-rank reconstruction
errors (Eckart-Young), the LU example, Gram-Schmidt QR, and the
PCA-via-SVD listing -- prints them, and writes the same text to
outputs/ch06_decompositions.txt.
"""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)

import numpy as np  # noqa: E402

from mathpowersai.decompositions import (  # noqa: E402
    eigendecomposition,
    gram_schmidt_qr,
    low_rank_error,
    lu_decomposition,
    lu_solve,
    pca_svd,
    rank_k_storage,
    svd,
    symmetric_eigendecomposition,
    truncated_svd,
)


def main():
    rng = np.random.default_rng(42)
    lines = []

    # --- Eigendecomposition (Example 6.1) -------------------
    lines.append("== Eigendecomposition ==")
    A = np.array([[4.0, 1.0], [2.0, 3.0]])
    eigenvalues, _ = eigendecomposition(A)
    order = np.argsort(eigenvalues.real)[::-1]
    lines.append(
        "A = [[4, 1], [2, 3]]  eigenvalues: "
        + str(np.round(eigenvalues.real[order], 6))
    )  # book: lambda_1 = 5, lambda_2 = 2

    # Spectral theorem (Try It): B = [[2, 1], [1, 2]]
    B = np.array([[2.0, 1.0], [1.0, 2.0]])
    evals, Q = symmetric_eigendecomposition(B)
    lines.append(
        "B = [[2, 1], [1, 2]]  eigh eigenvalues: "
        + str(np.round(evals, 6))
    )  # book: lambda = 1, 3 with orthogonal eigenvectors
    lines.append(
        "Q^T Q = I check: "
        + str(np.allclose(Q.T @ Q, np.eye(2)))
    )
    lines.append("")

    # --- SVD worked examples ---------------------------------
    lines.append("== Singular Value Decomposition ==")
    M = np.array([[3.0, 1.0], [1.0, 3.0]])
    U, S, Vt = svd(M)
    lines.append(
        "singular values of [[3, 1], [1, 3]]: "
        + str(np.round(S, 6))
    )  # book trace: sigma_1 = 4, sigma_2 = 2
    M2 = np.array([[3.0, 2.0], [2.0, 3.0], [2.0, -2.0]])
    U2, S2, Vt2 = svd(M2)
    lines.append(
        "singular values of [[3, 2], [2, 3], [2, -2]]: "
        + str(np.round(S2, 6))
    )  # book example: sigma_1 = 5, sigma_2 = 3
    recon = np.linalg.norm(M2 - U2 @ np.diag(S2) @ Vt2)
    lines.append(
        f"reconstruction ||A - U S V^T||_F: {recon:.2e}"
    )
    lines.append("")

    # --- Low-rank approximation (Eckart-Young) ---------------
    lines.append("== Low-Rank Approximation (Eckart-Young) ==")
    G = rng.standard_normal((12, 8))
    _, sigma, _ = svd(G)
    lines.append("seeded 12 x 8 Gaussian matrix:")
    for k in (1, 2, 4, 6):
        err = low_rank_error(G, k)
        predicted = float(np.sqrt(np.sum(sigma[k:] ** 2)))
        lines.append(
            f"  k = {k}: ||A - A_k||_F = {err:.6f}, "
            f"sqrt(sum_(i>k) sigma_i^2) = {predicted:.6f}"
        )
    A_2 = truncated_svd(G, 2)
    lines.append(
        f"  rank of A_2: {np.linalg.matrix_rank(A_2)}"
    )
    lines.append("")

    # --- Storage savings (chapter examples) ------------------
    lines.append("== Rank-k Storage Savings ==")
    full = 1000 * 1000
    compressed = rank_k_storage(1000, 1000, 50)
    lines.append(
        f"1000 x 1000 image, k = 50: original {full:,} values, "
        f"compressed {compressed:,} values "
        f"(ratio {full / compressed:.1f}:1)"
    )  # book: 100,050 values, approximately 10:1
    rank10 = 1000 * 10 + 1000 * 10
    lines.append(
        f"rank-10 factorization A = U V^T of 1000 x 1000: "
        f"{rank10:,} elements ({full // rank10}x reduction)"
    )  # book checkpoint: 20,000 elements, 50x reduction
    lines.append("")

    # --- LU decomposition (Example 6.4) ----------------------
    lines.append("== LU Decomposition ==")
    A_lu = np.array(
        [[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]
    )
    L, Uu = lu_decomposition(A_lu)
    lines.append("A = [[2,1,1],[4,3,3],[8,7,9]]")
    lines.append("L =\n" + str(L))
    lines.append("U =\n" + str(Uu))
    lines.append(
        "L U == A: " + str(np.allclose(L @ Uu, A_lu))
    )
    b = np.array([1.0, 2.0, 3.0])
    x = lu_solve(A_lu, b)
    lines.append(
        "solve A x = [1, 2, 3]: x = " + str(np.round(x, 6))
    )
    lines.append(
        "residual ||A x - b||: "
        + f"{np.linalg.norm(A_lu @ x - b):.2e}"
    )
    lines.append("")

    # --- QR via Gram-Schmidt (Algorithm 6.2) ------------------
    lines.append("== QR Decomposition (Gram-Schmidt) ==")
    A_qr = rng.standard_normal((6, 4))
    Q, R = gram_schmidt_qr(A_qr)
    lines.append(
        "Q^T Q = I check: "
        + str(np.allclose(Q.T @ Q, np.eye(4)))
    )
    lines.append(
        "R upper triangular: "
        + str(np.allclose(R, np.triu(R)))
    )
    lines.append(
        "Q R == A: " + str(np.allclose(Q @ R, A_qr))
    )
    lines.append("")

    # --- PCA via SVD (the chapter's pythoncode listing) -------
    lines.append("== PCA via SVD (chapter listing, Method 2) ==")
    # Seeded data: 3 latent dimensions embedded in 20 features.
    latent = rng.standard_normal((200, 3)) * [10.0, 5.0, 2.0]
    mixing = rng.standard_normal((3, 20))
    X = latent @ mixing + 0.01 * rng.standard_normal((200, 20))
    Z_svd, k, variance_ratio = pca_svd(X, variance=0.95)
    lines.append(
        f"Reduced from {X.shape[1]} to {Z_svd.shape[1]} "
        "dimensions"
    )  # mirrors the listing's print statement
    lines.append(
        f"components kept: k = {k}, variance explained = "
        f"{variance_ratio[:k].sum():.4f}"
    )
    lines.append(
        "top-5 variance ratios: "
        + str(np.round(variance_ratio[:5], 4))
    )

    text = "\n".join(lines) + "\n"
    print(text, end="")

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ch06_decompositions.txt"
    out_path.write_text(text)
    return out_path


if __name__ == "__main__":
    main()
