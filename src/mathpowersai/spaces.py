"""Chapter 7: Vector Spaces -- companion code.

Importable versions of the chapter's code: the linear-independence
check via matrix rank, change of basis (solving B c = x), the
Gram-Schmidt process, orthogonal projection onto a vector and onto
a subspace, and the Word2Vec-style embedding analogy demo
(king - man + woman ~= queen).
"""

import numpy as np

__all__ = [
    "is_linearly_independent",
    "change_of_basis",
    "gram_schmidt",
    "project_onto_vector",
    "projection_matrix",
    "project_onto_subspace",
    "make_toy_embeddings",
    "embedding_analogy",
]

_TOL = 1e-10


def is_linearly_independent(vectors):
    """Check linear independence of a set of vectors via rank.

    Vectors v_1, ..., v_k are linearly independent if the only
    solution to

        c_1 v_1 + c_2 v_2 + ... + c_k v_k = 0

    is c_1 = c_2 = ... = c_k = 0.  Equivalently, the matrix with
    these vectors as rows has rank k.

    Parameters
    ----------
    vectors : array_like, shape (k, n)
        The vectors v_1, ..., v_k as the rows of a matrix.

    Returns
    -------
    bool
        True if the vectors are linearly independent.
    """
    V = np.atleast_2d(np.asarray(vectors, dtype=float))
    if V.size == 0:
        raise ValueError("need at least one vector")
    return np.linalg.matrix_rank(V) == V.shape[0]


def change_of_basis(point, basis):
    """Coordinates of a point with respect to a new basis.

    If B = [b_1 | ... | b_n] has the basis vectors as columns,
    the coordinates c of x with respect to the basis solve

        B c = x

    so that x = c_1 b_1 + ... + c_n b_n (the unique
    representation theorem).

    The book's example: the point (1.5, 1.2) in the basis
    b_1 = (1.2, 0.4), b_2 = (0.3, 1.0) has coordinates
    (1.06, 0.78)_B.

    >>> import numpy as np
    >>> B = np.column_stack([[1.2, 0.4], [0.3, 1.0]])
    >>> change_of_basis(np.array([1.5, 1.2]), B).round(2)
    array([1.06, 0.78])

    Parameters
    ----------
    point : array_like, shape (n,)
        The point x in standard coordinates.
    basis : array_like, shape (n, n)
        Matrix B with the basis vectors as columns.

    Returns
    -------
    ndarray, shape (n,)
        The coordinates c with respect to the basis.

    Raises
    ------
    ValueError
        If the basis matrix is not square or is singular
        (its columns do not form a basis).
    """
    x = np.asarray(point, dtype=float)
    B = np.asarray(basis, dtype=float)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("basis must be a square matrix")
    if x.shape != (B.shape[0],):
        raise ValueError("point and basis dimensions differ")
    if np.linalg.matrix_rank(B) < B.shape[1]:
        raise ValueError("basis is singular: columns are "
                         "linearly dependent")
    return np.linalg.solve(B, x)


def gram_schmidt(vectors):
    """Gram-Schmidt: orthonormalize linearly independent vectors.

    Given linearly independent vectors {v_1, ..., v_k}, produce
    an orthonormal set {q_1, ..., q_k} with the same span:

        u_j = v_j - sum_{i=1}^{j-1} <v_j, q_i> q_i
        q_j = u_j / ||u_j||

    Parameters
    ----------
    vectors : array_like, shape (n, k)
        Matrix with the vectors v_1, ..., v_k as columns.

    Returns
    -------
    ndarray, shape (n, k)
        Matrix Q with orthonormal columns q_1, ..., q_k
        satisfying Q^T Q = I.

    Raises
    ------
    ValueError
        If the input vectors are linearly dependent.
    """
    V = np.asarray(vectors, dtype=float)
    if V.ndim != 2:
        raise ValueError("vectors must form a 2-D matrix")
    n, k = V.shape
    Q = np.zeros((n, k))
    for j in range(k):
        u = V[:, j].copy()
        for i in range(j):
            u -= np.dot(V[:, j], Q[:, i]) * Q[:, i]
        norm = np.linalg.norm(u)
        scale = max(1.0, np.linalg.norm(V[:, j]))
        if norm <= _TOL * scale:
            raise ValueError(
                f"vector {j} is linearly dependent on the "
                "previous vectors"
            )
        Q[:, j] = u / norm
    return Q


def project_onto_vector(v, u):
    """Project vector v onto vector u.

    proj_u(v) = (<v, u> / <u, u>) u = (<v, u> / ||u||^2) u

    Raises
    ------
    ValueError
        If u is the zero vector.
    """
    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)
    denom = np.dot(u, u)
    if denom <= _TOL:
        raise ValueError("cannot project onto the zero vector")
    return (np.dot(v, u) / denom) * u


def projection_matrix(A):
    """Projection matrix onto the column space of A.

    For A (m x n) with linearly independent columns,

        P = A (A^T A)^{-1} A^T

    P is idempotent (P^2 = P), symmetric (P^T = P), and
    col(P) = col(A).

    Raises
    ------
    ValueError
        If the columns of A are linearly dependent.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if np.linalg.matrix_rank(A) < A.shape[1]:
        raise ValueError("columns of A are linearly dependent")
    return A @ np.linalg.inv(A.T @ A) @ A.T


def project_onto_subspace(v, basis):
    """Orthogonal projection of v onto the subspace W = col(A).

    With A holding the spanning vectors as columns,

        proj_W(v) = A (A^T A)^{-1} A^T v

    Equivalently, with an orthonormal basis {q_1, ..., q_k}
    (e.g. from Gram-Schmidt),

        proj_W(v) = sum_{i=1}^k <v, q_i> q_i

    The residual v - proj_W(v) lies in the orthogonal
    complement W^perp.

    Parameters
    ----------
    v : array_like, shape (m,)
        The vector to project.
    basis : array_like, shape (m, k)
        Matrix A whose columns span the subspace W.

    Returns
    -------
    ndarray, shape (m,)
        The projection proj_W(v).

    Raises
    ------
    ValueError
        If the columns of `basis` are linearly dependent.
    """
    v = np.asarray(v, dtype=float)
    P = projection_matrix(basis)
    if v.shape != (P.shape[0],):
        raise ValueError("v and basis dimensions differ")
    return P @ v


def make_toy_embeddings(rng, dim=8, noise=0.05):
    """Build a tiny word-embedding space with separable directions.

    Mimics the Word2Vec structure from the chapter's war story:
    the "royal" direction and the "gender" direction are
    separable components in vector space, so

        king - man + woman ~= queen

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator for the base directions and noise.
    dim : int
        Embedding dimension.
    noise : float
        Scale of per-word Gaussian noise.

    Returns
    -------
    dict[str, ndarray]
        Word embeddings for king, queen, man, woman, apple.
    """
    royal = rng.normal(size=dim)
    gender = rng.normal(size=dim)
    fruit = rng.normal(size=dim)
    base = {
        "king": royal + gender,
        "queen": royal - gender,
        "man": gender,
        "woman": -gender,
        "apple": fruit,
    }
    return {
        word: vec + noise * rng.normal(size=dim)
        for word, vec in base.items()
    }


def embedding_analogy(embeddings, a, b, c):
    """Solve the analogy a - b + c ~= ? in an embedding space.

    Computes the target vector vec(a) - vec(b) + vec(c) and
    returns the vocabulary word (excluding a, b, c) whose
    embedding has the highest cosine similarity to it.

    Raises
    ------
    ValueError
        If a, b, or c is missing from `embeddings`, or no
        candidate words remain.
    """
    for word in (a, b, c):
        if word not in embeddings:
            raise ValueError(f"unknown word: {word!r}")
    target = (
        np.asarray(embeddings[a], dtype=float)
        - np.asarray(embeddings[b], dtype=float)
        + np.asarray(embeddings[c], dtype=float)
    )
    t_norm = np.linalg.norm(target)
    if t_norm <= _TOL:
        raise ValueError("analogy target is the zero vector")
    best_word, best_sim = None, -np.inf
    for word, vec in embeddings.items():
        if word in (a, b, c):
            continue
        vec = np.asarray(vec, dtype=float)
        sim = np.dot(target, vec) / (
            t_norm * np.linalg.norm(vec)
        )
        if sim > best_sim:
            best_word, best_sim = word, sim
    if best_word is None:
        raise ValueError("no candidate words in vocabulary")
    return best_word
