"""Chapter 1: Vectors, Matrices, and Linear Maps.

Companion code for "The Math That Powers AI" (2nd ed), Chapter 1.
Implements every ``pythoncode`` listing in the chapter as importable
functions, plus the dot-product / cosine-similarity / projection /
word-analogy helpers the chapter develops.

All functions use numpy only and have no module-level side effects.
"""

import numpy as np

# Toy word embedding in R^2 (Table 1.1 of the chapter).
TOY_EMBEDDING = {
    "king": np.array([3.0, 2.5]),
    "queen": np.array([2.8, 2.7]),
    "man": np.array([2.0, 0.5]),
    "woman": np.array([1.8, 0.7]),
    "apple": np.array([-1.0, 1.5]),
    "orange": np.array([-0.8, 1.3]),
}


def _as_vector(x, name):
    """Validate and return ``x`` as a 1-D float array."""
    v = np.asarray(x, dtype=float)
    if v.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D vector, got shape {v.shape}"
        )
    return v


def dot(u, v):
    """Dot product of two vectors.

    Book formula:
        u . v = <u, v> = sum_{i=1}^{n} u_i v_i
              = u_1 v_1 + u_2 v_2 + ... + u_n v_n
    """
    u = _as_vector(u, "u")
    v = _as_vector(v, "v")
    if u.shape != v.shape:
        raise ValueError(
            f"shape mismatch: u has shape {u.shape}, "
            f"v has shape {v.shape}"
        )
    return float(np.dot(u, v))


def norm(v, p=2):
    """The l_p norm of a vector, for p >= 1 (or p = np.inf).

    Book formula:
        ||v||_p = ( sum_{i=1}^{n} |v_i|^p )^(1/p)
    Special cases:
        ||v||_1   = sum_i |v_i|        (Manhattan norm)
        ||v||_2   = sqrt(sum_i v_i^2)  (Euclidean norm)
        ||v||_inf = max_i |v_i|        (max norm)
    """
    v = _as_vector(v, "v")
    # Value test, not identity: float('inf') and math.inf are equal
    # to np.inf but are different objects.
    if np.isinf(p):
        return float(np.max(np.abs(v)))
    if p < 1:
        raise ValueError(f"l_p norm requires p >= 1, got p={p}")
    return float(np.sum(np.abs(v) ** p) ** (1.0 / p))


def cosine_similarity(u, v):
    """Cosine similarity between two vectors.

    Book formula:
        sim(u, v) = (u . v) / (||u|| ||v||) = cos(theta)
    This ranges from -1 (opposite) through 0 (orthogonal) to +1
    (identical direction). Undefined for zero vectors: the
    denominator ||u|| ||v|| would be zero.
    """
    u = _as_vector(u, "u")
    v = _as_vector(v, "v")
    if u.shape != v.shape:
        raise ValueError(
            f"shape mismatch: u has shape {u.shape}, "
            f"v has shape {v.shape}"
        )
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        raise ValueError(
            "cosine similarity is undefined for the zero vector: "
            "the denominator ||u|| ||v|| is zero"
        )
    return float(np.dot(u, v) / (norm_u * norm_v))


def project(u, v):
    """Vector projection of ``u`` onto ``v``.

    Book formula:
        proj_v(u) = ((u . v) / (v . v)) v
    The result is the component of u along the direction of v; the
    residual u - proj_v(u) is orthogonal to v.
    """
    u = _as_vector(u, "u")
    v = _as_vector(v, "v")
    if u.shape != v.shape:
        raise ValueError(
            f"shape mismatch: u has shape {u.shape}, "
            f"v has shape {v.shape}"
        )
    vv = np.dot(v, v)
    if vv == 0.0:
        raise ValueError(
            "cannot project onto the zero vector: v . v is zero"
        )
    return (np.dot(u, v) / vv) * v


def softmax(x):
    """Row-wise softmax, exactly as in the chapter's listing.

    Book formula (simplified, per row):
        softmax(x)_ij = exp(x_ij - max_j x_ij)
                        / sum_j exp(x_ij - max_j x_ij)
    Each row of the result is non-negative and sums to 1.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(
            f"softmax expects a 2-D array of scores, got shape "
            f"{x.shape}"
        )
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def self_attention(V):
    """Simplified self-attention over embeddings ``V`` (n x d).

    Book formulas:
        score_ij = v_i . v_j   (dot product: how relevant is j to i?)
        v'_i = sum_{j=1}^{n} alpha_ij v_j
               (linear combination of all words)
    where alpha_ij are normalized scores (via softmax). Returns
    (scores, weights, V_new) with scores = V @ V.T,
    weights = softmax(scores), V_new = weights @ V.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise ValueError(
            f"V must be a 2-D array of embeddings, got shape "
            f"{V.shape}"
        )
    scores = V @ V.T
    weights = softmax(scores)
    V_new = weights @ V
    return scores, weights, V_new


def attention_demo():
    """The chapter's 3-word self-attention listing.

    Book formulas:
        score_ij = v_i . v_j
        alpha = softmax(scores)  (per row)
        v'_i = sum_j alpha_ij v_j
    Uses the three 4-dimensional word embeddings for "the", "cat",
    "sat". Returns (V, scores, weights, V_new).
    """
    # Three word embeddings (each 4-dimensional)
    V = np.array([[1, 0, 1, 0],    # "the"
                  [0, 1, 0, 1],    # "cat"
                  [1, 1, 0, 0]])   # "sat"
    scores, weights, V_new = self_attention(V)
    return V, scores, weights, V_new


def word_analogy(a, b, c, embeddings=None):
    """Solve the analogy "a - b + c = ?" by nearest neighbor.

    Book formula (the word analogy task):
        v_king - v_man + v_woman ~= v_queen
    Computes the target vector v_a - v_b + v_c, then returns the
    vocabulary word (excluding a, b, c) whose embedding has the
    highest cosine similarity sim(u, v) = (u . v) / (||u|| ||v||)
    with the target.
    """
    if embeddings is None:
        embeddings = TOY_EMBEDDING
    for word in (a, b, c):
        if word not in embeddings:
            raise ValueError(
                f"word {word!r} is not in the embedding vocabulary"
            )
    target = embeddings[a] - embeddings[b] + embeddings[c]
    candidates = {
        w: v for w, v in embeddings.items() if w not in (a, b, c)
    }
    if not candidates:
        raise ValueError(
            "no candidate words remain after excluding the inputs"
        )
    return max(
        candidates,
        key=lambda w: cosine_similarity(target, candidates[w]),
    )


def similarity_demo():
    """The chapter's cosine-similarity listing (Table 1.1 vectors).

    Book formula:
        sim(u, v) = (u . v) / (||u|| ||v||) = cos(theta)
    Returns (king_queen_sim, king_apple_sim) for the toy embedding;
    king-queen is high, king-apple is low.
    """
    king = TOY_EMBEDDING["king"]
    queen = TOY_EMBEDDING["queen"]
    apple = TOY_EMBEDDING["apple"]
    return (
        cosine_similarity(king, queen),
        cosine_similarity(king, apple),
    )


def two_layer_demo(rng=None):
    """The chapter's 2-layer transformation listing (no activation).

    Book formulas:
        z = W x + b     (a dense layer)
        z = W_3 W_2 W_1 x   (layers compose by multiplication)
    Uses the fixed weight matrices from the listing; ``rng`` is
    accepted for interface consistency (defaults to
    np.random.default_rng()) but the listing's weights are
    deterministic. Returns (x, z1, z2, W_composite).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Input: 4-dimensional feature vector
    x = np.array([1.0, 0.5, -0.3, 0.8])

    # Layer 1: Transform from 4D to 3D
    W1 = np.array([[0.2, 0.5, -0.1, 0.3],
                   [0.4, -0.2, 0.6, 0.1],
                   [-0.3, 0.1, 0.4, 0.5]])
    b1 = np.array([0.1, -0.1, 0.2])

    # Layer 2: Transform from 3D to 2D
    W2 = np.array([[0.5, -0.3, 0.2],
                   [0.1, 0.4, -0.5]])
    b2 = np.array([0.0, 0.1])

    # Forward pass: apply transformations
    z1 = W1 @ x + b1       # First transformation: R^4 -> R^3
    z2 = W2 @ z1 + b2      # Second transformation: R^3 -> R^2

    # The composite transformation (without biases)
    W_composite = W2 @ W1
    return x, z1, z2, W_composite
