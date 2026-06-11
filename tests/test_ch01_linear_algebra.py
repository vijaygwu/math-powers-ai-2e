"""Property tests for Chapter 1's claims (linear algebra)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)

from mathpowersai.linear_algebra import (  # noqa: E402
    TOY_EMBEDDING,
    attention_demo,
    cosine_similarity,
    dot,
    norm,
    project,
    self_attention,
    similarity_demo,
    softmax,
    two_layer_demo,
    word_analogy,
)


def test_dot_product_matches_book_example():
    # u . v = 1(4) + 2(-1) + 3(2) = 8 (chapter checkpoint)
    assert dot([1, 2, 3], [4, -1, 2]) == pytest.approx(8.0)


def test_dot_product_king_queen_and_king_apple():
    king = TOY_EMBEDDING["king"]
    queen = TOY_EMBEDDING["queen"]
    apple = TOY_EMBEDDING["apple"]
    assert dot(king, queen) == pytest.approx(15.15)
    assert dot(king, apple) == pytest.approx(0.75)


def test_dot_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dot([1, 2], [1, 2, 3])


def test_norms_match_book_checkpoint():
    # For v = (3, -4, 0): ||v||_1 = 7, ||v||_2 = 5, ||v||_inf = 4
    v = [3, -4, 0]
    assert norm(v, 1) == pytest.approx(7.0)
    assert norm(v, 2) == pytest.approx(5.0)
    assert norm(v, np.inf) == pytest.approx(4.0)


def test_norm_rejects_p_below_one():
    with pytest.raises(ValueError):
        norm([1.0, 2.0], p=0.5)


def test_cosine_similarity_in_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 10))
        u = rng.normal(size=n)
        v = rng.normal(size=n)
        s = cosine_similarity(u, v)
        assert -1.0 - 1e-12 <= s <= 1.0 + 1e-12


def test_cosine_similarity_identical_vectors_is_one():
    rng = np.random.default_rng(1)
    for _ in range(50):
        v = rng.normal(size=8)
        assert cosine_similarity(v, v) == pytest.approx(1.0)
        # Scaling does not change direction.
        assert cosine_similarity(v, 3.5 * v) == pytest.approx(1.0)


def test_cosine_similarity_opposite_and_orthogonal():
    assert cosine_similarity([1, 0], [-2, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([1, 0], [0, 5]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_raises():
    # Chapter "what breaks" exercise 1: undefined denominator.
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [0.0, 0.0])


def test_king_queen_more_similar_than_king_apple():
    king_queen, king_apple = similarity_demo()
    assert king_queen > king_apple
    assert king_queen > 0.99  # nearly identical direction


def test_projection_residual_is_orthogonal():
    rng = np.random.default_rng(2)
    for _ in range(100):
        u = rng.normal(size=5)
        v = rng.normal(size=5)
        p = project(u, v)
        # proj_v(u) is parallel to v; residual orthogonal to v.
        assert dot(u - p, v) == pytest.approx(0.0, abs=1e-10)
        assert cosine_similarity(p, v) == pytest.approx(
            np.sign(dot(u, v)) * 1.0
        )


def test_projection_is_idempotent():
    u = np.array([3.0, 2.5, -1.0])
    v = np.array([1.0, 1.0, 0.0])
    p = project(u, v)
    assert np.allclose(project(p, v), p)


def test_projection_onto_zero_vector_raises():
    with pytest.raises(ValueError):
        project([1.0, 2.0], [0.0, 0.0])


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 9)) * 10
    w = softmax(x)
    assert np.all(w > 0)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_softmax_rejects_non_2d():
    with pytest.raises(ValueError):
        softmax(np.array([1.0, 2.0, 3.0]))


def test_attention_demo_matches_book_listing():
    V, scores, weights, V_new = attention_demo()
    # scores = V @ V.T with the book's "the"/"cat"/"sat" vectors.
    assert np.array_equal(
        scores, np.array([[2, 0, 1], [0, 2, 1], [1, 1, 2]])
    )
    # Attention weights are a valid distribution per row.
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights > 0)
    # New embeddings are convex combinations of the rows of V.
    assert np.allclose(V_new, weights @ V)
    # Each output coordinate lies within the range of V's columns.
    assert np.all(V_new >= V.min(axis=0) - 1e-12)
    assert np.all(V_new <= V.max(axis=0) + 1e-12)


def test_self_attention_rows_sum_to_one_random():
    rng = np.random.default_rng(4)
    V = rng.normal(size=(5, 7))
    _, weights, V_new = self_attention(V)
    assert weights.shape == (5, 5)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert V_new.shape == V.shape


def test_word_analogy_king_man_woman_is_queen():
    # king - man + woman = (2.8, 2.7) = queen exactly in the toy
    # embedding (chapter Example 1.x).
    assert word_analogy("king", "man", "woman") == "queen"


def test_word_analogy_reverse_direction():
    assert word_analogy("queen", "woman", "man") == "king"


def test_word_analogy_unknown_word_raises():
    with pytest.raises(ValueError):
        word_analogy("king", "man", "zebra")


def test_two_layer_demo_composition():
    x, z1, z2, W_composite = two_layer_demo()
    assert z1.shape == (3,)
    assert z2.shape == (2,)
    # Composite W2 @ W1 is a single 2x4 matrix (book listing).
    assert W_composite.shape == (2, 4)
    # Without biases, the composition is just W_composite @ x.
    W1 = np.array([[0.2, 0.5, -0.1, 0.3],
                   [0.4, -0.2, 0.6, 0.1],
                   [-0.3, 0.1, 0.4, 0.5]])
    W2 = np.array([[0.5, -0.3, 0.2],
                   [0.1, 0.4, -0.5]])
    assert np.allclose(W_composite @ x, W2 @ (W1 @ x))
