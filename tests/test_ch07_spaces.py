"""Tests for Chapter 7: Vector Spaces."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)

from mathpowersai.spaces import (  # noqa: E402
    change_of_basis,
    gram_schmidt,
    is_linearly_independent,
    project_onto_subspace,
    project_onto_vector,
    projection_matrix,
)


class TestChangeOfBasis:
    def test_book_example(self):
        """The book's corrected example: (1.5, 1.2) in basis
        b1=(1.2, 0.4), b2=(0.3, 1.0) is (1.06, 0.78)_B."""
        B = np.column_stack([[1.2, 0.4], [0.3, 1.0]])
        coords = change_of_basis(np.array([1.5, 1.2]), B)
        np.testing.assert_array_almost_equal(
            coords.round(2), [1.06, 0.78]
        )

    def test_reconstruction(self):
        """B @ coords reconstructs the original point."""
        B = np.column_stack([[1.2, 0.4], [0.3, 1.0]])
        x = np.array([1.5, 1.2])
        coords = change_of_basis(x, B)
        np.testing.assert_allclose(B @ coords, x)

    def test_singular_basis_raises(self):
        B = np.array([[1.0, 2.0], [2.0, 4.0]])
        with pytest.raises(ValueError):
            change_of_basis(np.array([1.0, 1.0]), B)


class TestGramSchmidt:
    def test_orthonormal(self):
        """Q^T Q = I for the Gram-Schmidt output."""
        rng = np.random.default_rng(0)
        V = rng.normal(size=(5, 3))
        Q = gram_schmidt(V)
        np.testing.assert_allclose(
            Q.T @ Q, np.eye(3), atol=1e-10
        )

    def test_same_span(self):
        """span{q_1,...,q_k} = span{v_1,...,v_k}: projecting
        each v_j onto the q's reproduces v_j exactly."""
        rng = np.random.default_rng(1)
        V = rng.normal(size=(6, 3))
        Q = gram_schmidt(V)
        np.testing.assert_allclose(
            Q @ (Q.T @ V), V, atol=1e-10
        )
        assert np.linalg.matrix_rank(
            np.hstack([V, Q])
        ) == np.linalg.matrix_rank(V)

    def test_book_example(self):
        """v1=(1,1,0), v2=(1,0,1) gives q1 = (1,1,0)/sqrt(2),
        q2 = (1,-1,2)/sqrt(6)."""
        V = np.column_stack([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        Q = gram_schmidt(V)
        np.testing.assert_allclose(
            Q[:, 0], np.array([1, 1, 0]) / np.sqrt(2)
        )
        np.testing.assert_allclose(
            Q[:, 1], np.array([1, -1, 2]) / np.sqrt(6)
        )

    def test_dependent_raises(self):
        V = np.column_stack(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]
        )
        with pytest.raises(ValueError):
            gram_schmidt(V)


class TestProjection:
    def test_vector_in_subspace_is_fixed(self):
        """Projecting a vector already in W is the identity."""
        rng = np.random.default_rng(2)
        A = rng.normal(size=(5, 2))
        w = A @ np.array([0.7, -1.3])  # w is in col(A)
        np.testing.assert_allclose(
            project_onto_subspace(w, A), w, atol=1e-10
        )

    def test_idempotent(self):
        """P^2 = P: projecting twice equals projecting once."""
        rng = np.random.default_rng(3)
        A = rng.normal(size=(6, 3))
        P = projection_matrix(A)
        np.testing.assert_allclose(P @ P, P, atol=1e-10)
        v = rng.normal(size=6)
        p = project_onto_subspace(v, A)
        np.testing.assert_allclose(
            project_onto_subspace(p, A), p, atol=1e-10
        )

    def test_residual_orthogonal(self):
        rng = np.random.default_rng(4)
        A = rng.normal(size=(5, 2))
        v = rng.normal(size=5)
        r = v - project_onto_subspace(v, A)
        np.testing.assert_allclose(A.T @ r, 0, atol=1e-10)

    def test_chapter_listing_values(self):
        """proj of v=(2,3) onto u=(4,1) is (11/17)*(4,1)."""
        proj = project_onto_vector(
            np.array([2, 3]), np.array([4, 1])
        )
        np.testing.assert_allclose(
            proj, 11 / 17 * np.array([4.0, 1.0])
        )

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError):
            project_onto_vector(
                np.array([1.0, 2.0]), np.zeros(2)
            )

    def test_dependent_basis_raises(self):
        A = np.column_stack(
            [[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]]
        )
        with pytest.raises(ValueError):
            project_onto_subspace(np.ones(3), A)


class TestLinearIndependence:
    def test_dependent_pair(self):
        assert not is_linearly_independent(
            [[1, 2, 3], [2, 4, 6]]
        )

    def test_standard_basis(self):
        assert is_linearly_independent(np.eye(3))
