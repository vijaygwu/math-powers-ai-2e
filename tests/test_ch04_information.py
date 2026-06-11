"""Tests for Chapter 4: information theory."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, "src"),
)

from mathpowersai.information import (
    bsc_capacity,
    cross_entropy,
    entropy,
    kl_divergence,
    mutual_information,
)


def test_entropy_maximal_at_uniform():
    """Uniform distribution maximizes entropy: H = log2(n)."""
    n = 8
    uniform = [1 / n] * n
    assert entropy(uniform) == pytest.approx(np.log2(n))
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.ones(n))
        assert entropy(p) <= entropy(uniform) + 1e-12


def test_entropy_zero_for_point_mass():
    """A deterministic outcome has zero entropy."""
    assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_cross_entropy_identity():
    """Checkpoint: H(p, q) = H(p) + KL(p || q) for the book's
    Bernoulli(0.7) / Bernoulli(0.5) example."""
    p = [0.7, 0.3]
    q = [0.5, 0.5]
    assert cross_entropy(p, q) == pytest.approx(
        entropy(p) + kl_divergence(p, q)
    )
    # H(p, 0.5-uniform) is exactly 1 bit
    assert cross_entropy(p, q) == pytest.approx(1.0)


def test_kl_nonnegative_zero_iff_equal():
    """KL(p || q) >= 0 with equality iff p == q."""
    p = [0.1, 0.9]
    q = [0.5, 0.5]
    assert kl_divergence(p, q) > 0
    assert kl_divergence(p, p) == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.dirichlet(np.ones(4))
        b = rng.dirichlet(np.ones(4))
        assert kl_divergence(a, b) >= 0
        assert kl_divergence(a, a) == pytest.approx(0.0)


def test_kl_asymmetric():
    """KL divergence is not symmetric."""
    p = [0.1, 0.9]
    q = [0.5, 0.5]
    assert kl_divergence(p, q) != pytest.approx(
        kl_divergence(q, p)
    )


def test_mutual_information_independent_is_zero():
    """I(X; Y) = 0 when the joint factorizes: p(x,y)=p(x)p(y)."""
    p_x = np.array([0.2, 0.5, 0.3])
    p_y = np.array([0.6, 0.4])
    joint = np.outer(p_x, p_y)
    assert mutual_information(joint) == pytest.approx(0.0)


def test_bsc_capacity_book_example():
    """Book example: C = 1 - H(0.1) ~ 0.531 bits per symbol."""
    assert round(bsc_capacity(0.1), 3) == 0.531


def test_invalid_distributions_raise():
    """Negative entries or wrong normalization raise ValueError."""
    with pytest.raises(ValueError):
        entropy([0.5, 0.6])  # does not sum to 1
    with pytest.raises(ValueError):
        entropy([-0.1, 1.1])  # negative entry
    with pytest.raises(ValueError):
        kl_divergence([0.5, 0.5], [0.3, 0.3])
    with pytest.raises(ValueError):
        # q(x) = 0 where p(x) > 0: p not absolutely
        # continuous w.r.t. q
        kl_divergence([0.5, 0.5], [1.0, 0.0])
