"""The book's Chapter 5 claims, as executable properties."""

import numpy as np
import pytest

from mathpowersai.optimizers import (
    adam,
    compare_optimizers,
    gradient_descent,
    momentum,
    rmsprop,
    sgd,
)


def quad_grad(p):
    # f(x, y) = x^2 + 4y^2  (the book's worked example)
    return np.array([2 * p[0], 8 * p[1]])


def test_gd_walkthrough_matches_printed_numbers():
    """Ch 5 worked example: start (2,1), lr 0.1."""
    path = gradient_descent(quad_grad, np.array([2.0, 1.0]), lr=0.1,
                            n_steps=10)
    assert np.allclose(path[1], [1.6, 0.2])           # printed step 1
    assert abs(path[-1][0] - 2 * 0.8**10) < 1e-12     # x: factor 0.8
    assert abs(path[-1][1] - 0.2**10) < 1e-12         # y: factor 0.2
    f = path[-1][0] ** 2 + 4 * path[-1][1] ** 2
    assert abs(f - 0.046) < 5e-4                      # printed f ~ 0.046


def test_gd_divergence_threshold():
    """Ch 5: eta < 2/lambda_max avoids divergence; above it diverges.

    At eta = 0.26 (just past 2/8 = 0.25) the y-component grows by
    |1 - 0.26 * 8| = 1.08 per step: 1.08^200 ~ 5e6.
    """
    stable = gradient_descent(quad_grad, np.array([1.0, 1.0]), lr=0.09,
                              n_steps=200)
    diverging = gradient_descent(quad_grad, np.array([1.0, 1.0]), lr=0.26,
                                 n_steps=200)
    assert np.linalg.norm(stable[-1]) < 1e-3
    assert np.linalg.norm(diverging[-1]) > 1e3
    assert abs(diverging[-1][1]) == pytest.approx(1.08**200, rel=1e-6)


def test_condition_number_rate_at_optimal_step():
    """Ch 5 Principle box: ||x_k - x*|| <= ((k-1)/(k+1))^k ||x_0 - x*||
    with the optimal fixed step eta = 2/(l_min + l_max)."""
    lmin, lmax = 2.0, 8.0
    kappa = lmax / lmin
    eta = 2 / (lmin + lmax)
    grad = lambda p: np.array([lmin * p[0], lmax * p[1]])
    path = gradient_descent(grad, np.array([1.0, 1.0]), lr=eta, n_steps=20)
    bound = ((kappa - 1) / (kappa + 1)) ** 20 * np.linalg.norm([1.0, 1.0])
    assert np.linalg.norm(path[-1]) <= bound + 1e-12


def test_validation_guards():
    x0 = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        rmsprop(quad_grad, x0, epsilon=0.0, n_steps=1)
    with pytest.raises(ValueError):
        adam(quad_grad, x0, lr=-1.0, n_steps=1)
    with pytest.raises(ValueError):
        momentum(quad_grad, x0, beta=1.0, n_steps=1)
    with pytest.raises(ValueError):
        sgd(quad_grad, x0, noise_scale=-0.1, n_steps=1)
    with pytest.raises(ValueError):
        compare_optimizers(quad_grad, x0, optimizers=["not_an_optimizer"])


def test_sgd_reproducible_with_seeded_rng():
    x0 = np.array([2.0, 1.0])
    p1 = sgd(quad_grad, x0, n_steps=5, rng=np.random.default_rng(42))
    p2 = sgd(quad_grad, x0, n_steps=5, rng=np.random.default_rng(42))
    assert all(np.allclose(a, b) for a, b in zip(p1, p2))


def test_path_cap_keeps_final_point():
    path = gradient_descent(quad_grad, np.array([2.0, 1.0]), lr=0.1,
                            n_steps=50, max_path_len=5)
    assert len(path) == 5
    full = gradient_descent(quad_grad, np.array([2.0, 1.0]), lr=0.1,
                            n_steps=50)
    assert np.allclose(path[-1], full[-1])


def test_compare_optimizers_uses_per_optimizer_defaults():
    res = compare_optimizers(quad_grad, np.array([1.0, 1.0]), n_steps=5)
    assert set(res) == {"gradient_descent", "momentum", "rmsprop", "adam"}
