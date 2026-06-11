"""Tests for Chapter 3: Calculus Foundations."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "src"),
)

from mathpowersai.calculus import (
    classify_critical_point,
    numerical_gradient,
    numerical_hessian,
    sigmoid_derivative,
    vanishing_gradient_demo,
)


def test_numerical_gradient_matches_analytic():
    # f(x, y) = x^2 + 4y^2  =>  grad f = [2x, 8y]
    f = lambda v: v[0]**2 + 4*v[1]**2
    rng = np.random.default_rng(0)
    for _ in range(5):
        point = rng.uniform(-3.0, 3.0, size=2)
        analytic = np.array([2*point[0], 8*point[1]])
        numeric = numerical_gradient(f, point)
        assert np.allclose(numeric, analytic, atol=1e-6)


def test_numerical_gradient_rejects_bad_eps():
    f = lambda v: v[0]**2
    with pytest.raises(ValueError):
        numerical_gradient(f, np.array([1.0]), eps=0.0)


def test_sigmoid_derivative_max_at_zero():
    # sigma'(0) = sigma(0)(1 - sigma(0)) = 0.5 * 0.5 = 0.25 exactly
    assert sigmoid_derivative(0.0) == 0.25


def test_sigmoid_derivative_bounded_by_quarter():
    grid = np.linspace(-10.0, 10.0, 401)
    values = sigmoid_derivative(grid)
    assert np.all(values <= 0.25)
    assert np.all(values > 0.0)


def test_vanishing_gradient_decay():
    factors = vanishing_gradient_demo(10)
    assert np.allclose(factors, 0.25 ** np.arange(1, 11))
    assert factors[-1] == pytest.approx(1e-6, rel=0.1)
    with pytest.raises(ValueError):
        vanishing_gradient_demo(0)


def test_classifier_minimum():
    f = lambda v: v[0]**2 + v[1]**2
    hess = numerical_hessian(f, np.array([0.0, 0.0]))
    assert classify_critical_point(hess, tol=1e-3) == "minimum"


def test_classifier_maximum():
    f = lambda v: -v[0]**2 - v[1]**2
    hess = numerical_hessian(f, np.array([0.0, 0.0]))
    assert classify_critical_point(hess, tol=1e-3) == "maximum"


def test_classifier_saddle():
    f = lambda v: v[0]**2 - v[1]**2
    hess = numerical_hessian(f, np.array([0.0, 0.0]))
    assert classify_critical_point(hess, tol=1e-3) == "saddle"


def test_classifier_rejects_nonsquare():
    with pytest.raises(ValueError):
        classify_critical_point(np.ones((2, 3)))
