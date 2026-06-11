"""Tests for Chapter 8: Numerical Methods."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "src"),
)

from mathpowersai.numerics import (
    condition_number,
    fp32_stall_demo,
    kahan_sum,
    log_sum_exp,
    machine_epsilon,
    naive_softmax,
    simpson_rule,
    stable_softmax,
    trapezoid_rule,
    with_nan_check,
)


def test_naive_softmax_overflows_to_nan_on_large_logits():
    z = np.array([1000.0, 1001.0, 999.0])
    result = naive_softmax(z)
    assert np.all(np.isnan(result))


def test_stable_softmax_matches_book_values():
    z = np.array([1000.0, 1001.0, 999.0])
    result = stable_softmax(z)
    assert np.all(np.isfinite(result))
    expected = np.array([0.245, 0.665, 0.090])
    assert np.allclose(result, expected, atol=5e-4)
    assert np.isclose(np.sum(result), 1.0)


def test_softmax_shift_invariance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=7)
    base = stable_softmax(x)
    for c in (-100.0, 3.5, 250.0):
        assert np.allclose(stable_softmax(x + c), base)


def test_stable_matches_naive_for_small_inputs():
    x = np.array([0.5, 1.0, -0.5])
    assert np.allclose(stable_softmax(x), naive_softmax(x))


def test_fp32_stall_at_2_to_24():
    assert np.float32(2 ** 24) + np.float32(1.0) == 2 ** 24
    demo = fp32_stall_demo()
    assert demo["stall_value"] == 16_777_216.0
    assert demo["after_adding_one"] == 16_777_216.0
    assert demo["stalled"] is True


def test_machine_epsilon_values():
    assert np.finfo(np.float64).eps == 2.0 ** -52
    assert np.finfo(np.float32).eps == 2.0 ** -23
    assert machine_epsilon(np.float64) == 2.0 ** -52
    assert machine_epsilon(np.float32) == 2.0 ** -23


def test_kahan_sum_recovers_lost_bits():
    # 1e16 + 1 - 1e16: naive FP addition loses the 1.
    vals = [1e16, 1.0, -1e16]
    assert kahan_sum(vals) == 1.0


def test_log_sum_exp_matches_direct_for_small_inputs():
    rng = np.random.default_rng(1)
    x = rng.normal(size=10)
    direct = np.log(np.sum(np.exp(x)))
    assert np.isclose(log_sum_exp(x), direct)


def test_log_sum_exp_finite_for_large_inputs():
    result = log_sum_exp([1000.0, 1001.0])
    assert np.isfinite(result)
    # = 1001 + log(1 + e^-1)
    expected = 1001.0 + np.log1p(np.exp(-1.0))
    assert np.isclose(result, expected)


def test_simpson_rule_exact_for_x_squared():
    # Simpson with coefficient h/6 integrates x^2 on [0, 2]
    # exactly: int_0^2 x^2 dx = 8/3.
    f = lambda x: x ** 2
    assert np.isclose(simpson_rule(f, 0.0, 2.0), 8.0 / 3.0)


def test_simpson_rule_exact_for_cubics():
    f = lambda x: x ** 3 - 2.0 * x
    # int_0^2 (x^3 - 2x) dx = 4 - 4 = 0
    assert np.isclose(simpson_rule(f, 0.0, 2.0), 0.0)


def test_trapezoid_rule_linear_exact():
    f = lambda x: 3.0 * x + 1.0
    # int_0^2 (3x + 1) dx = 6 + 2 = 8
    assert np.isclose(trapezoid_rule(f, 0.0, 2.0), 8.0)


def test_condition_number_identity_and_scaling():
    assert np.isclose(condition_number(np.eye(3)), 1.0)
    D = np.diag([100.0, 1.0, 0.01])
    assert np.isclose(condition_number(D), 1e4)


def test_condition_number_singular_matrix():
    # Rank-1 matrix: kappa ~ 10^16 is essentially singular for
    # FP64 (SVD may return inf or a tiny nonzero sigma_min).
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert condition_number(A) > 1e15


def test_with_nan_check_raises_on_nan_output():
    checked = with_nan_check(naive_softmax)
    z = np.array([1000.0, 1001.0, 999.0])
    with pytest.raises(RuntimeError, match="NaN in naive_softmax"):
        checked(z)


def test_with_nan_check_passes_through_clean_output():
    checked = with_nan_check(stable_softmax)
    out = checked(np.array([1.0, 2.0, 3.0]))
    assert np.all(np.isfinite(out))


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        naive_softmax([])
    with pytest.raises(ValueError):
        stable_softmax([[1.0, 2.0]])
    with pytest.raises(ValueError):
        log_sum_exp([])
    with pytest.raises(ValueError):
        machine_epsilon(np.int32)
    with pytest.raises(ValueError):
        condition_number([1.0, 2.0])
    with pytest.raises(ValueError):
        simpson_rule(42, 0.0, 1.0)
    with pytest.raises(ValueError):
        trapezoid_rule("not callable", 0.0, 1.0)
    with pytest.raises(ValueError):
        with_nan_check(None)
