"""Chapter 8: Numerical Methods.

Companion code for "The Math That Powers AI" (2nd ed).

Implements the chapter's numerical-stability toolkit: naive vs.
stable softmax, the log-sum-exp trick, the FP32 summation stall,
Kahan (compensated) summation, machine epsilon, condition numbers,
Newton-Cotes quadrature (trapezoidal and Simpson's rules), and the
NaN-detection forward-hook pattern adapted to plain callables.

Conventions:
- numpy only; ValueError on invalid input.
- No module-level side effects; all demos are functions.
"""

import numpy as np


def _as_1d_array(x, name="x"):
    """Validate and return x as a non-empty 1-D float array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got ndim={arr.ndim})")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def naive_softmax(x):
    """Unstable: overflows for large inputs.

    softmax(z)_i = e^{z_i} / sum_j e^{z_j}

    For z = [1000, 1001, 999], e^{1000} = inf in floating point,
    yielding inf/inf = NaN.
    """
    arr = _as_1d_array(x)
    with np.errstate(over="ignore", invalid="ignore"):
        exp_x = np.exp(arr)
        return exp_x / np.sum(exp_x)


def stable_softmax(x):
    """Stable: subtract max first.

    softmax(z)_i = e^{z_i - max_j z_j} / sum_j e^{z_j - max_j z_j}

    Mathematically identical to the naive form since
    e^{z_i - c} / sum_j e^{z_j - c} = e^{z_i} / sum_j e^{z_j}.
    For z = [1000, 1001, 999] this gives [0.245, 0.665, 0.090].
    """
    arr = _as_1d_array(x)
    x_shifted = arr - np.max(arr)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def log_sum_exp(x):
    """Log-sum-exp trick: compute log sum_i e^{x_i} stably.

    log sum_i e^{x_i} = x_max + log sum_i e^{x_i - x_max}

    where x_max = max_i x_i. This prevents overflow when x_i is
    large (direct computation overflows near x_i ~ 700 in FP64).
    """
    arr = _as_1d_array(x)
    x_max = np.max(arr)
    return float(x_max + np.log(np.sum(np.exp(arr - x_max))))


def fp32_stall_demo():
    """Demonstrate the FP32 summation stall at 2^24 = 16,777,216.

    Summing 10^8 copies of 1.0 in FP32 with naive sequential
    accumulation stalls at 2^24: once the running sum reaches
    16,777,216, adding 1.0 no longer changes it because FP32
    lacks the mantissa bits to represent the increment:

        np.float32(2**24) + np.float32(1.0) == 2**24

    Returns a dict with the stall value, the (unchanged) sum
    after adding 1.0, and whether the stall occurred.
    """
    stall = np.float32(2.0 ** 24)
    after = stall + np.float32(1.0)
    return {
        "stall_value": float(stall),
        "after_adding_one": float(after),
        "stalled": bool(after == stall),
    }


def kahan_sum(values):
    """Compensated (Kahan) summation.

    Carries a correction term c that recovers the low-order bits
    lost when adding terms of very different magnitudes, so sums
    like 10^16 + 1 - 10^16 return 1 instead of 0 and 10^8 copies
    of 1.0 do not stall as in naive FP32 accumulation. Uses the
    Kahan-Babuska (Neumaier) variant of the compensation step.
    """
    arr = _as_1d_array(values, name="values")
    s = 0.0
    c = 0.0
    for v in arr:
        v = float(v)
        t = s + v
        if abs(s) >= abs(v):
            c += (s - t) + v
        else:
            c += (v - t) + s
        s = t
    return s + c


def machine_epsilon(dtype=np.float64):
    """Machine epsilon: gap between 1 and the next float.

    eps_mach = fl(1+) - 1

    where fl(1+) is the smallest representable number greater
    than 1. For FP64, eps_mach = 2^-52 ~ 2.2e-16; for FP32,
    eps_mach = 2^-23 ~ 1.2e-7.
    """
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(
            f"dtype must be float32 or float64 (got {dt})"
        )
    one = dt.type(1.0)
    return float(np.nextafter(one, dt.type(2.0)) - one)


def condition_number(A):
    """Condition number kappa_2(A) = sigma_max(A) / sigma_min(A).

    A problem with kappa = 10^k may lose up to k digits of
    accuracy, regardless of the algorithm used. kappa ~ 1 is
    well-conditioned; kappa ~ 10^16 is essentially singular for
    FP64. Returns inf for a singular matrix.
    """
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"A must be 2-D (got ndim={arr.ndim})")
    if arr.size == 0:
        raise ValueError("A must be non-empty")
    sigma = np.linalg.svd(arr, compute_uv=False)
    sigma_max = float(sigma[0])
    sigma_min = float(sigma[-1])
    if sigma_min == 0.0:
        return float("inf")
    return sigma_max / sigma_min


def trapezoid_rule(f, a, b):
    """Trapezoidal rule: (h/2)[f(a) + f(b)], where h = b - a.

    Newton-Cotes quadrature for int_a^b f(x) dx using linear
    interpolation at the endpoints. Composite error is O(h^2).
    """
    if not callable(f):
        raise ValueError("f must be callable")
    a = float(a)
    b = float(b)
    h = b - a
    return (h / 2.0) * (f(a) + f(b))


def simpson_rule(f, a, b):
    """Simpson's rule: (h/6)[f(a) + 4 f(m) + f(b)].

    Here h = b - a and m = (a + b)/2. Newton-Cotes quadrature
    for int_a^b f(x) dx using quadratic interpolation; exact for
    polynomials up to degree 3. Composite error is O(h^4).
    """
    if not callable(f):
        raise ValueError("f must be callable")
    a = float(a)
    b = float(b)
    h = b - a
    m = (a + b) / 2.0
    return (h / 6.0) * (f(a) + 4.0 * f(m) + f(b))


def with_nan_check(fn, name=None):
    """NaN-detection hook for plain callables.

    Adapted from the chapter's torch forward-hook pattern
    (check_nan_hook): wraps fn so that after each call the
    output is checked, and a RuntimeError reporting "NaN in
    <name>" plus the input min/max is raised if any output
    element is NaN. Returns the wrapped callable.
    """
    if not callable(fn):
        raise ValueError("fn must be callable")
    label = name or getattr(fn, "__name__", "function")

    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        out_arr = np.asarray(out, dtype=float)
        if np.isnan(out_arr).any():
            msg = f"NaN in {label}"
            if args:
                inp = np.asarray(args[0], dtype=float)
                if inp.size:
                    msg += (
                        f" (input: min={inp.min():.2e},"
                        f" max={inp.max():.2e})"
                    )
            raise RuntimeError(msg)
        return out

    return wrapped
