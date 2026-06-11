"""Companion code for Chapter 3: Calculus Foundations.

Importable versions of the chapter's code listings, from
"The Math That Powers AI" (2nd ed.).
"""

import numpy as np

__all__ = [
    "numerical_gradient",
    "numerical_hessian",
    "chain_rule_demo",
    "sigmoid",
    "sigmoid_derivative",
    "vanishing_gradient_demo",
    "classify_critical_point",
]


def numerical_gradient(f, x, eps=1e-5):
    """Compute gradient via finite differences.

    Uses the central-difference approximation

        grad[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2 * eps)

    Parameters
    ----------
    f : callable
        Function mapping a 1-D array to a scalar.
    x : ndarray
        1-D point at which to evaluate the gradient.
    eps : float
        Finite-difference step size (must be positive).

    Returns
    -------
    ndarray
        Gradient of ``f`` at ``x``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array")
    if eps <= 0:
        raise ValueError("eps must be positive")
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


def numerical_hessian(f, x, eps=1e-5):
    """Compute the Hessian matrix via central finite differences.

    The Hessian is the n x n matrix of second partial derivatives

        H[i, j] = d^2 f / (dx_i dx_j)

    Parameters
    ----------
    f : callable
        Function mapping a 1-D array to a scalar.
    x : ndarray
        1-D point at which to evaluate the Hessian.
    eps : float
        Finite-difference step size (must be positive).

    Returns
    -------
    ndarray
        Symmetrized Hessian of ``f`` at ``x``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array")
    if eps <= 0:
        raise ValueError("eps must be positive")
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
            hess[i, j] = (
                f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)
            ) / (4 * eps**2)
    return (hess + hess.T) / 2


def chain_rule_demo(x):
    """Chain-rule composition demo: y = e^{x^2}.

    Following the chapter's example, write y = e^u with u = x^2:

        dy/du = e^u = e^{x^2}
        du/dx = 2x
        dy/dx = dy/du * du/dx = 2x e^{x^2}

    Parameters
    ----------
    x : float or ndarray
        Point(s) at which to evaluate the composition.

    Returns
    -------
    dict
        Keys ``"y"``, ``"dy_du"``, ``"du_dx"``, ``"dy_dx"``.
    """
    x = np.asarray(x, dtype=float)
    u = x**2
    y = np.exp(u)
    dy_du = np.exp(u)
    du_dx = 2 * x
    dy_dx = dy_du * du_dx
    return {"y": y, "dy_du": dy_du, "du_dx": du_dx, "dy_dx": dy_dx}


def sigmoid(x):
    """Sigmoid activation: sigma(x) = 1 / (1 + e^{-x}).

    Parameters
    ----------
    x : float or ndarray
        Input value(s).

    Returns
    -------
    float or ndarray
        Sigmoid of ``x``, in (0, 1).
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of the sigmoid: sigma'(x) = sigma(x)(1 - sigma(x)).

    The derivative is maximal at x = 0, where sigma'(0) = 0.25, and
    vanishes as |x| -> infinity. This causes the vanishing gradient
    problem for deep networks.

    Parameters
    ----------
    x : float or ndarray
        Input value(s).

    Returns
    -------
    float or ndarray
        sigma'(x), in (0, 0.25].
    """
    s = sigmoid(x)
    return s * (1.0 - s)


def vanishing_gradient_demo(n_layers):
    """Vanishing-gradient product demo: 0.25^n decay.

    When backpropagating through many layers of sigmoid activations,
    gradients multiply together. Since sigmoid derivatives are at most
    0.25, the activation factor alone shrinks exponentially:
    0.25^{10} ~ 0.000001.

    Parameters
    ----------
    n_layers : int
        Number of sigmoid layers (must be a positive integer).

    Returns
    -------
    ndarray
        Array of length ``n_layers`` whose k-th entry (1-indexed) is
        the cumulative gradient factor 0.25^k after k layers.
    """
    if not isinstance(n_layers, (int, np.integer)) or n_layers < 1:
        raise ValueError("n_layers must be a positive integer")
    return np.cumprod(np.full(n_layers, 0.25))


def classify_critical_point(hessian, tol=1e-8):
    """Classify a critical point from its Hessian's eigenvalues.

    Second derivative test (multivariate), at a critical point where
    grad f(x*) = 0:

    - All positive eigenvalues: local minimum (bowl shape)
    - All negative eigenvalues: local maximum (inverted bowl)
    - Mixed signs: saddle point (horse saddle shape)
    - Singular Hessian (zero eigenvalue): test is inconclusive

    Parameters
    ----------
    hessian : ndarray
        Symmetric n x n Hessian matrix.
    tol : float
        Eigenvalues within ``tol`` of zero are treated as zero.

    Returns
    -------
    str
        One of ``"minimum"``, ``"maximum"``, ``"saddle"``, or
        ``"inconclusive"``.
    """
    hessian = np.asarray(hessian, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square matrix")
    if not np.allclose(hessian, hessian.T, atol=1e-6):
        raise ValueError("hessian must be symmetric")
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.any(np.abs(eigenvalues) <= tol):
        return "inconclusive"
    if np.all(eigenvalues > 0):
        return "minimum"
    if np.all(eigenvalues < 0):
        return "maximum"
    return "saddle"
