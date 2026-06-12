"""
Optimization algorithms for gradient-based learning.

This module provides clean implementations of common optimizers used in machine learning,
including vanilla gradient descent, SGD with momentum, RMSprop, and Adam.
"""

import numpy as np
from typing import Callable, Dict, List, Optional


def _check_common(lr: float, n_steps: int, max_path_len: int) -> None:
    """Validate the parameters shared by every optimizer."""
    if lr <= 0:
        raise ValueError(f"lr must be positive; got {lr}.")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative; got {n_steps}.")
    if max_path_len < 1:
        raise ValueError(
            f"max_path_len must be at least 1; got {max_path_len}."
        )


def gradient_descent(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    lr: float = 0.01,
    n_steps: int = 100,
    max_path_len: int = 10000
) -> List[np.ndarray]:
    """
    Vanilla gradient descent optimizer.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    lr : float
        Learning rate (step size).
    n_steps : int
        Number of optimization steps.
    max_path_len : int
        Maximum number of points retained in the returned path.
        Memory grows as O(max_path_len * dim); once the cap is
        reached intermediate points stop being recorded so the
        history cannot grow without bound. The final point is
        always kept.

    Returns
    -------
    List[np.ndarray]
        List of parameter values at each step (optimization path).

    Example
    -------
    >>> def grad_f(x):
    ...     return np.array([2*x[0], 4*x[1]])
    >>> path = gradient_descent(grad_f, np.array([4.0, 3.0]), lr=0.1, n_steps=50)
    >>> print(f"Final: {path[-1]}")

    Notes
    -----
    Update rule:
        x_t = x_{t-1} - lr * grad

    Raises
    ------
    ValueError
        If lr <= 0, n_steps < 0, or max_path_len < 1.
    """
    _check_common(lr, n_steps, max_path_len)
    x = x_init.copy().astype(float)
    path = [x.copy()]

    for _ in range(n_steps):
        grad = grad_fn(x)
        x = x - lr * grad
        if len(path) < max_path_len:
            path.append(x.copy())

    # Always retain the final point even if the cap was hit.
    if not np.array_equal(path[-1], x):
        path[-1] = x.copy()

    return path


def sgd(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    lr: float = 0.01,
    n_steps: int = 100,
    noise_scale: float = 0.1,
    max_path_len: int = 10000,
    rng: Optional[np.random.Generator] = None
) -> List[np.ndarray]:
    """
    Stochastic Gradient Descent with simulated noise.

    In practice, the stochasticity comes from mini-batch sampling.
    Here we simulate it by adding noise to the gradient.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    lr : float
        Learning rate (step size).
    n_steps : int
        Number of optimization steps.
    noise_scale : float
        Scale of Gaussian noise added to gradients.
    max_path_len : int
        Maximum number of points retained in the returned path;
        the final point is always kept.
    rng : np.random.Generator, optional
        Random generator for the simulated noise. Pass a seeded
        generator (e.g. ``np.random.default_rng(42)``) for
        reproducible paths; defaults to an unseeded generator.

    Returns
    -------
    List[np.ndarray]
        List of parameter values at each step.
    """
    _check_common(lr, n_steps, max_path_len)
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative; got "
                         f"{noise_scale}.")
    if rng is None:
        rng = np.random.default_rng()

    x = x_init.copy().astype(float)
    path = [x.copy()]

    for _ in range(n_steps):
        grad = grad_fn(x)
        # Add stochastic noise to simulate mini-batch variance
        noise = rng.standard_normal(grad.shape) * noise_scale
        x = x - lr * (grad + noise)
        if len(path) < max_path_len:
            path.append(x.copy())

    # Always retain the final point even if the cap was hit.
    if not np.array_equal(path[-1], x):
        path[-1] = x.copy()

    return path


def momentum(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    lr: float = 0.01,
    beta: float = 0.9,
    n_steps: int = 100,
    max_path_len: int = 10000
) -> List[np.ndarray]:
    """
    Gradient descent with momentum.

    Momentum accumulates past gradients to accelerate convergence
    and dampen oscillations.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    lr : float
        Learning rate (step size).
    beta : float
        Momentum coefficient (typically 0.9).
    n_steps : int
        Number of optimization steps.
    max_path_len : int
        Maximum number of points retained in the returned path;
        the final point is always kept.

    Returns
    -------
    List[np.ndarray]
        List of parameter values at each step.

    Notes
    -----
    Update rule:
        v_t = beta * v_{t-1} + grad
        x_t = x_{t-1} - lr * v_t
    """
    _check_common(lr, n_steps, max_path_len)
    if not 0 <= beta < 1:
        raise ValueError(f"beta must be in [0, 1); got {beta}.")

    x = x_init.copy().astype(float)
    v = np.zeros_like(x)  # Velocity (momentum term)
    path = [x.copy()]

    for _ in range(n_steps):
        grad = grad_fn(x)
        v = beta * v + grad
        x = x - lr * v
        if len(path) < max_path_len:
            path.append(x.copy())

    # Always retain the final point even if the cap was hit.
    if not np.array_equal(path[-1], x):
        path[-1] = x.copy()

    return path


def rmsprop(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    lr: float = 0.01,
    beta: float = 0.9,
    epsilon: float = 1e-8,
    n_steps: int = 100,
    max_path_len: int = 10000
) -> List[np.ndarray]:
    """
    RMSprop optimizer.

    Adapts learning rates based on a running average of squared gradients.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    lr : float
        Learning rate (step size).
    beta : float
        Decay rate for squared gradient average (typically 0.9).
    epsilon : float
        Small constant for numerical stability.
    n_steps : int
        Number of optimization steps.
    max_path_len : int
        Maximum number of points retained in the returned path;
        the final point is always kept.

    Returns
    -------
    List[np.ndarray]
        List of parameter values at each step.

    Notes
    -----
    Update rule:
        s_t = beta * s_{t-1} + (1 - beta) * grad^2
        x_t = x_{t-1} - lr * grad / (sqrt(s_t) + epsilon)

    Raises
    ------
    ValueError
        If epsilon <= 0 or lr <= 0; a non-positive epsilon can
        turn a zero gradient into a silent 0/0 NaN that then
        propagates through the whole path.
    """
    _check_common(lr, n_steps, max_path_len)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive; got "
                         f"{epsilon}.")
    if not 0 <= beta < 1:
        raise ValueError(f"beta must be in [0, 1); got {beta}.")

    x = x_init.copy().astype(float)
    s = np.zeros_like(x)  # Running average of squared gradients
    path = [x.copy()]

    for _ in range(n_steps):
        grad = grad_fn(x)
        s = beta * s + (1 - beta) * grad**2
        x = x - lr * grad / (np.sqrt(s) + epsilon)
        if len(path) < max_path_len:
            path.append(x.copy())

    # Always retain the final point even if the cap was hit.
    if not np.array_equal(path[-1], x):
        path[-1] = x.copy()

    return path


def adam(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    n_steps: int = 100,
    max_path_len: int = 10000
) -> List[np.ndarray]:
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Combines momentum with adaptive learning rates. The most widely used
    optimizer in deep learning.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    lr : float
        Learning rate (step size).
    beta1 : float
        Decay rate for first moment (momentum).
    beta2 : float
        Decay rate for second moment (squared gradients).
    epsilon : float
        Small constant for numerical stability.
    n_steps : int
        Number of optimization steps.
    max_path_len : int
        Maximum number of points retained in the returned path;
        the final point is always kept.

    Returns
    -------
    List[np.ndarray]
        List of parameter values at each step.

    Notes
    -----
    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m_hat = m_t / (1 - beta1^t)  (bias correction)
        v_hat = v_t / (1 - beta2^t)  (bias correction)
        x_t = x_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)

    Raises
    ------
    ValueError
        If epsilon <= 0 or lr <= 0; a non-positive epsilon can
        turn a zero gradient into a silent 0/0 NaN that then
        propagates through the whole path.

    References
    ----------
    Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
    arXiv preprint arXiv:1412.6980.
    """
    _check_common(lr, n_steps, max_path_len)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive; got "
                         f"{epsilon}.")
    for name, b in (("beta1", beta1), ("beta2", beta2)):
        if not 0 <= b < 1:
            raise ValueError(f"{name} must be in [0, 1); got {b}.")

    x = x_init.copy().astype(float)
    m = np.zeros_like(x)  # First moment (mean of gradients)
    v = np.zeros_like(x)  # Second moment (variance of gradients)
    path = [x.copy()]

    for t in range(1, n_steps + 1):
        grad = grad_fn(x)

        # Update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update parameters
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        if len(path) < max_path_len:
            path.append(x.copy())

    # Always retain the final point even if the cap was hit.
    if not np.array_equal(path[-1], x):
        path[-1] = x.copy()

    return path


def compare_optimizers(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    n_steps: int = 100,
    optimizers: Optional[List[str]] = None,
    lr: Optional[float] = None,
    optimizer_kwargs: Optional[Dict[str, dict]] = None
) -> Dict[str, List[np.ndarray]]:
    """
    Run multiple optimizers on the same problem for comparison.

    Parameters
    ----------
    grad_fn : Callable
        Function that computes the gradient at a given point.
    x_init : np.ndarray
        Initial parameter values.
    n_steps : int
        Number of optimization steps.
    optimizers : List[str], optional
        List of optimizer names to compare. Defaults to
        gradient_descent, momentum, rmsprop, and adam.
    lr : float, optional
        Learning rate applied to every optimizer. When None
        (the default), each optimizer uses its own documented
        default (0.01, except adam's 0.001), which avoids
        running adam at 100x its intended step size.
    optimizer_kwargs : dict, optional
        Per-optimizer extra keyword arguments, keyed by optimizer
        name, e.g. ``{'adam': {'beta1': 0.95}, 'sgd':
        {'noise_scale': 0.05}}``. Merged on top of ``lr``.

    Returns
    -------
    dict
        Dictionary mapping optimizer names to their optimization paths.
    """
    if optimizers is None:
        optimizers = ['gradient_descent', 'momentum', 'rmsprop', 'adam']

    results = {}

    lr_kwargs = {} if lr is None else {'lr': lr}
    optimizer_fns = {
        'gradient_descent': gradient_descent,
        'sgd': sgd,
        'momentum': momentum,
        'rmsprop': rmsprop,
        'adam': adam,
    }

    for name in optimizers:
        if name not in optimizer_fns:
            raise ValueError(
                f"Unknown optimizer {name!r}; "
                f"valid: {sorted(optimizer_fns)}"
            )
        extra = (optimizer_kwargs or {}).get(name, {})
        results[name] = optimizer_fns[name](
            grad_fn, x_init, n_steps=n_steps, **{**lr_kwargs, **extra}
        )

    return results
