"""
Visualization utilities for optimization and machine learning concepts.

This module provides helper functions for creating clear, publication-quality
visualizations of loss landscapes, optimization paths, and convergence behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Dict

# Floor applied to losses on log-scale plots: log scale silently
# drops non-positive values, and an exactly-zero loss (a legitimate
# converged result) would otherwise vanish from the curve.
LOG_SCALE_FLOOR = 1e-16


def _tab10_colors(n: int) -> np.ndarray:
    """n visually distinct colors from tab10, cycling past 10
    instead of producing duplicates via linspace interpolation."""
    return plt.cm.tab10(np.arange(n) % 10)


def plot_loss_landscape(
    f: Callable[[np.ndarray], float],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 100,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis',
    levels: int = 30,
    title: str = 'Loss Landscape'
) -> plt.Axes:
    """
    Plot a 2D loss landscape as a contour plot.

    Parameters
    ----------
    f : Callable
        Loss function that takes a 2D array and returns a scalar.
    x_range : Tuple[float, float]
        Range for x-axis (min, max).
    y_range : Tuple[float, float]
        Range for y-axis (min, max).
    resolution : int
        Number of points per axis for the grid.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. Creates new figure if None.
    cmap : str
        Colormap name.
    levels : int
        Number of contour levels.
    title : str
        Plot title.

    Returns
    -------
    plt.Axes
        The matplotlib axes object.

    Notes
    -----
    When ``ax`` is None this creates a new figure that the caller
    owns. The figure is intentionally left open so it can be shown
    or returned; callers that generate many plots in a loop should
    close it (e.g. ``plt.close(ax.figure)``) to avoid leaking
    figures and Matplotlib's "more than 20 figures opened" warning.

    Example
    -------
    >>> def rosenbrock(x):
    ...     return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    >>> ax = plot_loss_landscape(rosenbrock, x_range=(-2, 2), y_range=(-1, 3))
    >>> plt.show()
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Compute loss at each grid point. Note: this calls f once
    # per point, i.e. resolution**2 scalar evaluations (10,000
    # by default). That is fine for a one-off book figure but
    # slow for interactive/high-resolution use; if f is
    # vectorizable, evaluate it on the stacked grid instead.
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    # Create filled contour plot
    contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
    ax.contour(X, Y, Z, levels=levels, colors='k', alpha=0.3, linewidths=0.5)

    # Bind explicitly to the axes' own figure rather than the
    # pyplot-global current figure, which may differ.
    ax.figure.colorbar(contour, ax=ax, label='Loss')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')

    return ax


def plot_optimization_path(
    path: List[np.ndarray],
    ax: Optional[plt.Axes] = None,
    color='red',
    label: str = 'Optimization path',
    marker_size: int = 20,
    line_width: float = 1.5,
    alpha: float = 0.8,
    show_start_end: bool = True
) -> plt.Axes:
    """
    Plot an optimization path on a 2D landscape.

    Parameters
    ----------
    path : List[np.ndarray]
        List of 2D parameter values from optimization.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. Creates new figure if None.
    color : color-like
        Color for the path: any Matplotlib color spec (name, hex,
        or RGBA array, as passed by create_comparison_figure).
    label : str
        Label for the legend.
    marker_size : int
        Size of start/end markers.
    line_width : float
        Width of the path line.
    alpha : float
        Transparency of the path.
    show_start_end : bool
        Whether to show start/end markers.

    Returns
    -------
    plt.Axes
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    path_array = np.array(path)

    # Validate shape: this plotter only handles 2D paths of
    # shape (N, 2). Higher-dimensional optimizations cannot be
    # drawn on a 2D landscape.
    if path_array.ndim != 2 or path_array.shape[1] != 2:
        raise ValueError(
            "path must be a sequence of 2D points with shape "
            f"(N, 2); got array of shape {path_array.shape}."
        )
    if path_array.shape[0] == 0:
        raise ValueError("path must contain at least one point.")

    # Plot the path
    ax.plot(path_array[:, 0], path_array[:, 1],
            color=color, linewidth=line_width, alpha=alpha, label=label)

    # Plot intermediate points (skip when the path is too short
    # to have any interior points, e.g. length 1 or 2).
    if path_array.shape[0] > 2:
        ax.scatter(path_array[1:-1, 0], path_array[1:-1, 1],
                   color=color, s=marker_size//2,
                   alpha=alpha*0.5, zorder=5)

    if show_start_end:
        # Mark start point
        ax.scatter(path_array[0, 0], path_array[0, 1],
                   color=color, s=marker_size*3, marker='o',
                   edgecolors='white', linewidths=2, zorder=10, label=f'{label} (start)')
        # Mark end point
        ax.scatter(path_array[-1, 0], path_array[-1, 1],
                   color=color, s=marker_size*3, marker='*',
                   edgecolors='white', linewidths=1, zorder=10)

    return ax


def plot_convergence(
    paths: Dict[str, List[np.ndarray]],
    f: Callable[[np.ndarray], float],
    ax: Optional[plt.Axes] = None,
    title: str = 'Convergence Comparison',
    log_scale: bool = True
) -> plt.Axes:
    """
    Plot convergence curves for multiple optimizers.

    Parameters
    ----------
    paths : Dict[str, List[np.ndarray]]
        Dictionary mapping optimizer names to their optimization paths.
    f : Callable
        Loss function to evaluate convergence.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. Creates new figure if None.
    title : str
        Plot title.
    log_scale : bool
        Whether to use logarithmic scale for y-axis. Assumes
        non-negative losses: values below LOG_SCALE_FLOOR
        (including negatives) are clamped to the floor.

    Returns
    -------
    plt.Axes
        The matplotlib axes object.
    """
    if not paths:
        raise ValueError("paths must contain at least one optimizer path.")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    colors = _tab10_colors(len(paths))

    for (name, path), color in zip(paths.items(), colors):
        losses = [f(x) for x in path]
        if log_scale:
            losses = list(np.maximum(losses, LOG_SCALE_FLOOR))
        ax.plot(losses, label=name, color=color, linewidth=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    return ax


def plot_learning_rate_comparison(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    f: Callable[[np.ndarray], float],
    x_init: np.ndarray,
    learning_rates: List[float],
    n_steps: int = 50,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Compare the effect of different learning rates.

    Parameters
    ----------
    grad_fn : Callable
        Gradient function.
    f : Callable
        Loss function.
    x_init : np.ndarray
        Initial parameter values.
    learning_rates : List[float]
        List of learning rates to compare.
    n_steps : int
        Number of optimization steps.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    from .optimizers import gradient_descent

    if not learning_rates:
        raise ValueError("learning_rates must contain at least one value.")

    fig, axes = plt.subplots(1, len(learning_rates), figsize=figsize)
    if len(learning_rates) == 1:
        axes = [axes]

    for ax, lr in zip(axes, learning_rates):
        # Plot loss landscape
        plot_loss_landscape(f, ax=ax, title=f'lr = {lr}')

        # Run optimization
        path = gradient_descent(grad_fn, x_init, lr=lr, n_steps=n_steps)

        # Plot path
        plot_optimization_path(path, ax=ax, color='red', label='GD path')

        ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def plot_gradient_field(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 20,
    ax: Optional[plt.Axes] = None,
    title: str = 'Gradient Field'
) -> plt.Axes:
    """
    Plot the gradient vector field.

    Parameters
    ----------
    grad_fn : Callable
        Gradient function.
    x_range : Tuple[float, float]
        Range for x-axis.
    y_range : Tuple[float, float]
        Range for y-axis.
    resolution : int
        Number of arrows per axis.
    ax : plt.Axes, optional
        Matplotlib axes to plot on.
    title : str
        Plot title.

    Returns
    -------
    plt.Axes
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Compute gradients
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    first = np.asarray(grad_fn(np.array([X[0, 0], Y[0, 0]])))
    if first.shape != (2,):
        raise ValueError(
            "grad_fn must return a gradient of shape (2,) for a "
            f"2D field; got shape {first.shape}."
        )

    for i in range(resolution):
        for j in range(resolution):
            # Reuse the gradient already computed for the shape check.
            grad = first if (i, j) == (0, 0) else \
                grad_fn(np.array([X[i, j], Y[i, j]]))
            # Negative gradient points toward minimum
            U[i, j] = -grad[0]
            V[i, j] = -grad[1]

    # Normalize arrows for visibility
    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude == 0] = 1  # Avoid division by zero
    U_norm = U / magnitude
    V_norm = V / magnitude

    ax.quiver(X, Y, U_norm, V_norm, magnitude, cmap='coolwarm', alpha=0.8)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')

    return ax


def create_comparison_figure(
    f: Callable[[np.ndarray], float],
    paths: Dict[str, List[np.ndarray]],
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    figsize: Tuple[int, int] = (16, 6)
) -> plt.Figure:
    """
    Create a side-by-side comparison of paths and convergence.

    Parameters
    ----------
    f : Callable
        Loss function.
    paths : Dict[str, List[np.ndarray]]
        Dictionary mapping optimizer names to their paths.
    x_range : Tuple[float, float]
        Range for x-axis on landscape plot.
    y_range : Tuple[float, float]
        Range for y-axis on landscape plot.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    if not paths:
        raise ValueError("paths must contain at least one optimizer path.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot loss landscape with all paths
    plot_loss_landscape(f, x_range=x_range, y_range=y_range, ax=ax1,
                        title='Optimizer Trajectories')

    colors = _tab10_colors(len(paths))
    for (name, path), color in zip(paths.items(), colors):
        plot_optimization_path(path, ax=ax1, color=color, label=name)

    ax1.legend(loc='upper right')

    # Plot convergence
    plot_convergence(paths, f, ax=ax2, title='Convergence Comparison')

    plt.tight_layout()
    return fig
