"""Chapter 3 examples: Calculus Foundations.

Reproduces the chapter's printed outputs and writes them to
outputs/ch03_calculus.txt.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "src"),
)

from mathpowersai.calculus import (
    chain_rule_demo,
    classify_critical_point,
    numerical_gradient,
    numerical_hessian,
    sigmoid,
    sigmoid_derivative,
    vanishing_gradient_demo,
)


def main():
    lines = []

    # --- Numerical vs. analytical gradient (chapter listing) ---
    # f(x) = x1^2 + 2*x2^2  =>  grad f = [2*x1, 4*x2]
    f = lambda x: x[0]**2 + 2*x[1]**2
    x = np.array([3.0, 2.0])

    lines.append(
        f"Numerical gradient: {numerical_gradient(f, x)}"
    )  # [6, 8]
    lines.append(
        f"Analytical gradient: {[float(2*x[0]), float(4*x[1])]}"
    )  # [6, 8]

    # --- Chain rule: y = e^{x^2} => dy/dx = 2x e^{x^2} ---
    demo = chain_rule_demo(1.0)
    lines.append("")
    lines.append("Chain rule for y = e^(x^2) at x = 1:")
    lines.append(f"  dy/du = e^(x^2)   = {demo['dy_du']:.6f}")
    lines.append(f"  du/dx = 2x        = {demo['du_dx']:.6f}")
    lines.append(f"  dy/dx = 2x e^(x^2) = {demo['dy_dx']:.6f}")
    numeric = numerical_gradient(
        lambda v: np.exp(v[0]**2), np.array([1.0])
    )[0]
    lines.append(f"  numerical check   = {numeric:.6f}")

    # --- Sigmoid and its derivative ---
    lines.append("")
    lines.append("Sigmoid sigma(x) = 1 / (1 + e^(-x)):")
    lines.append(f"  sigma(0)  = {sigmoid(0.0):.4f}")
    lines.append(
        f"  sigma'(0) = sigma(0)(1 - sigma(0)) = "
        f"{sigmoid_derivative(0.0):.4f} (maximum value)"
    )
    lines.append(f"  sigma'(5) = {sigmoid_derivative(5.0):.6f}")

    # --- Vanishing gradients: 0.25^n decay ---
    lines.append("")
    lines.append("Vanishing gradients through sigmoid layers:")
    factors = vanishing_gradient_demo(10)
    for k in (1, 2, 5, 10):
        lines.append(f"  after {k:2d} layers: 0.25^{k} = "
                     f"{factors[k - 1]:.10f}")

    # --- Hessian-based critical-point classification ---
    lines.append("")
    lines.append("Second derivative test at the critical point (0, 0):")
    cases = [
        ("f(x, y) = x^2 + y^2", lambda v: v[0]**2 + v[1]**2),
        ("f(x, y) = -x^2 - y^2", lambda v: -v[0]**2 - v[1]**2),
        ("f(x, y) = x^2 - y^2", lambda v: v[0]**2 - v[1]**2),
    ]
    origin = np.array([0.0, 0.0])
    for name, func in cases:
        hess = numerical_hessian(func, origin)
        eigenvalues = np.linalg.eigvalsh(hess)
        label = classify_critical_point(hess, tol=1e-3)
        lines.append(
            f"  {name}: eigenvalues "
            f"[{eigenvalues[0]:.2f}, {eigenvalues[1]:.2f}] "
            f"-> {label}"
        )

    text = "\n".join(lines) + "\n"
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ch03_calculus.txt")
    with open(out_path, "w") as handle:
        handle.write(text)

    print(text, end="")
    print(f"\nWrote {os.path.normpath(out_path)}")


if __name__ == "__main__":
    main()
