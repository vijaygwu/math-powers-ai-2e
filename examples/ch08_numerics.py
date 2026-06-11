"""Chapter 8 examples: numerical methods.

Reproduces the chapter's printed numeric examples and writes them
to outputs/ch08_numerics.txt.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, "src"),
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


def main():
    lines = []

    # --- Softmax: naive vs. stable on large logits ---
    lines.append("Softmax with large logits z = [1000, 1001, 999]:")
    z = np.array([1000.0, 1001.0, 999.0])
    lines.append(f"naive_softmax(z)  = {naive_softmax(z)}")
    stable = stable_softmax(z)
    lines.append(
        "stable_softmax(z) = ["
        + ", ".join(f"{p:.3f}" for p in stable)
        + "]"
    )

    # --- Log-sum-exp trick ---
    lines.append("")
    lines.append("Log-sum-exp trick:")
    lines.append(
        f"log_sum_exp([1000, 1001]) = "
        f"{log_sum_exp([1000.0, 1001.0]):.4f}"
    )
    small = np.array([0.5, 1.0, -0.5])
    lines.append(
        f"log_sum_exp([0.5, 1.0, -0.5]) = {log_sum_exp(small):.6f}"
        f" (direct: {np.log(np.sum(np.exp(small))):.6f})"
    )

    # --- FP32 summation stall at 2^24 ---
    lines.append("")
    lines.append("FP32 summation stall:")
    demo = fp32_stall_demo()
    lines.append(
        f"np.float32(2**24) = {demo['stall_value']:,.0f}"
    )
    lines.append(
        f"np.float32(2**24) + 1.0 = "
        f"{demo['after_adding_one']:,.0f} (stalled: "
        f"{demo['stalled']})"
    )
    lines.append(
        "Summing 1e8 ones in FP32 naively stalls at 16,777,216."
    )

    # --- Kahan (compensated) summation ---
    lines.append("")
    lines.append("Kahan (compensated) summation:")
    vals = np.array([1e16, 1.0, -1e16])
    lines.append(
        f"naive sum(1e16 + 1 - 1e16)  = {sum(vals.tolist()):.1f}"
    )
    lines.append(
        f"kahan_sum(1e16 + 1 - 1e16)  = {kahan_sum(vals):.1f}"
    )

    # --- Machine epsilon ---
    lines.append("")
    lines.append("Machine epsilon (gap between 1 and next float):")
    lines.append(
        f"FP64: {machine_epsilon(np.float64):.6e} = 2^-52"
    )
    lines.append(
        f"FP32: {machine_epsilon(np.float32):.6e} = 2^-23"
    )

    # --- Condition number ---
    lines.append("")
    lines.append("Condition number kappa_2 = sigma_max/sigma_min:")
    A = np.array([[1.0, 1.0], [1.0, 1.0001]])
    lines.append(
        f"kappa([[1, 1], [1, 1.0001]]) = {condition_number(A):.3e}"
    )

    # --- Quadrature: trapezoid and Simpson ---
    lines.append("")
    lines.append("Quadrature for int_0^2 x^2 dx (true value 8/3):")
    f = lambda x: x ** 2
    lines.append(
        f"trapezoid_rule = {trapezoid_rule(f, 0.0, 2.0):.6f}"
    )
    lines.append(
        f"simpson_rule   = {simpson_rule(f, 0.0, 2.0):.6f}"
        f" (8/3 = {8.0 / 3.0:.6f})"
    )

    # --- NaN-detection hook (forward-hook pattern) ---
    lines.append("")
    lines.append("NaN-detection hook on naive softmax:")
    checked = with_nan_check(naive_softmax)
    try:
        checked(z)
    except RuntimeError as e:
        lines.append(f"Failure: {e}")
    lines.append("Diagnosis: logits ~1000 cause exp() overflow.")
    lines.append("Fix: clip logits or use stable softmax.")

    text = "\n".join(lines)
    print(text)

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, "outputs",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ch08_numerics.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nWrote {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
