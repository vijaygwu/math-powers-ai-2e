"""Chapter 4 examples: information theory.

Reproduces the chapter's printed numeric examples and writes them
to outputs/ch04_information.txt.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, "src"),
)

from mathpowersai.information import (
    binary_entropy,
    bsc_capacity,
    cross_entropy,
    entropy,
    kl_divergence,
    perplexity,
    self_information,
)


def main():
    lines = []

    # --- Self-information (surprisal) ---
    lines.append("Self-information:")
    lines.append(
        f"Fair coin heads (p=0.5): "
        f"{self_information(0.5):.3f} bits"
    )
    lines.append(
        f"Fair die six (p=1/6):    "
        f"{self_information(1 / 6):.3f} bits"
    )
    lines.append(
        f"Certain event (p=1.0):   "
        f"{self_information(1.0):.3f} bits"
    )

    # --- Entropy of coin flips ---
    lines.append("")
    lines.append("Entropy:")
    lines.append(
        f"Fair coin (p=0.5): {entropy([0.5, 0.5]):.3f} bits"
    )
    lines.append(
        f"Biased (p=0.9):    {entropy([0.9, 0.1]):.3f} bits"
    )
    lines.append(
        f"Certain (p=1.0):   {entropy([1.0, 0.0]):.3f} bits"
    )
    # Uniform over 8 outcomes: log2(8) = 3 bits
    lines.append(
        f"Uniform over 8:    {entropy([1 / 8] * 8):.3f} bits"
    )

    # --- KL divergence (asymmetric!) ---
    lines.append("")
    lines.append("KL divergence:")
    p = [0.1, 0.9]  # True: biased toward outcome 2
    q = [0.5, 0.5]  # Model: assumes uniform
    # Cost of using q when p is true
    lines.append(f"KL(p || q) = {kl_divergence(p, q):.4f} bits")
    # Asymmetric!
    lines.append(f"KL(q || p) = {kl_divergence(q, p):.4f} bits")

    # --- BSC capacity: 1 - H(0.1) ~ 0.531 ---
    lines.append("")
    lines.append("Binary symmetric channel:")
    lines.append(
        f"H(0.1) = {binary_entropy(0.1):.3f} bits"
    )
    lines.append(
        f"Capacity = 1 - H(0.1) = {bsc_capacity(0.1):.3f} "
        f"bits per symbol"
    )

    # --- Checkpoint: H(p, q) = H(p) + KL(p || q) ---
    lines.append("")
    lines.append("Identity H(p, q) = H(p) + KL(p || q):")
    p_b = [0.7, 0.3]  # Bernoulli(0.7)
    q_b = [0.5, 0.5]  # Bernoulli(0.5)
    h_p = entropy(p_b)
    kl_pq = kl_divergence(p_b, q_b)
    h_pq = cross_entropy(p_b, q_b)
    lines.append(f"H(p)        = {h_p:.4f} bits")
    lines.append(f"KL(p || q)  = {kl_pq:.4f} bits")
    lines.append(f"H(p, q)     = {h_pq:.4f} bits")
    lines.append(
        f"H(p) + KL(p || q) = {h_p + kl_pq:.4f} bits "
        f"(matches: {np.isclose(h_pq, h_p + kl_pq)})"
    )

    # --- Perplexity from cross-entropy ---
    lines.append("")
    lines.append("Perplexity:")
    ce_nats = cross_entropy(p_b, q_b, base=np.e)
    lines.append(
        f"Cross-entropy = {ce_nats:.4f} nats -> "
        f"perplexity = {perplexity(ce_nats):.4f}"
    )
    lines.append(
        f"Same in bits: 2**{h_pq:.4f} = "
        f"{perplexity(h_pq, base=2):.4f}"
    )

    text = "\n".join(lines)
    print(text)

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, "outputs",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ch04_information.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nWrote {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
