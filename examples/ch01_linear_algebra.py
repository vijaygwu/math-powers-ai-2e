"""Reproduce the printed outputs of Chapter 1's code listings.

Runs the three ``pythoncode`` listings from Chapter 1 (cosine
similarity, the 2-layer transformation, and simplified self-
attention), prints their output, and writes the same text to
outputs/ch01_linear_algebra.txt.
"""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)

import numpy as np  # noqa: E402

from mathpowersai.linear_algebra import (  # noqa: E402
    attention_demo,
    similarity_demo,
    two_layer_demo,
    word_analogy,
)


def main():
    lines = []

    # Listing 1: cosine similarity on the Table 1.1 toy embedding.
    king_queen, king_apple = similarity_demo()
    lines.append(f"king-queen similarity: {king_queen:.3f}")  # High
    lines.append(f"king-apple similarity: {king_apple:.3f}")  # Low

    # Listing 2: a simple 2-layer transformation (no activation).
    x, z1, z2, W_composite = two_layer_demo()
    lines.append(f"Input x (4D):  {x}")
    lines.append(f"After layer 1: {z1.round(3)}")  # 3D intermediate
    lines.append(f"After layer 2: {z2.round(3)}")  # 2D output
    lines.append("")
    lines.append(
        f"Composite W2 @ W1 shape: {W_composite.shape}"
    )  # 2x4 matrix

    # Listing 3: simplified self-attention over three words.
    V, scores, weights, V_new = attention_demo()
    lines.append("Attention scores:\n" + str(scores))
    lines.append("Attention weights:\n" + str(weights.round(3)))
    lines.append("New embeddings:\n" + str(V_new.round(3)))

    # Word analogy from the chapter's running example.
    answer = word_analogy("king", "man", "woman")
    lines.append(f"king - man + woman ~= {answer}")

    text = "\n".join(lines) + "\n"
    print(text, end="")

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ch01_linear_algebra.txt"
    out_path.write_text(text)
    return out_path


if __name__ == "__main__":
    main()
