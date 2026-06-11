"""Chapter 7 examples: Vector Spaces.

Reproduces the chapter's printed output (the projection-onto-a-
vector listing) plus the worked examples: the linear-independence
check, the corrected change-of-basis example, Gram-Schmidt, and
projection onto a subspace.  Also runs the Word2Vec-style analogy
demo from the war story.  Writes outputs/ch07_spaces.txt.
"""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)

import numpy as np  # noqa: E402

from mathpowersai.spaces import (  # noqa: E402
    change_of_basis,
    embedding_analogy,
    gram_schmidt,
    is_linearly_independent,
    make_toy_embeddings,
    project_onto_subspace,
    project_onto_vector,
)


def main():
    lines = []

    # Listing: projection onto a vector (the chapter's printed
    # output).
    v = np.array([2, 3])
    u = np.array([4, 1])
    proj = project_onto_vector(v, u)
    residual = v - proj  # Perpendicular component
    lines.append(f"v = {v}")
    lines.append(f"proj_u(v) = {proj.round(3)}")
    lines.append(f"residual = {residual.round(3)}")
    lines.append(
        f"Check orthogonality: {np.dot(proj, residual):.6f}"
    )  # Should be ~0
    lines.append("")

    # Linear independence via matrix rank (Section 7.2).
    v1, v2 = [1, 2, 3], [2, 4, 6]
    lines.append(
        f"v1={v1}, v2={v2} independent? "
        f"{is_linearly_independent([v1, v2])}"
    )
    e = np.eye(3)
    lines.append(
        "standard basis e1,e2,e3 independent? "
        f"{is_linearly_independent(e)}"
    )
    lines.append("")

    # Change of basis: the book's corrected example.  The point
    # (1.5, 1.2) in basis b1=(1.2, 0.4), b2=(0.3, 1.0) has
    # coordinates (1.06, 0.78)_B.
    B = np.column_stack([[1.2, 0.4], [0.3, 1.0]])
    x = np.array([1.5, 1.2])
    coords = change_of_basis(x, B)
    lines.append(f"point {x} in basis B -> {coords.round(2)}")
    lines.append(
        f"reconstruction B @ coords = {(B @ coords).round(2)}"
    )
    lines.append("")

    # Gram-Schmidt on the chapter's worked example:
    # v1 = (1, 1, 0), v2 = (1, 0, 1).
    V = np.column_stack([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    Q = gram_schmidt(V)
    lines.append("Gram-Schmidt on v1=(1,1,0), v2=(1,0,1):")
    lines.append(f"q1 = {Q[:, 0].round(4)}")
    lines.append(f"q2 = {Q[:, 1].round(4)}")
    lines.append(
        f"<q1, q2> = {np.dot(Q[:, 0], Q[:, 1]):.6f}"
    )  # Should be ~0
    lines.append("")

    # Projection onto the subspace W = span{v1, v2}.
    b = np.array([1.0, 2.0, 3.0])
    p = project_onto_subspace(b, V)
    r = b - p
    lines.append(f"proj_W({b}) = {p.round(4)}")
    lines.append(
        "residual orthogonal to W? "
        f"{np.allclose(V.T @ r, 0)}"
    )
    lines.append("")

    # Embedding-space demo (war story): king - man + woman ~= ?
    rng = np.random.default_rng(7)
    embeddings = make_toy_embeddings(rng)
    answer = embedding_analogy(embeddings, "king", "man", "woman")
    lines.append(f"king - man + woman ~= {answer}")

    text = "\n".join(lines) + "\n"
    print(text, end="")

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ch07_spaces.txt"
    out_path.write_text(text)
    return out_path


if __name__ == "__main__":
    main()
