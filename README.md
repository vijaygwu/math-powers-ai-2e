# The Math That Powers AI — Companion Code (2nd Edition)

Executable companion to *The Math That Powers AI* (The AI Engineer's
Library, Book 1, 2nd edition) by Dr. Vijay Raghavan.

Every code listing printed in the book lives here as an importable,
tested module — one module per chapter — and every output box printed
in the book can be regenerated with one command.

## Install

```bash
pip install -e .            # NumPy + Matplotlib only
pip install -e ".[capstone]"  # adds scikit-learn for the MNIST capstone
```

## Layout

| Module | Chapter |
|---|---|
| `mathpowersai.linear_algebra` | Ch 1 — Vectors, Matrices, and Linear Maps |
| `mathpowersai.probability` | Ch 2 — Probability and Statistics |
| `mathpowersai.calculus` | Ch 3 — Calculus Foundations |
| `mathpowersai.information` | Ch 4 — Information Theory |
| `mathpowersai.optimizers` | Ch 5 — Optimization Basics |
| `mathpowersai.decompositions` | Ch 6 — Matrix Decomposition |
| `mathpowersai.spaces` | Ch 7 — Vector Spaces |
| `mathpowersai.numerics` | Ch 8 — Numerical Methods |
| `mathpowersai.pca` | Capstone — PCA from Scratch |
| `mathpowersai.visualization` | Plotting helpers used throughout |

## Reproduce the book's output boxes

```bash
python examples/ch05_gradient_descent.py   # prints the exact box from Ch 5
make outputs                               # regenerates outputs/ for every chapter
```

`outputs/` holds the canonical text of every output box printed in the
book. CI re-executes the examples on every push and fails if any
printed number drifts from the code that claims to produce it.

## Tests

```bash
pytest
```

The test suite encodes the book's mathematical claims as properties
(covariance eigenvalues are non-negative, truncated SVD is the best
rank-k approximation, softmax is shift-invariant, the Ch 5 gradient
descent walkthrough lands on the printed numbers, ...), so the book
and the code cannot silently disagree.

## Conventions

- Teaching code: NumPy + Matplotlib only (scikit-learn appears solely
  in the capstone, for validation against a reference implementation).
- Function names and docstring formulas match the book's notation
  exactly; the docstring formula IS the implementation's contract and
  is tested.
- All randomness flows through an explicit `rng: np.random.Generator`
  parameter; examples seed it so outputs are reproducible.
- Lines fit the book's 78-character listing width.

## Editions

This repo serves the 2nd edition. First-edition QR codes point at
[math-powers-ai](https://github.com/vijaygwu/math-powers-ai), which is
preserved unchanged. Tags `print-N` snapshot this repo at each printing.
