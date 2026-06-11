"""Chapter 4: Information Theory.

Companion code for "The Math That Powers AI" (2nd ed).

Implements self-information, entropy, conditional entropy, mutual
information, KL divergence, cross-entropy, binary symmetric channel
capacity, and perplexity, exactly as developed in the chapter.

Conventions:
- Distributions are 1-D arrays of probabilities that are
  non-negative and sum to 1 (within tolerance), else ValueError.
- 0 log 0 = 0 (zero-probability terms are masked out).
- base=2 gives bits (log2); base=np.e gives nats (ln).
"""

import numpy as np

_ATOL = 1e-6


def _validate_dist(p, name="p"):
    """Validate and return a probability distribution as an array.

    Raises ValueError if entries are negative or the entries do
    not sum to 1 within tolerance.
    """
    p = np.asarray(p, dtype=float)
    if np.any(p < 0):
        raise ValueError(f"{name} has negative entries")
    if not np.isclose(p.sum(), 1.0, atol=_ATOL):
        raise ValueError(f"{name} does not sum to 1 (got {p.sum()})")
    return p


def _validate_prob(p, name="p"):
    """Validate a single probability p in [0, 1]."""
    p = float(p)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"{name} must be in [0, 1] (got {p})")
    return p


def self_information(p, base=2):
    """Self-information (surprisal) of an event with probability p.

    I(x) = -log p(x) = log(1 / p(x))

    Bits for base=2 (log2); nats for base=e (ln). p must be in
    (0, 1]; a zero-probability event has infinite surprisal.
    """
    p = _validate_prob(p, "p")
    if p == 0.0:
        raise ValueError("self-information undefined for p = 0")
    return float(-np.log(p) / np.log(base)) + 0.0


def entropy(probs, base=2):
    """Shannon entropy of a distribution.

    H(X) = -sum_x p(x) log p(x)

    Bits for base=2 (log2); nats for base=e (ln). Terms with
    p(x) = 0 are skipped, following the convention 0 log 0 = 0.
    """
    probs = _validate_dist(probs, "probs")
    probs = probs[probs > 0]  # Avoid log(0)
    h = -np.sum(probs * np.log(probs) / np.log(base))
    return float(h) + 0.0  # +0.0 normalizes -0.0 to 0.0


def binary_entropy(p):
    """Binary entropy function in bits.

    H(p) = -p log2 p - (1 - p) log2 (1 - p)

    Maximized at p = 0.5 (1 bit); zero at p = 0 or p = 1.
    """
    p = _validate_prob(p, "p")
    return entropy([p, 1.0 - p], base=2)


def conditional_entropy(joint, base=2):
    """Conditional entropy H(Y|X) from a joint distribution.

    H(Y|X) = -sum_{x,y} p(x,y) log p(y|x)

    joint is a 2-D array with joint[i, j] = p(X=x_i, Y=y_j).
    Bits for base=2 (log2); nats for base=e (ln).
    """
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2:
        raise ValueError("joint must be a 2-D array p(x, y)")
    _validate_dist(joint.ravel(), "joint")
    p_x = joint.sum(axis=1, keepdims=True)
    mask = joint > 0
    cond = joint[mask] / np.broadcast_to(p_x, joint.shape)[mask]
    terms = joint[mask] * np.log(cond) / np.log(base)
    return float(-np.sum(terms))


def mutual_information(joint, base=2):
    """Mutual information I(X; Y) from a joint distribution.

    I(X; Y) = H(Y) - H(Y|X)
            = sum_{x,y} p(x,y) log [p(x,y) / (p(x) p(y))]

    joint is a 2-D array with joint[i, j] = p(X=x_i, Y=y_j).
    Bits for base=2 (log2); nats for base=e (ln).
    """
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2:
        raise ValueError("joint must be a 2-D array p(x, y)")
    _validate_dist(joint.ravel(), "joint")
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    outer = np.outer(p_x, p_y)
    mask = joint > 0
    ratio = joint[mask] / outer[mask]
    terms = joint[mask] * np.log(ratio) / np.log(base)
    return float(np.sum(terms))


def kl_divergence(p, q, base=2):
    """Kullback-Leibler divergence KL(p || q).

    KL(p || q) = sum_x p(x) log [p(x) / q(x)]

    The expected extra cost (in bits for base=2, nats for base=e)
    of coding samples from p using a code optimized for q. Requires
    p and q to share support: p must be absolutely continuous with
    respect to q, i.e. q(x) = 0 implies p(x) = 0. Asymmetric:
    KL(p || q) != KL(q || p) in general.
    """
    p = _validate_dist(p, "p")
    q = _validate_dist(q, "q")
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if np.any((q == 0) & (p > 0)):
        raise ValueError(
            "KL(p || q) undefined: q(x) = 0 where p(x) > 0 "
            "(p is not absolutely continuous w.r.t. q)"
        )
    mask = p > 0
    terms = p[mask] * np.log(p[mask] / q[mask]) / np.log(base)
    return float(np.sum(terms))


def cross_entropy(p, q, base=2):
    """Cross-entropy of q relative to p.

    H(p, q) = -sum_x p(x) log q(x) = H(p) + KL(p || q)

    Bits for base=2 (log2); nats for base=e (ln). Requires
    q(x) > 0 wherever p(x) > 0.
    """
    p = _validate_dist(p, "p")
    q = _validate_dist(q, "q")
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if np.any((q == 0) & (p > 0)):
        raise ValueError(
            "cross-entropy undefined: q(x) = 0 where p(x) > 0"
        )
    mask = p > 0
    terms = p[mask] * np.log(q[mask]) / np.log(base)
    return float(-np.sum(terms))


def bsc_capacity(error_p):
    """Capacity of a binary symmetric channel in bits per symbol.

    C = 1 - H(p)

    where H is the binary entropy function and p is the crossover
    (error) probability. For p = 0.1:
    C = 1 - H(0.1) approx 0.531 bits per symbol.
    """
    error_p = _validate_prob(error_p, "error_p")
    return 1.0 - binary_entropy(error_p)


def perplexity(cross_entropy_loss, base=np.e):
    """Perplexity from a cross-entropy loss.

    Perplexity = exp(L) for L in nats (base=e, the default,
    matching frameworks like PyTorch), or 2**H for H in bits
    (base=2). Both give the same value for the same quantity:
    2**H_bits = e**H_nats.

    A perplexity of k means the model is as uncertain as choosing
    uniformly among k equally likely words.
    """
    cross_entropy_loss = float(cross_entropy_loss)
    if cross_entropy_loss < 0:
        raise ValueError("cross-entropy loss must be non-negative")
    return float(base) ** cross_entropy_loss
