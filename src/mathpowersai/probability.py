"""Chapter 2: Probability and Statistics -- companion code.

Importable versions of the chapter's printed listings:

* ``bayes_theorem`` -- the medical-diagnosis / base-rate demo
* ``sample_gaussians`` -- the Gaussian sampling demo
* ``bernoulli_mle`` -- MLE for a Bernoulli parameter, p_hat = k / n
* ``sample_mean`` / ``mle_variance`` / ``sample_variance`` -- the
  1/n (MLE) vs 1/(n-1) (unbiased) distinction
* ``simulate_sample_means`` -- empirical Var(xbar) = sigma^2 / n
* ``NaiveBayesClassifier`` -- the chapter's complete spam classifier

No module-level side effects; all randomness flows through an
explicit ``rng: np.random.Generator`` parameter.
"""

from collections import defaultdict

import numpy as np


def _check_probability(value, name):
    """Raise ValueError unless ``value`` is a probability in [0, 1]."""
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")


def _check_rng(rng):
    """Raise ValueError unless ``rng`` is a numpy Generator."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a np.random.Generator "
                         "(e.g. np.random.default_rng(42))")


def bayes_theorem(prior, likelihood, false_positive_rate):
    """Compute posterior probability using Bayes' theorem.

    P(A | B) = P(B | A) * P(A) / P(B)

    where the evidence expands via the law of total probability:

    P(B) = P(B | A) * P(A) + P(B | A^c) * P(A^c)

    Parameters
    ----------
    prior : float
        P(A), e.g. disease prevalence (the base rate).
    likelihood : float
        P(B | A), e.g. test sensitivity P(+ | disease).
    false_positive_rate : float
        P(B | A^c), e.g. P(+ | healthy) = 1 - specificity.

    Returns
    -------
    float
        The posterior P(A | B).

    Raises
    ------
    ValueError
        If any argument is outside [0, 1] or the evidence is zero.
    """
    _check_probability(prior, "prior")
    _check_probability(likelihood, "likelihood")
    _check_probability(false_positive_rate, "false_positive_rate")
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    if evidence <= 0.0:
        raise ValueError("evidence P(B) is zero; posterior undefined")
    posterior = (likelihood * prior) / evidence
    return posterior


def sample_gaussians(rng, size=1000):
    """Sample from different Gaussian distributions (chapter demo).

    Draws ``size`` samples from each of N(0, 1), N(2, 1), and
    N(0, 0.25), so sample statistics can be checked against theory:
    E[X] = mu and Var(X) = sigma^2.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness, e.g. ``np.random.default_rng(42)``.
    size : int
        Number of samples per distribution (default 1000).

    Returns
    -------
    dict
        ``{"standard": N(0, 1) draws, "shifted": N(2, 1) draws,
        "narrow": N(0, 0.25) draws}``.

    Raises
    ------
    ValueError
        If ``rng`` is not a Generator or ``size`` is not positive.
    """
    _check_rng(rng)
    if int(size) != size or size <= 0:
        raise ValueError(f"size must be a positive integer, got {size}")
    return {
        "standard": rng.normal(0, 1, int(size)),    # N(0, 1)
        "shifted": rng.normal(2, 1, int(size)),     # N(2, 1)
        "narrow": rng.normal(0, 0.5, int(size)),    # N(0, 0.25)
    }


def sample_mean(x):
    """Sample mean: xbar = (1/n) * sum_i x_i.

    This is the MLE for the Gaussian mean:
    mu_hat_MLE = (1/n) * sum_i x_i = xbar.

    Raises
    ------
    ValueError
        If ``x`` is empty or not one-dimensional.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1-D array of samples")
    return float(np.sum(x) / x.size)


def mle_variance(x):
    """MLE (1/n) variance: sigma2_hat = (1/n) * sum_i (x_i - xbar)^2.

    This is the maximum likelihood estimator for the Gaussian
    variance. It is *biased*:

    E[(1/n) * sum_i (x_i - xbar)^2] = ((n - 1) / n) * sigma^2

    so the MLE slightly underestimates sigma^2. For large n this
    doesn't matter; compare ``sample_variance`` (the 1/(n-1)
    unbiased estimator).

    Raises
    ------
    ValueError
        If ``x`` is empty or not one-dimensional.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1-D array of samples")
    xbar = np.sum(x) / x.size
    return float(np.sum((x - xbar) ** 2) / x.size)


def sample_variance(x):
    """Unbiased sample variance: s^2 = (1/(n-1)) * sum (x_i - xbar)^2.

    Dividing by n - 1 instead of n corrects the bias of the MLE,
    giving E[s^2] = sigma^2. In ML we typically use the MLE (see
    ``mle_variance``) because we care about prediction, not
    parameter recovery.

    Raises
    ------
    ValueError
        If ``x`` has fewer than two samples or is not 1-D.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1-D array with at least 2 "
                         "samples")
    xbar = np.sum(x) / x.size
    return float(np.sum((x - xbar) ** 2) / (x.size - 1))


def bernoulli_mle(k, n):
    """MLE for a Bernoulli parameter: p_hat_MLE = k / n.

    Maximizes the log-likelihood
    ell(p) = k * log p + (n - k) * log(1 - p),
    whose stationary point d ell / dp = k/p - (n-k)/(1-p) = 0
    yields p_hat = k / n.

    Example: if 800 of 1000 spam emails contain "free", then
    p_hat = 0.80. This is exactly how Naive Bayes learns: count
    word frequencies in each class.

    Parameters
    ----------
    k : int
        Number of successes (e.g. heads, or emails with the word).
    n : int
        Number of trials.

    Raises
    ------
    ValueError
        If ``n <= 0``, ``k < 0``, ``k > n``, or non-integers given.
    """
    if int(n) != n or int(k) != k:
        raise ValueError("k and n must be integers")
    k, n = int(k), int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}")
    return k / n


def simulate_sample_means(rng, mu, sigma, n, n_trials):
    """Simulate ``n_trials`` sample means of size-``n`` Gaussian draws.

    Each trial draws x_1, ..., x_n ~ N(mu, sigma^2) i.i.d. and
    records xbar = (1/n) * sum_i x_i. Across trials the means
    satisfy Var(xbar) = sigma^2 / n: the variance of the sample
    mean shrinks as the sample size grows (and by the Central
    Limit Theorem, xbar is approximately Gaussian).

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    mu, sigma : float
        Population mean and standard deviation (sigma > 0).
    n : int
        Sample size per trial.
    n_trials : int
        Number of independent sample means to simulate.

    Returns
    -------
    np.ndarray
        Array of shape (n_trials,) holding the sample means.

    Raises
    ------
    ValueError
        If ``sigma <= 0`` or ``n``/``n_trials`` are not positive
        integers.
    """
    _check_rng(rng)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    for name, value in (("n", n), ("n_trials", n_trials)):
        if int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer, "
                             f"got {value}")
    draws = rng.normal(mu, sigma, size=(int(n_trials), int(n)))
    return draws.mean(axis=1)


class NaiveBayesClassifier:
    """
    A Naive Bayes classifier for text classification.
    Demonstrates: priors, Bayes' theorem, conditional independence,
    MLE.
    """

    def __init__(self, alpha=1.0):
        """Initialize with Laplace smoothing parameter alpha."""
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = alpha  # Laplace smoothing avoids zero probs
        self.class_priors = {}      # P(class) - prior probabilities
        self.word_probs = {}        # P(word | class) - likelihoods
        self.classes = []
        self.vocabulary = set()

    def fit(self, documents, labels):
        """Learn parameters using Maximum Likelihood Estimation."""
        if len(documents) != len(labels):
            raise ValueError("documents and labels must have the "
                             "same length")
        if len(documents) == 0:
            raise ValueError("cannot fit on an empty dataset")
        self.classes = list(set(labels))
        n_docs = len(documents)

        # Step 1: Estimate prior probabilities P(class) via MLE
        # MLE for categorical: count(class) / total
        class_counts = defaultdict(int)
        for label in labels:
            class_counts[label] += 1
        for c in self.classes:
            self.class_priors[c] = class_counts[c] / n_docs

        # Step 2: Estimate word likelihoods P(word | class) via MLE
        # With Laplace smoothing to handle unseen words
        word_counts = {c: defaultdict(int) for c in self.classes}
        total_words = {c: 0 for c in self.classes}

        for doc, label in zip(documents, labels):
            for word in doc:
                self.vocabulary.add(word)
                word_counts[label][word] += 1
                total_words[label] += 1

        if len(self.vocabulary) == 0:
            raise ValueError("documents contain no words")

        # MLE with Laplace smoothing:
        # (count + alpha) / (total + alpha * |V|)
        vocab_size = len(self.vocabulary)
        self.word_probs = {c: {} for c in self.classes}
        for c in self.classes:
            for word in self.vocabulary:
                count = word_counts[c][word]
                self.word_probs[c][word] = (
                    (count + self.alpha) /
                    (total_words[c] + self.alpha * vocab_size)
                )

    def predict_proba(self, document):
        """
        Compute P(class | document) using Bayes' theorem.

        P(class | words) ~ P(class) * Prod P(word | class)
                           ^          ^
                         prior    likelihood (cond. independence)
        """
        if not self.classes:
            raise ValueError("classifier is not fitted; call fit() "
                             "first")
        log_probs = {}
        for c in self.classes:
            # Start with log prior: log P(class)
            log_prob = np.log(self.class_priors[c])

            # Add log likelihoods: sum of log P(word | class)
            # This is the "naive" conditional independence assumption
            for word in document:
                if word in self.vocabulary:
                    log_prob += np.log(self.word_probs[c][word])

            log_probs[c] = log_prob

        # Convert to probabilities via softmax (normalization)
        max_log = max(log_probs.values())
        probs = {c: np.exp(log_probs[c] - max_log)
                 for c in self.classes}
        total = sum(probs.values())
        return {c: p / total for c, p in probs.items()}

    def predict(self, document):
        """Return the class with highest posterior probability."""
        probs = self.predict_proba(document)
        return max(probs, key=probs.get)
