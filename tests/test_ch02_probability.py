"""The book's Chapter 2 claims, as executable properties."""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

import numpy as np
import pytest

from mathpowersai.probability import (
    NaiveBayesClassifier,
    bayes_theorem,
    bernoulli_mle,
    mle_variance,
    sample_variance,
    simulate_sample_means,
)

SPAM_DOCS = [
    ["free", "money", "click", "now"],
    ["free", "winner", "prize", "claim"],
    ["click", "free", "offer", "limited"],
]
HAM_DOCS = [
    ["meeting", "schedule", "tomorrow", "office"],
    ["project", "deadline", "review", "meeting"],
    ["lunch", "team", "meeting", "friday"],
]
DOCS = SPAM_DOCS + HAM_DOCS
LABELS = ["spam"] * 3 + ["ham"] * 3


def fitted_classifier(alpha=1.0):
    clf = NaiveBayesClassifier(alpha=alpha)
    clf.fit(DOCS, LABELS)
    return clf


def test_naive_bayes_posteriors_sum_to_one():
    """predict_proba returns a valid distribution over classes."""
    clf = fitted_classifier()
    for doc in (["free", "meeting", "tomorrow"],
                ["free", "money"],
                ["lunch", "friday"],
                ["unseen", "words", "only"]):
        probs = clf.predict_proba(doc)
        assert set(probs) == {"spam", "ham"}
        assert all(p >= 0 for p in probs.values())
        assert sum(probs.values()) == pytest.approx(1.0)


def test_laplace_smoothing_keeps_probabilities_nonzero():
    """(count + alpha) / (total + alpha * |V|) > 0 for every word,
    even words never seen in a class (e.g. 'meeting' in spam)."""
    clf = fitted_classifier(alpha=1.0)
    for c in clf.classes:
        for word in clf.vocabulary:
            assert clf.word_probs[c][word] > 0.0
    # A class never paired with these words still gets nonzero
    # posterior mass.
    probs = clf.predict_proba(["meeting", "schedule", "office"])
    assert probs["spam"] > 0.0
    assert probs["ham"] > probs["spam"]


def test_naive_bayes_predicts_obvious_classes():
    clf = fitted_classifier()
    assert clf.predict(["free", "money", "click"]) == "spam"
    assert clf.predict(["meeting", "schedule"]) == "ham"


def test_naive_bayes_invalid_input_raises():
    with pytest.raises(ValueError):
        NaiveBayesClassifier(alpha=-0.5)
    with pytest.raises(ValueError):
        NaiveBayesClassifier().fit(DOCS, LABELS[:-1])
    with pytest.raises(ValueError):
        NaiveBayesClassifier().predict_proba(["free"])  # unfitted


def test_bernoulli_mle_equals_heads_over_n():
    """p_hat_MLE = k / n (the chapter's coin-flip derivation)."""
    assert bernoulli_mle(800, 1000) == pytest.approx(0.80)
    assert bernoulli_mle(8, 10) == pytest.approx(0.8)  # H-heavy flips
    assert bernoulli_mle(0, 5) == 0.0
    assert bernoulli_mle(5, 5) == 1.0
    with pytest.raises(ValueError):
        bernoulli_mle(6, 5)
    with pytest.raises(ValueError):
        bernoulli_mle(1, 0)


def test_mle_variance_vs_sample_variance():
    """sigma2_MLE = ((n-1)/n) * s^2: the 1/n vs 1/(n-1) split."""
    x = np.arange(10.0)
    n = x.size
    assert mle_variance(x) == pytest.approx(
        (n - 1) / n * sample_variance(x))
    assert mle_variance(x) < sample_variance(x)


def test_variance_of_sample_mean_shrinks_as_sigma2_over_n():
    """Var(xbar) = sigma^2 / n, checked empirically (seeded)."""
    rng = np.random.default_rng(42)
    sigma = 2.0
    empirical = []
    for n in (4, 16, 64):
        means = simulate_sample_means(rng, mu=0.0, sigma=sigma,
                                      n=n, n_trials=4000)
        var = means.var()
        empirical.append(var)
        assert var == pytest.approx(sigma**2 / n, rel=0.15)
    # Shrinks monotonically as n grows.
    assert empirical[0] > empirical[1] > empirical[2]


def test_bayes_theorem_matches_chapter_numbers():
    """Medical test: ~0.088; spam given 'free': ~0.77."""
    assert bayes_theorem(0.01, 0.95, 0.10) == pytest.approx(
        0.0876, abs=1e-3)
    assert bayes_theorem(0.30, 0.80, 0.10) == pytest.approx(
        24 / 31, abs=1e-3)
    with pytest.raises(ValueError):
        bayes_theorem(1.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        bayes_theorem(0.0, 0.5, 0.0)  # zero evidence
