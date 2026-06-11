"""Reproduce the printed outputs of Chapter 2 (Probability).

Runs the chapter's demos -- Bayes' theorem for medical diagnosis,
Gaussian sampling, Bernoulli MLE, variance of the sample mean, and
the Naive Bayes spam classifier -- writes the transcript to
outputs/ch02_probability.txt and prints it to stdout.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "src"))

import numpy as np

from mathpowersai.probability import (
    NaiveBayesClassifier,
    bayes_theorem,
    bernoulli_mle,
    mle_variance,
    sample_gaussians,
    sample_variance,
    simulate_sample_means,
)


def main():
    rng = np.random.default_rng(42)
    lines = []

    # --- Bayes' theorem: the medical test / base rate fallacy ---
    lines.append("=== Bayes' theorem: medical diagnosis ===")
    prior_disease = 0.01
    sensitivity = 0.95       # P(+|disease)
    false_positive = 0.10    # P(+|healthy)
    posterior = bayes_theorem(prior_disease, sensitivity,
                              false_positive)
    lines.append(f"P(disease | positive test) = {posterior:.3f}")

    # --- Spam version of the same computation (running example) ---
    p_spam = bayes_theorem(prior=0.30, likelihood=0.80,
                           false_positive_rate=0.10)
    lines.append(f"P(spam | 'free')           = {p_spam:.3f}")

    # --- Gaussian sampling demo ---
    lines.append("")
    lines.append("=== Gaussian sampling (N(0,1), N(2,1), N(0,0.25)) "
                 "===")
    samples = sample_gaussians(rng, size=1000)
    standard = samples["standard"]
    shifted = samples["shifted"]
    narrow = samples["narrow"]
    lines.append(f"Standard: mean={standard.mean():.2f}, "
                 f"std={standard.std():.2f}")
    lines.append(f"Shifted:  mean={shifted.mean():.2f}, "
                 f"std={shifted.std():.2f}")
    lines.append(f"Narrow:   mean={narrow.mean():.2f}, "
                 f"std={narrow.std():.2f}")

    # --- Bernoulli MLE: learning spam word probabilities ---
    lines.append("")
    lines.append("=== Bernoulli MLE: p_hat = k / n ===")
    p_hat = bernoulli_mle(k=800, n=1000)
    lines.append("800 of 1000 spam emails contain 'free' -> "
                 f"p_hat = {p_hat:.2f}")

    # --- 1/n (MLE) vs 1/(n-1) (unbiased) variance ---
    lines.append("")
    lines.append("=== Variance: MLE (1/n) vs unbiased (1/(n-1)) ===")
    x = rng.normal(0, 1, 10)
    lines.append(f"n=10 sample: MLE variance      = "
                 f"{mle_variance(x):.4f}")
    lines.append(f"n=10 sample: unbiased variance = "
                 f"{sample_variance(x):.4f}")

    # --- Var(xbar) = sigma^2 / n shrinks with n ---
    lines.append("")
    lines.append("=== Variance of the sample mean: sigma^2 / n ===")
    sigma = 2.0
    for n in (4, 16, 64):
        means = simulate_sample_means(rng, mu=0.0, sigma=sigma,
                                      n=n, n_trials=2000)
        lines.append(f"n={n:3d}: empirical Var(xbar) = "
                     f"{means.var():.4f}  "
                     f"(theory sigma^2/n = {sigma**2 / n:.4f})")

    # --- Naive Bayes spam classifier ---
    lines.append("")
    lines.append("=== Naive Bayes spam classifier ===")
    spam_docs = [
        ["free", "money", "click", "now"],
        ["free", "winner", "prize", "claim"],
        ["click", "free", "offer", "limited"],
    ]
    ham_docs = [
        ["meeting", "schedule", "tomorrow", "office"],
        ["project", "deadline", "review", "meeting"],
        ["lunch", "team", "meeting", "friday"],
    ]

    documents = spam_docs + ham_docs
    labels = ["spam"] * 3 + ["ham"] * 3

    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(documents, labels)

    # Classify a new email
    test_doc = ["free", "meeting", "tomorrow"]
    probs = clf.predict_proba(test_doc)
    lines.append(f"P(spam | doc) = {probs['spam']:.3f}")
    lines.append(f"P(ham | doc)  = {probs['ham']:.3f}")
    lines.append(f"Prediction: {clf.predict(test_doc)}")

    text = "\n".join(lines) + "\n"
    out_dir = os.path.join(_HERE, os.pardir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ch02_probability.txt")
    with open(out_path, "w") as fh:
        fh.write(text)

    print(text, end="")
    print(f"[written to {os.path.relpath(out_path)}]")


if __name__ == "__main__":
    main()
