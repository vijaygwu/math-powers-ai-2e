"""Companion code for The Math That Powers AI, 2nd edition.

One module per chapter; every function mirrors a listing printed in
the book, and every docstring formula is tested in tests/.
"""

__version__ = "2.0.0a1"

from . import (
    calculus,
    decompositions,
    information,
    linear_algebra,
    numerics,
    optimizers,
    pca,
    probability,
    spaces,
    visualization,
)

# Curated flat namespace: the names readers reach for most often.
from .calculus import (
    classify_critical_point,
    numerical_gradient,
    sigmoid,
    sigmoid_derivative,
)
from .decompositions import low_rank_error, svd, truncated_svd
from .information import (
    binary_entropy,
    bsc_capacity,
    cross_entropy,
    entropy,
    kl_divergence,
    mutual_information,
    perplexity,
)
from .linear_algebra import cosine_similarity, self_attention, word_analogy
from .numerics import (
    condition_number,
    kahan_sum,
    log_sum_exp,
    stable_softmax,
)
from .optimizers import (
    adam,
    compare_optimizers,
    gradient_descent,
    momentum,
    rmsprop,
    sgd,
)
from .pca import (
    PCA,
    compare_methods,
    pca_via_eigendecomposition,
    pca_via_svd,
    variance_threshold_components,
)
from .probability import NaiveBayesClassifier, bernoulli_mle
from .spaces import change_of_basis, gram_schmidt, project_onto_subspace

__all__ = [
    "__version__",
    # submodules
    "calculus", "decompositions", "information", "linear_algebra",
    "numerics", "optimizers", "pca", "probability", "spaces",
    "visualization",
    # curated names
    "PCA", "NaiveBayesClassifier", "adam", "bernoulli_mle",
    "binary_entropy", "bsc_capacity", "change_of_basis",
    "classify_critical_point", "compare_methods", "compare_optimizers",
    "condition_number", "cosine_similarity", "cross_entropy", "entropy",
    "gradient_descent", "gram_schmidt", "kahan_sum", "kl_divergence",
    "log_sum_exp", "low_rank_error", "momentum", "mutual_information",
    "numerical_gradient", "pca_via_eigendecomposition", "pca_via_svd",
    "perplexity", "project_onto_subspace", "rmsprop", "self_attention",
    "sgd", "sigmoid", "sigmoid_derivative", "stable_softmax", "svd",
    "truncated_svd", "variance_threshold_components", "word_analogy",
]
