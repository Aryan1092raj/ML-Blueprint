"""Shared foundation used by every algorithm.

``core`` holds only what more than one algorithm needs: the estimator API,
input validation, seeding, common metrics, small synthetic datasets, and the
visualisation panel registry.

It is kept small on purpose. Everything depends on it, so adding something
here should be discussed in an issue first. If only one algorithm needs it,
it belongs in that algorithm's own folder instead.

See ``docs/ARCHITECTURE.md``.
"""

from .base import Estimator, NotFittedError, Predictor, Transformer
from .metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from .random import DEFAULT_SEED, check_random_state
from .validation import check_array, check_consistent_length, check_X_y

__all__ = [
    # base
    "Estimator",
    "Predictor",
    "Transformer",
    "NotFittedError",
    # validation
    "check_array",
    "check_X_y",
    "check_consistent_length",
    # random
    "check_random_state",
    "DEFAULT_SEED",
    # metrics
    "accuracy_score",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
]
