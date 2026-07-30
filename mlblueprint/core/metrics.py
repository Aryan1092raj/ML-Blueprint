"""Evaluation metrics shared across algorithms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .validation import check_consistent_length

__all__ = [
    "accuracy_score",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
]


def _as_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Validate a ``(y_true, y_pred)`` pair and return both as arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred, names=("y_true", "y_pred"))
    return y_true, y_pred


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of predictions that are correct.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    score : float
        Accuracy in [0, 1], where 1.0 means every prediction was correct.

    Notes
    -----
    Accuracy is misleading on imbalanced data: predicting the majority class for
    a 99:1 split scores 0.99 while learning nothing. Report it alongside a
    confusion matrix or a per-class metric when the classes are unbalanced.
    """
    y_true, y_pred = _as_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the mean squared error between targets and predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    error : float
        Mean of the squared residuals. Lower is better; 0.0 is exact.
    """
    y_true, y_pred = _as_pair(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the mean absolute error between targets and predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    error : float
        Mean of the absolute residuals.

    Notes
    -----
    Less sensitive to outliers than :func:`mean_squared_error`, because the
    residuals are not squared.
    """
    y_true, y_pred = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the coefficient of determination, R².

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    score : float
        1.0 for a perfect fit, 0.0 for a model no better than predicting the
        mean, and negative for one that is worse than the mean.

    Notes
    -----
    Defined as ``1 - SS_res / SS_tot``. When the targets are constant ``SS_tot``
    is zero: this returns 1.0 if the predictions are also exact and 0.0
    otherwise, rather than dividing by zero.
    """
    y_true, y_pred = _as_pair(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return 1.0 - ss_res / ss_tot
