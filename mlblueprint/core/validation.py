"""Input validation shared by every algorithm."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["check_array", "check_X_y", "check_consistent_length"]


def check_array(
    X: ArrayLike,
    *,
    name: str = "X",
    ensure_2d: bool = True,
    dtype: type = np.float64,
    allow_empty: bool = False,
) -> NDArray:
    """Validate an array-like input and return it as a NumPy array.

    Parameters
    ----------
    X : array-like
        The input to validate.
    name : str, default="X"
        Name used in error messages, so the user knows which argument is wrong.
    ensure_2d : bool, default=True
        Require a 2-D array. A 1-D input is rejected rather than silently
        reshaped, because guessing whether the user meant one sample or one
        feature is how bugs get hidden.
    dtype : type, default=np.float64
        The dtype to cast to.
    allow_empty : bool, default=False
        Whether zero samples is acceptable.

    Returns
    -------
    X : ndarray
        The validated array.

    Raises
    ------
    ValueError
        If the input is not numeric, has the wrong number of dimensions, is
        empty when that is not allowed, or contains NaN or infinity.
    """
    try:
        array = np.asarray(X, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} could not be converted to a numeric array of dtype "
            f"{np.dtype(dtype).name}. Got type {type(X).__name__}."
        ) from exc

    if ensure_2d and array.ndim != 2:
        raise ValueError(
            f"{name} must be 2-dimensional with shape (n_samples, n_features), "
            f"but it has {array.ndim} dimension(s) and shape {array.shape}. "
            f"If you have a single feature, reshape with {name}.reshape(-1, 1); "
            f"for a single sample, use {name}.reshape(1, -1)."
        )

    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} is empty (shape {array.shape}).")

    if not np.all(np.isfinite(array)):
        n_nan = int(np.isnan(array).sum())
        n_inf = int(np.isinf(array).sum())
        raise ValueError(
            f"{name} contains values that are not finite: "
            f"{n_nan} NaN and {n_inf} infinite."
        )

    return array


def check_consistent_length(
    *arrays: ArrayLike, names: tuple[str, ...] | None = None
) -> None:
    """Check that all arrays have the same number of samples.

    Parameters
    ----------
    *arrays : array-like
        Arrays whose first dimensions must match.
    names : tuple of str, optional
        Names for the error message. Defaults to positional labels.

    Raises
    ------
    ValueError
        If the first dimensions differ.
    """
    lengths = [len(np.asarray(a)) for a in arrays]
    if len(set(lengths)) <= 1:
        return

    labels = names or tuple(f"argument {i}" for i in range(len(arrays)))
    detail = ", ".join(
        f"{label}: {n}" for label, n in zip(labels, lengths, strict=False)
    )
    raise ValueError(f"Inputs have inconsistent numbers of samples: {detail}.")


def check_X_y(
    X: ArrayLike,
    y: ArrayLike,
    *,
    y_numeric: bool = True,
    allow_empty: bool = False,
) -> tuple[NDArray, NDArray]:
    """Validate a supervised ``(X, y)`` pair.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Targets.
    y_numeric : bool, default=True
        Cast ``y`` to float. Set to ``False`` for classification targets, which
        may be strings or arbitrary labels.
    allow_empty : bool, default=False
        Whether zero samples is acceptable.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Validated training data.
    y : ndarray of shape (n_samples,)
        Validated targets.

    Raises
    ------
    ValueError
        If either input is invalid or their lengths disagree.
    """
    X = check_array(X, name="X", allow_empty=allow_empty)

    y = np.asarray(y, dtype=np.float64) if y_numeric else np.asarray(y)

    if y.ndim != 1:
        raise ValueError(
            f"y must be 1-dimensional with shape (n_samples,), but it has "
            f"shape {y.shape}. Use y.ravel() if you have a column vector."
        )

    check_consistent_length(X, y, names=("X", "y"))
    return X, y
