"""Base classes defining the estimator API every algorithm follows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["Estimator", "Predictor", "Transformer", "NotFittedError"]


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before ``fit`` has been called."""


class Estimator(ABC):
    """Base class for everything that learns from data.

    Subclasses must implement :meth:`fit`. They must also store their
    hyperparameters unchanged in ``__init__``, no validation and no computation
    there, so that the object can always be inspected and copied.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any = None) -> Estimator:
        """Fit the estimator to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Targets. Unused by unsupervised estimators, but accepted so that
            every estimator has the same signature.

        Returns
        -------
        self : Estimator
            The fitted estimator.
        """

    def get_params(self) -> dict[str, Any]:
        """Return the estimator's hyperparameters.

        Reads the arguments of ``__init__`` and returns their current values.
        This is why ``__init__`` must store its arguments unchanged.

        Returns
        -------
        params : dict
            Mapping of parameter name to value.
        """
        import inspect

        signature = inspect.signature(type(self).__init__)
        names = [p for p in signature.parameters if p != "self"]
        return {name: getattr(self, name) for name in names}

    def _check_is_fitted(self) -> None:
        """Raise :class:`NotFittedError` if the estimator has not been fitted.

        An estimator counts as fitted once it has at least one attribute ending
        in a single trailing underscore.

        Raises
        ------
        NotFittedError
            If no fitted attributes are present.
        """
        fitted = [
            attr
            for attr in vars(self)
            if attr.endswith("_")
            and not attr.startswith("_")
            and not attr.endswith("__")
        ]
        if not fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def __repr__(self) -> str:
        """Return a readable representation showing the hyperparameters."""
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"


class Predictor(Estimator):
    """Base class for estimators that predict a target, the supervised ones."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict targets for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """

    def fit_predict(self, X: Any, y: Any = None) -> Any:
        """Fit the estimator and return predictions on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Targets.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predictions on ``X``.
        """
        return self.fit(X, y).predict(X)


class Transformer(Estimator):
    """Base class for estimators that map data into a new representation."""

    @abstractmethod
    def transform(self, X: Any) -> Any:
        """Transform ``X`` into the learned representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        """Fit the transformer and transform the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Targets, if the transformer is supervised.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X, y).transform(X)
