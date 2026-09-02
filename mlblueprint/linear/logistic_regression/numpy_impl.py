"""NumPy implementation of Logistic Regression with L2 regularization."""

import numpy as np

from mlblueprint.core.validation import check_X_y, check_array
from mlblueprint.core.random import check_random_state


class LogisticRegression:
    """
    Logistic Regression (L2 Regularized) using Gradient Descent.

    Loss = -(1/N) * sum(y * log(y_hat) + (1-y) * log(1-y_hat)) + lam * sum(w^2)

    Parameters
    ----------
    lr : float
        Learning rate. Default 0.1.
    n_iters : int
        Number of gradient descent iterations. Default 1000.
    lam : float
        L2 regularization strength. Set to 0 for plain Logistic Regression.
    random_state : int or np.random.Generator or None
        Seed for reproducibility.
    """

    def __init__(self, lr=0.1, n_iters=1000, lam=0.0, random_state=None):
        """Initialize hyperparameters."""
        self.lr = lr
        self.n_iters = n_iters
        self.lam = lam
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the model using Gradient Descent.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training input data.
        y : array-like, shape (n_samples,)
            Target values (0 or 1).

        Returns
        -------
        self : LogisticRegression
            The trained model.
        """
        X, y = check_X_y(X, y, y_numeric=False)

        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        # init parameters
        self.weights = rng.normal(scale=0.01, size=n_features)
        self.bias = 0.0

        # training loop
        for epoch in range(self.n_iters):
            # forward pass
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            # log-loss with clipping for numerical stability
            eps = 1e-15
            y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
            loss += self.lam * np.sum(self.weights**2)

            # print loss
            if epoch % 100 == 0:
                print(f"epoch {epoch} loss: {loss:.4f}")

            # gradients
            error = y_hat - y
            dw = (1 / n_samples) * (X.T @ error) + 2 * self.lam * self.weights
            db = (1 / n_samples) * np.sum(error)

            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def _sigmoid(self, z):
        """Sigmoid function with numerical stability."""
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def predict_proba(self, X):
        """
        Predict probabilities for input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to predict on.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted probabilities of class 1.
        """
        X = check_array(X)
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predict class labels for input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to predict on.
        threshold : float, default=0.5
            Decision threshold for class 1.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)