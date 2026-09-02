"""Pure Python implementation of Logistic Regression without external libraries."""

import math
import random


def sigmoid(z):
    """Sigmoid function: squashes any number into (0, 1)."""
    # Clip to avoid overflow in exp
    if z > 20:
        return 1.0
    if z < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def log_loss(y_true, y_pred):
    """Binary cross-entropy loss with clipping for numerical stability."""
    eps = 1e-15
    y_pred = [max(eps, min(1 - eps, p)) for p in y_pred]
    return -sum(
        y * math.log(p) + (1 - y) * math.log(1 - p)
        for y, p in zip(y_true, y_pred)
    ) / len(y_true)


class LogisticRegression:
    """Logistic Regression using plain Python lists and Gradient Descent."""

    def __init__(self, lr=0.1, n_iters=1000, random_state=None):
        self.lr = lr
        self.n_iters = n_iters
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        """
        Train the model using Gradient Descent.

        Parameters
        ----------
        X : list of list of float
            Training input data, shape (n_samples, n_features).
        y : list of float
            Target values (0 or 1), shape (n_samples,).

        Returns
        -------
        self : LogisticRegression
            The trained model.
        """
        if self.random_state is not None:
            rng = random.Random(self.random_state)
        else:
            rng = random

        n_samples, n_features = len(X), len(X[0])

        # Initialize weights and bias
        self.weights = [rng.random() * 0.01 for _ in range(n_features)]
        self.bias = 0.0

        for epoch in range(self.n_iters):
            # Forward pass
            y_hat = []
            for i in range(n_samples):
                z = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                y_hat.append(sigmoid(z))

            # Compute gradients
            dw = [0.0 for _ in range(n_features)]
            db = 0.0

            for i in range(n_samples):
                error = y_hat[i] - y[i]
                for j in range(n_features):
                    dw[j] += X[i][j] * error
                db += error

            # Average gradients
            dw = [g / n_samples for g in dw]
            db = db / n_samples

            # Update parameters
            for j in range(n_features):
                self.weights[j] -= self.lr * dw[j]
            self.bias -= self.lr * db

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = log_loss(y, y_hat)
                print(f"epoch {epoch} loss: {loss:.4f}")

        return self

    def predict_proba(self, X):
        """
        Predict probabilities for input data.

        Parameters
        ----------
        X : list of list of float
            Input data, shape (n_samples, n_features).

        Returns
        -------
        list of float
            Predicted probabilities of class 1.
        """
        if self.weights is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        n_samples, n_features = len(X), len(X[0])
        probs = []

        for i in range(n_samples):
            z = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
            probs.append(sigmoid(z))

        return probs

    def predict(self, X, threshold=0.5):
        """
        Predict class labels for input data.

        Parameters
        ----------
        X : list of list of float
            Input data, shape (n_samples, n_features).
        threshold : float, default=0.5
            Decision threshold for class 1.

        Returns
        -------
        list of int
            Predicted class labels (0 or 1).
        """
        probs = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probs]