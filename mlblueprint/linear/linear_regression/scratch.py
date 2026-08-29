"""Pure Python implementation of Linear Regression without external libraries."""

import random


class LinearRegression:
    """Linear Regression using plain Python lists and Gradient Descent."""

    def __init__(self, lr=0.01, n_iters=1000):
        """Initialize hyperparameters."""
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the model using Gradient Descent.

        Parameters
        ----------
        X : list of list of float
            Training input data, shape (n_samples, n_features).
        y : list of float
            Target values, shape (n_samples,).

        Returns
        -------
        self : LinearRegression
            The trained model.
        """
        n_samples, n_features = len(X), len(X[0])

        # init weights
        self.weights = [random.random() for _ in range(n_features)]
        self.bias = random.random()

        # training loop
        for epoch in range(self.n_iters):
            # forward pass
            y_hat = []
            for i in range(n_samples):
                out = self.bias
                for j in range(n_features):
                    out += X[i][j] * self.weights[j]
                y_hat.append(out)

            # loss
            mse = sum([(y_cap - y_i) ** 2 for y_cap, y_i in zip(y_hat, y)]) / (
                2 * n_samples
            )

            # print loss
            if epoch % 100 == 0:
                print(f"epoch {epoch} loss: {mse}")

            # backpropagation
            dw = [0] * n_features
            for j in range(n_features):
                total = 0
                for i in range(n_samples):
                    total += X[i][j] * (y_hat[i] - y[i])
                dw[j] = (1 / n_samples) * total
            db = 0
            for i in range(n_samples):
                db += y_hat[i] - y[i]
            db = db / n_samples

            # update
            for j in range(n_features):
                self.weights[j] -= self.lr * dw[j]
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters
        ----------
        X : list of list of float
            Input data, shape (n_samples, n_features).

        Returns
        -------
        list of float
            Predicted values, shape (n_samples,).
        """
        n_samples, n_features = len(X), len(X[0])
        y_hat = []
        for i in range(n_samples):
            out = self.bias
            for j in range(n_features):
                out += X[i][j] * self.weights[j]
            y_hat.append(out)
        return y_hat
