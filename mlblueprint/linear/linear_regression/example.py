import numpy as np
from numpy_impl import LinearRegression


def generate_data(n_samples):
    """
    Generate synthetic data: y = 3*x1 + 2*x2 + 5 + noise

    Parameters
    ----------
    n_samples : int
        Number of data points to generate.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
        Input features.
    y : np.ndarray, shape (n_samples,)
        Target values.
    """
    np.random.seed(42)

    X = np.random.rand(n_samples, 2) * 10
    noise = np.random.randn(n_samples) * 2
    y = 3 * X[:, 0] + 2 * X[:, 1] + 5 + noise

    return X, y


def r2_score(y_true, y_pred):
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    # data generation
    X, y = generate_data(100)

    # cross validation
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # model
    model = LinearRegression(lr=0.1, n_iters=1000, lam=0)
    model.fit(X_train, y_train)

    # results
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"r2 score : {r2}")
