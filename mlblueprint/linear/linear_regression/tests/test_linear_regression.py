"""Tests for Linear Regression implementation."""
import sys
import os
import numpy as np

# parent dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_impl import LinearRegression
from scratch import LinearRegression as LinearRegressionScratch

class TestLinearRegression:
    """Test cases for Linear Regression."""

    def test_fit_simple(self):
        """Test that model can learn y = 3x + 2."""
        np.random.seed(42)

        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([5, 8, 11, 14, 17])

        model = LinearRegression(lr=0.1, n_iters=2000, lam=0)
        model.fit(X, y)

        # wt check
        assert abs(model.weights[0] - 3.0) < 0.5
        # bias check
        assert abs(model.bias - 2.0) < 0.5

    def test_predict_shape(self):
        """Test that predictions have correct shape."""
        np.random.seed(42)

        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])

        model = LinearRegression(lr=0.01, n_iters=100, lam=0)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert len(y_pred)==len(X)

    def test_r2_score_perfect(self):
        """Test R² score on perfectly predicted data."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        SSR = np.sum((y_true-y_pred)**2)
        SST = np.sum((y_true-np.mean(y_true))**2)
        r2 = 1-(SSR / SST)

        assert abs(r2 - 1.0) < 0.001

    def test_scratch_vs_numpy(self):
        """Test that scratch and numpy implementations give similar results."""
        import random
        random.seed(42)
        np.random.seed(42)

        X = [[1], [2], [3], [4], [5]]
        y = [5, 8, 11, 14, 17]

        # train both models
        model_scratch = LinearRegressionScratch(lr=0.1, n_iters=2000)
        model_scratch.fit(X, y)

        model_numpy = LinearRegression(lr=0.1, n_iters=2000, lam=0)
        model_numpy.fit(X, y)

        # predictions should be similar
        pred_scratch = model_scratch.predict(X)
        pred_numpy = model_numpy.predict(X)

        for ps, pn in zip(pred_scratch, pred_numpy):
            assert abs(ps - pn) < 1.5  