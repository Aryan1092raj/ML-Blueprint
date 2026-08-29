"""Tests for Linear Regression implementation."""
import sys
import os
import random
import numpy as np
from sklearn.linear_model import LinearRegression as sklearnLR

# Add parent directory to path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_impl import LinearRegression
from scratch import LinearRegression as LinearRegressionScratch


class TestLinearRegression:
    """Test cases for Linear Regression."""

    def test_fit_simple(self):
        """Test that model can learn y = 3x+2."""
        np.random.seed(42)

        X = np.array([[1],[2],[3],[4],[5]])
        y = np.array([5,8,11,14,17])

        model = LinearRegression(lr=0.1, n_iters=2000, lam=0)
        model.fit(X, y)

        # weight should be close to 3 with threshold 0.5
        assert abs(model.weights[0] - 3.0) < 0.5
        # bias should be close to 2 with threshold 0.5
        assert abs(model.bias - 2.0) < 0.5


    def test_scratch_vs_numpy(self):
        """Test that scratch and numpy implementations give similar results."""
        random.seed(42)
        np.random.seed(42)

        X = [[1],[2],[3],[4],[5]]
        y = [5,8,11,14,17]

        # train both models
        model_scratch = LinearRegressionScratch(lr=0.1, n_iters=2000)
        model_scratch.fit(X, y)

        model_numpy = LinearRegression(lr=0.1, n_iters=2000, lam=0)
        model_numpy.fit(X, y)

        # predictions should be similar
        pred_scratch = model_scratch.predict(X)
        pred_numpy = model_numpy.predict(X)

        for ps, pn in zip(pred_scratch, pred_numpy):
            assert abs(ps - pn) < 1.5  # some tolerance

    def test_against_sklearn(self):
        """Test that our implementation matches scikit-learn's Linear Regression."""
        np.random.seed(42)
        random.seed(42)

        # Generate data
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
        y = np.array([2.1, 3.9, 6.2, 8.1, 9.8, 12.1, 14.0, 15.9])

        # train our model 
        our_model = LinearRegression(lr=0.01, n_iters=5000, lam=0)
        our_model.fit(X, y)

        # train scikit-learn model
        sklearn_model = sklearnLR()
        sklearn_model.fit(X, y)

        # compare weights (slope)
        assert abs(our_model.weights[0] - sklearn_model.coef_[0]) < 0.1, "Weights don't match sklearn!"
        
        # compare bias (intercept)
        assert abs(our_model.bias - sklearn_model.intercept_) < 0.1, "Bias doesn't match sklearn!"

        # Compare predictions
        our_preds = our_model.predict(X)
        sklearn_preds = sklearn_model.predict(X)

        for ours, theirs in zip(our_preds, sklearn_preds):
            assert abs(ours - theirs) < 0.1, "Predictions don't match sklearn!"