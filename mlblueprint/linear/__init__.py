"""
Linear models family.

This family contains algorithms that learn linear relationships between
features and targets, including both regression and classification models.

Algorithms
----------
- LinearRegression: Linear regression with gradient descent
"""

from .linear_regression import LinearRegression, LinearRegressionScratch

__all__ = [
    "LinearRegression",
    "LinearRegressionScratch",
]
