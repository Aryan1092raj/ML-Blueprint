"""Linear models.

Models that predict using a straight-line combination of the features. The foundation
most other algorithms build on or compare against.
"""

from .numpy_impl import LinearRegression
from .scratch import LinearRegression as LinearRegressionScratch

__all__ = [
    "LinearRegression",
    "LinearRegressionScratch",
]
