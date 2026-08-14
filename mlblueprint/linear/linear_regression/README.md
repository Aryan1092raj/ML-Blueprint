# Linear Regression

Family: linear  
Completion & Scope: Fully Complete  
Maintainers: @Frozen-afk

Linear Regression is a fundamental supervised learning algorithm that models 
the relationship between a dependent variable and one or more independent 
variables by fitting a linear equation to observed data. It is used for 
predicting continuous values such as house prices, sales figures, or temperatures.

This implementation uses Gradient Descent for optimization and includes 
optional L2 regularization (Ridge Regression) to prevent overfitting.

## Files

| Version | File | Done? |
|---------|------|-------|
| Plain Python | `scratch.py` | ✅ yes |
| Fast version | `numpy_impl.py` | ✅ yes |
| PyTorch | `torch_impl.py` | ❌ no |

## Documentation

- [Intuition](docs/intuition.md) — The idea in plain words
- [Derivation](docs/derivation.md) — The maths, step by step
- [Complexity](docs/complexity.md) — Time and space analysis
- [References](docs/references.md) — Papers, books, links

## Usage

```python
from mlblueprint.linear.linear_regression import LinearRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Create and train the model
model = LinearRegression(lr=0.01, n_iters=1000, lam=0)
model.fit(X, y)

# Make a prediction
prediction = model.predict(np.array([[6]]))
print(prediction)