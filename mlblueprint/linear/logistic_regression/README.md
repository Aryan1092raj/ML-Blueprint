# Logistic Regression

Family: linear
Completion & Scope: draft
Maintainers: @aryan-raj

Binary classification using a linear model passed through the sigmoid function, trained by gradient descent on log-loss. Teaches numerical stability of the loss, gradient computation, and how a simple linear model becomes a probability estimator.

## Files

| Version | File | Done? |
|---|---|---|
| Plain Python | `scratch.py` | no |
| Fast version | `numpy_impl.py` | no |
| PyTorch | `torch_impl.py` | no |

## Docs

[Intuition](docs/intuition.md), [Derivation](docs/derivation.md), [Complexity](docs/complexity.md), [References](docs/references.md)

## Usage

```python
from mlblueprint.linear import LogisticRegression
```