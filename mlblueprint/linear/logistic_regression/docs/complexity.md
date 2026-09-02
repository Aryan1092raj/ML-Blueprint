# Logistic Regression — Complexity Analysis

## Overview

This document analyzes the time and space complexity of the Logistic Regression
implementation using Gradient Descent.

## Notation

- **n** = number of training samples
- **d** = number of features
- **T** = number of training iterations (epochs)
- **m** = number of new samples passed to `predict`

## Time Complexity

### Training (fit)

| Operation | Complexity | Explanation |
|-----------|-----------|-------------|
| Forward pass (z = X @ w + b) | O(n × d) | Matrix-vector multiply |
| Sigmoid | O(n) | Element-wise on predictions |
| Compute gradients | O(n × d) | Matrix multiply: X.T @ (y_hat - y) |
| Update weights | O(d) | Subtract gradient from weights |
| **Per iteration** | **O(n × d)** | Dominated by matrix operations |
| **Total training** | **O(T × n × d)** | T iterations of the above |

### Prediction (predict)

| Operation | Complexity | Explanation |
|-----------|-----------|-------------|
| Single prediction | O(d) | Dot product + sigmoid |
| Batch prediction | O(m × d) | m new samples, each requires O(d) |

## Space Complexity

| Component | Space | Explanation |
|-----------|-------|-------------|
| Weights | O(d) | One weight per feature |
| Bias | O(1) | Single scalar |
| Input data | O(n × d) | Training matrix |
| Predictions (y_hat) | O(n) | Probabilities during training |
| Gradients | O(d) | One gradient per weight |
| **Total** | **O(n × d)** | Dominated by input storage |

## Summary

| Aspect | Complexity |
|--------|-----------|
| Training | O(T × n × d) |
| Prediction | O(m × d) |
| Space | O(n × d) |

## Notes

The complexity is identical to Linear Regression with Gradient Descent. The sigmoid
and log-loss add only O(n) operations per iteration, which is absorbed by the
O(n × d) matrix multiply term. Memory usage is the same since we store the same
intermediate arrays (predictions, gradients).
