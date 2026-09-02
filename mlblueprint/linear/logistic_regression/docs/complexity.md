# Logistic Regression — Complexity Analysis

## Overview

This document analyzes the time and space complexity of the Logistic Regression
implementation using Gradient Descent.

## Notation

- **N** = number of training samples
- **n** = number of features
- **T** = number of training iterations (epochs)

## Time Complexity

### Training (fit)

| Operation | Complexity | Explanation |
|-----------|-----------|-------------|
| Forward pass (z = X @ w + b) | O(N × n) | Matrix-vector multiply |
| Sigmoid | O(N) | Element-wise on predictions |
| Compute gradients | O(N × n) | Matrix multiply: X.T @ (y_hat - y) |
| Update weights | O(n) | Subtract gradient from weights |
| **Per iteration** | **O(N × n)** | Dominated by matrix operations |
| **Total training** | **O(T × N × n)** | T iterations of the above |

### Prediction (predict)

| Operation | Complexity | Explanation |
|-----------|-----------|-------------|
| Single prediction | O(n) | Dot product + sigmoid |
| Batch prediction | O(M × n) | M new samples, each requires O(n) |

## Space Complexity

| Component | Space | Explanation |
|-----------|-------|-------------|
| Weights | O(n) | One weight per feature |
| Bias | O(1) | Single scalar |
| Input data | O(N × n) | Training matrix |
| Predictions (y_hat) | O(N) | Probabilities during training |
| Gradients | O(n) | One gradient per weight |
| **Total** | **O(N × n)** | Dominated by input storage |

## Summary

| Aspect | Complexity |
|--------|-----------|
| Training | O(T × N × n) |
| Prediction | O(M × n) |
| Space | O(N × n) |

## Notes

The complexity is identical to Linear Regression with Gradient Descent. The sigmoid
and log-loss add only O(N) operations per iteration, which is absorbed by the
O(N × n) matrix multiply term. Memory usage is the same since we store the same
intermediate arrays (predictions, gradients).