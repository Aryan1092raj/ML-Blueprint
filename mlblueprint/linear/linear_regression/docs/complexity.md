# Linear Regression — Complexity Analysis

## Overview

This document analyzes the time and space complexity of the Linear Regression 
implementation using Gradient Descent.

## Notation

- **N** = number of training samples
- **n** = number of features
- **T** = number of training iterations (epochs)

## Time Complexity

### Training (fit)

| Operation | Complexity | Explanation |
|-----------|-----------|-------------|
| Forward pass | O(N × n) | Compute predictions: X @ w + b |
| Compute gradients | O(N × n) | Matrix multiply: X.T @ errors |
| Update weights | O(n) | Subtract gradient from weights |
| **Per iteration** | **O(N × n)** | Dominated by matrix operations |
| **Total training** | **O(T × N × n)** | T iterations of the above |

### Prediction (predict)

| Operation | Complexity | Explanation |
|-----------|-----------|-------------|
| Single prediction | O(n) | Dot product of one sample with weights |
| Batch prediction | O(M × n) | M new samples, each requires O(n) |

## Space Complexity

| Component | Space | Explanation |
|-----------|-------|-------------|
| Weights | O(n) | One weight per feature |
| Bias | O(1) | Single scalar |
| Input data | O(N × n) | Training matrix |
| Predictions | O(N) | Output array during training |
| Gradients | O(n) | One gradient per weight |
| **Total** | **O(N × n)** | Dominated by input storage |

## Summary

| Aspect | Complexity |
|--------|-----------|
| Training | O(T × N × n) |
| Prediction | O(M × n) |
| Space | O(N × n) |

The implementation scales linearly with the number of samples, features, 
and iterations, making it suitable for medium to large datasets.