"""Small synthetic datasets for examples, tests and visualisations.

Everything here is generated, never downloaded.
"""

from __future__ import annotations

import numpy as np

from .random import check_random_state

__all__ = ["make_blobs", "make_regression", "make_moons"]


def make_blobs(
    n_samples: int = 300,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate isotropic Gaussian blobs, for clustering.

    Parameters
    ----------
    n_samples : int, default=300
        Total number of points, split as evenly as possible between centres.
    n_features : int, default=2
        Dimensionality. Two is the useful default because it can be plotted.
    centers : int, default=3
        Number of blobs.
    cluster_std : float, default=1.0
        Standard deviation of each blob. Raise it to make the clusters overlap.
    random_state : int, Generator or None, default=None
        Seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Generated points.
    y : ndarray of shape (n_samples,)
        Index of the blob each point came from, the ground truth a clustering
        algorithm is trying to recover without seeing it.
    """
    rng = check_random_state(random_state)

    centroids = rng.uniform(-10.0, 10.0, size=(centers, n_features))
    counts = np.full(centers, n_samples // centers)
    counts[: n_samples % centers] += 1

    X = np.vstack(
        [
            rng.normal(loc=centroid, scale=cluster_std, size=(count, n_features))
            for centroid, count in zip(centroids, counts, strict=True)
        ]
    )
    y = np.repeat(np.arange(centers), counts)

    order = rng.permutation(n_samples)
    return X[order], y[order]


def make_regression(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 10.0,
    bias: float = 0.0,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data from a linear model plus Gaussian noise.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.
    n_features : int, default=1
        Number of features.
    noise : float, default=10.0
        Standard deviation of the noise added to the targets. Set to 0.0 for an
        exact linear relationship, which is useful in tests.
    bias : float, default=0.0
        Intercept of the underlying model.
    random_state : int, Generator or None, default=None
        Seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Generated features.
    y : ndarray of shape (n_samples,)
        Targets.
    """
    rng = check_random_state(random_state)

    X = rng.normal(size=(n_samples, n_features))
    true_coef = rng.uniform(-100.0, 100.0, size=n_features)
    y = X @ true_coef + bias + rng.normal(scale=noise, size=n_samples)

    return X, y


def make_moons(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving half-circles.

    Parameters
    ----------
    n_samples : int, default=200
        Total number of points, split between the two moons.
    noise : float, default=0.1
        Standard deviation of Gaussian noise added to the coordinates.
    random_state : int, Generator or None, default=None
        Seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Generated points.
    y : ndarray of shape (n_samples,)
        Class labels, 0 or 1.

    Notes
    -----
    The classic non-linearly-separable dataset. No straight line separates the
    two classes, which makes it the standard demonstration of why kernels,
    trees and neural networks exist.
    """
    rng = check_random_state(random_state)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    outer_angle = np.linspace(0, np.pi, n_outer)
    inner_angle = np.linspace(0, np.pi, n_inner)

    outer = np.column_stack([np.cos(outer_angle), np.sin(outer_angle)])
    inner = np.column_stack([1 - np.cos(inner_angle), 0.5 - np.sin(inner_angle)])

    X = np.vstack([outer, inner]) + rng.normal(scale=noise, size=(n_samples, 2))
    y = np.hstack([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)])

    order = rng.permutation(n_samples)
    return X[order], y[order]
