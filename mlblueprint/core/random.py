"""Random state handling, so that results are reproducible."""

from __future__ import annotations

import numpy as np

__all__ = ["check_random_state", "DEFAULT_SEED"]

#: Seed used by examples and documentation so their output is stable.
DEFAULT_SEED = 42


def check_random_state(
    seed: int | np.random.Generator | None = None,
) -> np.random.Generator:
    """Turn a seed into a NumPy random generator.

    Parameters
    ----------
    seed : int, Generator or None, default=None
        If ``None``, a fresh generator seeded from the operating system, results
        will differ between runs. If an ``int``, a generator seeded with it,
        which is reproducible. If already a ``Generator``, it is returned
        unchanged, so that a caller can thread one generator through several
        estimators.

    Returns
    -------
    rng : numpy.random.Generator
        The generator to draw from.

    Raises
    ------
    TypeError
        If ``seed`` is not one of the accepted types.

    Examples
    --------
    ::

        rng = check_random_state(self.random_state)
        idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, (int, np.integer)):
        return np.random.default_rng(int(seed))

    raise TypeError(
        f"random_state must be None, an int, or a numpy.random.Generator, "
        f"but got {type(seed).__name__}."
    )
