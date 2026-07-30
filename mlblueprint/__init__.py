"""ML Blueprint, machine learning algorithms written so you can understand them.

Every algorithm lives in its own folder together with its maths, its
intuition, its tests, and a runnable example. See ``docs/ARCHITECTURE.md``.

The library is organised by algorithm family:

    mlblueprint.core             shared foundation
    mlblueprint.linear           linear and logistic models
    mlblueprint.tree             decision trees and ensembles
    mlblueprint.cluster          clustering
    mlblueprint.decomposition    dimensionality reduction
    mlblueprint.svm              support vector machines
    mlblueprint.probabilistic    probabilistic models
    mlblueprint.optim            optimisers
    mlblueprint.neural           neural networks and autograd

This is a teaching library, not built for speed.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
