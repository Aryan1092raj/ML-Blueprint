# Optimisers

The algorithms that actually do the learning, by nudging a model's parameters to reduce its
error.

**Import path:** `mlblueprint.optim`

## A couple of notes

Every optimiser here needs a test that checks its gradient by hand, see docs/TESTING.md. This family also makes some of the best visualiser panels.

Every algorithm here follows the same layout, see
[docs/ALGORITHM_TEMPLATE.md](../../docs/ALGORITHM_TEMPLATE.md).

This family can use code from `mlblueprint.core` and from outside libraries, but not from a
different family folder. See [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) for why.
