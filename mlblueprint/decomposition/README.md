# Dimensionality reduction

Algorithms that take data with many features and find a simpler way to represent it, useful for
visualising and for speeding other algorithms up.

**Import path:** `mlblueprint.decomposition`

## A couple of notes

These lean more on linear algebra than most of the project. Take the derivation slowly, don't
skip steps.

Every algorithm here follows the same layout, see
[docs/ALGORITHM_TEMPLATE.md](../../docs/ALGORITHM_TEMPLATE.md).

This family can use code from `mlblueprint.core` and from outside libraries, but not from a
different family folder. See [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) for why.
