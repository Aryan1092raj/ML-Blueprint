# Neural networks

Automatic differentiation, and the layers and networks built on top of it.

**Import path:** `mlblueprint.neural`

## A couple of notes

Build the autograd engine first, everything else here depends on it. This is meant to be the
project's flagship piece, the thing that explains how PyTorch works underneath. This family is
allowed to use code from `mlblueprint.optim`.

**Testing is different here.** Most of this family has no scikit-learn equivalent to check
against, there's no `sklearn.neural_network.Autograd`. Instead, check your gradients against
finite differences, and if you write a `torch_impl.py`, use it as your reference the way other
families use scikit-learn. See [docs/TESTING.md](../../docs/TESTING.md).

Every algorithm here follows the same layout, see
[docs/ALGORITHM_TEMPLATE.md](../../docs/ALGORITHM_TEMPLATE.md).

This family can use code from `mlblueprint.core` and from outside libraries, but not from a
different family folder. See [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) for why.
