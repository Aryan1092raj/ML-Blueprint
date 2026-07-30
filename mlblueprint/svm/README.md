# Support vector machines

Classifiers that try to draw the widest possible boundary between classes, and the kernel trick
that lets them handle curved boundaries too.

**Import path:** `mlblueprint.svm`

## A couple of notes

The maths here (the dual problem) trips people up more than anywhere else in the project. A
clear derivation here would help a lot of people.

Every algorithm here follows the same layout, see
[docs/ALGORITHM_TEMPLATE.md](../../docs/ALGORITHM_TEMPLATE.md).

This family can use code from `mlblueprint.core` and from outside libraries, but not from a
different family folder. See [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) for why.
