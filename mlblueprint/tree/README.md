# Trees and ensembles

Models that split the data into smaller and smaller groups, plus the methods that combine many
of them together.

**Import path:** `mlblueprint.tree`

## A couple of notes

Ensembles are allowed to reuse code from `decision_tree` since they're built on top of it.
Nothing here should reuse code from a different family folder.

Every algorithm here follows the same layout, see
[docs/ALGORITHM_TEMPLATE.md](../../docs/ALGORITHM_TEMPLATE.md).

This family can use code from `mlblueprint.core` and from outside libraries, but not from a
different family folder. See [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) for why.
