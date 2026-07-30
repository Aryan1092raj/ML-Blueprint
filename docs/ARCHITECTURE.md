# Why the folders are laid out this way

Quick read. Worth it before you move anything around.

## The one big idea

**One algorithm, one folder. Everything about it lives there.**

The maths, the code, the tests, the example, and the visualisation for k-means are all inside
`mlblueprint/cluster/kmeans/`. Nothing about k-means lives anywhere else.

This means:

- You never have to hunt for where something goes. It goes in the folder.
- A reviewer can see your whole contribution at once.
- If something is missing, it is obviously missing, a gap in one folder, not a mystery
  spread across the project.

## The family folders

```
mlblueprint/
├── core/            shared code, algorithms can implement their own based on needs
├── linear/
├── tree/
├── cluster/
├── decomposition/
├── svm/
├── probabilistic/
├── optim/
└── neural/
```

**Rule:** a family folder can use code from `core`, and from outside libraries like NumPy.
It should not import code from a different family folder (like `tree` using something from
`cluster`). If that ever seems necessary, ask in an issue first.

This keeps each family self contained, so it stays easy to understand, and easy to eventually
split off into its own project if it ever gets big enough.

## What goes in `core`

Only things that more than one algorithm needs:

| File | What it holds |
|---|---|
| `base.py` | The shared shape every algorithm follows (`fit`, `predict`, and so on) |
| `validation.py` | Checking that input data makes sense, with clear error messages |
| `random.py` | Making random results repeatable |
| `metrics.py` | Accuracy, error scores, and the like |
| `datasets.py` | Small generated datasets for examples and tests |
| `viz.py` | The visualiser's plugin system |

If you're not sure whether something belongs in `core`, it probably does not. Put it in your
algorithm's folder instead, and ask a maintainer if you think it should move.

