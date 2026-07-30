# Adding an algorithm

This is the shape every algorithm folder follows. Copy it, don't reinvent it, that way anyone
can open any folder in this project and already know where everything is.

## The folder

```
mlblueprint/<family>/<algorithm_name>/
├── README.md              what it is, and how finished it is
├── __init__.py            what gets exported
├── scratch.py             plain Python, written to be read
├── numpy_impl.py          the fast, vectorised version
├── torch_impl.py          optional
├── docs/
│   ├── intuition.md       the idea, in plain words
│   ├── derivation.md      the maths, step by step
│   ├── complexity.md      how fast or slow it is, and why
│   └── references.md      papers, books, links
├── tests/
├── example.py             a script that runs and shows the algorithm working
└── viz.py                 optional, adds it to the visualiser
```

Folder name in `snake_case`, matching the algorithm: `decision_tree`, `kmeans`.

## How finished does it need to be?

You do not have to build the whole thing in one pull request. 

Write the completion level at the top of the algorithm's `README.md`, and add a row to
[ALGORITHMS.md](ALGORITHMS.md). Partial work is genuinely welcome, adding a missing derivation
to something already at `draft` is a great first contribution.

## The steps

1. **Open a "New algorithm" issue** and wait for a thumbs up. Takes about a day, and it stops
   two people accidentally building the same thing.
2. **Branch:** `git checkout -b algo/decision-tree`
3. **Make the folder**, following the layout above.
4. **Write the intuition page first.** If you can't explain it simply, you're not ready to
   code it yet, and that's useful to find out now rather than later.
5. **Write the derivation**, from sources you actually read.
6. **Write `scratch.py`**, test it against a couple of examples you worked out by hand.
7. **Write `numpy_impl.py`**, and check it gives the same answer as `scratch.py`.
8. **Add more tests**: edge cases, plus something that proves the answer is right, see
   [TESTING.md](TESTING.md).
9. **Write `example.py`.**
10. **Write the README** and add the row to `ALGORITHMS.md`.
11. **Export it** from the family's `__init__.py`.
12. `ruff format . && ruff check . --fix && pytest`, then open the pull request.


## README template

```markdown
# Decision Tree

Family: tree
Completion & Scope: Fully Complete
Maintainers: @your-handle

One paragraph: what it does, when you'd use it.

## Files
| Version | File | Done? |
|---|---|---|
| Plain Python | `scratch.py` | yes |
| Fast version | `numpy_impl.py` | yes |
| PyTorch | `torch_impl.py` | no |

## Docs
[Intuition](docs/intuition.md), [Derivation](docs/derivation.md),
[Complexity](docs/complexity.md), [References](docs/references.md)

## Usage
\`\`\`python
from mlblueprint.tree import DecisionTreeClassifier
\`\`\`
```
