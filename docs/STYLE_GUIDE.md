# How to write code and docs here

Everything in one place: how to write the code, how to write the explanations, and the maths
notation we all share. Read once, keep it open while you work.

## Code

**Formatting and checking is automatic.** Run this before you push:

```bash
ruff format .
ruff check . --fix
```

Do not hand format code or argue about spacing. The tool decides.

**Match the maths.** If your derivation calls something `w`, call it `w` in the code too, not
`weight_vector`. If a name needs explaining, say so once in the docstring:

```python
def fit(self, X, y):
    """Fit the model.

    X is the data (n_samples, n_features), y is the targets,
    w is the weight vector, b is the intercept.
    """
```

Short names like `X`, `y`, `w`, `b`, `K` are fine here. `tmp`, `data2`, `res` are not, they
don't tell you anything.

**Follow the same shape every time:**

```python
class KMeans:
    def __init__(self, n_clusters=8, random_state=None):
        # __init__ just stores settings, nothing else happens here
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        # all the real work happens here
        self.cluster_centers_ = ...  # things learned during fit end with _
        return self  # fit always returns self

    def predict(self, X): ...
```

**Type hints on public functions.** Helps everyone, including you in six months.

```python
def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
```

**Docstrings on everything public**, in this shape:

```python
def fit(self, X, y):
    """Fit the model to the data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    self : LinearRegression
        The fitted model.
    """
```

**Comments explain why, not what.** The code already shows what it does.

```python
# not useful
i += 1  # increment i

# useful
# Numbers this large would overflow when we exponentiate them, so we
# shift everything down first. Doesn't change the final answer.
z = z - z.max()
```

**Keep it readable over clever.** If a line needs a second read to understand, split it up.
We are not trying to be as fast as scikit-learn, we are trying to be understandable. A trick
that saves time but costs clarity is not a win here.

**Error messages should say what to do:**

```python
# not useful
raise ValueError("bad input")

# useful
raise ValueError(f"X has {len(X)} rows but y has {len(y)}. They need to match.")
```

**Same input, same seed, same output, every time.** Never use plain `random` or the global
NumPy random state. Use `check_random_state` from `mlblueprint.core.random` instead.

**No new dependencies without asking first.** NumPy is required. Everything else (SciPy,
scikit-learn, matplotlib, Streamlit, PyTorch) is optional and imported only inside the
function that needs it, never at the top of the file. That way the library keeps working for
someone who only has NumPy installed.

## Writing the docs pages

Each algorithm has four small pages. They are separate on purpose, different readers want
different things.

**`intuition.md`, almost no equations.** What problem does this solve? What is the basic idea?
When does it not work well? Write it like you're explaining it to a friend who hasn't taken
the course.

**`derivation.md`, the maths, step by step.** Never skip a step with "it can be shown that".
If a step is genuinely standard (like a matrix identity), say exactly where it comes from.
Write it in your own words, from a source you actually read, don't copy from a textbook.

**`complexity.md`, how slow or fast it is, and why.** The answer (`O(n log n)`) matters less
than the reasoning that gets you there. Show your working.

**`references.md`, where you learned it from.** The original paper if there is one, a
textbook chapter, and a couple of good extra links. Every single one has to be something you
actually opened and read. Made up citations are worse than no citations.

## Maths notation

So thirty pages written by different people still read like one project.

| Symbol | Means |
|---|---|
| `n` | number of samples |
| `d` | number of features |
| `X` | the data, shape (n, d) |
| `y` | the targets |
| `w` | weights |
| `b` | bias / intercept |
| `L` | the loss (what we're trying to shrink) |

Write maths with `$...$` for inline and `$$...$$` for a bigger equation:

```markdown
The gradient is $\nabla_w L = -2 X^\top (y - Xw)$.
```

If an algorithm has its own very standard notation (like `alpha` for SVM), it's fine to use
it, just say so at the top of the page.
