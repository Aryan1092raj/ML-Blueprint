# Writing tests

Tests are the proof that your code actually matches your maths. Run them with:

```bash
pytest
```

Each algorithm has its own `tests/` folder inside it. Only tests for the shared `core` code, or
tests that check the whole project's rules, go in the top level `tests/`.

## The kinds of test to write

**1. Hand worked examples.** Pick an input where you can work out the right answer yourself,
and explain why in the test.

```python
def test_fits_a_perfect_line():
    """y = 2x + 1 exactly, so the model should recover slope 2, intercept 1."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([3.0, 5.0, 7.0])

    model = LinearRegression().fit(X, y)

    assert model.coef_ == pytest.approx([2.0])
    assert model.intercept_ == pytest.approx(1.0)
```

**2. Checking against a known-good implementation, if one exists.** For anything scikit-learn
also has (linear regression, k-means, decision trees, SVMs, and most of the classic stuff),
this is our strongest proof of correctness. We don't reimplement scikit-learn, we use it to
check our own answers.

```python
sklearn = pytest.importorskip("sklearn")


def test_matches_sklearn():
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0])

    ours = LinearRegression().fit(X, y)
    theirs = SkLinearRegression().fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-6)
```

**Not every algorithm has a scikit-learn version.** Most of what goes in `mlblueprint/neural/`
(autograd, CNN layers, RNNs, attention) doesn't exist in scikit-learn at all. That's fine, it
just means you lean on the other three kinds of test instead, and especially on these two:

- **Check your gradients numerically.** If you wrote a backward pass by hand, compare it
  against a finite-difference estimate of the same gradient. This catches almost every autograd
  bug there is, and it doesn't need any other library to compare against.

  ```python
  def test_gradient_matches_finite_differences():
      """A hand-derived backward pass should agree with a numerical approximation."""
      x = np.array([1.0, 2.0, -0.5])
      eps = 1e-6

      analytic = my_layer.backward(x)
      numeric = np.array(
          [
              (my_layer.forward(x + eps * e_i) - my_layer.forward(x - eps * e_i))
              / (2 * eps)
              for e_i in np.eye(len(x))
          ]
      )

      np.testing.assert_allclose(analytic, numeric, atol=1e-4)
  ```

- **Check against PyTorch instead, if you wrote a `torch_impl.py`.** Same idea as the
  scikit-learn check, just a different reference. Build the same tiny network in both, copy the
  weights across, and check the forward and backward passes agree.

If neither of those fits, hand-worked examples (small enough to compute on paper) and the edge
case tests below still apply, they're just more work to construct for something like a CNN.

**3. Edge cases.** The inputs that actually break things:

- one sample, or one feature
- empty input
- `X` and `y` with mismatched lengths
- calling `predict` before `fit`
- very large or very small numbers

**4. Checking `scratch.py` and `numpy_impl.py` agree**, if you wrote both:

```python
def test_scratch_and_numpy_agree():
    a = ScratchLinearRegression().fit(X.tolist(), y.tolist())
    b = LinearRegression().fit(X, y)
    np.testing.assert_allclose(a.coef_, b.coef_, rtol=1e-8)
```

## A few rules

- **Always use a fixed seed** (like `np.random.default_rng(0)`), so tests give the same result
  every time they run.
- **Keep tests fast.** The whole suite should run in a couple of minutes.
- **One thing per test**, with a name that says what it checks:
  `test_handles_empty_clusters`, not `test_kmeans_2`.
- **No internet access in tests.** Use `mlblueprint.core.datasets` to generate data.

That's really it. If you're not sure how to test something, open a draft pull request and ask.
