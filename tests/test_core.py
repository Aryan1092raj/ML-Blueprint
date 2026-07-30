"""Tests for the shared foundation in ``mlblueprint.core``.

Algorithm tests live inside the algorithm's own folder.
Only cross-cutting tests belong here. See ``docs/TESTING.md``.
"""

import numpy as np
import pytest

from mlblueprint.core import (
    Estimator,
    NotFittedError,
    Predictor,
    Transformer,
    accuracy_score,
    check_array,
    check_random_state,
    check_X_y,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from mlblueprint.core.datasets import make_blobs, make_moons, make_regression
from mlblueprint.core.viz import Frame, Parameter, VizPanel, get_panels, register_panel


class DummyPredictor(Predictor):
    """Minimal estimator used to exercise the base class behaviour."""

    def __init__(self, offset=0.0):
        self.offset = offset

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = float(np.mean(X))
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = check_array(X)
        return np.full(X.shape[0], self.mean_ + self.offset)


class DummyTransformer(Transformer):
    """Minimal transformer used to exercise the base class behaviour."""

    def __init__(self, shift=0.0):
        self.shift = shift

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = check_array(X)
        return X - self.mean_ + self.shift


# Base


def test_estimator_cannot_be_instantiated_directly():
    """Estimator is abstract, fit has no sensible default."""
    with pytest.raises(TypeError):
        Estimator()  # type: ignore[abstract]


def test_fit_returns_self():
    model = DummyPredictor()
    assert model.fit(np.ones((4, 2))) is model


def test_get_params_reads_init_arguments():
    """get_params relies on __init__ storing its arguments unchanged."""
    assert DummyPredictor(offset=2.0).get_params() == {"offset": 2.0}


def test_repr_shows_hyperparameters():
    assert repr(DummyPredictor(offset=1.5)) == "DummyPredictor(offset=1.5)"


def test_predict_before_fit_raises_a_clear_error():
    """The message must tell the user what to do, not just that something failed."""
    with pytest.raises(NotFittedError, match="not fitted yet"):
        DummyPredictor().predict(np.ones((3, 2)))


def test_fit_predict_is_fit_then_predict():
    X = np.array([[1.0, 3.0], [2.0, 2.0]])
    np.testing.assert_allclose(
        DummyPredictor().fit_predict(X), DummyPredictor().fit(X).predict(X)
    )


def test_transform_before_fit_raises_a_clear_error():
    with pytest.raises(NotFittedError, match="not fitted yet"):
        DummyTransformer().transform(np.ones((3, 2)))


def test_fit_transform_is_fit_then_transform():
    X = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    np.testing.assert_allclose(
        DummyTransformer().fit_transform(X), DummyTransformer().fit(X).transform(X)
    )


def test_transform_centres_the_data():
    """shift=0 means the transformed columns should average to zero."""
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    centred = DummyTransformer().fit_transform(X)
    np.testing.assert_allclose(centred.mean(axis=0), [0.0, 0.0], atol=1e-10)


# Validation


def test_check_array_rejects_1d_input_with_a_helpful_message():
    """A 1-D array is rejected rather than reshaped: guessing hides bugs."""
    with pytest.raises(ValueError, match="reshape"):
        check_array(np.array([1.0, 2.0, 3.0]))


def test_check_array_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        check_array(np.empty((0, 3)))


def test_check_array_rejects_non_finite_values():
    with pytest.raises(ValueError, match="NaN"):
        check_array(np.array([[1.0, np.nan]]))


def test_check_array_converts_lists():
    result = check_array([[1, 2], [3, 4]])
    assert result.dtype == np.float64
    assert result.shape == (2, 2)


def test_check_X_y_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="inconsistent"):
        check_X_y(np.ones((5, 2)), np.ones(3))


def test_check_X_y_rejects_2d_targets():
    with pytest.raises(ValueError, match="ravel"):
        check_X_y(np.ones((5, 2)), np.ones((5, 1)))


def test_check_X_y_allows_non_numeric_labels():
    _, y = check_X_y(np.ones((3, 2)), ["cat", "dog", "cat"], y_numeric=False)
    assert list(y) == ["cat", "dog", "cat"]


# Random


def test_same_seed_gives_identical_draws():
    """Reproducibility is a hard requirement across the library."""
    a = check_random_state(42).normal(size=10)
    b = check_random_state(42).normal(size=10)
    np.testing.assert_array_equal(a, b)


def test_generator_is_passed_through_unchanged():
    rng = np.random.default_rng(0)
    assert check_random_state(rng) is rng


def test_invalid_seed_type_is_rejected():
    with pytest.raises(TypeError, match="random_state"):
        check_random_state("42")


# Metrics


def test_accuracy_score_counts_correct_predictions():
    assert accuracy_score([1, 0, 1, 1], [1, 0, 0, 1]) == pytest.approx(0.75)


def test_mean_squared_error_is_zero_for_a_perfect_prediction():
    assert mean_squared_error([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_mean_squared_error_squares_the_residuals():
    """Residuals of 1 and 3 give (1 + 9) / 2 = 5."""
    assert mean_squared_error([0.0, 0.0], [1.0, 3.0]) == pytest.approx(5.0)


def test_mean_absolute_error_does_not_square():
    """The same residuals give (1 + 3) / 2 = 2."""
    assert mean_absolute_error([0.0, 0.0], [1.0, 3.0]) == pytest.approx(2.0)


def test_r2_is_one_for_a_perfect_fit():
    assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_r2_is_zero_when_predicting_the_mean():
    """Predicting the mean everywhere is the baseline R² is measured against."""
    y = [1.0, 2.0, 3.0]
    assert r2_score(y, [2.0, 2.0, 2.0]) == pytest.approx(0.0)


def test_r2_handles_constant_targets_without_dividing_by_zero():
    assert r2_score([5.0, 5.0], [5.0, 5.0]) == 1.0
    assert r2_score([5.0, 5.0], [4.0, 6.0]) == 0.0


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="inconsistent"):
        accuracy_score([1, 0, 1], [1, 0])


# Datasets


@pytest.mark.parametrize(
    ("generator", "kwargs", "expected_features"),
    [
        (make_blobs, {"n_samples": 60, "centers": 3}, 2),
        (make_regression, {"n_samples": 60, "n_features": 4}, 4),
        (make_moons, {"n_samples": 60}, 2),
    ],
)
def test_generators_return_the_requested_shapes(generator, kwargs, expected_features):
    X, y = generator(random_state=0, **kwargs)
    assert X.shape == (60, expected_features)
    assert y.shape == (60,)


@pytest.mark.parametrize("generator", [make_blobs, make_regression, make_moons])
def test_generators_are_reproducible(generator):
    X1, y1 = generator(n_samples=40, random_state=7)
    X2, y2 = generator(n_samples=40, random_state=7)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_make_blobs_splits_samples_evenly_when_not_divisible():
    """100 samples across 3 centres must still be exactly 100 samples."""
    _, y = make_blobs(n_samples=100, centers=3, random_state=0)
    counts = np.bincount(y)
    assert counts.sum() == 100
    assert counts.max() - counts.min() <= 1


def test_make_regression_with_no_noise_is_exactly_linear():
    """With noise=0 the targets lie exactly in the column space of X."""
    X, y = make_regression(n_samples=50, n_features=3, noise=0.0, random_state=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(X @ coef, y, atol=1e-8)


# Viz


def test_register_panel_rejects_non_panels():
    with pytest.raises(TypeError, match="VizPanel"):
        register_panel(object)


def test_registering_a_panel_makes_it_discoverable():
    @register_panel
    class ExamplePanel(VizPanel):
        name = "Example"
        family = "test"
        description = "Used only by the test suite."

        def parameters(self):
            return [Parameter("n", "Points", default=5, kind="int", min=1, max=10)]

        def frames(self, n=5):
            return [Frame(draw=lambda ax: None, caption=f"step {i}") for i in range(n)]

    assert "test.ExamplePanel" in get_panels()

    panel = ExamplePanel()
    assert len(panel.parameters()) == 1
    assert len(panel.frames(n=3)) == 3


def test_panel_is_abstract():
    """A panel must implement both parameters() and frames()."""
    with pytest.raises(TypeError):
        VizPanel()  # type: ignore[abstract]
