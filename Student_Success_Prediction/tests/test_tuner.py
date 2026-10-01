from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from student_success.config import Settings
from student_success.models.experiment_tracker import ExperimentTracker
from student_success.models.tuner import HyperparameterTuner, TuningResult


@pytest.fixture()
def toy_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(60, 2))
    y = 3 * X[:, 0] + 2 * X[:, 1] + rng.normal(0, 0.1, size=60)
    return X, y


def test_tune_returns_tuning_result_with_best_params(toy_data):
    X, y = toy_data
    registry = {
        "random_forest": {
            "estimator": __import__(
                "sklearn.ensemble", fromlist=["RandomForestRegressor"]
            ).RandomForestRegressor(random_state=42),
            "param_grid": {"n_estimators": [5, 10]},
        }
    }
    tuner = HyperparameterTuner(model_registry=registry, cv=2)
    result = tuner.tune("random_forest", X, y)

    assert isinstance(result, TuningResult)
    assert result.model_name == "random_forest"
    assert result.best_params["n_estimators"] in (5, 10)
    assert hasattr(result.best_estimator, "predict")


def test_tune_handles_empty_param_grid(toy_data):
    """LinearRegression has no hyperparameters to tune — tune() should still work."""
    X, y = toy_data
    registry = {"linear_regression": {"estimator": LinearRegression(), "param_grid": {}}}
    tuner = HyperparameterTuner(model_registry=registry, cv=2)
    result = tuner.tune("linear_regression", X, y)

    assert result.best_params == {}
    assert result.best_cv_score > 0.9  # near-perfect linear relationship


def test_tune_unknown_model_raises(toy_data):
    X, y = toy_data
    tuner = HyperparameterTuner(model_registry={}, cv=2)
    with pytest.raises(KeyError):
        tuner.tune("does_not_exist", X, y)


def test_tune_all_covers_every_registered_model(toy_data):
    X, y = toy_data
    registry = {
        "linear_regression": {"estimator": LinearRegression(), "param_grid": {}},
    }
    tuner = HyperparameterTuner(model_registry=registry, cv=2)
    results = tuner.tune_all(X, y)
    assert set(results.keys()) == {"linear_regression"}


def test_experiment_tracker_logs_a_run(test_settings: Settings, toy_data):
    X, y = toy_data
    tracker = ExperimentTracker(test_settings)

    with tracker.start_run(run_name="unit_test_run") as run:
        tracker.log_params({"n_estimators": 10})
        tracker.log_metrics({"r2": 0.99})
        run_id = run.info.run_id

    assert run_id is not None
