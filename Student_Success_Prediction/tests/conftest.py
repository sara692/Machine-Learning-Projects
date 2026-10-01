from __future__ import annotations

from pathlib import Path

import pytest

from student_success.config import Settings
from student_success.models.ml_model import StudentSuccessModel
from student_success.models.train import ModelTrainer
from student_success.models.tuner import HyperparameterTuner

# A tiny hyperparameter grid so tests stay fast — real training uses the full
# grids in models/tuner.py::MODEL_REGISTRY. Only random_forest gets more than
# one option so tests still exercise the "pick the best hyperparameters" path
# without doing a full grid search.
TEST_MODEL_REGISTRY = {
    "linear_regression": {"estimator": None, "param_grid": {}},
    "random_forest": {"estimator": None, "param_grid": {"n_estimators": [10, 20]}},
}


def _build_test_registry() -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    return {
        "linear_regression": {"estimator": LinearRegression(), "param_grid": {}},
        "random_forest": {
            "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
            "param_grid": {"n_estimators": [10, 20]},
        },
    }


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    """Settings pointed at an isolated tmp dir, so tests never touch real artifacts/."""
    settings = Settings()
    settings.data_dir = tmp_path / "data"
    settings.artifacts_dir = tmp_path / "artifacts"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Isolated, sqlite-backed MLflow tracking store per test — never touches
    # the real project's mlflow.db, and each test gets its own experiment.
    settings.mlflow_tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    settings.mlflow_experiment_name = f"test-{tmp_path.name}"
    settings.cv_folds = 2

    # No bi.csv placed -> train_and_save falls back to synthetic data.
    return settings


@pytest.fixture()
def fast_trainer(test_settings: Settings) -> ModelTrainer:
    """A ModelTrainer with a tiny hyperparameter grid, for fast tests."""
    tuner = HyperparameterTuner(model_registry=_build_test_registry(), cv=test_settings.cv_folds)
    return ModelTrainer(test_settings, tuner=tuner)


@pytest.fixture()
def trained_model(fast_trainer: ModelTrainer) -> StudentSuccessModel:
    fast_trainer.run()
    model = StudentSuccessModel(fast_trainer.settings)
    model.load()
    return model
