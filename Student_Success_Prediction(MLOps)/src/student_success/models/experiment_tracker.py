"""
Thin wrapper around MLflow so ModelTrainer never calls the `mlflow` module
directly. If MLflow's API changes, or you swap it for another tracker, only
this file needs to change.
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn

from student_success.config import Settings
from student_success.config import settings as default_settings

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Wraps MLflow run lifecycle, logging, and model registration."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)

    def start_run(self, run_name: str | None = None, nested: bool = False):
        """Context manager: `with tracker.start_run("random_forest"):`."""
        return mlflow.start_run(run_name=run_name, nested=nested)

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict) -> None:
        mlflow.log_metrics(metrics)

    #: skops (MLflow's sklearn serializer) refuses to deserialize certain
    #: compiled sklearn internals by default as an extra safety check. KNN's
    #: KDTree is one of them — these are standard scikit-learn classes, not
    #: arbitrary/unsafe code, so we explicitly trust them here.
    TRUSTED_SKOPS_TYPES = [
        "sklearn.metrics._dist_metrics.EuclideanDistance64",
        "sklearn.neighbors._kd_tree.KDTree",
    ]

    def log_model(self, model, artifact_path: str = "model") -> None:
        mlflow.sklearn.log_model(
            model, artifact_path, skops_trusted_types=self.TRUSTED_SKOPS_TYPES
        )

    def set_tags(self, tags: dict) -> None:
        mlflow.set_tags(tags)

    def register_model(self, run_id: str, artifact_path: str = "model") -> str:
        """Register a logged model under settings.registered_model_name; returns its version."""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, self.settings.registered_model_name)
        logger.info(
            "Registered model version %s as '%s'",
            result.version,
            self.settings.registered_model_name,
        )
        return result.version

    def load_registered_model(self, stage_or_alias: str | None = None):
        """
        Load the current registered model for serving.

        `stage_or_alias` can be an MLflow alias (e.g. "champion") or a legacy
        stage name (e.g. "Production"). Falls back to settings.
        """
        alias = stage_or_alias or self.settings.mlflow_model_stage_or_alias
        model_uri = f"models:/{self.settings.registered_model_name}@{alias}"
        return mlflow.sklearn.load_model(model_uri)
