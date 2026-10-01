"""
Centralized configuration for the Student Success Prediction service.

Values can be overridden via environment variables or configs/config.yaml.
Keeping config out of the code (per the project-structure convention) avoids
magic constants scattered across the model/controller/view layers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/student_success/config.py -> repo root)
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Settings:
    # Paths
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    artifacts_dir: Path = ROOT_DIR / "artifacts"

    raw_data_filename: str = "bi.csv"
    model_filename: str = "rf_student_model.pkl"
    scaler_filename: str = "scaler.pkl"
    encoders_filename: str = "encoding_maps.pkl"

    # Feature schema (must match training order)
    feature_order: list[str] = field(
        default_factory=lambda: ["Gender", "Residence", "Entry_Exam", "Study_Hours"]
    )
    gender_categories: list[str] = field(default_factory=lambda: ["Female", "Male"])
    residence_categories: list[str] = field(
        default_factory=lambda: ["BI Residence", "Home", "Other"]
    )

    # Training
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 50
    cv_folds: int = 3

    # MLflow experiment tracking + hyperparameter search.
    # A sqlite-backed store is used (not "file:./mlruns") because MLflow 3.x
    # put the plain filesystem backend into maintenance mode and it no longer
    # supports the Model Registry. sqlite needs no separate server process.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "student-success-prediction"
    registered_model_name: str = "student_success_regressor"
    #: "local" = predict using the joblib files in artifacts/ (default, no
    #: MLflow server needed at serving time). "mlflow" = load the latest
    #: registered model from the MLflow Model Registry instead.
    model_source: str = "local"
    mlflow_model_stage_or_alias: str = "champion"

    # Service
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    gradio_port: int = 7860
    api_base_url: str = "http://localhost:8000"

    @property
    def raw_data_path(self) -> Path:
        return self.data_dir / self.raw_data_filename

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / self.model_filename

    @property
    def scaler_path(self) -> Path:
        return self.artifacts_dir / self.scaler_filename

    @property
    def encoders_path(self) -> Path:
        return self.artifacts_dir / self.encoders_filename


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Build a Settings object from defaults, overlaid with YAML, overlaid with env vars."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    yaml_cfg = _load_yaml(path)

    settings = Settings()

    # YAML overrides
    for key, value in yaml_cfg.items():
        if hasattr(settings, key):
            setattr(settings, key, value)

    # Env var overrides (highest priority) — useful for Docker/K8s
    settings.api_host = os.getenv("API_HOST", settings.api_host)
    settings.api_port = int(os.getenv("API_PORT", settings.api_port))
    settings.gradio_port = int(os.getenv("GRADIO_PORT", settings.gradio_port))
    settings.api_base_url = os.getenv("API_BASE_URL", settings.api_base_url)
    settings.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
    settings.model_source = os.getenv("MODEL_SOURCE", settings.model_source)

    return settings


settings = load_settings()
