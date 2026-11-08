"""
The 'Model' layer: wraps the trained RandomForestRegressor + scaler + label
encodings produced by notebooks/ML_for_Student_Success_Prediction.ipynb.

This class knows nothing about HTTP, Gradio, or CLI concerns — it is a pure
prediction component that the Controller layer calls into.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from student_success.config import Settings
from student_success.config import settings as default_settings


class ModelNotTrainedError(RuntimeError):
    """Raised when prediction is requested but no trained artifacts exist."""


class StudentSuccessModel:
    """Loads trained artifacts and serves predictions."""

    #: Grade cutoffs, expressed as (minimum percentage, letter grade), checked
    #: in order — kept as class state so subclasses/tests can override the
    #: grading scale without touching the prediction logic.
    GRADE_BANDS: tuple[tuple[float, str], ...] = (
        (85, "A"),
        (70, "B"),
        (55, "C"),
        (0, "D"),
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self.model = None
        self.scaler = None
        self.encoding_maps: dict[str, dict[str, int]] | None = None

    @classmethod
    def grade_from_percentage(cls, pct: float) -> str:
        """Map a predicted total percentage to a letter grade (mirrors notebook logic)."""
        for threshold, grade in cls.GRADE_BANDS:
            if pct >= threshold:
                return grade
        return cls.GRADE_BANDS[-1][1]

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None and self.encoding_maps is not None

    def load(self) -> StudentSuccessModel:
        """
        Load the scaler + encoders from the local artifacts directory (always),
        and the model itself from either:
          - "local" (default): the .pkl file training saved to artifacts/
          - "mlflow": the current registered model in the MLflow Model Registry

        Set settings.model_source = "mlflow" (or env var MODEL_SOURCE=mlflow)
        to switch, e.g. once you're promoting models via MLflow aliases/stages
        instead of just retraining locally.
        """
        scaler_path = self.settings.scaler_path
        encoders_path = self.settings.encoders_path

        missing = [p for p in (scaler_path, encoders_path) if not p.exists()]
        if missing:
            raise ModelNotTrainedError(
                "Missing trained artifacts: "
                f"{[str(p) for p in missing]}. Run `python -m student_success.models.train` "
                "(or the train console script) first."
            )
        self.scaler = joblib.load(scaler_path)
        self.encoding_maps = joblib.load(encoders_path)

        if self.settings.model_source == "mlflow":
            self.model = self._load_model_from_mlflow()
        else:
            model_path = self.settings.model_path
            if not model_path.exists():
                raise ModelNotTrainedError(
                    f"Missing trained model artifact: {model_path}. "
                    "Run `python -m student_success.models.train` first."
                )
            self.model = joblib.load(model_path)
        return self

    def _load_model_from_mlflow(self):
        """Fetch the current champion/production model from the MLflow Model Registry."""
        from student_success.models.experiment_tracker import ExperimentTracker

        try:
            tracker = ExperimentTracker(self.settings)
            return tracker.load_registered_model()
        except Exception as exc:  # noqa: BLE001
            raise ModelNotTrainedError(
                "Could not load a model from the MLflow Model Registry "
                f"(model_source='mlflow'): {exc}. Either register a model first "
                "by running training, or set model_source back to 'local'."
            ) from exc

    def _encode_row(self, row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        for col, mapping in self.encoding_maps.items():
            if col in row:
                row[col] = mapping.get(row[col], 0)
        return row

    def predict_one(
        self, gender: str, residence: str, entry_exam: float, study_hours: float
    ) -> float:
        if not self.is_loaded:
            raise ModelNotTrainedError("Model artifacts are not loaded. Call .load() first.")

        row = self._encode_row(
            {
                "Gender": gender,
                "Residence": residence,
                "Entry_Exam": entry_exam,
                "Study_Hours": study_hours,
            }
        )
        df_input = pd.DataFrame([row])[self.settings.feature_order]
        scaled = self.scaler.transform(df_input)
        prediction = float(self.model.predict(scaled)[0])
        return round(prediction, 2)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch-predict for an uploaded CSV (mirrors the notebook's predict_file)."""
        if not self.is_loaded:
            raise ModelNotTrainedError("Model artifacts are not loaded. Call .load() first.")

        df = df.copy()
        for col, mapping in self.encoding_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)

        X = df[self.settings.feature_order]
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        df["Predicted_Total_Percentage"] = np.round(predictions, 2)
        df["Grade"] = df["Predicted_Total_Percentage"].apply(self.grade_from_percentage)
        return df


# Module-level alias kept for callers that prefer a plain function (e.g. quick
# scripting or notebooks) without instantiating the class.
def grade_from_percentage(pct: float) -> str:
    """Functional convenience wrapper around StudentSuccessModel.grade_from_percentage."""
    return StudentSuccessModel.grade_from_percentage(pct)
