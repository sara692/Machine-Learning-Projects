"""
The 'Controller' layer: sits between Views (FastAPI routes / Gradio callbacks)
and the Model (StudentSuccessModel). It validates/shapes I/O and contains the
business logic, so neither the API layer nor the Gradio UI needs to know
about pandas, joblib, or the artifact file layout.
"""

from __future__ import annotations

import io

import pandas as pd

from student_success.config import Settings
from student_success.config import settings as default_settings
from student_success.models.ml_model import StudentSuccessModel
from student_success.models.schemas import PredictionResult, StudentInput


class PredictionController:
    def __init__(self, model: StudentSuccessModel | None = None, settings: Settings | None = None):
        self.settings = settings or default_settings
        self.model = model or StudentSuccessModel(self.settings)

    def ensure_ready(self) -> None:
        if not self.model.is_loaded:
            self.model.load()

    def predict_single(self, student: StudentInput) -> PredictionResult:
        self.ensure_ready()
        pct = self.model.predict_one(
            gender=student.gender,
            residence=student.residence,
            entry_exam=student.entry_exam,
            study_hours=student.study_hours,
        )
        grade = self.model.grade_from_percentage(pct)
        return PredictionResult(predicted_total_percentage=pct, grade=grade)

    def predict_batch(self, csv_bytes: bytes) -> pd.DataFrame:
        self.ensure_ready()
        df = pd.read_csv(io.BytesIO(csv_bytes))
        missing = [c for c in self.settings.feature_order if c not in df.columns]
        if missing:
            raise ValueError(f"Uploaded CSV is missing required columns: {missing}")
        return self.model.predict_dataframe(df)


# Module-level singleton used by both the FastAPI view and the Gradio view,
# so the model is loaded once per process rather than per-request.
controller = PredictionController()
