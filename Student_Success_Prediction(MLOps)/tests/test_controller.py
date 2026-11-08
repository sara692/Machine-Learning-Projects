from __future__ import annotations

import io

import pandas as pd
import pytest

from student_success.controllers.prediction_controller import PredictionController
from student_success.models.ml_model import StudentSuccessModel
from student_success.models.schemas import StudentInput
from student_success.models.train import ModelTrainer


@pytest.fixture()
def ctrl(fast_trainer: ModelTrainer) -> PredictionController:
    fast_trainer.run()
    return PredictionController(
        model=StudentSuccessModel(fast_trainer.settings), settings=fast_trainer.settings
    )


def test_predict_single_end_to_end(ctrl: PredictionController):
    result = ctrl.predict_single(
        StudentInput(gender="Female", residence="Home", entry_exam=88, study_hours=30)
    )
    assert 0 <= result.predicted_total_percentage <= 100
    assert result.grade in {"A", "B", "C", "D"}


def test_predict_batch_end_to_end(ctrl: PredictionController):
    df = pd.DataFrame(
        {
            "Gender": ["Male", "Female", "Male"],
            "Residence": ["Home", "BI Residence", "Other"],
            "Entry_Exam": [55, 92, 70],
            "Study_Hours": [5, 35, 18],
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    result = ctrl.predict_batch(buf.getvalue().encode("utf-8"))
    assert len(result) == 3
    assert "Predicted_Total_Percentage" in result.columns


def test_predict_batch_missing_columns_raises(ctrl: PredictionController):
    df = pd.DataFrame({"Gender": ["Male"]})
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    with pytest.raises(ValueError):
        ctrl.predict_batch(buf.getvalue().encode("utf-8"))
