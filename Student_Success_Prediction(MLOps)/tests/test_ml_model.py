from __future__ import annotations

import pandas as pd
import pytest

from student_success.config import Settings
from student_success.models.ml_model import (
    ModelNotTrainedError,
    StudentSuccessModel,
    grade_from_percentage,
)
from student_success.models.train import ModelTrainer


@pytest.mark.parametrize(
    "pct,expected",
    [(90, "A"), (85, "A"), (75, "B"), (70, "B"), (60, "C"), (55, "C"), (40, "D"), (0, "D")],
)
def test_grade_from_percentage(pct, expected):
    assert grade_from_percentage(pct) == expected


def test_model_not_loaded_raises(test_settings: Settings):
    model = StudentSuccessModel(test_settings)
    with pytest.raises(ModelNotTrainedError):
        model.predict_one("Male", "Home", 70, 15)


def test_train_and_save_creates_artifacts(fast_trainer: ModelTrainer):
    model_path = fast_trainer.run()
    assert model_path.exists()
    assert fast_trainer.settings.scaler_path.exists()
    assert fast_trainer.settings.encoders_path.exists()


def test_predict_one_returns_float_in_range(trained_model: StudentSuccessModel):
    pred = trained_model.predict_one("Male", "BI Residence", entry_exam=80, study_hours=20)
    assert isinstance(pred, float)
    assert 0 <= pred <= 100


def test_predict_dataframe_adds_expected_columns(trained_model: StudentSuccessModel):
    df = pd.DataFrame(
        {
            "Gender": ["Male", "Female"],
            "Residence": ["Home", "BI Residence"],
            "Entry_Exam": [65, 90],
            "Study_Hours": [10, 25],
        }
    )
    result = trained_model.predict_dataframe(df)
    assert "Predicted_Total_Percentage" in result.columns
    assert "Grade" in result.columns
    assert len(result) == 2
    assert result["Grade"].isin(["A", "B", "C", "D"]).all()


def test_clean_data_normalizes_gender_typos(test_settings: Settings):
    trainer = ModelTrainer(test_settings)
    df = pd.DataFrame({"gender": ["M", "F", "male", "female"]})
    cleaned = trainer.clean_data(df)
    assert set(cleaned["Gender"]) == {"Male", "Female"}


def test_engineer_features_encodes_gender_residence_consistently(test_settings: Settings):
    trainer = ModelTrainer(test_settings)
    df = pd.DataFrame(
        {
            "Gender": ["Male", "Female"],
            "Residence": ["Home", "Other"],
            "Python": [80, 60],
            "Db": [70, 50],
        }
    )
    result = trainer.engineer_features(df)
    assert result["Gender"].tolist() == [1, 0]
    assert result["Residence"].tolist() == [1, 2]
    assert result["Total_Percentage"].round(6).tolist() == [75.0, 55.0]


def test_model_trainer_run_end_to_end(fast_trainer: ModelTrainer):
    model_path = fast_trainer.run()
    assert model_path.exists()
    assert fast_trainer.best_model_name in fast_trainer.evaluation_results
    assert fast_trainer.best_model is not None


def test_tuning_picks_hyperparameters_and_records_cv_score(fast_trainer: ModelTrainer):
    fast_trainer.run()
    best = fast_trainer.evaluation_results[fast_trainer.best_model_name]
    assert "cv_score" in best
    assert isinstance(fast_trainer.best_params, dict)
    if fast_trainer.best_model_name == "random_forest":
        assert best["model"].n_estimators in (10, 20)
