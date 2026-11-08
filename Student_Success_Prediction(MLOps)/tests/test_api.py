from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from student_success.models.ml_model import StudentSuccessModel
from student_success.models.train import ModelTrainer


@pytest.fixture()
def client(fast_trainer: ModelTrainer) -> TestClient:
    fast_trainer.run()

    # Build a fresh app wired to an isolated test controller/settings, rather
    # than mutating the process-wide singleton — AppFactory makes this a
    # one-liner.
    from student_success.controllers.prediction_controller import PredictionController
    from student_success.main import AppFactory

    test_controller = PredictionController(
        model=StudentSuccessModel(fast_trainer.settings), settings=fast_trainer.settings
    )
    app = AppFactory(settings=fast_trainer.settings, controller=test_controller).create_app()

    return TestClient(app)


def test_health_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_endpoint(client: TestClient):
    resp = client.post(
        "/predict",
        json={"gender": "Male", "residence": "BI Residence", "entry_exam": 75, "study_hours": 20},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["predicted_total_percentage"] <= 100
    assert body["grade"] in {"A", "B", "C", "D"}


def test_predict_endpoint_validation_error(client: TestClient):
    resp = client.post(
        "/predict",
        json={"gender": "Unknown", "residence": "Home", "entry_exam": 75, "study_hours": 20},
    )
    assert resp.status_code == 422


def test_predict_batch_endpoint(client: TestClient):
    df = pd.DataFrame(
        {
            "Gender": ["Male", "Female"],
            "Residence": ["Home", "Other"],
            "Entry_Exam": [60, 95],
            "Study_Hours": [8, 32],
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    files = {"file": ("students.csv", buf.getvalue(), "text/csv")}
    resp = client.post("/predict/batch", files=files)
    assert resp.status_code == 200
    result_df = pd.read_csv(io.StringIO(resp.text))
    assert "Predicted_Total_Percentage" in result_df.columns
    assert len(result_df) == 2


def test_predict_batch_rejects_non_csv(client: TestClient):
    files = {"file": ("students.txt", "not,a,csv", "text/plain")}
    resp = client.post("/predict/batch", files=files)
    assert resp.status_code == 400
