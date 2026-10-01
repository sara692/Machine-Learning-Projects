"""
Data schemas (the 'M' in MVC as far as request/response shape is concerned).

These Pydantic models define the contract between the View layer (FastAPI
routes / Gradio UI) and the Controller layer. Keeping them separate from
ml_model.py means validation rules can evolve independently of the model
implementation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    """A single student's raw features, as entered by a user."""

    gender: Literal["Male", "Female"] = Field(..., description="Student gender")
    residence: Literal["BI Residence", "Home", "Other"] = Field(
        ..., description="Student residence type"
    )
    entry_exam: float = Field(..., ge=0, le=100, description="Entry exam score (0-100)")
    study_hours: float = Field(..., ge=0, le=168, description="Weekly study hours")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Male",
                    "residence": "BI Residence",
                    "entry_exam": 72.5,
                    "study_hours": 15,
                }
            ]
        }
    }


class PredictionResult(BaseModel):
    """Result of a single prediction."""

    predicted_total_percentage: float
    grade: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
