"""
FastAPI routes: the HTTP 'View' onto the prediction controller.

Organized as a `PredictionAPI` class that owns an `APIRouter` and registers
its own bound methods as route handlers. Kept thin on purpose — request
parsing/response shaping only. All business logic lives in
controllers/prediction_controller.py.
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from student_success.controllers.prediction_controller import (
    PredictionController,
)
from student_success.controllers.prediction_controller import (
    controller as default_controller,
)
from student_success.models.ml_model import ModelNotTrainedError
from student_success.models.schemas import HealthResponse, PredictionResult, StudentInput


class PredictionAPI:
    """Groups all prediction-related HTTP routes and wires them to a controller."""

    def __init__(self, controller: PredictionController | None = None) -> None:
        self.controller = controller or default_controller
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self) -> None:
        self.router.add_api_route(
            "/health", self.health, methods=["GET"], response_model=HealthResponse, tags=["system"]
        )
        self.router.add_api_route(
            "/predict",
            self.predict,
            methods=["POST"],
            response_model=PredictionResult,
            tags=["prediction"],
        )
        self.router.add_api_route(
            "/predict/batch", self.predict_batch, methods=["POST"], tags=["prediction"]
        )

    def health(self) -> HealthResponse:
        model_loaded = self.controller.model.is_loaded
        if not model_loaded:
            try:
                self.controller.ensure_ready()
                model_loaded = True
            except ModelNotTrainedError:
                model_loaded = False
        return HealthResponse(status="ok", model_loaded=model_loaded)

    def predict(self, student: StudentInput) -> PredictionResult:
        try:
            return self.controller.predict_single(student)
        except ModelNotTrainedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def predict_batch(self, file: UploadFile = File(...)) -> StreamingResponse:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv files are supported")

        content = await file.read()
        try:
            result_df: pd.DataFrame = self.controller.predict_batch(content)
        except ModelNotTrainedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        buffer = io.StringIO()
        result_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions_output.csv"},
        )


# Module-level instance + exposed router, so `main.py` can keep doing
# `app.include_router(router)` without knowing about the class.
prediction_api = PredictionAPI()
router = prediction_api.router
