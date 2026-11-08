"""
FastAPI application entrypoint, built via an `AppFactory` class so app
construction (middleware, routers, lifespan) is grouped and reusable (e.g.
for tests that need a fresh app wired to isolated settings).

Run with:
    uvicorn student_success.main:app --host 0.0.0.0 --port 8000
or:
    uv run uvicorn student_success.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from student_success.config import Settings
from student_success.config import settings as default_settings
from student_success.controllers.prediction_controller import (
    PredictionController,
)
from student_success.controllers.prediction_controller import (
    controller as default_controller,
)
from student_success.models.ml_model import ModelNotTrainedError
from student_success.views.api import PredictionAPI


class AppFactory:
    """Builds a configured FastAPI application around a PredictionController."""

    def __init__(
        self,
        settings: Settings | None = None,
        controller: PredictionController | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.controller = controller or default_controller
        self.prediction_api = PredictionAPI(self.controller)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        # Best-effort model load at startup; /health and endpoints still
        # report a clear 503 if artifacts are missing, so the container can
        # boot even before `train-model` has been run.
        try:
            self.controller.ensure_ready()
        except ModelNotTrainedError:
            pass
        yield

    def create_app(self) -> FastAPI:
        app = FastAPI(
            title="Student Success Prediction API",
            description=(
                "Predicts a student's total percentage score from study habits and background."
            ),
            version="0.1.0",
            lifespan=self._lifespan,
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.include_router(self.prediction_api.router)
        app.add_api_route("/", self.root, methods=["GET"], tags=["system"])
        return app

    def root(self) -> dict:
        return {"message": "Student Success Prediction API", "docs": "/docs"}

    def serve(self) -> None:
        """Console-script entrypoint: `serve-api` (see pyproject.toml)."""
        import uvicorn

        uvicorn.run(
            "student_success.main:app",
            host=self.settings.api_host,
            port=self.settings.api_port,
            reload=False,
        )


# Module-level app instance, so `uvicorn student_success.main:app` keeps working.
app_factory = AppFactory()
app = app_factory.create_app()


def serve() -> None:
    """Functional convenience wrapper used by the `serve-api` console script."""
    app_factory.serve()


if __name__ == "__main__":
    serve()
