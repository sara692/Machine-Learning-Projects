# 🎓 Student Success Prediction

A productionized, MVC-structured rewrite of `ML_for_Student_Success_Prediction.ipynb`.
Predicts a student's **Total Percentage** score (average of Python + DB exam scores)
from `Gender`, `Residence`, `Entry_Exam`, and `Study_Hours`.

Training grid-searches hyperparameters for 5 model families (Linear Regression,
Random Forest, KNN, Decision Tree, SVR), tracks every run in **MLflow**, and
registers the best-performing tuned model in the MLflow Model Registry.

- **Backend**: FastAPI (`/predict`, `/predict/batch`, `/health`)
- **Frontend**: Gradio (talks to the API over HTTP, or in-process with `STANDALONE=1`)
- **ML pipeline**: hyperparameter tuning (`GridSearchCV`) + experiment tracking
  and model registry (`MLflow`)
- **Packaging**: `pyproject.toml` + `uv` (no `requirements.txt`)
- **Tests**: `pytest`
- **Containerization**: Docker + docker-compose (API, UI, and MLflow server)

## Project structure

```
student-success-prediction/
├── src/
│   └── student_success/
│       ├── models/          # Model (M): ML model wrapper, schemas, training pipeline
│       │   ├── ml_model.py            # loads a trained model and predicts
│       │   ├── schemas.py             # request/response shapes
│       │   ├── train.py               # ModelTrainer — orchestrates the pipeline
│       │   ├── tuner.py               # HyperparameterTuner — GridSearchCV per model
│       │   └── experiment_tracker.py  # ExperimentTracker — thin MLflow wrapper
│       ├── controllers/     # Controller (C): business logic between Model and Views
│       │   └── prediction_controller.py
│       ├── views/           # View (V): FastAPI routes + Gradio UI
│       │   ├── api.py
│       │   └── gradio_app.py
│       ├── config.py        # Single source of settings (YAML + env overrides)
│       └── main.py          # FastAPI app entrypoint
├── tests/                   # pytest — auto-discovered, CI-ready
│   ├── conftest.py
│   ├── test_ml_model.py
│   ├── test_tuner.py
│   ├── test_controller.py
│   └── test_api.py
├── configs/
│   └── config.yaml          # Config separated from code — no magic constants
├── notebooks/
│   └── ML_for_Student_Success_Prediction.ipynb   # Original exploration notebook (reference only)
├── data/                    # Place the real bi.csv here (not committed)
├── artifacts/               # Trained model/scaler/encoders land here (serving source of truth)
├── mlflow.db                # Local MLflow tracking DB (created on first training run)
├── Dockerfile                # FastAPI backend image
├── Dockerfile.gradio         # Gradio frontend image
├── docker-compose.yml        # Runs API + UI + MLflow server together
├── pyproject.toml            # Single source of truth for deps and tools
└── README.md
```

## Why this split?

- `notebooks/` is exploration only — nothing here imports from it.
- `src/` is an installable package (`student_success`) so both the API and the
  UI import shared model/controller code cleanly, instead of duplicating logic.
- `tuner.py` and `experiment_tracker.py` are separate from `train.py` so "how
  do we search one model's hyperparameters" and "how do we talk to MLflow"
  don't get tangled into the pipeline orchestration logic.
- `tests/` is auto-discovered by pytest and safe to run in CI from a fresh clone
  (falls back to a small synthetic dataset if `data/bi.csv` isn't present).

## Getting started (local, with `uv`)

```bash
# 1. Install dependencies (creates .venv, resolves + locks automatically)
uv sync --all-extras

# 2. (Optional) Drop the real dataset in place
cp /path/to/bi.csv data/bi.csv

# 3. Train the model — writes artifacts/rf_student_model.pkl, scaler.pkl, encoding_maps.pkl
uv run train-model
# equivalent: uv run python -m student_success.models.train

# 4. Run the API
uv run serve-api
# equivalent: uv run uvicorn student_success.main:app --reload

# 5. In another terminal, run the Gradio UI (defaults to calling the API above)
uv run serve-ui
# equivalent: uv run python -m student_success.views.gradio_app

# Or run the UI standalone, without a separate API process:
STANDALONE=1 uv run serve-ui
```

- API docs: http://localhost:8000/docs
- Gradio UI: http://localhost:7860

## Hyperparameter tuning + MLflow

Every `uv run train-model` run:

1. Grid-searches hyperparameters for each model family in
   `models/tuner.py::MODEL_REGISTRY` (edit that dict to widen/narrow the search).
2. Cross-validates each configuration (`cv_folds` in `configs/config.yaml`, default 3).
3. Scores every tuned model on a held-out test set and logs params + metrics +
   the fitted model to MLflow as a nested run under one parent
   `training_pipeline` run.
4. Registers the overall best model as a new version of
   `student_success_regressor` in the MLflow Model Registry.
5. Still saves the winning model + scaler + encoders to `artifacts/*.pkl` —
   this is what the API/Gradio app actually load by default, so **serving
   never requires a running MLflow server.**

**View the results:**

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000 to compare runs, see the leaderboard across model
families, and browse the registered model's versions.

**Switching where predictions load the model from** (`configs/config.yaml` or env vars):

| `model_source` | Behavior |
|---|---|
| `local` (default) | Loads `artifacts/rf_student_model.pkl` etc. No MLflow server needed at serving time. |
| `mlflow` | Loads the model tagged `@champion` (or `mlflow_model_stage_or_alias`) from the MLflow Model Registry instead. |

To promote a specific registered version to `champion` after reviewing it in the MLflow UI:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.MlflowClient()
client.set_registered_model_alias("student_success_regressor", "champion", version=<N>)
```

Then set `MODEL_SOURCE=mlflow` (env var) or `model_source: "mlflow"` in
`configs/config.yaml` and restart the API.

## Running tests

```bash
uv run pytest
uv run pytest --cov=student_success --cov-report=term-missing
```

Tests train a small model into an isolated `tmp_path` for each test run, so they
never touch your real `artifacts/` or `data/` directories and don't require the
original dataset.

## Docker

```bash
# Build & run API + UI + MLflow tracking server together
docker compose up --build
```

This starts three services:
- `mlflow` — tracking server + registry UI at http://localhost:5000 (sqlite-backed)
- `api` — FastAPI backend at http://localhost:8000, pointed at the `mlflow` service
- `ui` — Gradio frontend at http://localhost:7860

```bash
# API only, without compose
docker build -t student-success-api .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts student-success-api
```

Train the model into `./artifacts` on the host first (`uv run train-model`) and
mount that directory into the container — the image itself doesn't bundle
trained weights. To train *inside* Docker against the shared MLflow server,
run `docker compose exec api uv run train-model` after `docker compose up`.

## API reference

| Method | Path             | Description                                   |
|--------|------------------|------------------------------------------------|
| GET    | `/health`        | Service + model-loaded status                 |
| POST   | `/predict`       | Single prediction (JSON body, see `/docs`)    |
| POST   | `/predict/batch` | Upload a CSV, get back predictions as CSV     |

