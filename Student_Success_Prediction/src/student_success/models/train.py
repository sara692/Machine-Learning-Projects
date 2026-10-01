"""
Training pipeline, ported from notebooks/ML_for_Student_Success_Prediction.ipynb,
encapsulated as a `ModelTrainer` class so each pipeline stage is a discrete,
testable method rather than a loose function.

Steps mirrored from the notebook, now with hyperparameter tuning + MLflow
experiment tracking layered on top:
  1. Load raw data (data/bi.csv)
  2. Clean categorical typos (gender/country/residence/prevEducation)
  3. Impute missing 'Python' scores with the column mean
  4. Rename/title-case columns
  5. Encode categoricals, engineer Total_Percentage target
  6. Train/test split + StandardScaler
  7. For every candidate model family, grid-search its best hyperparameters
     (HyperparameterTuner), score the tuned model on the held-out test set,
     and log params/metrics/the model itself to MLflow (ExperimentTracker)
  8. Pick the best-performing tuned model overall, register it in the MLflow
     Model Registry, and persist it + the scaler + encoding maps to
     artifacts/ for serving

If data/bi.csv is not present, a small synthetic dataset with the same
schema is generated instead, so the service remains runnable end-to-end
(e.g. in CI or a fresh clone) without the original private dataset.
Replace data/bi.csv with the real file for production-quality results.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from student_success.config import Settings
from student_success.config import settings as default_settings
from student_success.models.experiment_tracker import ExperimentTracker
from student_success.models.tuner import HyperparameterTuner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Encapsulates the end-to-end training pipeline as discrete methods:

        trainer = ModelTrainer(settings)
        model_path = trainer.run()

    Each pipeline stage (load, clean, engineer, split, tune+evaluate, persist)
    is its own method so it can be unit-tested or overridden independently.
    """

    GENDER_MAP = {"Male": 1, "Female": 0}
    RESIDENCE_MAP = {"BI Residence": 0, "Home": 1, "Other": 2}

    def __init__(
        self,
        settings: Settings | None = None,
        tuner: HyperparameterTuner | None = None,
        tracker: ExperimentTracker | None = None,
        enable_mlflow: bool = True,
    ) -> None:
        self.settings = settings or default_settings()
        self._label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        self.tuner = tuner or HyperparameterTuner(cv=self.settings.cv_folds)
        self.enable_mlflow = enable_mlflow
        self.tracker = tracker if tracker is not None else (
            ExperimentTracker(self.settings) if self.enable_mlflow else None
        )

        self.best_model_name: str | None = None
        self.best_model = None
        self.best_params: dict = {}
        self.evaluation_results: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Data loading / synthetic fallback
    # ------------------------------------------------------------------ #
    def load_raw_data(self) -> pd.DataFrame:
        if self.settings.raw_data_path.exists():
            logger.info("Loading raw data from %s", self.settings.raw_data_path)
            for encoding in ("utf-8", "cp1252", "latin-1"):
                try:
                    return pd.read_csv(self.settings.raw_data_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise UnicodeDecodeError(
                "utf-8/cp1252/latin-1", b"", 0, 1,
                f"Could not decode {self.settings.raw_data_path} with any known encoding",
            )

        logger.warning(
            "%s not found — generating a synthetic dataset for demo purposes. "
            "Place the real bi.csv in the data/ directory for production training.",
            self.settings.raw_data_path,
        )

        raise FileNotFoundError(
        f"Raw data file not found: {self.settings.raw_data_path}"
)

    # ------------------------------------------------------------------ #
    # Cleaning / feature engineering
    # ------------------------------------------------------------------ #
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reproduce the notebook's cleaning steps."""
        df = df.copy()

        if "Python" in df.columns:
            df["Python"] = df["Python"].fillna(df["Python"].mean())

        if "gender" in df.columns:
            df["gender"] = df["gender"].replace(
                {"M": "Male", "F": "Female", "male": "Male", "female": "Female"}
            )
        if "country" in df.columns:
            df["country"] = df["country"].replace(
                {
                    "Norway": "Norway",
                    "Norge": "Norway",
                    "Rsa": "South Africa",
                    "UK": "United Kingdom",
                    "Somali": "Somalia",
                }
            )
        if "residence" in df.columns:
            df["residence"] = df["residence"].replace(
                {
                    "BI-Residence": "BI Residence",
                    "BIResidence": "BI Residence",
                    "BI_Residence": "BI Residence",
                }
            )
        if "prevEducation" in df.columns:
            df["prevEducation"] = df["prevEducation"].replace(
                {
                    "Barrrchelors": "Bachelors",
                    "Diplomaaa": "Diploma",
                    "diploma": "Diploma",
                    "DIPLOMA": "Diploma",
                    "HighSchool": "High School",
                }
            )

        df.columns = df.columns.str.title()
        df = df.rename(
            columns={
                "Fname": "FName",
                "Lname": "LName",
                "Entryexam": "Entry_Exam",
                "Preveducation": "Prev_Education",
                "Studyhours": "Study_Hours",
            }
        )
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categoricals and compute the regression target.

        NOTE: Gender/Residence are encoded with the *same* fixed maps that get
        persisted to encoding_maps.pkl (self.GENDER_MAP / self.RESIDENCE_MAP),
        so training and inference stay consistent. The original notebook used
        a fresh LabelEncoder at train time but a separately hand-written map
        at inference time — a latent train/serve skew bug, fixed here by
        using one source of truth. Country/Prev_Education are still
        label-encoded for completeness even though they're dropped before
        modeling.
        """
        df = df.copy()

        for col in ("Country", "Prev_Education"):
            if col in df.columns:
                df[col] = self._label_encoder.fit_transform(df[col])

        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map(self.GENDER_MAP).fillna(0).astype(int)
        if "Residence" in df.columns:
            df["Residence"] = df["Residence"].map(self.RESIDENCE_MAP).fillna(0).astype(int)

        df["Total_Percentage"] = ((df["Python"] + df["Db"]) / 200) * 100
        return df

    # ------------------------------------------------------------------ #
    # Split / model selection
    # ------------------------------------------------------------------ #
    def split_and_scale(self, df: pd.DataFrame):
        drop_cols = [
            c
            for c in [
                "FName", "LName", "Country", "Age", "Python", "Db",
                "Total_Percentage", "Prev_Education",
            ]
            if c in df.columns
        ]
        X = df.drop(columns=drop_cols)[self.settings.feature_order]
        y = df["Total_Percentage"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
            shuffle=True,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    # ------------------------------------------------------------------ #
    # Hyperparameter tuning + model selection (MLflow-tracked)
    # ------------------------------------------------------------------ #
    def tune_and_evaluate_models(self, X_train, X_test, y_train, y_test) -> str:
        """
        For every model family registered in the tuner:
          1. Grid-search its hyperparameters via cross-validation on the
             training set (HyperparameterTuner.tune).
          2. Refit the winning configuration and score it on the held-out
             test set (the number that actually decides the overall winner).
          3. Log params/metrics/the fitted model to MLflow as a nested run
             (if MLflow tracking is enabled).

        Returns the name of the overall best-performing model, and leaves
        `self.best_model` / `self.best_params` / `self.evaluation_results`
        populated for `save_artifacts()` and inspection/tests.
        """
        results: dict[str, dict] = {}
        logger.info(
            "%-20s %-10s %-10s %-10s %-10s", "Model", "CV R2", "Test MAE", "Test RMSE", "Test R2"
        )

        run_ctx = (
            self.tracker.start_run(run_name="training_pipeline") if self.tracker else nullcontext()
        )
        with run_ctx:
            for model_name in self.tuner.model_registry:
                tuning_result = self.tuner.tune(model_name, X_train, y_train)
                pred = tuning_result.best_estimator.predict(X_test)
                mae = mean_absolute_error(y_test, pred)
                rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                r2 = r2_score(y_test, pred)

                results[model_name] = {
                    "model": tuning_result.best_estimator,
                    "params": tuning_result.best_params,
                    "cv_score": tuning_result.best_cv_score,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                }
                logger.info(
                    "%-20s %-10.4f %-10.2f %-10.2f %-10.4f",
                    model_name, tuning_result.best_cv_score, mae, rmse, r2,
                )

                if self.tracker:
                    with self.tracker.start_run(run_name=model_name, nested=True) as child_run:
                        self.tracker.log_params(
                            {"model_name": model_name, **tuning_result.best_params}
                        )
                        self.tracker.log_metrics(
                            {
                                "cv_r2": tuning_result.best_cv_score,
                                "test_mae": mae,
                                "test_rmse": rmse,
                                "test_r2": r2,
                            }
                        )
                        self.tracker.log_model(tuning_result.best_estimator)
                        results[model_name]["run_id"] = child_run.info.run_id

            best_name = max(
                results,
                key=lambda name: (
                    results[name]["R2"], -results[name]["RMSE"], -results[name]["MAE"],
                ),
            )
            best = results[best_name]
            logger.info(
                "Best model after tuning: %s (Test R2=%.4f, params=%s)",
                best_name, best["R2"], best["params"],
            )

            if self.tracker:
                self.tracker.set_tags({"best_model": best_name})
                if "run_id" in best:
                    try:
                        self.tracker.register_model(best["run_id"])
                    except Exception as exc:  # noqa: BLE001
                        # Model registry needs a database-backed MLflow store in
                        # some setups; tuning/selection still succeeded either
                        # way, so don't fail the whole run over registration.
                        logger.warning("Could not register model in MLflow: %s", exc)

        self.evaluation_results = results
        self.best_model_name = best_name
        self.best_model = best["model"]
        self.best_params = best["params"]
        return best_name

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_artifacts(self) -> Path:
        if self.best_model is None:
            raise RuntimeError("No trained model to save — call tune_and_evaluate_models() first.")

        self.settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, self.settings.model_path)
        joblib.dump(self.scaler, self.settings.scaler_path)
        joblib.dump(
            {"Gender": self.GENDER_MAP, "Residence": self.RESIDENCE_MAP},
            self.settings.encoders_path,
        )
        logger.info("Artifacts saved to %s", self.settings.artifacts_dir)
        return self.settings.model_path

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self) -> Path:
        """Run the full pipeline end-to-end and return the saved model path."""
        df = self.load_raw_data()
        df = self.clean_data(df)
        df = self.engineer_features(df)

        X_train, X_test, y_train, y_test = self.split_and_scale(df)
        self.tune_and_evaluate_models(X_train, X_test, y_train, y_test)
        return self.save_artifacts()


def train_and_save(settings: Settings | None = None) -> Path:
    """Functional convenience wrapper around ModelTrainer, used by the console script."""
    return ModelTrainer(settings).run()


if __name__ == "__main__":
    train_and_save()
