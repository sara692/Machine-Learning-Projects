"""
Hyperparameter search, kept separate from ModelTrainer so "how do we search
this one model's hyperparameters" and "how do we run the whole pipeline"
aren't tangled together.

PARAM_GRIDS defines the search space per candidate model family. Grids are
intentionally small — this is a teaching/demo pipeline, not a full sweep.
Widen them freely for a real search; ModelTrainer just iterates whatever is
in this dict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)

# Model family -> (estimator factory, hyperparameter grid).
# LinearRegression has no meaningful hyperparameters here, so its grid is
# empty — GridSearchCV still runs (with cv folds) purely to score it
# consistently alongside the tuned models.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "linear_regression": {
        "estimator": LinearRegression(),
        "param_grid": {},
    },
    "random_forest": {
        "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
    },
    "knn": {
        "estimator": KNeighborsRegressor(),
        "param_grid": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
        },
    },
    "decision_tree": {
        "estimator": DecisionTreeRegressor(random_state=42),
        "param_grid": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
    },
    "svr": {
        "estimator": SVR(),
        "param_grid": {
            "C": [1, 10, 100],
            "epsilon": [0.01, 0.1, 0.2],
            "kernel": ["rbf", "linear"],
        },
    },
}


@dataclass
class TuningResult:
    """Everything ModelTrainer/ExperimentTracker need about one tuned model family."""

    model_name: str
    best_estimator: Any
    best_params: dict[str, Any]
    best_cv_score: float  # cross-validated score on the training set (higher is better)


class HyperparameterTuner:
    """
    Runs GridSearchCV for one or more model families and reports the winner
    within each family.

        tuner = HyperparameterTuner(cv=3, scoring="r2")
        result = tuner.tune("random_forest", X_train, y_train)
        result = tuner.tune_all(X_train, y_train)  # every registered family
    """

    def __init__(
        self,
        model_registry: dict[str, dict[str, Any]] | None = None,
        cv: int = 3,
        scoring: str = "r2",
        n_jobs: int = -1,
    ) -> None:
        self.model_registry = model_registry or MODEL_REGISTRY
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def tune(self, model_name: str, X_train, y_train) -> TuningResult:
        """Grid-search a single registered model family."""
        if model_name not in self.model_registry:
            raise KeyError(
                f"Unknown model '{model_name}'. Known models: {list(self.model_registry)}"
            )

        spec = self.model_registry[model_name]
        param_grid = spec["param_grid"]

        if not param_grid:
            # Nothing to search (e.g. plain LinearRegression) — still
            # cross-validate so it's scored consistently alongside tuned
            # models, then fit once on the full training set.
            logger.info("No hyperparameters to tune for %s — cross-validating as-is.", model_name)
            estimator = clone(spec["estimator"])
            scores = cross_val_score(
                estimator, X_train, y_train, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs
            )
            best_estimator = clone(spec["estimator"]).fit(X_train, y_train)
            best_cv_score = float(scores.mean())
            logger.info("%-20s CV %s=%.4f (no params)", model_name, self.scoring, best_cv_score)
            return TuningResult(
                model_name=model_name,
                best_estimator=best_estimator,
                best_params={},
                best_cv_score=best_cv_score,
            )

        search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        logger.info(
            "Tuning %s over %d parameter combination(s)...",
            model_name,
            self._grid_size(param_grid),
        )
        search.fit(X_train, y_train)

        logger.info(
            "%-20s best CV %s=%.4f params=%s",
            model_name, self.scoring, search.best_score_, search.best_params_,
        )
        return TuningResult(
            model_name=model_name,
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            best_cv_score=search.best_score_,
        )

    def tune_all(self, X_train, y_train) -> dict[str, TuningResult]:
        """Grid-search every registered model family. Returns name -> TuningResult."""
        return {name: self.tune(name, X_train, y_train) for name in self.model_registry}

    @staticmethod
    def _grid_size(param_grid: dict) -> int:
        size = 1
        for values in param_grid.values():
            size *= len(values)
        return size
