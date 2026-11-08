import hmac
import json
import os
import pickle
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    GCCollector,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "customer-segmentation")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
API_KEY = os.getenv("API_KEY", "")
MAX_BATCH = int(os.getenv("MAX_BATCH", "5000"))
SEGMENT_PROFILE = os.getenv("SEGMENT_PROFILE", "reports/cluster_profile.csv")
FEATURES = ["recency", "frequency", "monetary"]

# Own registry (instead of the global one) so the module can be imported/reloaded safely in tests.
REGISTRY = CollectorRegistry()
ProcessCollector(registry=REGISTRY)
PlatformCollector(registry=REGISTRY)
GCCollector(registry=REGISTRY)

REQUESTS = Counter("seg_requests_total", "Total prediction requests", registry=REGISTRY)
ERRORS = Counter("seg_errors_total", "Failed prediction requests", registry=REGISTRY)
LATENCY = Histogram("seg_latency_seconds", "Prediction latency in seconds", registry=REGISTRY)
CLUSTER_COUNT = Counter("seg_predictions_by_cluster_total", "Predictions per cluster", ["cluster"],
                        registry=REGISTRY)
DIST_TO_CENTROID = Histogram("seg_distance_to_centroid", "Distance to nearest centroid (KMeans only)",
                             buckets=[0.25, 0.5, 1, 2, 3, 5, 10, 20], registry=REGISTRY)
MODEL_VERSION = Gauge("seg_model_version", "Loaded champion model version (0 = none loaded)",
                      registry=REGISTRY)
MODEL_VERSION.set(0)

_state = {"pipe": None, "final": None, "version": None, "algorithm": None,
          "source": None, "info": {}, "segments": None, "warnings": []}
_lock = threading.Lock()


# ----------------------------------------------------------------------------- model loading
def _read_profile(path) -> Optional[list]:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def _load_local(path: str):
    p = Path(path)
    with open(p / "model.pkl", "rb") as f:
        pipe = pickle.load(f)  # file baked into the image by us; only load files you created yourself
    info = json.loads((p / "model_info.json").read_text())
    return pipe, int(info["version"]), info, _read_profile(p / "segment_profile.csv"), "local"


def _load_mlflow():
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(TRACKING_URI)
    mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    pipe = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    info = {"model_name": MODEL_NAME, "version": int(mv.version)}
    try:  # the registering run records which algorithm won (the last pipeline step can be a KNN wrapper)
        info["algorithm"] = mlflow.MlflowClient().get_run(mv.run_id).data.tags.get("best_algorithm")
    except Exception:
        pass
    return pipe, int(mv.version), info, _read_profile(SEGMENT_PROFILE), "mlflow"


def _version_warnings(info: dict) -> list:
    """Pickled models can misbehave if library versions differ from training time."""
    import sklearn
    warnings = []
    trained = (info.get("library_versions") or {}).get("scikit-learn")
    if trained and trained != sklearn.__version__:
        warnings.append(f"scikit-learn {sklearn.__version__} installed, model trained with {trained}")
    return warnings


def load_model():
    """(Re)load the champion. Raises if there is none yet."""
    pipe, version, info, segments, source = _load_local(MODEL_PATH) if MODEL_PATH else _load_mlflow()
    with _lock:
        _state.update(pipe=pipe, final=pipe[-1], version=version,
                      algorithm=info.get("algorithm") or type(pipe[-1]).__name__,
                      source=source, info=info, segments=segments, warnings=_version_warnings(info))
    MODEL_VERSION.set(version)


def get_model():
    if _state["pipe"] is None:
        try:
            load_model()
        except Exception as exc:  # no champion registered yet / MLflow unreachable / bad folder
            raise HTTPException(503, f"Model not available: {exc}") from exc
    return _state


def _preload():
    try:
        load_model()
    except Exception as exc:
        print(f"[startup] model not loaded yet: {exc}", flush=True)


@asynccontextmanager
async def lifespan(_app):
    # load in the background so the server answers /live immediately (matters for Lambda start-up)
    threading.Thread(target=_preload, daemon=True).start()
    yield


app = FastAPI(title="Customer Segmentation API", lifespan=lifespan)


def require_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and not (x_api_key and hmac.compare_digest(x_api_key.encode(), API_KEY.encode())):
        raise HTTPException(401, "Invalid or missing API key")


# ----------------------------------------------------------------------------- schemas
class Customer(BaseModel):
    recency: float = Field(ge=0, description="days since last purchase")
    frequency: float = Field(ge=0, description="number of invoices")
    monetary: float = Field(ge=0, description="total spend")


def _predict(rows: pd.DataFrame):
    st = get_model()
    X = rows[FEATURES].astype(float).to_numpy()
    clusters = st["pipe"].predict(X)
    dists = [None] * len(clusters)
    if hasattr(st["final"], "transform"):  # KMeans: distance to nearest centroid
        d = st["final"].transform(st["pipe"][:-1].transform(X))
        dists = np.min(d, axis=1).tolist()
    return clusters, dists


# ----------------------------------------------------------------------------- routes
@app.get("/live")
def live():
    return {"status": "alive"}


@app.get("/health")
def health():
    st = get_model()
    return {"status": "ok", "model": st["info"].get("model_name", MODEL_NAME), "version": st["version"],
            "algorithm": st["algorithm"], "source": st["source"], "warnings": st["warnings"]}


@app.get("/segments", dependencies=[Depends(require_key)])
def segments():
    """Description of each segment (size, average RFM, persona hint) for the loaded model."""
    st = get_model()
    if not st["segments"]:
        raise HTTPException(404, "No segment profile available for this model")
    return {"version": st["version"], "segments": st["segments"]}


@app.post("/reload", dependencies=[Depends(require_key)])
def reload_model():
    """Pick up a newly promoted champion without restarting."""
    try:
        load_model()
    except Exception as exc:
        raise HTTPException(503, f"Reload failed: {exc}") from exc
    return {"version": _state["version"], "algorithm": _state["algorithm"]}


@app.post("/predict", dependencies=[Depends(require_key)])
def predict(c: Customer):
    REQUESTS.inc()
    start = time.time()
    try:
        clusters, dists = _predict(pd.DataFrame([c.model_dump()]))
        cluster, dist = int(clusters[0]), dists[0]
        CLUSTER_COUNT.labels(cluster=str(cluster)).inc()
        if dist is not None:
            DIST_TO_CENTROID.observe(dist)
        return {"cluster": cluster, "distance_to_centroid": dist, "model_version": _state["version"]}
    except Exception:
        ERRORS.inc()
        raise
    finally:
        LATENCY.observe(time.time() - start)


@app.post("/predict/batch", dependencies=[Depends(require_key)])
def predict_batch(customers: List[Customer]):
    if len(customers) > MAX_BATCH:
        raise HTTPException(413, f"Too many rows ({len(customers)}); the limit is {MAX_BATCH} per request")
    REQUESTS.inc(len(customers))
    start = time.time()
    try:
        if not customers:
            return {"clusters": [], "model_version": get_model()["version"]}
        clusters, dists = _predict(pd.DataFrame([c.model_dump() for c in customers]))
        for cl, d in zip(clusters, dists):
            CLUSTER_COUNT.labels(cluster=str(int(cl))).inc()
            if d is not None:
                DIST_TO_CENTROID.observe(d)
        return {"clusters": [int(x) for x in clusters], "model_version": _state["version"]}
    except Exception:
        ERRORS.inc()
        raise
    finally:
        LATENCY.observe(time.time() - start)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
