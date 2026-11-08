# Streamlit UI and AWS deployment - Phases 12 to 18

This continues `PHASES.md`. Same layout as before: **Goal / Why / What to do / Code to write / Check / Commit**. Commands are for **Git Bash** on Windows with the `retail-mlops` conda environment active.

**What you need before starting:** phases 0-5 done (a champion model is registered in MLflow and `reports/cluster_profile.csv` exists) and phase 7 (the FastAPI service works). Phases 6 and 8-11 are not required for this part.

## The plan in one picture

```
                         your PC                                          AWS
  dvc repro -> MLflow -> export_champion.py -> docker build/push -> ECR (segmentation-api, segmentation-ui)
                                                                       |                    |
                                                    Lambda (container + Function URL)   ECS Express Mode
                                                    FastAPI + model baked in            Streamlit (Fargate + load balancer)
                                                                  ^                              |
                                                                  +---- HTTPS + x-api-key -------+
  Browser ---------------------------------------------------------------------------------------> Streamlit URL
```

## How this maps to the course you followed

| Course step | This project | Why different |
|---|---|---|
| ECR repository `consultation-app` | Two repositories: `segmentation-api`, `segmentation-ui` | Two services instead of one |
| `aws configure`, push image | Same | - |
| Lambda from container image + Function URL | **Same, for the FastAPI model API** | - |
| Function URL invoke mode `RESPONSE_STREAM` | `BUFFERED` | The API returns whole answers, not streamed text |
| Env vars `CLERK_*`, `OPENAI_API_KEY` | `API_KEY` | Our only secret is the key that protects the API |
| Next.js static files inside the image | Not needed | The UI is Streamlit, a separate service |
| App Runner (replaced by Lambda in the course) | **ECS Express Mode** for Streamlit | Streamlit needs a long-running server with WebSockets, which Lambda cannot provide; App Runner no longer accepts new customers |

**Tested:** I replayed phases 12-14 from your actual starting point (your phase 7 API), applying each phase's files in order and running its checks, plus the API and the UI in clean Python 3.11 environments that contain only what the Docker images would install. **Not tested** (no Docker engine or AWS account available to me): the `docker build` commands and everything in phases 15-18. The AWS console steps are based on your course and on AWS's current documentation; screen labels can change, so if a screen differs, go by the field names.

---

## Phase 12 - Streamlit user interface

**Goal:** A web page where someone types a customer's Recency / Frequency / Monetary values (or uploads a CSV) and sees the segment.

**Why:** **A model needs a front door.** The Streamlit app is a thin client: it holds no model, it only calls your FastAPI service. That separation is what lets the two halves be deployed to different AWS services in the next phases.

### What to do

1. Install Streamlit in your environment:
   ```bash
   pip install -r requirements-ui.txt
   ```

2. Create the three files below. `segment_client.py` holds the code that talks to the API and checks uploaded files (kept apart so it can be tested); `streamlit_app.py` is the page itself. It has three tabs: *One customer*, *Upload a file* (CSV with columns recency, frequency, monetary; shows a segment per customer, a bar chart, and a download button), and *About the segments*.

3. Start your API as in phase 7 (MLflow server running), in one window:
   ```bash
   MLFLOW_TRACKING_URI=http://localhost:5000 uvicorn src.serving.app:app --port 8000
   ```

4. Start the UI in a second window:
   ```bash
   API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
   ```
   It opens http://localhost:8501. Click *Check connection* in the sidebar, then try the three tabs. For a test upload, use the *Download a template* button on the *Upload a file* tab.

### Code to write

**`requirements-ui.txt`**

```text
streamlit>=1.40
requests>=2.31
pandas>=2.2
```

**`ui/segment_client.py`**

```python
"""Small client + helpers used by the Streamlit app (kept separate so they can be unit-tested)."""
from typing import Optional

import pandas as pd
import requests

REQUIRED = ["recency", "frequency", "monetary"]
TEMPLATE_CSV = (
    "customer_id,recency,frequency,monetary\n"
    "1001,12,25,5400.50\n"
    "1002,310,1,89.90\n"
    "1003,45,6,820.00\n"
)


class ApiError(Exception):
    """An error message that is safe and useful to show to the person using the app."""


class ApiClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 60, session=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs):
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        try:
            resp = getattr(self.session, method)(f"{self.base_url}{path}", headers=headers,
                                                 timeout=self.timeout, **kwargs)
        except requests.exceptions.Timeout as exc:
            raise ApiError("The API did not answer in time. If it was idle, the first request can take "
                           "up to 30 seconds - please try again.") from exc
        except requests.exceptions.RequestException as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url}. Check that it is running and "
                           "that API_URL is correct.") from exc
        if resp.status_code == 401:
            raise ApiError("The API rejected the API key. Check API_KEY.")
        if resp.status_code == 503:
            raise ApiError("The API is running but has no model loaded yet.")
        if resp.status_code == 413:
            raise ApiError("Too many rows in one request.")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise ApiError(f"API error {resp.status_code}: {detail}")
        return resp

    def health(self) -> dict:
        return self._request("get", "/health").json()

    def segments(self) -> Optional[pd.DataFrame]:
        """Segment descriptions, or None if this API has none."""
        try:
            data = self._request("get", "/segments").json()["segments"]
        except ApiError as exc:
            if "404" in str(exc):
                return None
            raise
        return pd.DataFrame(data)

    def predict(self, recency: float, frequency: float, monetary: float) -> dict:
        return self._request("post", "/predict", json={"recency": recency, "frequency": frequency,
                                                       "monetary": monetary}).json()

    def predict_batch(self, df: pd.DataFrame, chunk: int = 500) -> list:
        """Segment for every row of df, sent in chunks to stay far below request-size limits."""
        out = []
        for i in range(0, len(df), chunk):
            rows = df.iloc[i:i + chunk][REQUIRED].astype(float).to_dict(orient="records")
            out.extend(self._request("post", "/predict/batch", json=rows).json()["clusters"])
        return out


def normalize_batch(df: pd.DataFrame):
    """Check an uploaded table. Returns (clean_df, n_rows_dropped); raises ValueError with a clear message."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Required columns: {', '.join(REQUIRED)}")
    before = len(df)
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    df = df[(df[REQUIRED] >= 0).all(axis=1)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows: recency, frequency and monetary must be numbers >= 0.")
    return df, before - len(df)


def persona_map(segments: Optional[pd.DataFrame]) -> dict:
    """segment number -> persona hint text"""
    if segments is None or "segment" not in segments or "persona_hint" not in segments:
        return {}
    return {int(r["segment"]): str(r["persona_hint"]) for _, r in segments.iterrows()}
```

**`ui/streamlit_app.py`**

```python
"""Streamlit front end for the customer-segmentation API.

Env vars:
  API_URL      base URL of the API, e.g. http://localhost:8000 or the Lambda Function URL (no trailing slash)
  API_KEY      sent as the x-api-key header (leave empty if the API has no key)
  API_TIMEOUT  seconds to wait for the API (default 60; Lambda can need ~30 s after being idle)

Run:  streamlit run ui/streamlit_app.py
"""
import os

import pandas as pd
import streamlit as st

from segment_client import TEMPLATE_CSV, ApiClient, ApiError, normalize_batch, persona_map

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "60"))

st.set_page_config(page_title="Customer Segmentation", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient(API_URL, API_KEY, API_TIMEOUT)


@st.cache_data(ttl=300, show_spinner=False)
def load_segments(api_url: str):
    """Segment descriptions; cached for 5 minutes. Returns None if the API has none."""
    return get_client().segments()


client = get_client()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    if st.button("Check connection"):
        try:
            with st.spinner("Contacting the API (the first call after idle can take up to 30 seconds)..."):
                h = client.health()
            st.success(f"Connected - model v{h['version']} ({h['algorithm']})")
            for w in h.get("warnings", []):
                st.warning(w)
        except ApiError as exc:
            st.error(str(exc))
    st.caption(f"API: {API_URL}")
    st.caption("Segments are groups of customers with similar buying behaviour, based on Recency "
               "(days since last purchase), Frequency (number of orders) and Monetary value (total spend).")

st.title("Customer segmentation")
tab_one, tab_batch, tab_segments = st.tabs(["One customer", "Upload a file", "About the segments"])

# ------------------------------------------------------------------ one customer
with tab_one:
    with st.form("single"):
        c1, c2, c3 = st.columns(3)
        recency = c1.number_input("Recency - days since last purchase", min_value=0, value=30, step=1)
        frequency = c2.number_input("Frequency - number of orders", min_value=0, value=5, step=1)
        monetary = c3.number_input("Monetary - total spend", min_value=0.0, value=500.0, step=50.0)
        submitted = st.form_submit_button("Find segment")
    if submitted:
        try:
            with st.spinner("Asking the model (the first call after idle can take up to 30 seconds)..."):
                res = client.predict(recency, frequency, monetary)
                segs = load_segments(API_URL)
            seg = int(res["cluster"])
            persona = persona_map(segs).get(seg)
            m1, m2 = st.columns(2)
            m1.metric("Segment", seg)
            m2.metric("Looks like", persona or "n/a")
            if segs is not None and "segment" in segs:
                row = segs[segs["segment"] == seg]
                if not row.empty:
                    st.caption("Typical customer in this segment")
                    cols = [c for c in ["customers", "share", "recency_median", "frequency_median",
                                        "monetary_median"] if c in row]
                    st.dataframe(row[cols], hide_index=True)
        except ApiError as exc:
            st.error(str(exc))

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write("Upload a CSV with the columns **recency**, **frequency** and **monetary** "
             "(and optionally **customer_id**).")
    st.download_button("Download a template", TEMPLATE_CSV, file_name="customers_template.csv",
                       mime="text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            raw = pd.read_csv(up)
            data, dropped = normalize_batch(raw)
            if dropped:
                st.warning(f"{dropped} row(s) were skipped: a value was missing, not a number, or negative.")
            st.write(f"{len(data)} customers ready.")
            if st.button("Segment these customers"):
                with st.spinner("Segmenting (the first call after idle can take up to 30 seconds)..."):
                    data["segment"] = client.predict_batch(data)
                    segs = load_segments(API_URL)
                names = persona_map(segs)
                if names:
                    data["looks_like"] = data["segment"].map(names)
                st.subheader("Customers per segment")
                counts = data["segment"].value_counts().sort_index()
                st.bar_chart(counts)
                st.subheader("Result")
                st.dataframe(data)
                st.download_button("Download result as CSV", data.to_csv(index=False),
                                   file_name="customers_segmented.csv", mime="text/csv")
        except ValueError as exc:
            st.error(str(exc))
        except ApiError as exc:
            st.error(str(exc))

# ------------------------------------------------------------------ about the segments
with tab_segments:
    try:
        with st.spinner("Loading segment descriptions..."):
            segs = load_segments(API_URL)
        if segs is None:
            st.info("This API does not provide segment descriptions. Export the model after running "
                    "src/models/evaluate.py to include them.")
        else:
            st.dataframe(segs, hide_index=True)
            if {"segment", "customers"} <= set(segs.columns):
                st.bar_chart(segs.set_index("segment")["customers"])
            st.caption("'Looks like' labels are rule-of-thumb hints computed from each segment's "
                       "median Recency / Frequency / Monetary values, not a precise classification.")
    except ApiError as exc:
        st.error(str(exc))
```

### Check

*Check connection* shows `Connected - model v1 (<algorithm>)`. *Find segment* shows a segment number. Uploading a CSV with one bad row shows a warning that the row was skipped and still segments the rest.

With your phase 7 API, *Looks like* shows `n/a` and the *About the segments* tab says the API has no segment descriptions; that is fixed in the next phase.

### Commit

```bash
git add . && git commit -m "Phase 12: Streamlit UI"
```

---

## Phase 13 - Make the model portable and protect the API

**Goal:** The API can run **without MLflow**, loading a model from a plain folder, and it can require an API key.

**Why:** **Deployment packaging.** On AWS your laptop's MLflow server does not exist. So you export the champion model into a folder (`deploy/model/`) that is baked into the container image. The export also records the exact library versions used in training, because a pickled scikit-learn model can break if the libraries differ. The API key stops strangers from using a public URL.

### What to do

1. **Replace** `src/serving/app.py` with the new version below. It is a superset of the old one: with no `MODEL_PATH` it behaves as before (MLflow registry); with `MODEL_PATH` it loads from the exported folder. New: `API_KEY` protection, `GET /segments`, a batch-size limit, and the real algorithm name in `/health`.

2. Create `scripts/export_champion.py`, then export your champion (MLflow server running):
   ```bash
   python scripts/export_champion.py
   ls deploy/model
   ```
   You should see `model.pkl`, `model_info.json`, `requirements-model.txt` and `segment_profile.csv` (the last one only if you ran `evaluate.py`, phase 5; if it is missing, run that first and export again).

3. **Replace** `pytest.ini`, `ruff.toml` and `.gitignore` with the versions below (small updates: the UI folder is added to the import path, and the exported model and `.env.*` secret files are ignored by Git). Create the two new test files.

4. Run the API from the exported folder, with a key, **with the MLflow server stopped** to prove it no longer needs it:
   ```bash
   MODEL_PATH=deploy/model API_KEY=test-key uvicorn src.serving.app:app --port 8000
   ```
   In another window:
   ```bash
   curl localhost:8000/health
   curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"recency":5,"frequency":30,"monetary":20000}'   # 401: no key
   curl -X POST localhost:8000/predict -H "x-api-key: test-key" -H "Content-Type: application/json" -d '{"recency":5,"frequency":30,"monetary":20000}'
   ```

5. Point the UI at it:
   ```bash
   API_URL=http://localhost:8000 API_KEY=test-key streamlit run ui/streamlit_app.py
   ```
   The *Looks like* label and *About the segments* tab now work.

6. Run the checks:
   ```bash
   pip install -r requirements-dev.txt
   ruff check src tests scripts ui
   pytest -q
   ```

### Code to write

**`src/serving/app.py`**  *(replace your existing file with this)*

```python
"""FastAPI service for the customer-segmentation model, with Prometheus metrics.

The model is loaded from one of two places:
  * MODEL_PATH=<folder>  a folder written by scripts/export_champion.py (model.pkl + model_info.json).
                         Used on AWS Lambda; needs no MLflow.
  * otherwise            the MLflow Model Registry: models:/<MODEL_NAME>@<MODEL_ALIAS> at MLFLOW_TRACKING_URI.
                         Used locally and in docker compose.

Env vars:
  MODEL_PATH           folder with an exported model (optional)
  MLFLOW_TRACKING_URI  default http://localhost:5000
  MODEL_NAME           default customer-segmentation
  MODEL_ALIAS          default champion
  API_KEY              if set, /predict, /predict/batch, /segments, /reload need header x-api-key: <key>
  MAX_BATCH            max rows in one /predict/batch call (default 5000)
  SEGMENT_PROFILE      csv with segment descriptions for MLflow mode (default reports/cluster_profile.csv)

Run:  uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
"""
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
```

**`scripts/export_champion.py`**

```python
"""Export the champion model from the MLflow registry into a plain folder, ready to bake into a container.

Writes (default folder deploy/model):
  model.pkl               the fitted pipeline (log1p -> scaler -> best model)
  model_info.json         version, algorithm, features, library versions it was trained with
  requirements-model.txt  exact versions of scikit-learn / numpy / pandas / scipy used for training
  segment_profile.csv     segment descriptions (copied from reports/cluster_profile.csv, if it exists)

The deployed API then needs no MLflow server. Re-run this after every new champion.

Run:  python scripts/export_champion.py [--config configs/tuning.yaml] [--tracking-uri URI]
"""
import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

try:
    import cloudpickle as pickler  # ships with MLflow
except ImportError:  # pragma: no cover
    import pickle as pickler

LIBS = ["scikit-learn", "numpy", "pandas", "scipy"]


def export_pipeline(pipe, out_dir, *, model_name, version, features, profile_csv=None,
                    algorithm=None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "model.pkl", "wb") as f:
        pickler.dump(pipe, f)

    libs = {name: pkg_version(name) for name in LIBS}
    info = {
        "model_name": model_name,
        "version": int(version),
        "algorithm": algorithm or type(pipe[-1]).__name__,
        "features": list(features),
        "library_versions": libs,
        "python_version": platform.python_version(),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "model_info.json").write_text(json.dumps(info, indent=2))
    (out / "requirements-model.txt").write_text("".join(f"{k}=={v}\n" for k, v in libs.items()))

    prof = Path(profile_csv) if profile_csv else None
    if prof and prof.exists():
        shutil.copy(prof, out / "segment_profile.csv")
        info["segment_profile"] = True
    else:
        (out / "segment_profile.csv").unlink(missing_ok=True)
        info["segment_profile"] = False
    return info


def main():
    import mlflow
    import mlflow.sklearn
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tuning.yaml")
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--out", default="deploy/model")
    ap.add_argument("--profile", default="reports/cluster_profile.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name, alias = cfg["mlflow"]["registered_model"], cfg["mlflow"]["champion_alias"]
    mlflow.set_tracking_uri(args.tracking_uri or cfg["mlflow"]["tracking_uri"])

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    pipe = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    # the run that produced the model records which algorithm actually won (DBSCAN/Agglomerative are
    # wrapped in a nearest-neighbours classifier for serving, so the pipeline's last step is not informative)
    algorithm = client.get_run(mv.run_id).data.tags.get("best_algorithm")
    info = export_pipeline(pipe, args.out, model_name=name, version=mv.version,
                           features=cfg["data"]["features"], profile_csv=args.profile, algorithm=algorithm)

    print(f"Exported '{name}' v{info['version']} ({info['algorithm']}) to {args.out}/")
    print("  libraries:", ", ".join(f"{k} {v}" for k, v in info["library_versions"].items()))
    if not info["segment_profile"]:
        print(f"  note: {args.profile} not found - run src/models/evaluate.py first if you want "
              "segment descriptions in the UI")


if __name__ == "__main__":
    main()
```

**`pytest.ini`**  *(replace your existing file with this)*

```ini
[pytest]
pythonpath = . ui
testpaths = tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::FutureWarning
    ignore::UserWarning
```

**`ruff.toml`**  *(replace your existing file with this)*

```toml
line-length = 110
src = [".", "ui"]
extend-exclude = ["notebooks", "mlruns", "mlartifacts"]

[lint]
select = ["E", "F", "I"]
```

**`.gitignore`**  *(replace your existing file with this)*

```text
# python
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/

# local MLflow stores
mlflow.db
mlflow_smoke.db
mlruns/
mlartifacts/

# data is tracked by DVC, not git (DVC adds the exact .gitignore entries for its outputs)
data/raw/*
!data/raw/.gitkeep
!data/raw/*.dvc
reports/drift.json
models/*
!models/.gitkeep
.env

# deployment artifacts and secrets
deploy/model/
.env.*
```

**`tests/test_deploy.py`**

```python
"""Tests for the portable (no-MLflow) model mode, the API key and the export script."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from scripts.export_champion import export_pipeline


def make_pipe(model):
    rng = np.random.default_rng(0)
    X = rng.lognormal(3, 1, size=(300, 3))
    pipe = Pipeline([("log", FunctionTransformer(np.log1p, validate=True)), ("scaler", StandardScaler()),
                     ("model", model)]).fit(X)
    return pipe, X


@pytest.fixture()
def exported(tmp_path):
    pipe, X = make_pipe(KMeans(3, n_init=5, random_state=0))
    profile = tmp_path / "profile.csv"
    pd.DataFrame({"segment": [0, 1, 2], "customers": [100, 120, 80], "persona_hint": ["A", "B", "C"]}
                 ).to_csv(profile, index=False)
    out = tmp_path / "model"
    info = export_pipeline(pipe, out, model_name="customer-segmentation", version=7,
                           features=["recency", "frequency", "monetary"], profile_csv=profile)
    return pipe, X, out, info


def load_app(monkeypatch, **env):
    for k in ["MODEL_PATH", "API_KEY", "MAX_BATCH"]:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    import src.serving.app as app_module
    return importlib.reload(app_module)


def test_export_writes_expected_files(exported):
    _, _, out, info = exported
    assert {"model.pkl", "model_info.json", "requirements-model.txt", "segment_profile.csv"} <= \
        {p.name for p in out.iterdir()}
    meta = json.loads((out / "model_info.json").read_text())
    assert meta["version"] == 7 and meta["algorithm"] == "KMeans" and info["segment_profile"]
    reqs = (out / "requirements-model.txt").read_text()
    assert "scikit-learn==" in reqs and "numpy==" in reqs and "pandas==" in reqs and "scipy==" in reqs


def test_local_mode_matches_original_pipeline(exported, monkeypatch):
    from fastapi.testclient import TestClient
    pipe, X, out, _ = exported
    app_module = load_app(monkeypatch, MODEL_PATH=out)
    c = TestClient(app_module.app)

    h = c.get("/health").json()
    assert h["source"] == "local" and h["version"] == 7 and h["algorithm"] == "KMeans" and h["warnings"] == []

    rows = [{"recency": float(r[0]), "frequency": float(r[1]), "monetary": float(r[2])} for r in X[:20]]
    got = c.post("/predict/batch", json=rows).json()["clusters"]
    assert got == [int(v) for v in pipe.predict(X[:20])]

    seg = c.get("/segments").json()
    assert seg["segments"][0]["persona_hint"] == "A"


def test_model_without_predict_still_works_and_has_no_distance(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    pipe, _ = make_pipe(GaussianMixture(3, random_state=0))
    out = tmp_path / "gmm"
    export_pipeline(pipe, out, model_name="m", version=1, features=["recency", "frequency", "monetary"])
    app_module = load_app(monkeypatch, MODEL_PATH=out)
    c = TestClient(app_module.app)
    r = c.post("/predict", json={"recency": 10, "frequency": 3, "monetary": 100}).json()
    assert r["distance_to_centroid"] is None and r["cluster"] in (0, 1, 2)
    assert c.get("/segments").status_code == 404  # exported without a profile


def test_api_key_is_enforced(exported, monkeypatch):
    from fastapi.testclient import TestClient
    _, _, out, _ = exported
    app_module = load_app(monkeypatch, MODEL_PATH=out, API_KEY="secret-key")
    c = TestClient(app_module.app)
    body = {"recency": 1, "frequency": 1, "monetary": 1}

    assert c.get("/live").status_code == 200 and c.get("/health").status_code == 200  # open
    assert c.post("/predict", json=body).status_code == 401
    assert c.post("/predict", json=body, headers={"x-api-key": "wrong"}).status_code == 401
    assert c.get("/segments").status_code == 401
    assert c.post("/reload").status_code == 401
    ok = c.post("/predict", json=body, headers={"x-api-key": "secret-key"})
    assert ok.status_code == 200


def test_batch_limit(exported, monkeypatch):
    from fastapi.testclient import TestClient
    _, _, out, _ = exported
    app_module = load_app(monkeypatch, MODEL_PATH=out, MAX_BATCH=3)
    c = TestClient(app_module.app)
    row = {"recency": 1, "frequency": 1, "monetary": 1}
    assert c.post("/predict/batch", json=[row] * 3).status_code == 200
    assert c.post("/predict/batch", json=[row] * 4).status_code == 413
    assert c.post("/predict/batch", json=[]).json()["clusters"] == []


def test_bad_model_folder_gives_503(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    app_module = load_app(monkeypatch, MODEL_PATH=tmp_path / "does-not-exist")
    c = TestClient(app_module.app)
    assert c.get("/live").status_code == 200
    assert c.get("/health").status_code == 503
```

**`tests/test_ui.py`**

```python
"""Tests for the Streamlit app's client and helpers (no browser needed)."""
import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from scripts.export_champion import export_pipeline
from segment_client import ApiClient, ApiError, normalize_batch, persona_map


# ------------------------------------------------------------------ helpers
def test_normalize_batch_ok_and_drops_bad_rows():
    raw = pd.DataFrame({" Recency ": [1, "x", 3, -2], "FREQUENCY": [1, 2, 3, 4], "monetary": [5, 6, 7, 8],
                        "customer_id": [1, 2, 3, 4]})
    df, dropped = normalize_batch(raw)
    assert list(df["customer_id"]) == [1, 3] and dropped == 2


def test_normalize_batch_errors():
    with pytest.raises(ValueError, match="Missing column"):
        normalize_batch(pd.DataFrame({"recency": [1], "frequency": [1]}))
    with pytest.raises(ValueError, match="No valid rows"):
        normalize_batch(pd.DataFrame({"recency": ["a"], "frequency": [1], "monetary": [1]}))


def test_persona_map():
    segs = pd.DataFrame({"segment": [0, 1], "persona_hint": ["VIP", "Lost"]})
    assert persona_map(segs) == {0: "VIP", 1: "Lost"}
    assert persona_map(None) == {} and persona_map(pd.DataFrame({"a": [1]})) == {}


# ------------------------------------------------------------------ client against the real API app
@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    rng = np.random.default_rng(0)
    X = rng.lognormal(3, 1, size=(300, 3))
    pipe = Pipeline([("log", FunctionTransformer(np.log1p, validate=True)), ("scaler", StandardScaler()),
                     ("model", KMeans(3, n_init=5, random_state=0))]).fit(X)
    prof = tmp_path / "p.csv"
    pd.DataFrame({"segment": [0, 1, 2], "customers": [1, 2, 3], "persona_hint": ["a", "b", "c"]}
                 ).to_csv(prof, index=False)
    export_pipeline(pipe, tmp_path / "m", model_name="m", version=2,
                    features=["recency", "frequency", "monetary"], profile_csv=prof)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("API_KEY", "k")
    import src.serving.app as app_module
    importlib.reload(app_module)
    session = TestClient(app_module.app)  # same get/post interface as requests.Session
    return lambda key="k": ApiClient("http://testserver", key, session=session)


def test_client_roundtrip(api_client):
    c = api_client()
    assert c.health()["version"] == 2
    assert c.predict(10, 5, 300)["cluster"] in (0, 1, 2)
    segs = c.segments()
    assert list(segs["persona_hint"]) == ["a", "b", "c"]

    df = pd.DataFrame({"recency": np.arange(1, 1201), "frequency": 3, "monetary": 100.0})
    clusters = c.predict_batch(df, chunk=500)  # 3 chunks
    assert len(clusters) == 1200 and set(clusters) <= {0, 1, 2}


def test_client_wrong_key_gives_friendly_error(api_client):
    with pytest.raises(ApiError, match="API key"):
        api_client("nope").predict(1, 1, 1)


def test_client_unreachable_gives_friendly_error():
    c = ApiClient("http://127.0.0.1:9", "", timeout=1)  # nothing listens on port 9
    with pytest.raises(ApiError, match="Cannot reach the API"):
        c.health()
```

### Check

`/health` shows `"source":"local"` and your algorithm; the request without a key returns 401 and with the key works; the UI shows persona labels; `pytest` reports `24 passed`.

If `/health` shows a `warnings` entry about library versions, the environment running the API differs from the one that trained the model; in the container (next phase) this cannot happen because the image installs the exact versions from `requirements-model.txt`.

### Commit

```bash
git add . && git commit -m "Phase 13: portable model and API key"
```

---

## Phase 14 - Build the two container images

**Goal:** Two Docker images: the API (for Lambda) and the Streamlit UI, tested on your own PC before touching AWS.

**Why:** **Same artifact everywhere.** What you test locally in a container is exactly what AWS runs. Lambda can run a container image if it contains the *Lambda Web Adapter*, a small program that lets Lambda talk to an ordinary web server (uvicorn) inside the image; that is what your course's Dockerfile did.

### What to do

1. Install and start **Docker Desktop**.

2. Create the files below. Key points: `Dockerfile.lambda` copies in the adapter, installs the *exact* training versions of scikit-learn/numpy/pandas/scipy from the exported folder, and bakes in `deploy/model`. `Dockerfile.ui` is plain Streamlit. `.dockerignore` keeps data and caches out of the build. *(Optional, only if you did phase 10)* to also start the UI in the local docker-compose stack, add this service to `docker/docker-compose.yml`, between `api:` and `prometheus:`:
   ```yaml
   ui:
     build:
       context: ..
       dockerfile: docker/Dockerfile.ui
     environment:
       API_URL: http://api:8000
     ports: ["8501:8501"]
     depends_on: [api]
     restart: unless-stopped
   ```

3. Build the API image. Flags matter: `--platform linux/amd64` matches Lambda, and `--provenance=false --sbom=false` stop recent Docker versions from adding extra metadata that Lambda rejects (error *"image manifest, config or layer media type ... is not supported"*):
   ```bash
   docker build --platform linux/amd64 --provenance=false --sbom=false -f docker/Dockerfile.lambda -t segmentation-api .
   ```

4. Build the UI image:
   ```bash
   docker build --platform linux/amd64 --provenance=false --sbom=false -f docker/Dockerfile.ui -t segmentation-ui .
   ```

5. Run both and test them together:
   ```bash
   docker run --rm -p 9000:8000 -e API_KEY=test-key segmentation-api
   ```
   Second window:
   ```bash
   docker run --rm -p 8501:8501 -e API_URL=http://host.docker.internal:9000 -e API_KEY=test-key segmentation-ui
   ```
   Third window: `curl localhost:9000/health`, then open http://localhost:8501 and try the UI. Stop both with Ctrl+C.

### Code to write

**`requirements-lambda.txt`**

```text
# Runtime dependencies of the API image used on AWS Lambda.
# The model libraries (scikit-learn, numpy, pandas, scipy) are NOT listed here: they are pinned to the exact
# versions used for training in deploy/model/requirements-model.txt (written by scripts/export_champion.py).
fastapi>=0.110
uvicorn>=0.29
prometheus-client>=0.20
```

**`docker/Dockerfile.lambda`**

```dockerfile
# API image for AWS Lambda (container image + Lambda Web Adapter).
# Build from the project root AFTER running scripts/export_champion.py:
#   docker build --platform linux/amd64 --provenance=false --sbom=false -f docker/Dockerfile.lambda -t segmentation-api .
FROM public.ecr.aws/docker/library/python:3.11-slim

# Lambda Web Adapter: lets Lambda talk to a normal web server (uvicorn) running inside the container.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/live \
    MODEL_PATH=/app/model \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Exact library versions the model was trained with, then the API's own dependencies.
COPY deploy/model/requirements-model.txt ./requirements-model.txt
COPY requirements-lambda.txt ./requirements-lambda.txt
RUN pip install --no-cache-dir -r requirements-model.txt -r requirements-lambda.txt

COPY src/__init__.py ./src/__init__.py
COPY src/serving ./src/serving
COPY deploy/model ./model

CMD ["python", "-m", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker/Dockerfile.ui`**

```dockerfile
# Streamlit UI image (runs on ECS Express Mode / Fargate, or locally).
# Build from the project root:
#   docker build --platform linux/amd64 --provenance=false --sbom=false -f docker/Dockerfile.ui -t segmentation-ui .
FROM public.ecr.aws/docker/library/python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt
COPY ui ./ui

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**`.dockerignore`**

```text
# Keep the Docker build context small (the project root is the context). Do NOT ignore src, ui, deploy, docker or requirements*.
.git
.dvc
.venv
**/__pycache__
.pytest_cache
.ruff_cache
data
reports
models
mlruns
mlartifacts
*.db
docs
notebooks
tests
.github
monitoring
.env*
```

### Check

Both builds finish without errors, `/health` on port 9000 answers, and the UI on port 8501 works against it.

**Honest note:** I could not run Docker where I wrote this. What I did test: the API's and the UI's dependency lists in two clean Python 3.11 environments (the API environment had no MLflow installed), the compose file's syntax, and the apps themselves. The `docker build` steps are the part most likely to need a small fix on your machine; if a build fails, send me the error.

### Commit

```bash
git add . && git commit -m "Phase 14: container images"
```

---

## Phase 15 - AWS account preparation

**Goal:** A spending alarm, the AWS CLI configured, and two empty image repositories (ECR) waiting for your images.

**Why:** **Guard rails first.** Cloud mistakes cost money; set the alarm before you create anything. ECR is where AWS stores your container images; both Lambda and ECS pull from it. This phase mirrors steps 1-2 of your course's "Deploy to AWS" part.

### What to do

1. **Spending alarm.** In the AWS console search *Budgets* -> *Create budget* -> use the *Monthly cost budget* template -> set an amount you are comfortable with (for example 10 USD) -> enter your email -> create. You will get an email when actual or forecast spending reaches the threshold.

2. **Install the AWS CLI** (Windows installer from https://aws.amazon.com/cli/ ), then close and reopen Git Bash and check `aws --version`.

3. **Permissions.** Do not use the root account for daily work. Use an IAM user (your course's `aiengineer` user is fine). For this project the user needs: `AmazonEC2ContainerRegistryFullAccess`, `AWSLambda_FullAccess`, `AmazonECS_FullAccess`, and `IAMFullAccess` (so the console can create the roles that Lambda and ECS need). In a personal learning account, giving the user `AdministratorAccess` plus MFA is the simplest alternative. In the console: *IAM* -> *Users* -> your user -> *Add permissions*.

4. **Access key** (same as your course): *IAM* -> *Users* -> your user -> *Security credentials* -> *Create access key* -> *Command Line Interface (CLI)* -> confirm -> description `Docker push access` -> *Create*. Save the two values. Then:
   ```bash
   aws configure
   ```
   Enter the access key ID, the secret, a region close to you that is enabled by default (for example `eu-central-1` Frankfurt or `eu-west-1` Ireland) and `json`. Use **one region for everything** below. Check:
   ```bash
   aws sts get-caller-identity
   ```

5. **Settings file.** Create `.env.aws` in the project folder (it is ignored by Git; it holds no secrets, only your account number and region):
   ```bash
   AWS_ACCOUNT_ID=123456789012
   DEFAULT_AWS_REGION=eu-west-1
   ```
   Use the 12-digit `Account` number that `aws sts get-caller-identity` printed. Load it in every new Git Bash window:
   ```bash
   set -a; source .env.aws; set +a
   export ECR="$AWS_ACCOUNT_ID.dkr.ecr.$DEFAULT_AWS_REGION.amazonaws.com"
   ```

6. **Create two ECR repositories** (console: *ECR* -> *Create repository* -> *Private*; check the region at the top right matches). Names exactly: `segmentation-api` and `segmentation-ui`; leave the other settings as default. Or by command:
   ```bash
   aws ecr create-repository --repository-name segmentation-api --region $DEFAULT_AWS_REGION
   aws ecr create-repository --repository-name segmentation-ui --region $DEFAULT_AWS_REGION
   ```

### Code to write

None in this phase: it is all console and command-line work.

### Check

`aws sts get-caller-identity` prints your account; the ECR console lists `segmentation-api` and `segmentation-ui` in your region; you received (or can see) the budget in *Billing -> Budgets*.

Never put the access key or secret in a file inside the project or commit them. `aws configure` stores them in your home folder.

### Commit

```bash
git status    # .env.aws must NOT appear: it is ignored
```

---

## Phase 16 - Deploy the API to AWS Lambda

**Goal:** A public HTTPS address (Lambda Function URL) that answers `/predict` using your champion model.

**Why:** **Serverless inference.** This is the same pattern as your course: image in ECR -> Lambda container function -> Function URL. Lambda costs nothing while idle and a few cents under light use, and the model is small and fast, which suits it. Differences from the course: no Clerk/OpenAI keys (our secret is `API_KEY`) and the URL uses *BUFFERED* mode, because the API returns whole answers rather than streaming text.

### What to do

1. Export the latest champion again (MLflow server running) so the image contains it:
   ```bash
   python scripts/export_champion.py
   ```

2. Load your settings, log Docker in to ECR, build, tag and push the API image (login lasts 12 hours):
   ```bash
   set -a; source .env.aws; set +a
   export ECR="$AWS_ACCOUNT_ID.dkr.ecr.$DEFAULT_AWS_REGION.amazonaws.com"
   aws ecr get-login-password --region $DEFAULT_AWS_REGION | docker login --username AWS --password-stdin $ECR
   docker build --platform linux/amd64 --provenance=false --sbom=false -f docker/Dockerfile.lambda -t segmentation-api .
   docker tag segmentation-api:latest $ECR/segmentation-api:latest
   docker push $ECR/segmentation-api:latest
   ```
   The first push takes a few minutes. Check the ECR console shows the image with tag `latest`.

3. **Create the function.** Console: *Lambda* -> check the region -> *Create function* -> **Container image** -> name `segmentation-api` -> *Browse images* -> repository `segmentation-api`, tag `latest` -> *Select image* -> *Create function*.

4. **Memory and timeout.** *Configuration* tab -> *General configuration* -> *Edit*: Memory **1024 MB**, Timeout **1 min 0 sec** -> *Save*. (Your course used 5 minutes for streaming; this API answers in milliseconds, but the very first request must load the model.)

5. **Cap concurrency** as in the course: *Configuration* -> *Concurrency and recursion detection* -> *Edit* -> *Reserve concurrency* -> **2** -> *Save*. Do not touch *Provisioned concurrency* (that one costs money). If the console refuses (new accounts often have a total limit of 10 and Lambda insists that 10 stay unreserved), skip this step; your budget alarm and the API key still protect you.

6. **API key.** Generate a long random key and keep it somewhere safe (a password manager):
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   *Configuration* -> *Environment variables* -> *Edit* -> *Add*: `API_KEY` = (the key) -> *Save*. You do not need `MODEL_PATH` or `PORT`: they are already set inside the image.

7. **Function URL.** *Configuration* -> *Function URL* -> *Create function URL*: Auth type **NONE** (our API key does the protecting; `AWS_IAM` would block the Streamlit app), expand *Additional settings*: leave CORS unchecked (the Streamlit server calls the API, not the browser), Invoke mode **BUFFERED** -> *Save*. Copy the URL: `https://<id>.lambda-url.<region>.on.aws/`.

8. **Test it** (the first call after idle can take 10-30 seconds, so allow a long timeout):
   ```bash
   export API_URL=https://<id>.lambda-url.<region>.on.aws     # no trailing slash
   export API_KEY=<your key>
   curl --max-time 90 $API_URL/live
   curl --max-time 90 $API_URL/health
   curl --max-time 90 -X POST $API_URL/predict -H "x-api-key: $API_KEY" -H "Content-Type: application/json" -d '{"recency":5,"frequency":30,"monetary":20000}'
   curl -s -o /dev/null -w "%{http_code}\n" -X POST $API_URL/predict -H "Content-Type: application/json" -d '{"recency":5,"frequency":30,"monetary":20000}'   # expect 401
   ```

### Code to write

None in this phase: it is all console and command-line work.

### Check

`/live` answers, `/health` shows `"source":"local"` and your model version, the call with the key returns a segment, and the call without it returns 401. In the Lambda console, *Monitor* -> *View CloudWatch logs* shows the requests.

**If you get 403 Forbidden on every call:** a public function URL needs a resource-based policy allowing `lambda:InvokeFunctionUrl` and `lambda:InvokeFunction`. The console normally adds both; check *Configuration* -> *Permissions* -> *Resource-based policy statements*. **"The image manifest ... is not supported" when creating the function:** rebuild with `--provenance=false --sbom=false` and push again. **Timeout on the first call:** raise the Timeout or retry; it is the cold start.

### Commit

```bash
git commit --allow-empty -m "Phase 16: API deployed to Lambda"
```

---

## Phase 17 - Deploy the Streamlit UI to Amazon ECS Express Mode

**Goal:** A public HTTPS address for the Streamlit page that talks to your Lambda API.

**Why:** **Right tool for each half.** Streamlit keeps a live WebSocket connection between the browser and a long-running server process. Lambda (and its Function URLs) answers single request/response calls and cannot hold that connection, so Streamlit cannot run on Lambda. Your course's service for this job, App Runner, no longer accepts new customers; AWS recommends **Amazon ECS Express Mode** as its replacement. It gives you the same experience: you supply a container image from ECR and it creates the Fargate service, load balancer, HTTPS address and scaling for you.

**Cost warning:** unlike Lambda, this runs 24 hours a day. You pay for the Fargate task plus an Application Load Balancer (the load balancer alone is roughly 16-20 USD a month in many regions; see the AWS pricing pages for yours). Plan to delete it when you are not demonstrating (phase 18).

### What to do

1. Build and push the UI image:
   ```bash
   set -a; source .env.aws; set +a
   export ECR="$AWS_ACCOUNT_ID.dkr.ecr.$DEFAULT_AWS_REGION.amazonaws.com"
   aws ecr get-login-password --region $DEFAULT_AWS_REGION | docker login --username AWS --password-stdin $ECR
   docker build --platform linux/amd64 --provenance=false --sbom=false -f docker/Dockerfile.ui -t segmentation-ui .
   docker tag segmentation-ui:latest $ECR/segmentation-ui:latest
   docker push $ECR/segmentation-ui:latest
   ```

2. Open the **Amazon ECS console** -> **Express Mode** -> **Create**.

3. **Container image URI:** `<account>.dkr.ecr.<region>.amazonaws.com/segmentation-ui:latest` (shown on the image in the ECR console).

4. **Task execution role** and **Infrastructure role:** for each, choose *Create new role* from the dropdown (the console creates them from AWS managed policies).

5. Expand **Additional configurations** and set:
   - Container port: **8501**
   - Health check path: **/_stcore/health**
   - Environment variables: `API_URL` = your Lambda Function URL **without** a trailing slash; `API_KEY` = the same key you set on Lambda
   - CPU / Memory: **0.5 vCPU / 1 GB**
   - Scaling: keep the minimum at 1 task and set the maximum low (for example 2) to cap cost
   - CloudWatch logs: leave enabled

6. Choose **Create** and watch the *Timeline view* on the *Resources* tab until provisioning finishes (a few minutes). The application URL then appears in the console.

7. Open the URL. In the sidebar click *Check connection* (the first call can take up to 30 seconds while Lambda wakes up), then try *One customer* and *Upload a file*.

### Code to write

None in this phase: it is all console and command-line work.

### Check

The page loads over HTTPS, *Check connection* shows `Connected - model v<N> (<algorithm>)`, and a segment appears for a customer.

**"Cannot reach the API":** `API_URL` must start with `https://` and have no trailing slash; the ECS task needs outbound internet access (Express Mode normally provides it). **"API rejected the API key":** the two `API_KEY` values differ. **Page loads but never finishes loading / WebSocket errors:** check the service's health check path is exactly `/_stcore/health` and the container port is 8501. **Upload returns 403:** this can happen when Streamlit runs behind some load balancers or proxies; only if you hit it, add `--server.enableXsrfProtection=false` to the command in `docker/Dockerfile.ui`, rebuild and push, and be aware it lowers a protection.

**Be aware:** the UI address is public and has no login. Anyone who finds it can use your model (that is why concurrency is capped and a budget alarm exists). Adding sign-in (for example Amazon Cognito with the load balancer) is outside this guide.

### Commit

```bash
git commit --allow-empty -m "Phase 17: UI deployed to ECS Express Mode"
```

---

## Phase 18 - Operate, update and clean up

**Goal:** Know how to ship a new model, what AWS shows you, what is *not* on AWS, and how to switch everything off.

**Why:** **Operations.** Deployment is not the end: models get retrained, and unused cloud resources keep billing.

### What to do

1. **Ship a new model** (after `dvc repro` produced a better champion):
   ```bash
   python scripts/export_champion.py
   set -a; source .env.aws; set +a
   export ECR="$AWS_ACCOUNT_ID.dkr.ecr.$DEFAULT_AWS_REGION.amazonaws.com"
   aws ecr get-login-password --region $DEFAULT_AWS_REGION | docker login --username AWS --password-stdin $ECR
   docker build --platform linux/amd64 --provenance=false --sbom=false -f docker/Dockerfile.lambda -t segmentation-api .
   docker tag segmentation-api:latest $ECR/segmentation-api:latest
   docker push $ECR/segmentation-api:latest
   aws lambda update-function-code --function-name segmentation-api --image-uri $ECR/segmentation-api:latest
   aws lambda wait function-updated --function-name segmentation-api
   ```
   The UI needs no change for a new model. If you change the UI code, rebuild and push `segmentation-ui` the same way and open the service in the ECS console and use its update/redeploy option so it pulls the new image (the exact button label may differ).

2. **Monitoring on AWS.** Prometheus and Grafana from phase 10 stay on your PC. On AWS use CloudWatch: Lambda console -> *Monitor* shows invocations, errors, duration and throttles; *View CloudWatch logs* shows every request. The UI's logs are in CloudWatch too (ECS service -> *Logs*). Drift checks (phase 8) are still something you run yourself on new data.

3. **What is NOT on AWS:** the MLflow server and registry, DVC storage, the tuning runs, Prometheus, Grafana. Your laptop remains the place where models are trained and chosen; AWS only serves a snapshot of the champion. Moving MLflow to AWS (EC2/RDS/S3) is a separate, larger project.

4. **Rotate the API key:** change `API_KEY` in the Lambda environment variables *and* in the ECS service's environment variables.

5. **Switch everything off** (in this order), when you finish a demo or the course:
   1. ECS console -> Express Mode -> your service -> *Delete*. Then open *EC2* -> *Load Balancers* and confirm no load balancer is left (it is the main cost).
   2. Lambda console -> `segmentation-api` -> *Actions* -> *Delete function*.
   3. ECR console -> delete the two repositories (images are billed per GB-month).
   4. IAM -> your user -> *Security credentials* -> make the access key *Inactive* and delete it.
   5. Next day, open *Billing* -> *Cost Explorer* and check that new charges stopped.

### Code to write

None in this phase: it is all console and command-line work.

### Check

After step 5 above: no running services, no load balancer, no Lambda function, no ECR images, and no new charges the following day.

### Commit

```bash
git add . && git commit -m "Phase 18: operations notes"
```

---

## Quick reference

| Problem | Fix |
|---|---|
| `docker login` / push says unauthorized | Run the `aws ecr get-login-password ...` line again (the login lasts 12 hours); check region and account number in `.env.aws` |
| Lambda: image manifest not supported | Build with `--platform linux/amd64 --provenance=false --sbom=false` and push again |
| Lambda URL returns 403 | Check the function's resource-based policy has `lambda:InvokeFunctionUrl` and `lambda:InvokeFunction` statements (the console adds them when you create a NONE-auth URL) |
| First request is very slow | Lambda cold start; allow 30-60 s, later requests are fast |
| UI says "Cannot reach the API" | `API_URL` must start with `https://` and have no trailing slash |
| UI says "API rejected the API key" | The `API_KEY` values on Lambda and on the ECS service differ |
| Unexpected AWS charges | The ECS service and its load balancer run 24/7; delete them (phase 18) |
| Git Bash mangles an AWS CLI argument that starts with `/` | Prefix the command with `MSYS_NO_PATHCONV=1` |
