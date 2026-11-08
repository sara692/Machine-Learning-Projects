import importlib

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from scripts.make_synthetic_data import make
from src.data.prepare import clean, normalize_columns
from src.features.build_rfm import build_rfm
from src.monitoring.drift import category_psi, drift_report, psi


# ------------------------------------------------------------------ prepare
def raw_frame():
    return pd.DataFrame({
        "Invoice": ["1", "2", "C3", "4", "5", "1"],
        "StockCode": ["a"] * 6,
        "Description": ["x"] * 6,
        "Quantity": [2, 1, -1, 3, 1, 2],
        "InvoiceDate": ["2011-01-01 10:00"] * 6,
        "Price": [10.0, 5.0, 5.0, 0.0, 2.0, 10.0],
        "Customer ID": [1.0, 1.0, 1.0, 2.0, None, 1.0],
        "Country": ["UK"] * 6,
    })


def test_normalize_columns_maps_aliases():
    df = normalize_columns(raw_frame())
    assert {"invoice", "customer_id", "price", "invoice_date"} <= set(df.columns)


def test_normalize_columns_rejects_missing():
    with pytest.raises(ValueError):
        normalize_columns(pd.DataFrame({"foo": [1]}))


def test_clean_applies_all_rules():
    df, stats = clean(raw_frame())
    # duplicate row, cancelled invoice C3, zero price (inv 4) and missing customer (inv 5) are removed
    assert sorted(df["invoice"]) == ["1", "2"]
    assert (df["revenue"] == df["quantity"] * df["price"]).all()
    assert df["customer_id"].dtype == "int64"
    assert stats["customers"] == 1


def test_clean_raises_when_empty():
    with pytest.raises(ValueError):
        clean(raw_frame(), min_quantity=1000)


# ------------------------------------------------------------------ features
def test_build_rfm_values():
    df = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "invoice": ["a", "b", "c"],
        "invoice_date": pd.to_datetime(["2011-01-01", "2011-01-10", "2011-01-05"]),
        "revenue": [10.0, 20.0, 5.0],
    })
    rfm = build_rfm(df, snapshot_date="2011-01-11").set_index("customer_id")
    assert rfm.loc[1, "recency"] == 1 and rfm.loc[2, "recency"] == 6
    assert rfm.loc[1, "frequency"] == 2 and rfm.loc[1, "monetary"] == 30


def test_build_rfm_window_and_auto_snapshot():
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "invoice": ["a", "b"],
        "invoice_date": pd.to_datetime(["2011-01-01", "2011-03-01"]),
        "revenue": [1.0, 1.0],
    })
    rfm = build_rfm(df, "auto", start="2011-02-01")
    assert list(rfm["customer_id"]) == [2]


def test_end_to_end_synthetic_data():
    raw = make(200, seed=1)
    df, _ = clean(raw)
    rfm = build_rfm(df)
    assert len(rfm) > 150
    assert (rfm[["recency", "frequency", "monetary"]] >= 0).all().all()


# ------------------------------------------------------------------ drift
def test_psi_zero_for_same_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(2, 1, 5000)) > 0.5


def test_drift_report_flags_shifted_feature():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({f: rng.lognormal(3, 1, 3000) for f in ["recency", "frequency", "monetary"]})
    cur = ref.copy()
    cur["monetary"] = cur["monetary"] * 5
    rep = drift_report(ref, cur)
    assert rep["features"]["monetary"]["drifted"] and not rep["features"]["recency"]["drifted"]
    assert rep["share_of_drifted_features"] == pytest.approx(1 / 3)


def test_category_psi():
    a = np.array([0, 0, 1, 1])
    assert category_psi(a, a) == pytest.approx(0, abs=1e-9)
    assert category_psi(a, np.array([0, 0, 0, 1])) > 0


# ------------------------------------------------------------------ API
@pytest.fixture()
def client(tmp_path, monkeypatch):
    """API wired to a throw-away MLflow store holding a tiny registered champion model."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    exp = mlflow.create_experiment("t", artifact_location=(tmp_path / "art").as_uri())
    rng = np.random.default_rng(0)
    X = rng.lognormal(3, 1, size=(200, 3))
    pipe = Pipeline([("log", FunctionTransformer(np.log1p, validate=True)), ("scaler", StandardScaler()),
                     ("model", KMeans(3, n_init=5, random_state=0))]).fit(X)
    with mlflow.start_run(experiment_id=exp):
        info = mlflow.sklearn.log_model(pipe, "model", registered_model_name="customer-segmentation",
                                        serialization_format="cloudpickle")
    mlflow.MlflowClient().set_registered_model_alias("customer-segmentation", "champion",
                                                     info.registered_model_version)
    import src.serving.app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_api_predict_and_metrics(client):
    assert client.get("/live").status_code == 200
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["algorithm"] == "KMeans"

    r = client.post("/predict", json={"recency": 10, "frequency": 5, "monetary": 500})
    assert r.status_code == 200
    body = r.json()
    assert body["cluster"] in (0, 1, 2) and body["distance_to_centroid"] is not None

    rb = client.post("/predict/batch", json=[{"recency": 1, "frequency": 1, "monetary": 10},
                                             {"recency": 300, "frequency": 20, "monetary": 9000}])
    assert rb.status_code == 200 and len(rb.json()["clusters"]) == 2

    assert client.post("/predict", json={"recency": -1, "frequency": 1, "monetary": 1}).status_code == 422
    text = client.get("/metrics").text
    assert "seg_requests_total" in text and "seg_model_version 1.0" in text
    assert client.post("/reload").status_code == 200


def test_api_503_without_model(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    import src.serving.app as app_module
    importlib.reload(app_module)
    c = TestClient(app_module.app)
    assert c.get("/live").status_code == 200
    assert c.get("/health").status_code == 503
