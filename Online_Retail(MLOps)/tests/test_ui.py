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
