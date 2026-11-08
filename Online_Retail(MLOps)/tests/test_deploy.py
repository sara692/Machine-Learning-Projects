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
