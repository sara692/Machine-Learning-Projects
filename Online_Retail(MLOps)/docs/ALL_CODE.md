# All project files with full code

Every file of the project in one place, in the order of the folder tree. Create each file at the path shown (or just unzip `retail-mlops.zip`, which contains the same files). Empty `__init__.py` files and `.gitkeep` placeholders are listed at the end.

## `.gitignore`

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
```

## `dvc.yaml`

```yaml
stages:
  prepare:
    cmd: python src/data/prepare.py
    deps:
      - src/data/prepare.py
      - data/raw/online_retail_II.csv
    params:
      - configs/params.yaml:
          - prepare
    outs:
      - data/processed/clean.parquet

  features:
    cmd: python src/features/build_rfm.py
    deps:
      - src/features/build_rfm.py
      - data/processed/clean.parquet
    params:
      - configs/params.yaml:
          - features
    outs:
      - data/features/rfm.parquet

  tune:
    # needs the MLflow server running (see configs/tuning.yaml -> mlflow.tracking_uri)
    cmd: python src/models/tune.py
    deps:
      - src/models/tune.py
      - configs/tuning.yaml
      - data/features/rfm.parquet
    outs:
      - reports/leaderboard.csv
      - reports/cluster_sizes.csv
    metrics:
      - reports/metrics.json:
          cache: false

  evaluate:
    cmd: python src/models/evaluate.py
    deps:
      - src/models/evaluate.py
      - reports/metrics.json
      - data/features/rfm.parquet
    outs:
      - reports/cluster_profile.csv
      - reports/customer_segments.csv
```

## `Makefile`

```makefile
# On Windows Git Bash you can run the commands below by hand (or install make: `choco install make`).
.PHONY: setup data synthetic prepare features mlflow tune evaluate pipeline api drift-demo up down test lint smoke

setup:
	pip install -r requirements-dev.txt

data:            ## download the real dataset from Kaggle (needs ~/.kaggle/kaggle.json)
	kaggle datasets download -d mashlyn/online-retail-ii-uci -p data/raw --unzip

synthetic:       ## fake data with the same schema, for trying the pipeline
	python scripts/make_synthetic_data.py --customers 1500 --output data/raw/online_retail_II.csv

prepare:
	python src/data/prepare.py

features:
	python src/features/build_rfm.py

mlflow:          ## local MLflow server (UI at http://localhost:5000)
	mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

tune:
	python src/models/tune.py

evaluate:
	python src/models/evaluate.py

pipeline:        ## prepare -> features -> tune -> evaluate via DVC
	dvc repro

api:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

drift-demo:      ## compare two 12-month windows (shows the mechanism, see README caveat)
	python src/features/build_rfm.py --start 2009-12-01 --end 2010-12-01 --output data/features/rfm_ref.parquet
	python src/features/build_rfm.py --start 2010-12-01 --end 2011-12-10 --output data/features/rfm_current.parquet
	python src/monitoring/drift.py --reference data/features/rfm_ref.parquet --current data/features/rfm_current.parquet

up:
	docker compose -f docker/docker-compose.yml up -d --build

down:
	docker compose -f docker/docker-compose.yml down

test:
	pytest -q

lint:
	ruff check src tests scripts

smoke:           ## tiny end-to-end run on synthetic data (what CI does)
	python scripts/make_synthetic_data.py --customers 600 --output data/raw/synthetic.csv
	python src/data/prepare.py --input data/raw/synthetic.csv
	python src/features/build_rfm.py
	python src/models/tune.py --config configs/tuning_smoke.yaml
	python src/models/evaluate.py --config configs/tuning_smoke.yaml
	python scripts/check_metrics.py --min-silhouette 0.2
```

## `pytest.ini`

```ini
[pytest]
pythonpath = .
testpaths = tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::FutureWarning
    ignore::UserWarning
```

## `requirements-dev.txt`

```text
-r requirements.txt
pytest>=8
httpx>=0.27
ruff>=0.5
jupyter>=1.0
```

## `requirements-serve.txt`

```text
# Minimal dependencies for the API container.
# Keep scikit-learn / pandas / numpy / mlflow versions the same as the ones used for training
# (check with `pip freeze`), otherwise loading the registered model can fail.
pandas>=2.2
numpy>=1.26
scipy>=1.11
scikit-learn>=1.5
mlflow>=3.0,<4
fastapi>=0.110
uvicorn>=0.29
prometheus-client>=0.20
```

## `requirements.txt`

```text
pandas>=2.2
numpy>=1.26
scipy>=1.11
scikit-learn>=1.5
pyarrow>=15
pyyaml>=6
mlflow>=3.0,<4
dvc>=3.50
fastapi>=0.110
uvicorn>=0.29
prometheus-client>=0.20
matplotlib>=3.8
kaggle>=1.6
```

## `ruff.toml`

```toml
line-length = 110
extend-exclude = ["notebooks", "mlruns", "mlartifacts"]

[lint]
select = ["E", "F", "I"]
```

## `.github/workflows/ci.yml`

```yaml
name: ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: requirements-dev.txt

      - name: Install
        run: pip install -r requirements-dev.txt

      - name: Lint
        run: ruff check src tests scripts

      - name: Unit + API tests
        run: pytest -q

      # Runs the whole pipeline on synthetic data (the real dataset is not stored in git).
      - name: Smoke pipeline (synthetic data)
        run: |
          python scripts/make_synthetic_data.py --customers 600 --output data/raw/synthetic.csv
          python src/data/prepare.py --input data/raw/synthetic.csv
          python src/features/build_rfm.py
          python src/models/tune.py --config configs/tuning_smoke.yaml
          python src/models/evaluate.py --config configs/tuning_smoke.yaml

      - name: Quality gate
        run: python scripts/check_metrics.py --min-silhouette 0.20

  build-images:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build API image
        run: docker build -f docker/Dockerfile.api -t segmentation-api:${{ github.sha }} .
      - name: Build MLflow image
        run: docker build -f docker/Dockerfile.mlflow -t segmentation-mlflow:${{ github.sha }} docker
```

## `configs/params.yaml`

```yaml
prepare:
  input: data/raw/online_retail_II.csv
  output: data/processed/clean.parquet
  min_quantity: 1          # drop returns / negative quantities
  min_price: 0.01          # drop free items and price adjustments
  drop_cancelled: true     # drop invoices starting with "C"
  drop_duplicates: true    # drop exact duplicate rows

features:
  input: data/processed/clean.parquet
  output: data/features/rfm.parquet
  snapshot_date: auto      # "auto" = day after the last invoice, or e.g. "2011-12-10"
```

## `configs/tuning.yaml`

```yaml
data:
  path: data/features/rfm.parquet
  features: [recency, frequency, monetary]
  log_transform: true          # log1p before scaling (RFM is heavily skewed)

mlflow:
  tracking_uri: http://localhost:5000
  experiment: customer-segmentation-tuning
  registered_model: customer-segmentation
  champion_alias: champion

evaluation:
  silhouette_sample_size: 10000   # silhouette is O(n^2); sample for speed
  selection_metric: silhouette    # higher is better
  min_clusters: 2
  max_clusters: 12
  min_cluster_share: 0.02         # reject solutions with a cluster < 2% of customers
  max_noise_share: 0.10           # DBSCAN: reject if > 10% flagged as noise
  random_state: 42

# Grids are exhaustive; set max_trials_per_algo to randomly sample if too big
max_trials_per_algo: 60

search_space:
  kmeans:
    n_clusters: [2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_init: [10, 20]
    init: [k-means++, random]
  gmm:
    n_components: [2, 3, 4, 5, 6, 7, 8, 9, 10]
    covariance_type: [full, tied, diag, spherical]
  birch:
    n_clusters: [2, 3, 4, 5, 6, 7, 8, 9, 10]
    threshold: [0.1, 0.3, 0.5, 1.0]
    branching_factor: [50]
  agglomerative:
    n_clusters: [2, 3, 4, 5, 6, 7, 8, 9, 10]
    linkage: [ward, complete, average]
  dbscan:
    eps: [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    min_samples: [5, 10, 20, 50]
```

## `configs/tuning_smoke.yaml`

```yaml
# Small, fast config used by CI and quick local checks (sqlite store, tiny grids)
data:
  path: data/features/rfm.parquet
  features: [recency, frequency, monetary]
  log_transform: true

mlflow:
  tracking_uri: sqlite:///mlflow_smoke.db
  experiment: customer-segmentation-smoke
  registered_model: customer-segmentation
  champion_alias: champion

evaluation:
  silhouette_sample_size: 5000
  selection_metric: silhouette
  min_clusters: 2
  max_clusters: 12
  min_cluster_share: 0.02
  max_noise_share: 0.10
  random_state: 42

max_trials_per_algo: 6

search_space:
  kmeans:
    n_clusters: [3, 4, 5]
    n_init: [10]
    init: [k-means++]
  gmm:
    n_components: [3, 4]
    covariance_type: [full, diag]
  birch:
    n_clusters: [3, 4]
    threshold: [0.5]
    branching_factor: [50]
  agglomerative:
    n_clusters: [3, 4]
    linkage: [ward]
  dbscan:
    eps: [0.3, 0.5]
    min_samples: [10]
```

## `docker/docker-compose.yml`

```yaml
name: segmentation-mlops

services:
  mlflow:
    build:
      context: .
      dockerfile: Dockerfile.mlflow
    # --artifacts-destination + the default artifact proxy let clients (your PC, the API container)
    # upload/download models through the server, so they don't need access to the container's disk.
    # --allowed-hosts "*" lets other containers reach it as http://mlflow:5000 (fine for local dev only).
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --artifacts-destination /mlflow/artifacts
      --allowed-hosts "*"
    ports: ["5000:5000"]
    volumes: ["mlflow-data:/mlflow"]
    restart: unless-stopped

  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports: ["8000:8000"]
    depends_on: [mlflow]
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ../monitoring/prometheus:/etc/prometheus:ro
      - prometheus-data:/prometheus
    ports: ["9090:9090"]
    depends_on: [api]
    restart: unless-stopped

  pushgateway:
    image: prom/pushgateway:latest
    ports: ["9091:9091"]
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: admin   # change this
    volumes:
      - ../monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ../monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    ports: ["3000:3000"]
    depends_on: [prometheus]
    restart: unless-stopped

volumes:
  mlflow-data:
  prometheus-data:
  grafana-data:
```

## `docker/Dockerfile.api`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src ./src
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## `docker/Dockerfile.mlflow`

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir "mlflow>=3.0,<4"
EXPOSE 5000
```

## `monitoring/grafana/dashboards/segmentation.json`

```json
{
  "uid": "segmentation-overview",
  "title": "Customer Segmentation - API & Model Monitoring",
  "tags": [
    "mlops",
    "segmentation"
  ],
  "timezone": "browser",
  "schemaVersion": 39,
  "version": 1,
  "refresh": "10s",
  "time": {
    "from": "now-1h",
    "to": "now"
  },
  "panels": [
    {
      "id": 1,
      "type": "timeseries",
      "title": "Requests / sec",
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "gridPos": {
        "x": 0,
        "y": 0,
        "w": 8,
        "h": 8
      },
      "fieldConfig": {
        "defaults": {
          "unit": "reqps"
        },
        "overrides": []
      },
      "targets": [
        {
          "refId": "A",
          "expr": "rate(seg_requests_total[1m])",
          "legendFormat": "requests",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        }
      ]
    },
    {
      "id": 2,
      "type": "timeseries",
      "title": "Error rate",
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "gridPos": {
        "x": 8,
        "y": 0,
        "w": 8,
        "h": 8
      },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit"
        },
        "overrides": []
      },
      "targets": [
        {
          "refId": "A",
          "expr": "rate(seg_errors_total[5m]) / clamp_min(rate(seg_requests_total[5m]), 1e-9)",
          "legendFormat": "errors",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        }
      ]
    },
    {
      "id": 3,
      "type": "timeseries",
      "title": "Latency p50 / p95",
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "gridPos": {
        "x": 16,
        "y": 0,
        "w": 8,
        "h": 8
      },
      "fieldConfig": {
        "defaults": {
          "unit": "s"
        },
        "overrides": []
      },
      "targets": [
        {
          "refId": "A",
          "expr": "histogram_quantile(0.5, sum by (le) (rate(seg_latency_seconds_bucket[5m])))",
          "legendFormat": "p50",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        },
        {
          "refId": "B",
          "expr": "histogram_quantile(0.95, sum by (le) (rate(seg_latency_seconds_bucket[5m])))",
          "legendFormat": "p95",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        }
      ]
    },
    {
      "id": 4,
      "type": "timeseries",
      "title": "Segment mix (predictions / sec)",
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "gridPos": {
        "x": 0,
        "y": 8,
        "w": 8,
        "h": 8
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short"
        },
        "overrides": []
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (cluster) (rate(seg_predictions_by_cluster_total[15m]))",
          "legendFormat": "segment {{cluster}}",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        }
      ]
    },
    {
      "id": 5,
      "type": "timeseries",
      "title": "Distance to centroid p50 / p95 (KMeans)",
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "gridPos": {
        "x": 8,
        "y": 8,
        "w": 8,
        "h": 8
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short"
        },
        "overrides": []
      },
      "targets": [
        {
          "refId": "A",
          "expr": "histogram_quantile(0.5, sum by (le) (rate(seg_distance_to_centroid_bucket[15m])))",
          "legendFormat": "p50",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        },
        {
          "refId": "B",
          "expr": "histogram_quantile(0.95, sum by (le) (rate(seg_distance_to_centroid_bucket[15m])))",
          "legendFormat": "p95",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        }
      ]
    },
    {
      "id": 6,
      "type": "timeseries",
      "title": "Feature PSI (drift)",
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "gridPos": {
        "x": 16,
        "y": 8,
        "w": 8,
        "h": 8
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short"
        },
        "overrides": []
      },
      "targets": [
        {
          "refId": "A",
          "expr": "seg_feature_psi",
          "legendFormat": "{{feature}}",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        }
      ]
    },
    {
      "id": 7,
      "type": "stat",
      "title": "Champion model version",
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "gridPos": {
        "x": 0,
        "y": 16,
        "w": 6,
        "h": 5
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short"
        },
        "overrides": []
      },
      "targets": [
        {
          "refId": "A",
          "expr": "seg_model_version",
          "legendFormat": "version",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        }
      ]
    },
    {
      "id": 8,
      "type": "stat",
      "title": "Share of drifted features",
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "gridPos": {
        "x": 6,
        "y": 16,
        "w": 6,
        "h": 5
      },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit"
        },
        "overrides": []
      },
      "targets": [
        {
          "refId": "A",
          "expr": "seg_data_drift_share",
          "legendFormat": "drift share",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        }
      ]
    },
    {
      "id": 9,
      "type": "stat",
      "title": "Cluster-share PSI",
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "gridPos": {
        "x": 12,
        "y": 16,
        "w": 6,
        "h": 5
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short"
        },
        "overrides": []
      },
      "targets": [
        {
          "refId": "A",
          "expr": "seg_cluster_share_psi",
          "legendFormat": "psi",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        }
      ]
    },
    {
      "id": 10,
      "type": "stat",
      "title": "API up",
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "gridPos": {
        "x": 18,
        "y": 16,
        "w": 6,
        "h": 5
      },
      "fieldConfig": {
        "defaults": {
          "unit": "short"
        },
        "overrides": []
      },
      "targets": [
        {
          "refId": "A",
          "expr": "up{job=\"segmentation-api\"}",
          "legendFormat": "up",
          "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
          }
        }
      ]
    }
  ]
}
```

## `monitoring/grafana/provisioning/dashboards/dashboards.yml`

```yaml
apiVersion: 1

providers:
  - name: segmentation
    folder: Segmentation
    type: file
    disableDeletion: false
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

## `monitoring/grafana/provisioning/datasources/prometheus.yml`

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

## `monitoring/prometheus/alerts.yml`

```yaml
groups:
  - name: segmentation-api
    rules:
      - alert: ApiDown
        expr: up{job="segmentation-api"} == 0
        for: 2m
        labels: {severity: critical}
        annotations:
          summary: "Segmentation API is not reachable"

      - alert: ModelNotLoaded
        expr: seg_model_version == 0
        for: 5m
        labels: {severity: critical}
        annotations:
          summary: "API has no champion model loaded (run tune.py, then POST /reload)"

      - alert: HighErrorRate
        expr: rate(seg_errors_total[5m]) / clamp_min(rate(seg_requests_total[5m]), 1e-9) > 0.05
        for: 5m
        labels: {severity: critical}
        annotations:
          summary: "More than 5% of prediction requests are failing"

      - alert: HighLatencyP95
        expr: histogram_quantile(0.95, sum by (le) (rate(seg_latency_seconds_bucket[5m]))) > 0.3
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: "p95 prediction latency above 300 ms"

  - name: segmentation-model-quality
    rules:
      - alert: DataDriftDetected
        expr: seg_data_drift_share{job="drift-job"} >= 0.34
        for: 5m
        labels: {severity: warning}
        annotations:
          summary: "At least one RFM feature drifted (PSI >= 0.2) - consider retraining"

      - alert: ClusterShareShift
        expr: seg_cluster_share_psi{job="drift-job"} >= 0.2
        for: 5m
        labels: {severity: warning}
        annotations:
          summary: "Segment proportions shifted a lot versus training data"

      - alert: DistanceToCentroidGrowing
        expr: histogram_quantile(0.95, sum by (le) (rate(seg_distance_to_centroid_bucket[1h]))) > 5
        for: 1h
        labels: {severity: warning}
        annotations:
          summary: "Customers are landing far from their centroid (KMeans champion only)"
```

## `monitoring/prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/alerts.yml

scrape_configs:
  - job_name: segmentation-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]

  - job_name: pushgateway
    honor_labels: true          # keep the job="drift-job" label set by drift.py
    static_configs:
      - targets: ["pushgateway:9091"]
```

## `notebooks/01_eda.ipynb`
(notebook shown as its cells; the real file is a .ipynb JSON)

```python
# --- markdown ---
# # 01 - Exploratory analysis
# Run the notebook from the project root after `python src/data/prepare.py` and `python src/features/build_rfm.py`.

# --- cell ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

clean = pd.read_parquet("data/processed/clean.parquet")
rfm = pd.read_parquet("data/features/rfm.parquet")
print(clean.shape, rfm.shape)
clean.head()

# --- markdown ---
# ## Transactions

# --- cell ---
print(clean["invoice_date"].min(), "->", clean["invoice_date"].max())
print("customers:", clean["customer_id"].nunique(), " invoices:", clean["invoice"].nunique())
monthly = clean.set_index("invoice_date")["revenue"].resample("MS").sum()
monthly.plot(title="Monthly revenue", figsize=(9, 3)); plt.show()

# --- markdown ---
# ## RFM distributions (heavily right-skewed -> we log-transform before clustering)

# --- cell ---
fig, ax = plt.subplots(2, 3, figsize=(12, 6))
for i, c in enumerate(["recency", "frequency", "monetary"]):
    rfm[c].plot.hist(bins=50, ax=ax[0, i], title=c)
    np.log1p(rfm[c]).plot.hist(bins=50, ax=ax[1, i], title=f"log1p({c})")
plt.tight_layout(); plt.show()
rfm[["recency", "frequency", "monetary"]].describe().round(2)

# --- markdown ---
# ## Correlation

# --- cell ---
np.log1p(rfm[["recency", "frequency", "monetary"]]).corr().round(2)

# --- markdown ---
# ## Quick look at k (elbow + silhouette) - the real search is in `src/models/tune.py`

# --- cell ---
X = StandardScaler().fit_transform(np.log1p(rfm[["recency", "frequency", "monetary"]]))
ks, inertia, sil = range(2, 11), [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_, sample_size=min(5000, len(X)), random_state=42))
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(list(ks), inertia, marker="o"); ax[0].set_title("Elbow (inertia)")
ax[1].plot(list(ks), sil, marker="o"); ax[1].set_title("Silhouette")
plt.show()
```

## `scripts/check_metrics.py`

```python
"""Quality gate: fail (exit 1) if the tuned model is not good enough.

Run:  python scripts/check_metrics.py --min-silhouette 0.30 [--min-clusters 2]
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="reports/metrics.json")
    ap.add_argument("--min-silhouette", type=float, default=0.30)
    ap.add_argument("--min-clusters", type=int, default=2)
    args = ap.parse_args()

    m = json.load(open(args.metrics))
    ok = m["silhouette"] >= args.min_silhouette and m["n_clusters"] >= args.min_clusters
    print(f"{m['algorithm']}: silhouette={m['silhouette']:.3f} (min {args.min_silhouette}), "
          f"n_clusters={m['n_clusters']} (min {args.min_clusters}) -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

## `scripts/make_synthetic_data.py`

```python
"""Generate a fake Online-Retail-II-style transactions CSV (same columns, same kinds of mess).

Useful for trying the whole pipeline / CI without downloading the Kaggle file.
Run:  python scripts/make_synthetic_data.py --customers 1500 --output data/raw/synthetic.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# share of customers, mean #orders, mean basket value, typical days since last order
SEGMENTS = [(0.40, 1.5, 150, 300), (0.30, 4, 400, 120), (0.22, 10, 900, 40), (0.08, 30, 3000, 10)]
COUNTRIES = ["United Kingdom"] * 8 + ["Germany", "France", "EIRE"]


def make(n_customers: int, seed: int = 0, start="2009-12-01", end="2011-12-09") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    span = (t1 - t0).days
    shares = np.array([s[0] for s in SEGMENTS])
    rows, inv_no = [], 500000

    for cust in range(12347, 12347 + n_customers):
        _, orders, basket, rec = SEGMENTS[rng.choice(len(SEGMENTS), p=shares / shares.sum())]
        n_orders = 1 + rng.poisson(max(orders - 1, 0.1))
        last_gap = int(min(rng.exponential(rec), span - 1))
        last_day = t1 - pd.Timedelta(days=last_gap)
        first_day = t0 + pd.Timedelta(days=int(rng.integers(0, max(span - last_gap, 1))))
        first_day = min(first_day, last_day)
        country = COUNTRIES[int(rng.integers(len(COUNTRIES)))]
        for k in range(n_orders):
            frac = 1.0 if k == n_orders - 1 else rng.random()
            when = first_day + (last_day - first_day) * frac
            when = when.normalize() + pd.Timedelta(hours=int(rng.integers(8, 18)),
                                                   minutes=int(rng.integers(0, 60)))
            inv_no += 1
            n_lines = 1 + rng.poisson(3)
            line_value = basket / n_lines
            for _ in range(n_lines):
                price = round(float(rng.lognormal(1.2, 0.6)), 2)
                qty = max(1, int(round(rng.lognormal(np.log(max(line_value / price, 1)), 0.4))))
                rows.append((str(inv_no), f"{int(rng.integers(10000, 99999))}", "GIFT ITEM", qty,
                             when, price, float(cust), country))
            if rng.random() < 0.04:  # cancellation of the whole invoice
                inv_no += 1
                rows.append((f"C{inv_no}", "10001", "GIFT ITEM", -qty, when, price, float(cust), country))

    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "Price", "Customer ID", "Country"])
    # real-world mess
    guest = rng.random(len(df)) < 0.02
    df.loc[guest, "Customer ID"] = np.nan                    # guest checkouts
    free = rng.random(len(df)) < 0.005
    df.loc[free, "Price"] = 0.0                              # zero-price rows
    dupes = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)          # exact duplicates
    return df.sort_values("InvoiceDate").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/raw/synthetic.csv")
    args = ap.parse_args()
    df = make(args.customers, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(df)} rows, {df['Customer ID'].nunique()} customers")


if __name__ == "__main__":
    main()
```

## `src/data/prepare.py`

```python
"""Clean raw Online Retail II transactions.

Input : data/raw/online_retail_II.csv   (Kaggle: mashlyn/online-retail-ii-uci)
Output: data/processed/clean.parquet

Run:  python src/data/prepare.py [--input PATH] [--output PATH] [--params configs/params.yaml]
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

# Raw column names differ between releases of the dataset -> map them to one schema.
COLUMN_ALIASES = {
    "invoice": "invoice",
    "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "price",
    "unitprice": "price",
    "customerid": "customer_id",
    "country": "country",
}
REQUIRED = ["invoice", "quantity", "invoice_date", "price", "customer_id"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to snake_case names, e.g. 'Customer ID' -> 'customer_id'."""
    mapping = {}
    for col in df.columns:
        key = re.sub(r"[\s_]+", "", str(col).lower())
        if key in COLUMN_ALIASES:
            mapping[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=mapping)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)}")
    return df


def clean(df: pd.DataFrame, min_quantity=1, min_price=0.01,
          drop_cancelled=True, drop_duplicates=True):
    """Apply cleaning rules. Returns (clean_df, stats_dict)."""
    df = normalize_columns(df)
    stats = {"rows_raw": len(df)}

    if drop_duplicates:
        df = df.drop_duplicates()
    stats["rows_after_dedup"] = len(df)

    df = df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df["invoice"] = df["invoice"].astype(str)

    df = df.dropna(subset=["invoice_date", "quantity", "price", "customer_id"])
    stats["rows_after_dropna"] = len(df)

    if drop_cancelled:
        df = df[~df["invoice"].str.upper().str.startswith("C")]
    stats["rows_after_cancelled"] = len(df)

    df = df[(df["quantity"] >= min_quantity) & (df["price"] >= min_price)]
    stats["rows_after_filters"] = len(df)

    df["customer_id"] = df["customer_id"].astype("int64")
    df["revenue"] = df["quantity"] * df["price"]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows left after cleaning - check the input file and filters.")
    stats["customers"] = int(df["customer_id"].nunique())
    stats["date_min"] = str(df["invoice_date"].min())
    stats["date_max"] = str(df["invoice_date"].max())
    return df, stats


def read_raw(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download it with:\n"
            "  kaggle datasets download -d mashlyn/online-retail-ii-uci -p data/raw --unzip")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, encoding="utf-8", encoding_errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="configs/params.yaml")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    p = yaml.safe_load(open(args.params))["prepare"]
    src = args.input or p["input"]
    dst = args.output or p["output"]

    raw = read_raw(src)
    df, stats = clean(raw, p["min_quantity"], p["min_price"],
                      p["drop_cancelled"], p.get("drop_duplicates", True))

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    print(f"Wrote {dst}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

## `src/features/build_rfm.py`

```python
"""Build one Recency / Frequency / Monetary row per customer.

Input : data/processed/clean.parquet
Output: data/features/rfm.parquet   (customer_id, recency, frequency, monetary)

Run:  python src/features/build_rfm.py [--snapshot-date 2011-12-10] [--start D] [--end D] [--output PATH]
  --start/--end restrict transactions to a time window (used to simulate "new" data for drift checks).
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def build_rfm(df: pd.DataFrame, snapshot_date=None, start=None, end=None) -> pd.DataFrame:
    """recency  = days between snapshot and the customer's last purchase
    frequency = number of distinct invoices
    monetary  = total revenue"""
    if start:
        df = df[df["invoice_date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["invoice_date"] < pd.Timestamp(end)]
    if df.empty:
        raise ValueError("No transactions in the selected window.")

    if snapshot_date in (None, "auto"):
        snapshot = df["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    else:
        snapshot = pd.Timestamp(snapshot_date)

    rfm = (df.groupby("customer_id")
             .agg(last_purchase=("invoice_date", "max"),
                  frequency=("invoice", "nunique"),
                  monetary=("revenue", "sum"))
             .reset_index())
    rfm["recency"] = (snapshot - rfm["last_purchase"].dt.normalize()).dt.days.clip(lower=0)
    rfm = rfm[rfm["monetary"] > 0]
    return rfm[["customer_id", "recency", "frequency", "monetary"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="configs/params.yaml")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--snapshot-date", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    p = yaml.safe_load(open(args.params))["features"]
    src = args.input or p["input"]
    dst = args.output or p["output"]
    snap = args.snapshot_date or p.get("snapshot_date", "auto")

    df = pd.read_parquet(src)
    rfm = build_rfm(df, snap, args.start, args.end)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(dst, index=False)
    print(f"Wrote {dst}: {len(rfm)} customers")
    print(rfm[["recency", "frequency", "monetary"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
```

## `src/models/evaluate.py`

```python
"""Profile the registered champion model: segment sizes, RFM averages and persona hints.

Loads models:/<registered_model>@<champion_alias> from MLflow, assigns every customer to a segment and writes
  reports/cluster_profile.csv     one row per segment
  reports/customer_segments.csv   customer_id + segment
It also logs both files to MLflow (run "evaluate-champion").

Run:  python src/models/evaluate.py [--config configs/tuning.yaml] [--tracking-uri URI]
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient


def persona_hint(row, z) -> str:
    """Heuristic label from how a segment's median RFM compares with all customers (z-scores)."""
    r, f, m = -z["recency"], z["frequency"], z["monetary"]  # r high = bought recently
    if f > 0.5 and m > 0.5 and r > 0:
        return "Champions / VIP"
    if r < -0.5 and f < 0:
        return "Lost / Hibernating"
    if r > 0.3 and f <= 0.5:
        return "Recent / Promising"
    if f > 0.3 or m > 0.3:
        return "Loyal / Big spenders"
    return "Regular / Needs attention"


def profile_segments(rfm: pd.DataFrame, labels: np.ndarray, features) -> pd.DataFrame:
    df = rfm.copy()
    df["segment"] = labels
    grp = df.groupby("segment")
    prof = grp.size().rename("customers").to_frame()
    prof["share"] = (prof["customers"] / len(df)).round(4)
    for f in features:
        prof[f"{f}_mean"] = grp[f].mean().round(2)
        prof[f"{f}_median"] = grp[f].median().round(2)
    # z-score of each segment's median (in log space) against the whole population
    logs = np.log1p(df[features])
    mu, sd = logs.mean(), logs.std().replace(0, 1)
    hints = []
    for seg in prof.index:
        med = np.log1p(grp[features].median().loc[seg])
        z = ((med - mu) / sd).to_dict()
        hints.append(persona_hint(None, z))
    prof["persona_hint"] = hints
    return prof.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tuning.yaml")
    ap.add_argument("--tracking-uri", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name = cfg["mlflow"]["registered_model"]
    alias = cfg["mlflow"]["champion_alias"]
    feats = cfg["data"]["features"]

    mlflow.set_tracking_uri(args.tracking_uri or cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    rfm = pd.read_parquet(cfg["data"]["path"])
    pipe = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    version = MlflowClient().get_model_version_by_alias(name, alias).version
    labels = pipe.predict(rfm[feats].astype(float).to_numpy())

    prof = profile_segments(rfm[feats], labels, feats)
    Path("reports").mkdir(exist_ok=True)
    prof.to_csv("reports/cluster_profile.csv", index=False)
    out = rfm[["customer_id"]].copy() if "customer_id" in rfm else pd.DataFrame(index=rfm.index)
    out["segment"] = labels
    out.to_csv("reports/customer_segments.csv", index=False)

    with mlflow.start_run(run_name="evaluate-champion"):
        mlflow.set_tags({"model": name, "model_version": str(version), "alias": alias})
        mlflow.log_metric("n_segments", len(prof))
        mlflow.log_artifact("reports/cluster_profile.csv")
        mlflow.log_artifact("reports/customer_segments.csv")

    print(f"Champion '{name}' v{version}")
    print(prof.to_string(index=False))


if __name__ == "__main__":
    main()
```

## `src/models/tune.py`

```python
"""Try several clustering algorithms, tune hyperparameters, log everything to MLflow,
pick the best configuration, refit it and register it in the MLflow Model Registry.

Run:  python src/models/tune.py [--config configs/tuning.yaml] [--tracking-uri URI]
"""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.cluster import DBSCAN, AgglomerativeClustering, Birch, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED_DEFAULT = 42


# ----------------------------------------------------------------------------- models
def build_estimator(algo: str, params: dict, seed: int):
    if algo == "kmeans":
        return KMeans(random_state=seed, **params)
    if algo == "gmm":
        return GaussianMixture(random_state=seed, **params)
    if algo == "birch":
        return Birch(**params)
    if algo == "agglomerative":
        return AgglomerativeClustering(**params)
    if algo == "dbscan":
        return DBSCAN(**params)
    raise ValueError(f"Unknown algorithm: {algo}")


def get_labels(est, X):
    """fit and return labels; hides API differences between estimators."""
    if hasattr(est, "fit_predict"):
        return est.fit_predict(X)
    est.fit(X)
    return est.predict(X)


# ----------------------------------------------------------------------------- scoring
def score(X, labels, cfg, rng_seed):
    """Return metrics dict, or None if the solution is invalid."""
    ev = cfg["evaluation"]
    noise_mask = labels == -1
    noise_share = float(noise_mask.mean())
    keep = ~noise_mask
    kept_labels = labels[keep]
    uniq, counts = np.unique(kept_labels, return_counts=True)
    n_clusters = len(uniq)

    reason = None
    if n_clusters < ev["min_clusters"]:
        reason = "too_few_clusters"
    elif n_clusters > ev["max_clusters"]:
        reason = "too_many_clusters"
    elif noise_share > ev["max_noise_share"]:
        reason = "too_much_noise"
    elif counts.min() / keep.sum() < ev["min_cluster_share"]:
        reason = "tiny_cluster"

    base = {"n_clusters": n_clusters, "noise_share": noise_share,
            "min_cluster_share": float(counts.min() / keep.sum()) if keep.sum() else 0.0}
    if reason:
        return base, reason

    Xk = X[keep]
    sample = min(ev["silhouette_sample_size"], len(Xk))
    sil = silhouette_score(Xk, kept_labels, sample_size=sample, random_state=rng_seed)
    base.update({
        "silhouette": float(sil),
        "davies_bouldin": float(davies_bouldin_score(Xk, kept_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Xk, kept_labels)),
    })
    return base, None


# ----------------------------------------------------------------------------- provenance
def run_provenance(data_path: str) -> dict:
    """Tags that make a run reproducible: git commit + hash of the exact data file used."""
    tags = {}
    try:
        tags["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        tags["git_commit"] = "unknown"
    try:
        tags["data_sha256"] = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()[:16]
    except Exception:
        tags["data_sha256"] = "unknown"
    return tags


# ----------------------------------------------------------------------------- data
def load_features(cfg):
    d = cfg["data"]
    df = pd.read_parquet(d["path"]) if d["path"].endswith(".parquet") else pd.read_csv(d["path"])
    X_raw = df[d["features"]].astype(float).to_numpy()
    X = np.log1p(X_raw) if d["log_transform"] else X_raw
    X = StandardScaler().fit_transform(X)
    return df, X_raw, X


# ----------------------------------------------------------------------------- final model
def make_servable_pipeline(algo, est, X_raw, X_scaled, labels, cfg):
    """Pipeline(raw RFM -> log1p -> scaler -> model) with a .predict().
    Estimators without predict() (DBSCAN, Agglomerative) get a KNN surrogate
    trained on their labels so the registered model can score new customers."""
    log = cfg["data"]["log_transform"]
    pre = [("log", FunctionTransformer(np.log1p if log else None, validate=True)),
           ("scaler", StandardScaler())]
    if hasattr(est, "predict"):
        final = est
        note = "native predict"
    else:
        keep = labels != -1
        final = KNeighborsClassifier(n_neighbors=15).fit(X_scaled[keep], labels[keep])
        note = "KNN surrogate trained on cluster labels (algorithm has no predict)"
    pipe = Pipeline(pre + [("model", final)])
    # refit preprocessing on raw data so the pipeline is self-contained
    log_step = pipe.named_steps["log"].fit(X_raw)
    pipe.named_steps["scaler"].fit(log_step.transform(X_raw))
    return pipe, note


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tuning.yaml")
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--data", default=None, help="override data path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.data:
        cfg["data"]["path"] = args.data
    seed = cfg["evaluation"].get("random_state", SEED_DEFAULT)
    random.seed(seed)
    np.random.seed(seed)

    mlflow.set_tracking_uri(args.tracking_uri or cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    df, X_raw, X = load_features(cfg)
    print(f"Loaded {len(df)} customers, features={cfg['data']['features']}")

    sel = cfg["evaluation"]["selection_metric"]
    leaderboard = []  # one row per valid trial
    best = {"score": -np.inf}

    with mlflow.start_run(run_name="model-selection") as parent:
        mlflow.log_artifact(args.config)
        mlflow.set_tags(run_provenance(cfg["data"]["path"]))
        mlflow.log_params({"n_customers": len(df), "selection_metric": sel,
                           "log_transform": cfg["data"]["log_transform"]})

        for algo, space in cfg["search_space"].items():
            grid = list(ParameterGrid(space))
            cap = cfg.get("max_trials_per_algo")
            if cap and len(grid) > cap:
                grid = random.Random(seed).sample(grid, cap)
            print(f"[{algo}] {len(grid)} trials")

            with mlflow.start_run(run_name=algo, nested=True):
                mlflow.set_tag("algorithm", algo)
                algo_best = -np.inf
                for i, params in enumerate(grid):
                    with mlflow.start_run(run_name=f"{algo}-{i:03d}", nested=True):
                        mlflow.set_tag("algorithm", algo)
                        mlflow.log_params({"algorithm": algo, **params})
                        try:
                            est = build_estimator(algo, params, seed)
                            labels = get_labels(est, X)
                            metrics, reason = score(X, labels, cfg, seed)
                        except Exception as e:  # keep sweeping on failures
                            mlflow.set_tag("status_reason", f"error: {e}")
                            continue

                        mlflow.log_metrics({k: v for k, v in metrics.items()})
                        if reason:
                            mlflow.set_tag("status_reason", reason)
                            continue
                        mlflow.set_tag("status_reason", "valid")

                        row = {"algorithm": algo, **params, **metrics}
                        leaderboard.append(row)
                        s = metrics[sel]
                        algo_best = max(algo_best, s)
                        if s > best["score"]:
                            best = {"score": s, "algo": algo, "params": params, "metrics": metrics}

                if np.isfinite(algo_best):
                    mlflow.log_metric(f"best_{sel}", algo_best)

        if not leaderboard:
            raise SystemExit("No valid clustering found. Relax constraints in configs/tuning.yaml.")

        # ---- leaderboard artifacts
        lb = pd.DataFrame(leaderboard).sort_values(sel, ascending=False)
        Path("reports").mkdir(exist_ok=True)
        lb.to_csv("reports/leaderboard.csv", index=False)
        mlflow.log_artifact("reports/leaderboard.csv")
        per_algo = lb.groupby("algorithm")[sel].max().sort_values(ascending=False)
        print("\nBest score per algorithm:\n", per_algo.to_string())

        # ---- refit the winner and register it
        algo, params = best["algo"], best["params"]
        est = build_estimator(algo, params, seed)
        labels = get_labels(est, X)
        pipe, note = make_servable_pipeline(algo, est, X_raw, X, labels, cfg)

        feature_cols = cfg["data"]["features"]
        sample_in = pd.DataFrame(X_raw[:5], columns=feature_cols)
        sig = infer_signature(sample_in, pipe.predict(sample_in))

        mlflow.set_tags({"best_algorithm": algo, "serving_note": note})
        mlflow.log_params({f"best_{k}": v for k, v in params.items()})
        mlflow.log_param("best_algorithm", algo)
        mlflow.log_metrics({f"best_{k}": v for k, v in best["metrics"].items()})

        sizes = pd.Series(labels).value_counts().sort_index()
        sizes.rename_axis("cluster").rename("size").to_csv("reports/cluster_sizes.csv")
        mlflow.log_artifact("reports/cluster_sizes.csv")
        json.dump({"algorithm": algo, "params": params, **best["metrics"]},
                  open("reports/metrics.json", "w"), indent=2)
        mlflow.log_artifact("reports/metrics.json")

        info = mlflow.sklearn.log_model(
            pipe, artifact_path="model", signature=sig, input_example=sample_in,
            serialization_format="cloudpickle",  # skops can't serialize the np.log1p step
            registered_model_name=cfg["mlflow"]["registered_model"])

        client = MlflowClient()
        version = info.registered_model_version
        client.set_registered_model_alias(cfg["mlflow"]["registered_model"],
                                          cfg["mlflow"]["champion_alias"], version)
        print(f"\nWinner: {algo} {params}\n{best['metrics']}")
        print(f"Registered '{cfg['mlflow']['registered_model']}' v{version} "
              f"-> alias '{cfg['mlflow']['champion_alias']}'  ({note})")
        print(f"Parent run id: {parent.info.run_id}")


if __name__ == "__main__":
    main()
```

## `src/monitoring/drift.py`

```python
"""Data-drift check for the RFM features (PSI + Kolmogorov-Smirnov), no extra dependencies.

Compares a reference set (training data) with a current batch and
  * writes reports/drift.json
  * optionally pushes gauges to a Prometheus Pushgateway (seg_data_drift_share, seg_feature_psi{feature=...},
    seg_cluster_share_psi) so Grafana / Prometheus alerts can use them.

PSI rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, >= 0.2 significant drift.

Run:
  python src/features/build_rfm.py --start 2011-07-01 --output data/features/rfm_current.parquet
  python src/monitoring/drift.py --current data/features/rfm_current.parquet \
         [--pushgateway localhost:9091] [--model-uri models:/customer-segmentation@champion]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = ["recency", "frequency", "monetary"]


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index using quantile bins of the reference data."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(reference, edges)[0] / len(reference)
    cur_p = np.histogram(current, edges)[0] / len(current)
    ref_p, cur_p = np.clip(ref_p, eps, None), np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def category_psi(ref_labels, cur_labels, eps: float = 1e-4) -> float:
    """PSI between two label distributions (cluster-share shift)."""
    cats = sorted(set(ref_labels) | set(cur_labels))
    ref_p = np.array([np.mean(ref_labels == c) for c in cats])
    cur_p = np.array([np.mean(cur_labels == c) for c in cats])
    ref_p, cur_p = np.clip(ref_p, eps, None), np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, features=FEATURES, psi_threshold: float = 0.2) -> dict:
    per_feature = {}
    for f in features:
        ks = stats.ks_2samp(ref[f], cur[f])
        p = psi(ref[f], cur[f])
        per_feature[f] = {"psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                          "ks_pvalue": float(ks.pvalue), "drifted": bool(p >= psi_threshold)}
    n_drift = sum(v["drifted"] for v in per_feature.values())
    return {"n_reference": len(ref), "n_current": len(cur), "psi_threshold": psi_threshold,
            "share_of_drifted_features": n_drift / len(features), "features": per_feature}


def push_metrics(report: dict, gateway: str):
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    reg = CollectorRegistry()
    Gauge("seg_data_drift_share", "Share of drifted features", registry=reg).set(
        report["share_of_drifted_features"])
    g = Gauge("seg_feature_psi", "PSI per feature", ["feature"], registry=reg)
    for f, v in report["features"].items():
        g.labels(feature=f).set(v["psi"])
    if "cluster_share_psi" in report:
        Gauge("seg_cluster_share_psi", "PSI of cluster proportions", registry=reg).set(
            report["cluster_share_psi"])
    push_to_gateway(gateway, job="drift-job", registry=reg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/features/rfm.parquet")
    ap.add_argument("--current", default="data/features/rfm_current.parquet")
    ap.add_argument("--psi-threshold", type=float, default=0.2)
    ap.add_argument("--pushgateway", default=None, help="host:port, e.g. localhost:9091")
    ap.add_argument("--model-uri", default=None, help="also compare cluster shares using this model")
    ap.add_argument("--tracking-uri", default="http://localhost:5000")
    ap.add_argument("--output", default="reports/drift.json")
    args = ap.parse_args()

    ref, cur = pd.read_parquet(args.reference), pd.read_parquet(args.current)
    report = drift_report(ref, cur, FEATURES, args.psi_threshold)

    if args.model_uri:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(args.tracking_uri)
        pipe = mlflow.sklearn.load_model(args.model_uri)
        ref_l = pipe.predict(ref[FEATURES].astype(float).to_numpy())
        cur_l = pipe.predict(cur[FEATURES].astype(float).to_numpy())
        report["cluster_share_psi"] = round(category_psi(ref_l, cur_l), 4)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w"), indent=2)
    print(json.dumps(report, indent=2))

    if args.pushgateway:
        push_metrics(report, args.pushgateway)
        print(f"Pushed metrics to {args.pushgateway}")


if __name__ == "__main__":
    main()
```

## `src/serving/app.py`

```python
"""FastAPI service that serves the champion model from the MLflow Model Registry
and exposes Prometheus metrics.

Env vars: MLFLOW_TRACKING_URI (default http://localhost:5000), MODEL_NAME, MODEL_ALIAS
Run:      uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
"""
import os
import threading
import time
from typing import List

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
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

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "customer-segmentation")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
FEATURES = ["recency", "frequency", "monetary"]

app = FastAPI(title="Customer Segmentation API")

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

_state = {"pipe": None, "final": None, "version": None, "algorithm": None}
_lock = threading.Lock()


def load_model():
    """(Re)load the champion from the registry. Raises if there is none yet."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    pipe = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    with _lock:
        _state.update(pipe=pipe, final=pipe[-1], version=int(mv.version),
                      algorithm=type(pipe[-1]).__name__)
    MODEL_VERSION.set(int(mv.version))


def get_model():
    if _state["pipe"] is None:
        try:
            load_model()
        except Exception as exc:  # no champion registered yet / MLflow unreachable
            raise HTTPException(503, f"Model not available: {exc}") from exc
    return _state


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


@app.get("/live")
def live():
    return {"status": "alive"}


@app.get("/health")
def health():
    st = get_model()
    return {"status": "ok", "model": MODEL_NAME, "version": st["version"], "algorithm": st["algorithm"]}


@app.post("/reload")
def reload_model():
    """Pick up a newly promoted champion without restarting the container."""
    try:
        load_model()
    except Exception as exc:
        raise HTTPException(503, f"Reload failed: {exc}") from exc
    return {"version": _state["version"], "algorithm": _state["algorithm"]}


@app.post("/predict")
def predict(c: Customer):
    REQUESTS.inc()
    start = time.time()
    try:
        clusters, dists = _predict(pd.DataFrame([c.model_dump()]))
        cluster, dist = int(clusters[0]), dists[0]
        CLUSTER_COUNT.labels(cluster=str(cluster)).inc()
        if dist is not None:
            DIST_TO_CENTROID.observe(dist)
        return {"cluster": cluster, "distance_to_centroid": dist,
                "model_version": _state["version"]}
    except Exception:
        ERRORS.inc()
        raise
    finally:
        LATENCY.observe(time.time() - start)


@app.post("/predict/batch")
def predict_batch(customers: List[Customer]):
    REQUESTS.inc(len(customers))
    start = time.time()
    try:
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

## `tests/test_pipeline.py`

```python
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
```

## Empty placeholder files

- `src/__init__.py`
- `src/data/__init__.py`
- `src/features/__init__.py`
- `src/models/__init__.py`
- `src/monitoring/__init__.py`
- `src/serving/__init__.py`
- `data/*/.gitkeep`
- `reports/.gitkeep`
- `models/.gitkeep`
