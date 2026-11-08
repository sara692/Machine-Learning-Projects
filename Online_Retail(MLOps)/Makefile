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
