# Customer Churn Prediction - Phase 1 Local Baseline

## Overview
This repository contains the Phase 1 local baseline for a customer churn prediction workflow. The current scope covers deterministic local data preparation, local model training, standalone local evaluation, and a minimal FastAPI payload-based inference service backed by a saved local model bundle.

## Phase 1 Scope
Implemented in this phase:
- Raw CSV preprocessing into a normalized processed CSV
- Conversion of processed data into Feast-compatible parquet with deterministic timestamps
- Local RandomForest baseline training
- Standalone local evaluation against a saved model bundle
- Minimal FastAPI serving for direct payload inference

Not included yet:
- `customer_id` lookup
- Batch prediction
- Feast online serving
- Monitoring
- MLflow or registry integration
- CI/CD

## Quickstart
### 1. Install dependencies
```bash
python -m pip install -r requirements.txt
```

Alternative with Conda:
```bash
conda env create -f environment.yaml
conda activate churn-mlops
```

### 2. Preprocess raw CSV into processed CSV
```bash
python -m src.data.processing --input-path data/raw/train_period_1.csv --output-path data/processed/df_processed.csv
```

### 3. Convert processed CSV into Feast-ready parquet
```bash
python -m src.data.prepare_feast_data --input-path data/processed/df_processed.csv --output-path data/processed/processed_churn_data.parquet
```

### 4. Train the local model bundle
```bash
python -m src.scripts.train --config configs/random_forest.yaml
```

### 5. Evaluate the saved model bundle
```bash
python -m src.scripts.eval --model-path models/random_forest_bundle.pkl --data-path data/processed/processed_churn_data.parquet --report-dir reports
```

### 6. Start the FastAPI server
```bash
uvicorn api.main:app --reload
```

### 7. Test `/health`
```bash
curl http://127.0.0.1:8000/health
```

### 8. Test `/predict`
```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"age\":42,\"gender\":\"Female\",\"tenure_months\":24,\"subscription_type\":\"Premium\",\"contract_length\":\"Annual\",\"usage_frequency\":15,\"support_calls\":2,\"payment_delay_days\":3,\"total_spend\":1200,\"last_interaction_days\":7}"
```

POSIX shell equivalent:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":42,"gender":"Female","tenure_months":24,"subscription_type":"Premium","contract_length":"Annual","usage_frequency":15,"support_calls":2,"payment_delay_days":3,"total_spend":1200,"last_interaction_days":7}'
```

## Expected Artifacts
- `data/processed/df_processed.csv`
- `data/processed/processed_churn_data.parquet`
- `models/random_forest_bundle.pkl`
- `models/random_forest_metrics.json`
- `reports/evaluation_metrics.json`
- `reports/predictions.csv`
- `reports/confusion_matrix.png`
- `reports/feature_importance.png` if the loaded model supports feature importances

## API Endpoints
- `GET /health` returns a simple service status
- `POST /predict` accepts a direct feature payload and returns:
  - `churn_probability`
  - `churn_prediction`

## Local Feast Runtime
Phase 2 uses the local Feast repository under `feature_repo/`. The runtime is standardized so Feast commands are executed from inside `feature_repo/`, which keeps the relative parquet source path stable.

### Start local Redis
If Redis is installed locally:
```bash
redis-server --port 6379
```

If using Docker:
```bash
docker run --name churn-redis -p 6379:6379 redis:7
```

### Apply the Feast repo
```bash
python scripts/run_feast_apply.py
```

### Materialize features incrementally
```bash
python scripts/materialize_features.py
```

The materialization wrapper generates the current UTC timestamp at runtime and calls:
```bash
feast materialize-incremental <current-utc-timestamp>
```

## Phase 2 Runbook
Phase 2 adds local data/versioning and local Feast online feature retrieval by `customer_id`. It does not yet integrate FastAPI with Feast, and `/predict/{customer_id}` is not implemented yet.

### Prerequisites
- `data/processed/processed_churn_data.parquet` already exists
- DVC metadata has already been initialized locally
- Redis is running on `localhost:6379`
- Feast is installed in the active environment

### Phase 2 Workflow
1. Check local DVC-tracked data status:
```bash
dvc status
```

2. Start Redis:
```bash
redis-server --port 6379
```

Docker alternative:
```bash
docker run --name churn-redis -p 6379:6379 redis:7
```

3. Apply the Feast repo:
```bash
python scripts/run_feast_apply.py
```

4. Materialize online features:
```bash
python scripts/materialize_features.py
```

5. Retrieve online features for one customer:
```bash
python scripts/sample_retrieval.py --customer-id <id>
```

Optional debug retrieval including the target:
```bash
python scripts/sample_retrieval.py --customer-id <id> --include-target
```

### Current Phase 2 Scope
Supported now:
- Local DVC tracking for the main raw and processed data artifacts
- Local Feast repo apply/materialization
- Local online feature retrieval by `customer_id`

Not implemented yet:
- FastAPI to Feast integration
- `/predict/{customer_id}`
- Batch retrieval/prediction
- MLflow, monitoring, CI/CD, or remote DVC storage

## Local MLflow Runtime
Phase 3 starts with a local MLflow tracking server for development. This step only sets up the tracking environment; training and serving are not wired to MLflow yet.

### Start the MLflow server
Run from the project root:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

### Local storage choices
- Backend store: `sqlite:///mlflow.db`
- Artifact store: `./mlruns`

### Access the UI
Open:
```text
http://127.0.0.1:5000
```

## Shared Environment
Use one shared local environment for preprocessing, DVC, Feast, training, evaluation, FastAPI, and MLflow.

Python version:
```text
Python 3.11
```

Install the shared environment with:
```bash
python -m pip install -r requirements.txt
```

This shared environment is used for:
```bash
python -m src.data.processing --input-path data/raw/train_period_1.csv --output-path data/processed/df_processed.csv
python -m src.data.prepare_feast_data --input-path data/processed/df_processed.csv --output-path data/processed/processed_churn_data.parquet
python scripts/run_feast_apply.py
python scripts/materialize_features.py
python scripts/sample_retrieval.py --customer-id <id>
python -m src.scripts.train --config configs/random_forest.yaml
python -m src.scripts.eval --model-path models/random_forest_bundle.pkl --data-path data/processed/processed_churn_data.parquet --report-dir reports
uvicorn api.main:app --reload
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

## Phase 3 Runbook
Phase 3 adds local MLflow experiment tracking, local model registration, and registry-backed inference loading for the FastAPI service.

### Start the MLflow server
Run from the project root:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

MLflow UI:
```text
http://127.0.0.1:5000
```

### Run training with MLflow enabled
```bash
python -m src.scripts.train --config configs/random_forest.yaml
```

Training now logs to MLflow:
- model parameters
- split settings
- training feature list
- categorical feature list
- evaluation metrics
- logged sklearn model artifact

Training also registers the logged model under:
```text
churn_random_forest
```

Each successful training run creates a new MLflow model version.

### Serving model resolution
FastAPI serving now loads the inference estimator from MLflow Model Registry via `MODEL_URI`.

Default behavior:
```text
MODEL_URI=models:/churn_random_forest/latest
```

Preprocessing metadata still comes from the local bundle:
```text
models/random_forest_bundle.pkl
```

### Environment variables
Useful local settings:
```text
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MODEL_URI=models:/churn_random_forest/latest
MODEL_BUNDLE_PATH=models/random_forest_bundle.pkl
```

### Current Phase 3 Scope
Supported now:
- local MLflow tracking server
- experiment tracking during training
- local MLflow Model Registry registration
- FastAPI inference model loading from MLflow registry

Not implemented yet:
- stage transitions
- alias-based promotion
- automated deployment
- Feast-integrated API inference

## Phase 4 Runbook
Phase 4 adds customer_id-based API inference via Feast, JSON batch inference, richer health/readiness debugging, model info visibility, and on-demand local drift reporting.

### Serving endpoints
Available endpoints:
- `POST /predict`
- `GET /predict/{customer_id}`
- `POST /predict/batch`
- `GET /health`
- `GET /health/ready`
- `GET /model/info`
- `GET /monitor/drift`

### Direct payload inference
`POST /predict` accepts a single feature payload, rebuilds the inference frame in training feature order, applies local preprocessing metadata from the bundle, and runs the estimator loaded from MLflow Model Registry.

Example:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":42,"gender":"Female","tenure_months":24,"subscription_type":"Premium","contract_length":"Annual","usage_frequency":15,"support_calls":2,"payment_delay_days":3,"total_spend":1200,"last_interaction_days":7}'
```

### Feast customer_id inference
`GET /predict/{customer_id}` retrieves online features from Feast, validates the retrieved feature names against serving expectations, then scores the MLflow-loaded estimator.

Example:
```bash
curl "http://127.0.0.1:8000/predict/12345"
```

### Batch payload inference
`POST /predict/batch` accepts a JSON list of payload records using the same schema as `POST /predict`. Each record is processed independently, and failures are returned per-record instead of failing the whole batch.

Example:
```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[{"age":42,"gender":"Female","tenure_months":24,"subscription_type":"Premium","contract_length":"Annual","usage_frequency":15,"support_calls":2,"payment_delay_days":3,"total_spend":1200,"last_interaction_days":7},{"age":"bad","gender":"Male","tenure_months":12,"subscription_type":"Basic","contract_length":"Monthly","usage_frequency":8,"support_calls":1,"payment_delay_days":0,"total_spend":300,"last_interaction_days":14}]'
```

### Health, readiness, and model info
- `GET /health` checks that the API process is alive
- `GET /health/ready` reports whether:
  - model loading is ready
  - Feast repo is loadable
  - feature mapping is consistent
- `GET /model/info` shows the active serving model URI, local bundle path, training features, categorical features, and target column

Examples:
```bash
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/health/ready"
curl "http://127.0.0.1:8000/model/info"
```

### Drift reporting
`GET /monitor/drift` generates a local Evidently HTML drift report on request.

Default local paths:
- reference: `data/processed/df_processed.csv`
- current: `data/processed/processed_churn_data.parquet`
- output dir: `reports/drift/`

Example:
```bash
curl "http://127.0.0.1:8000/monitor/drift"
```

Example with explicit paths:
```bash
curl "http://127.0.0.1:8000/monitor/drift?reference_path=data/processed/df_processed.csv&current_path=data/processed/processed_churn_data.parquet"
```

### Serving configuration and paths
Key serving inputs:
```text
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MODEL_URI=models:/churn_random_forest/latest
MODEL_BUNDLE_PATH=models/random_forest_bundle.pkl
```

Important local paths:
- `feature_repo/`
- `reports/drift/`

### Current Phase 4 Scope
Supported now:
- direct payload inference
- Feast-based customer_id inference
- JSON batch inference
- local readiness/model-info endpoints
- on-demand local drift report generation

Not implemented yet:
- automated deployment
- alerting or monitoring scheduler
- stage-based promotion logic in serving
- production-grade auth or rate limiting
