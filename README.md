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
