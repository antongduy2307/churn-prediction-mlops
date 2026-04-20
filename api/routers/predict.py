"""Prediction router for local payload-based churn inference."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import CustomerPredictionResponse, PredictionRequest, PredictionResponse
from src.serving.feast_retrieval import retrieve_online_features
from src.serving.load_model import get_model_bundle
from src.serving.pre_processing import prepare_inference_dataframe

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Run local model inference for a direct feature payload."""

    try:
        bundle = get_model_bundle()
        payload_dict = payload.model_dump()
        inference_df = prepare_inference_dataframe(payload_dict, bundle)
        model = bundle["model"]

        churn_probability = float(model.predict_proba(inference_df)[0][1])
        churn_prediction = int(model.predict(inference_df)[0])
        return PredictionResponse(
            churn_probability=churn_probability,
            churn_prediction=churn_prediction,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (LookupError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@router.get("/predict/{customer_id}", response_model=CustomerPredictionResponse)
def predict_by_customer_id(customer_id: int) -> CustomerPredictionResponse:
    """Run inference by retrieving online features from Feast for one customer_id."""

    try:
        bundle = get_model_bundle()
        feature_payload = retrieve_online_features(customer_id=customer_id)
        inference_df = prepare_inference_dataframe(feature_payload, bundle)
        model = bundle["model"]

        churn_probability = float(model.predict_proba(inference_df)[0][1])
        churn_prediction = int(model.predict(inference_df)[0])
        return CustomerPredictionResponse(
            customer_id=customer_id,
            churn_probability=churn_probability,
            churn_prediction=churn_prediction,
        )
    except (FileNotFoundError, LookupError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
