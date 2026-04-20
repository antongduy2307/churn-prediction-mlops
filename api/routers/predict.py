"""Prediction router for local payload-based churn inference."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.schemas import (
    BatchPredictionItemResponse,
    BatchPredictionResponse,
    CustomerPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.serving.feast_retrieval import (
    retrieve_online_features,
    validate_feature_mapping_consistency,
)
from src.serving.load_model import get_model_bundle
from src.serving.pre_processing import prepare_inference_dataframe

router = APIRouter(tags=["prediction"])


def _run_payload_inference(payload_dict: dict[str, Any], bundle: dict[str, Any]) -> tuple[float, int]:
    """Run direct payload inference using the existing serving preprocessing path."""

    inference_df = prepare_inference_dataframe(payload_dict, bundle)
    model = bundle["model"]
    churn_probability = float(model.predict_proba(inference_df)[0][1])
    churn_prediction = int(model.predict(inference_df)[0])
    return churn_probability, churn_prediction


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Run local model inference for a direct feature payload."""

    try:
        bundle = get_model_bundle()
        payload_dict = payload.model_dump()
        churn_probability, churn_prediction = _run_payload_inference(payload_dict, bundle)
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


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payloads: list[dict[str, Any]]) -> BatchPredictionResponse:
    """Run batch inference for a JSON list of direct feature payloads."""

    try:
        bundle = get_model_bundle()
    except (FileNotFoundError, LookupError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    predictions: list[BatchPredictionItemResponse] = []
    success_count = 0
    error_count = 0

    for index, raw_payload in enumerate(payloads):
        try:
            validated_payload = PredictionRequest.model_validate(raw_payload)
            churn_probability, churn_prediction = _run_payload_inference(
                validated_payload.model_dump(),
                bundle,
            )
            predictions.append(
                BatchPredictionItemResponse(
                    index=index,
                    churn_probability=churn_probability,
                    churn_prediction=churn_prediction,
                )
            )
            success_count += 1
        except Exception as exc:
            predictions.append(
                BatchPredictionItemResponse(
                    index=index,
                    error=str(exc),
                )
            )
            error_count += 1

    return BatchPredictionResponse(
        total_records=len(payloads),
        success_count=success_count,
        error_count=error_count,
        predictions=predictions,
    )


@router.get("/predict/{customer_id}", response_model=CustomerPredictionResponse)
def predict_by_customer_id(customer_id: int) -> CustomerPredictionResponse:
    """Run inference by retrieving online features from Feast for one customer_id."""

    try:
        bundle = get_model_bundle()
        feature_payload = retrieve_online_features(customer_id=customer_id)
        validate_feature_mapping_consistency(
            training_features=bundle["training_features"],
            retrieved_features=feature_payload,
        )
        churn_probability, churn_prediction = _run_payload_inference(feature_payload, bundle)
        return CustomerPredictionResponse(
            customer_id=customer_id,
            churn_probability=churn_probability,
            churn_prediction=churn_prediction,
        )
    except (FileNotFoundError, LookupError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
