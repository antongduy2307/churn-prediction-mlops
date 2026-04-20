"""Health router for local churn prediction serving."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from api.schemas import HealthResponse, ModelInfoResponse, ReadinessResponse
from src.serving.feast_retrieval import load_feature_store, validate_feature_mapping_consistency
from src.serving.load_model import get_model_bundle, get_model_info

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a minimal service health status."""

    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse)
def health_ready() -> ReadinessResponse:
    """Return local readiness details for model and Feast-backed inference."""

    details: dict[str, str] = {}
    model_ready = False
    feast_ready = False
    feature_mapping_consistent = False

    try:
        bundle = get_model_bundle()
        model_ready = True
        details["model"] = "MLflow model and local metadata bundle loaded successfully."
    except Exception as exc:
        bundle = None
        details["model"] = str(exc)

    if bundle is not None:
        try:
            validate_feature_mapping_consistency(bundle["training_features"])
            feature_mapping_consistent = True
            details["feature_mapping"] = "Feast feature references match bundle training_features."
        except Exception as exc:
            details["feature_mapping"] = str(exc)
    else:
        details["feature_mapping"] = "Skipped because serving model bundle is not ready."

    try:
        load_feature_store(Path.cwd() / "feature_repo")
        feast_ready = True
        details["feast"] = "Feast repository is loadable for customer_id inference."
    except Exception as exc:
        details["feast"] = str(exc)

    status = (
        "ready"
        if model_ready and feast_ready and feature_mapping_consistent
        else "degraded"
    )
    return ReadinessResponse(
        status=status,
        api_alive=True,
        model_ready=model_ready,
        feast_ready=feast_ready,
        feature_mapping_consistent=feature_mapping_consistent,
        details=details,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return concise serving model metadata for local debugging."""

    return ModelInfoResponse(**get_model_info())
