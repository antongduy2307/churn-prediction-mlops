"""Pydantic request and response schemas for local churn prediction serving."""

from __future__ import annotations

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Direct payload schema for local churn prediction."""

    age: float
    gender: str
    tenure_months: float
    subscription_type: str
    contract_length: str
    usage_frequency: float
    support_calls: float
    payment_delay_days: float
    total_spend: float
    last_interaction_days: float


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    churn_probability: float
    churn_prediction: int


class CustomerPredictionResponse(BaseModel):
    """Prediction response schema for Feast customer_id lookups."""

    customer_id: int
    churn_probability: float
    churn_prediction: int


class BatchPredictionItemResponse(BaseModel):
    """Per-record batch prediction result."""

    index: int
    churn_probability: float | None = None
    churn_prediction: int | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""

    total_records: int
    success_count: int
    error_count: int
    predictions: list[BatchPredictionItemResponse]


class HealthResponse(BaseModel):
    """Minimal health status schema."""

    status: str


class ReadinessResponse(BaseModel):
    """Readiness response schema for local debugging."""

    status: str
    api_alive: bool
    model_ready: bool
    feast_ready: bool
    feature_mapping_consistent: bool
    details: dict[str, str]


class ModelInfoResponse(BaseModel):
    """Serving model metadata response schema."""

    model_uri: str
    registered_model_name: str | None
    model_bundle_path: str
    training_features: list[str]
    categorical_features: list[str]
    target_column: str
    mlflow_tracking_uri: str


class DriftReportResponse(BaseModel):
    """Response schema for on-demand drift report generation."""

    status: str
    reference_path: str
    current_path: str
    report_path: str
    compared_columns: list[str]
