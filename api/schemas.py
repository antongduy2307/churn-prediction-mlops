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


class HealthResponse(BaseModel):
    """Minimal health status schema."""

    status: str
