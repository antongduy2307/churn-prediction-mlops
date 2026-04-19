"""Health router for local churn prediction serving."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a minimal service health status."""

    return HealthResponse(status="ok")
