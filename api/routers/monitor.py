"""Monitoring router for local drift report generation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.schemas import DriftReportResponse
from src.serving.monitoring import generate_drift_report

router = APIRouter(tags=["monitoring"])


@router.get("/monitor/drift", response_model=DriftReportResponse)
def monitor_drift(
    reference_path: str | None = Query(default=None),
    current_path: str | None = Query(default=None),
    output_path: str | None = Query(default=None),
) -> DriftReportResponse:
    """Generate a local Evidently drift report on request."""

    try:
        result = generate_drift_report(
            reference_path=reference_path,
            current_path=current_path,
            output_path=output_path,
        )
        return DriftReportResponse(
            status=result.status,
            reference_path=result.reference_path,
            current_path=result.current_path,
            report_path=result.report_path,
            compared_columns=result.compared_columns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift monitoring failed: {exc}") from exc
