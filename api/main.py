"""Minimal FastAPI application for local churn prediction serving."""

from __future__ import annotations

from fastapi import FastAPI

from api.routers.health import router as health_router
from api.routers.predict import router as predict_router

app = FastAPI(title="Customer Churn Local API", version="0.1.0")
app.include_router(health_router)
app.include_router(predict_router)
