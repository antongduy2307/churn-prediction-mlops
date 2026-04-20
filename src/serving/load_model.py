"""Lazy model and metadata loading utilities for FastAPI inference."""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

REQUIRED_METADATA_KEYS = {
    "training_features",
    "categorical_features",
    "target_column",
    "label_encoders",
}

DEFAULT_MODEL_BUNDLE_PATH = Path("models/random_forest_bundle.pkl")
DEFAULT_REGISTERED_MODEL_NAME = "churn_random_forest"
DEFAULT_MODEL_URI = f"models:/{DEFAULT_REGISTERED_MODEL_NAME}/latest"
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


def _resolve_model_bundle_path(model_path: str | Path | None = None) -> Path:
    """Resolve the configured local metadata bundle path."""

    if model_path is not None:
        return Path(model_path)

    env_path = os.getenv("MODEL_BUNDLE_PATH")
    if env_path:
        return Path(env_path)

    return DEFAULT_MODEL_BUNDLE_PATH


def _resolve_mlflow_tracking_uri() -> str:
    """Resolve the MLflow tracking URI used for model loading."""

    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)


def _resolve_model_uri() -> str:
    """Resolve the configured MLflow model URI."""

    return os.getenv("MODEL_URI", DEFAULT_MODEL_URI)


def _validate_metadata_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate that the loaded local bundle exposes required preprocessing metadata."""

    missing_keys = sorted(REQUIRED_METADATA_KEYS - set(bundle.keys()))
    if missing_keys:
        raise ValueError(
            f"Model bundle is missing required preprocessing keys: {missing_keys}. "
            f"Available keys: {sorted(bundle.keys())}"
        )
    return bundle


@lru_cache(maxsize=4)
def _load_metadata_bundle_cached(resolved_path: str) -> dict[str, Any]:
    """Load and cache the local preprocessing metadata bundle."""

    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")

    with path.open("rb") as file:
        bundle = pickle.load(file)

    if not isinstance(bundle, dict):
        raise ValueError(f"Model bundle must be a dictionary payload: {path}")

    return _validate_metadata_bundle(bundle)


def _resolve_latest_model_uri(model_uri: str, tracking_uri: str) -> str:
    """Resolve `models:/name/latest` into a concrete latest version URI."""

    if not model_uri.startswith("models:/") or not model_uri.endswith("/latest"):
        return model_uri

    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed in the active environment. "
            "Install project dependencies and retry."
        ) from exc

    mlflow.set_tracking_uri(tracking_uri)
    model_name = model_uri.removeprefix("models:/").removesuffix("/latest")
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:
        raise RuntimeError(
            f"Unable to query MLflow Model Registry for '{model_name}' at {tracking_uri}."
        ) from exc

    if not versions:
        raise LookupError(
            f"No registered versions were found for model '{model_name}'. "
            "Train and register a model first."
        )

    latest_version = max(versions, key=lambda version: int(version.version))
    return f"models:/{model_name}/{latest_version.version}"


@lru_cache(maxsize=4)
def _load_registry_model_cached(model_uri: str, tracking_uri: str) -> Any:
    """Load and cache the inference model from MLflow."""

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed in the active environment. "
            "Install project dependencies and retry."
        ) from exc

    mlflow.set_tracking_uri(tracking_uri)
    resolved_model_uri = _resolve_latest_model_uri(model_uri, tracking_uri)

    try:
        return mlflow.sklearn.load_model(resolved_model_uri)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load inference model from MLflow model URI '{resolved_model_uri}'. "
            "Ensure the MLflow server is running and the model has been registered."
        ) from exc


def _validate_serving_components(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate the final serving components required for prediction."""

    model = bundle.get("model")
    if model is None:
        raise ValueError("Serving bundle is missing the inference model.")
    if not hasattr(model, "predict"):
        raise ValueError("Loaded inference model does not implement predict().")
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded inference model does not implement predict_proba().")
    return bundle


def get_model_bundle(model_path: str | Path | None = None) -> dict[str, Any]:
    """Return serving components with model from MLflow and metadata from the local bundle."""

    metadata_bundle = dict(
        _load_metadata_bundle_cached(str(_resolve_model_bundle_path(model_path)))
    )
    tracking_uri = _resolve_mlflow_tracking_uri()
    model_uri = _resolve_model_uri()
    metadata_bundle["model"] = _load_registry_model_cached(model_uri, tracking_uri)
    metadata_bundle["model_uri"] = model_uri
    metadata_bundle["mlflow_tracking_uri"] = tracking_uri
    return _validate_serving_components(metadata_bundle)
