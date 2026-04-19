"""Lazy local model-bundle loading utilities for FastAPI inference."""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

REQUIRED_BUNDLE_KEYS = {
    "model",
    "training_features",
    "categorical_features",
    "target_column",
    "label_encoders",
}

DEFAULT_MODEL_BUNDLE_PATH = Path("models/random_forest_bundle.pkl")


def _resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve the configured local model bundle path."""

    if model_path is not None:
        return Path(model_path)

    env_path = os.getenv("MODEL_BUNDLE_PATH")
    if env_path:
        return Path(env_path)

    return DEFAULT_MODEL_BUNDLE_PATH


def _validate_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate that the loaded bundle exposes the required inference keys."""

    missing_keys = sorted(REQUIRED_BUNDLE_KEYS - set(bundle.keys()))
    if missing_keys:
        raise ValueError(
            f"Model bundle is missing required keys: {missing_keys}. "
            f"Available keys: {sorted(bundle.keys())}"
        )
    return bundle


@lru_cache(maxsize=4)
def _load_bundle_cached(resolved_path: str) -> dict[str, Any]:
    """Load and cache a validated model bundle by path."""

    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")

    with path.open("rb") as file:
        bundle = pickle.load(file)

    if not isinstance(bundle, dict):
        raise ValueError(f"Model bundle must be a dictionary payload: {path}")

    return _validate_bundle(bundle)


def get_model_bundle(model_path: str | Path | None = None) -> dict[str, Any]:
    """Return a cached validated model bundle for local inference."""

    resolved_path = _resolve_model_path(model_path)
    return _load_bundle_cached(str(resolved_path))
