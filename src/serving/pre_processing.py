"""Inference-time payload preprocessing for local churn prediction serving."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

UNKNOWN_CATEGORY = "Unknown"


def _normalize_categorical_value(value: Any) -> str:
    """Normalize a categorical payload value using the training-time strategy."""

    if value is None:
        return UNKNOWN_CATEGORY

    normalized = str(value).strip()
    return normalized if normalized else UNKNOWN_CATEGORY


def _validate_required_fields(payload: Mapping[str, Any], training_features: list[str]) -> None:
    """Validate that the inference payload contains all required features."""

    missing_fields = [feature for feature in training_features if feature not in payload]
    if missing_fields:
        raise ValueError(f"Missing required inference fields: {missing_fields}")


def prepare_inference_dataframe(
    payload: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> pd.DataFrame:
    """Convert a prediction payload into a one-row encoded dataframe for inference."""

    training_features = list(bundle["training_features"])
    categorical_features = list(bundle["categorical_features"])
    label_encoders = dict(bundle["label_encoders"])

    _validate_required_fields(payload, training_features)

    numeric_features = [
        feature for feature in training_features if feature not in categorical_features
    ]
    row: dict[str, Any] = {}

    for feature in numeric_features:
        try:
            row[feature] = float(payload[feature])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric value for '{feature}': {payload[feature]!r}"
            ) from exc

    for feature in categorical_features:
        if feature not in label_encoders:
            raise ValueError(f"Missing label encoder for categorical feature '{feature}'.")

        encoder = label_encoders[feature]
        known_classes = set(encoder.classes_.tolist())
        if UNKNOWN_CATEGORY not in known_classes:
            raise ValueError(
                f"Label encoder for '{feature}' does not include '{UNKNOWN_CATEGORY}', "
                "so unseen-category handling is unsafe."
            )

        normalized_value = _normalize_categorical_value(payload[feature])
        safe_value = normalized_value if normalized_value in known_classes else UNKNOWN_CATEGORY
        row[feature] = int(encoder.transform([safe_value])[0])

    ordered_row = {feature: row[feature] for feature in training_features}
    return pd.DataFrame([ordered_row], columns=training_features)
