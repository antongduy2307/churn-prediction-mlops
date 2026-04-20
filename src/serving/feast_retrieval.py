"""Reusable local Feast online feature retrieval helpers for serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

BASE_FEATURE_REFERENCES: list[str] = [
    "customer_demographics:age",
    "customer_demographics:gender",
    "customer_demographics:tenure_months",
    "customer_demographics:subscription_type",
    "customer_demographics:contract_length",
    "customer_behavior:usage_frequency",
    "customer_behavior:support_calls",
    "customer_behavior:payment_delay_days",
    "customer_behavior:total_spend",
    "customer_behavior:last_interaction_days",
]

EXPECTED_FEATURE_NAMES: list[str] = [
    feature_ref.split(":", maxsplit=1)[1] for feature_ref in BASE_FEATURE_REFERENCES
]


def load_feature_store(repo_path: Path) -> Any:
    """Instantiate a Feast FeatureStore from the local feature repository."""

    if not repo_path.exists():
        raise FileNotFoundError(f"Feast repository path not found: {repo_path}")

    try:
        from feast import FeatureStore
    except ImportError as exc:
        raise RuntimeError(
            "Feast is not installed in the active environment. "
            "Install the project dependencies and rerun."
        ) from exc

    try:
        return FeatureStore(repo_path=str(repo_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Feast repository from {repo_path}. "
            "Ensure the feature repo is valid and `feast apply` has been run."
        ) from exc


def _normalize_online_response(
    response_dict: dict[str, list[Any]],
    feature_references: list[str],
) -> dict[str, Any]:
    """Convert Feast's online response into a flat feature mapping."""

    features: dict[str, Any] = {}
    for feature_ref in feature_references:
        short_name = feature_ref.split(":", maxsplit=1)[1]
        values = response_dict.get(short_name, response_dict.get(feature_ref))
        if values is None:
            raise RuntimeError(
                f"Feast response did not contain expected feature '{feature_ref}'."
            )
        features[short_name] = values[0]
    return features


def retrieve_online_features(
    customer_id: int,
    repo_path: Path | None = None,
) -> dict[str, Any]:
    """Retrieve online Feast features for a single customer_id."""

    feature_repo_path = repo_path or (Path.cwd() / "feature_repo")
    store = load_feature_store(feature_repo_path)

    try:
        response = store.get_online_features(
            features=BASE_FEATURE_REFERENCES,
            entity_rows=[{"customer_id": customer_id}],
        ).to_dict()
    except Exception as exc:
        raise RuntimeError(
            "Failed to retrieve online features from Feast. "
            "Ensure Redis is running and features have been materialized."
        ) from exc

    features = _normalize_online_response(response, BASE_FEATURE_REFERENCES)
    if all(value is None for value in features.values()):
        raise LookupError(
            f"No online features were found for customer_id={customer_id}. "
            "Either the entity does not exist in the online store or materialization has not been run."
        )

    return features


def validate_feature_mapping_consistency(
    training_features: list[str],
    retrieved_features: Mapping[str, Any] | None = None,
) -> None:
    """Validate that Feast feature names match serving/training expectations exactly."""

    expected_names = set(EXPECTED_FEATURE_NAMES)
    training_names = set(training_features)

    if training_names != expected_names:
        raise ValueError(
            "Feast feature references do not match bundle training_features. "
            f"training_features={sorted(training_names)}, feast_features={sorted(expected_names)}"
        )

    if retrieved_features is None:
        return

    retrieved_names = set(retrieved_features.keys())
    if retrieved_names != training_names:
        raise ValueError(
            "Retrieved Feast feature payload does not match serving expectations. "
            f"retrieved={sorted(retrieved_names)}, expected={sorted(training_names)}"
        )
