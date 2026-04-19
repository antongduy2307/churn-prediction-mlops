"""Sample local Feast online feature retrieval by customer_id."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

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

TARGET_FEATURE_REFERENCE = "churn_target:churned"


def setup_logging() -> None:
    """Configure logging for CLI execution."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Retrieve online Feast features for a single customer_id."
    )
    parser.add_argument(
        "--customer-id",
        required=True,
        type=int,
        help="Customer entity identifier to retrieve from the Feast online store.",
    )
    parser.add_argument(
        "--include-target",
        action="store_true",
        help="Include churn_target:churned in the online lookup for debugging.",
    )
    return parser.parse_args()


def get_feature_references(include_target: bool) -> list[str]:
    """Return the feature references to retrieve."""

    feature_refs = list(BASE_FEATURE_REFERENCES)
    if include_target:
        feature_refs.append(TARGET_FEATURE_REFERENCE)
    return feature_refs


def load_feature_store(repo_path: Path) -> Any:
    """Instantiate a Feast FeatureStore from the local feature repository."""

    if not repo_path.exists():
        raise FileNotFoundError(f"Feast repository path not found: {repo_path}")

    try:
        from feast import FeatureStore
    except ImportError as exc:
        raise RuntimeError(
            "Feast is not installed in the active environment. "
            "Install the project dependencies and rerun the script."
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
    store: Any,
    customer_id: int,
    feature_references: list[str],
) -> dict[str, Any]:
    """Retrieve online features for a single customer_id."""

    try:
        response = store.get_online_features(
            features=feature_references,
            entity_rows=[{"customer_id": customer_id}],
        ).to_dict()
    except Exception as exc:
        raise RuntimeError(
            "Failed to retrieve online features from Feast. "
            "Ensure Redis is running and features have been materialized."
        ) from exc

    features = _normalize_online_response(response, feature_references)
    if all(value is None for value in features.values()):
        raise LookupError(
            f"No online features were found for customer_id={customer_id}. "
            "Either the entity does not exist in the online store or materialization has not been run."
        )

    return features


def print_features(customer_id: int, features: dict[str, Any], include_target: bool) -> None:
    """Print retrieved feature values in a readable structure."""

    payload = {
        "customer_id": customer_id,
        "features": features,
        "target_included": include_target,
    }
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    """CLI entrypoint for local Feast sample retrieval."""

    setup_logging()
    args = parse_args()
    project_root = Path.cwd()
    feature_repo_path = project_root / "feature_repo"
    feature_references = get_feature_references(include_target=args.include_target)

    LOGGER.info("Using Feast repository at %s", feature_repo_path)
    LOGGER.info("Retrieving feature references: %s", feature_references)

    store = load_feature_store(feature_repo_path)
    features = retrieve_online_features(
        store=store,
        customer_id=args.customer_id,
        feature_references=feature_references,
    )
    print_features(
        customer_id=args.customer_id,
        features=features,
        include_target=args.include_target,
    )


if __name__ == "__main__":
    main()
