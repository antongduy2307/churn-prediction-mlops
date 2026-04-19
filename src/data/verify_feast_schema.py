"""Verify the schema of a Feast-ready parquet dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = [
    "customer_id",
    "event_timestamp",
    "created_timestamp",
    "age",
    "gender",
    "tenure_months",
    "subscription_type",
    "contract_length",
    "usage_frequency",
    "support_calls",
    "payment_delay_days",
    "total_spend",
    "last_interaction_days",
    "churned",
]


def setup_logging() -> None:
    """Configure logging for CLI usage."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Verify the schema of a Feast-ready parquet dataset."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to the Feast-ready parquet file.",
    )
    return parser.parse_args()


def verify_feast_schema(input_path: Path) -> pd.DataFrame:
    """Load and verify the required Feast-ready parquet schema."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {input_path}")

    df = pd.read_parquet(input_path)
    available_columns = list(df.columns)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    extra_columns = [column for column in df.columns if column not in REQUIRED_COLUMNS]

    if missing_columns:
        raise ValueError(
            "Feast-ready parquet is missing required columns: "
            f"{missing_columns}. Available columns: {available_columns}"
        )

    if extra_columns:
        LOGGER.warning("Feast-ready parquet contains extra columns: %s", extra_columns)

    LOGGER.info("Row count: %s", len(df))
    LOGGER.info("Column list: %s", available_columns)
    LOGGER.info("Dtype map: %s", df.dtypes.astype(str).to_dict())
    return df


def main() -> None:
    """CLI entrypoint for Feast-ready schema verification."""

    setup_logging()
    args = parse_args()
    verify_feast_schema(Path(args.input_path))


if __name__ == "__main__":
    main()
