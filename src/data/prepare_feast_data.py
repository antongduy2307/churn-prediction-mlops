"""Prepare deterministic Feast-ready parquet data from processed churn CSV data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_PROCESSED_COLUMNS: list[str] = [
    "customer_id",
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

FEAST_COLUMN_ORDER: list[str] = [
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
    """Configure module logging for CLI usage."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_processed_columns(df: pd.DataFrame) -> None:
    """Validate that the processed dataframe contains the required normalized schema."""

    missing_columns = [
        column for column in REQUIRED_PROCESSED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Processed input is missing required columns: "
            f"{missing_columns}. Available columns: {list(df.columns)}"
        )


def add_timestamps(df: pd.DataFrame, fixed_timestamp: str) -> pd.DataFrame:
    """Add deterministic Feast timestamps using a fixed user-supplied value."""

    try:
        timestamp = pd.Timestamp(fixed_timestamp)
    except ValueError as exc:
        raise ValueError(
            f"Invalid fixed timestamp '{fixed_timestamp}'. "
            "Expected a value parseable by pandas.Timestamp."
        ) from exc

    timestamped_df = df.copy()
    timestamped_df["event_timestamp"] = pd.Series(
        [timestamp] * len(timestamped_df),
        index=timestamped_df.index,
        dtype="datetime64[ns]",
    )
    timestamped_df["created_timestamp"] = pd.Series(
        [timestamp] * len(timestamped_df),
        index=timestamped_df.index,
        dtype="datetime64[ns]",
    )
    return timestamped_df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to the exact Feast FileSource-compatible layout."""

    return df.loc[:, FEAST_COLUMN_ORDER].copy()


def prepare_feast_dataframe(df: pd.DataFrame, fixed_timestamp: str) -> pd.DataFrame:
    """Build a Feast-ready dataframe from processed local churn data."""

    validate_processed_columns(df)
    feast_df = add_timestamps(df, fixed_timestamp=fixed_timestamp)
    return reorder_columns(feast_df)


def _ensure_parent_directory(output_path: Path) -> None:
    """Create the parent directory for an output file when needed."""

    output_path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare a deterministic Feast-ready parquet dataset."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to the processed input CSV file.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path where the Feast-ready parquet file will be written.",
    )
    parser.add_argument(
        "--fixed-timestamp",
        default="2024-01-01 00:00:00",
        help="Deterministic timestamp used for both event_timestamp and created_timestamp.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for deterministic Feast data preparation."""

    setup_logging()
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    LOGGER.info("Reading processed CSV from %s", input_path)
    processed_df = pd.read_csv(input_path)

    feast_df = prepare_feast_dataframe(
        processed_df,
        fixed_timestamp=args.fixed_timestamp,
    )

    _ensure_parent_directory(output_path)
    feast_df.to_parquet(output_path, index=False)

    LOGGER.info("Row count: %s", len(feast_df))
    LOGGER.info("Schema/dtypes: %s", feast_df.dtypes.astype(str).to_dict())
    LOGGER.info("Feast-ready parquet written to %s", output_path)


if __name__ == "__main__":
    main()
