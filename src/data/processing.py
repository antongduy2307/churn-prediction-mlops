"""Deterministic local preprocessing for customer churn raw CSV data.

This module normalizes the raw training schema into a stable processed schema
that can be reused by later local and feature-store preparation steps.
Duplicate handling is deterministic: exact duplicate rows are removed first,
then duplicate ``customer_id`` rows keep the first occurrence in file order.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

RAW_TO_NORMALIZED_COLUMNS: dict[str, str] = {
    "Age": "age",
    "Gender": "gender",
    "Tenure": "tenure_months",
    "Usage Frequency": "usage_frequency",
    "Support Calls": "support_calls",
    "Payment Delay": "payment_delay_days",
    "Subscription Type": "subscription_type",
    "Contract Length": "contract_length",
    "Total Spend": "total_spend",
    "Last Interaction": "last_interaction_days",
    "Churn": "churned",
}

NORMALIZED_COLUMN_ORDER: list[str] = [
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

TARGET_COLUMN = "churned"

FEATURE_NUMERIC_COLUMNS: list[str] = [
    "age",
    "tenure_months",
    "usage_frequency",
    "support_calls",
    "payment_delay_days",
    "total_spend",
    "last_interaction_days",
]

FEATURE_CATEGORICAL_COLUMNS: list[str] = [
    "gender",
    "subscription_type",
    "contract_length",
]


def build_required_raw_columns(id_column: str) -> list[str]:
    """Return the required raw CSV columns for validation."""

    return [id_column, *RAW_TO_NORMALIZED_COLUMNS.keys()]


def setup_logging() -> None:
    """Configure module logging for CLI usage."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def validate_raw_columns(df: pd.DataFrame, id_column: str) -> None:
    """Raise an error when required raw columns are missing."""

    required_columns = build_required_raw_columns(id_column)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Raw input is missing required columns: "
            f"{missing_columns}. Available columns: {list(df.columns)}"
        )


def rename_columns(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Rename raw columns into the normalized schema."""

    rename_map = {id_column: "customer_id", **RAW_TO_NORMALIZED_COLUMNS}
    renamed_df = df.rename(columns=rename_map)
    return renamed_df.loc[:, NORMALIZED_COLUMN_ORDER].copy()


def _normalize_object_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Strip whitespace from object-like columns and convert blanks to missing."""

    normalized_df = df.copy()
    for column in columns:
        if column not in normalized_df.columns:
            continue
        series = normalized_df[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized_df[column] = (
                series.astype("string")
                .str.strip()
                .replace("", pd.NA)
            )
    return normalized_df


def clean_dataframe(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Apply deterministic, non-feature-engineering cleanup steps."""

    cleaned_df = _normalize_object_columns(
        df,
        FEATURE_CATEGORICAL_COLUMNS,
    )

    if not drop_duplicates:
        return cleaned_df

    input_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(keep="first").copy()
    dropped_exact_duplicates = input_rows - len(cleaned_df)
    if dropped_exact_duplicates:
        LOGGER.info("Dropped %s exact duplicate rows.", dropped_exact_duplicates)

    duplicate_customer_ids = cleaned_df.duplicated(subset=["customer_id"], keep="first")
    dropped_duplicate_ids = int(duplicate_customer_ids.sum())
    if dropped_duplicate_ids:
        LOGGER.info(
            "Dropped %s duplicate customer_id rows using keep='first'.",
            dropped_duplicate_ids,
        )
        cleaned_df = cleaned_df.loc[~duplicate_customer_ids].copy()

    return cleaned_df


def cast_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns into predictable intermediate types before imputation."""

    cast_df = df.copy()
    cast_df["customer_id"] = pd.to_numeric(cast_df["customer_id"], errors="coerce")

    for column in FEATURE_NUMERIC_COLUMNS:
        cast_df[column] = pd.to_numeric(cast_df[column], errors="coerce")

    cast_df[TARGET_COLUMN] = pd.to_numeric(cast_df[TARGET_COLUMN], errors="coerce")

    for column in FEATURE_CATEGORICAL_COLUMNS:
        cast_df[column] = cast_df[column].astype("string")

    return cast_df


def validate_required_entity_and_target(df: pd.DataFrame) -> None:
    """Validate that identifier and target columns are present and usable."""

    missing_customer_ids = int(df["customer_id"].isna().sum())
    LOGGER.info("Missing values before validation for customer_id: %s", missing_customer_ids)
    if missing_customer_ids:
        raise ValueError(
            "customer_id contains missing or non-numeric values after casting. "
            "Entity identifiers must be present and numeric."
        )

    missing_target_values = int(df[TARGET_COLUMN].isna().sum())
    LOGGER.info(
        "Missing values before validation for %s: %s",
        TARGET_COLUMN,
        missing_target_values,
    )
    if missing_target_values:
        raise ValueError(
            f"{TARGET_COLUMN} contains missing or non-numeric values after casting. "
            "Target values must be present before preprocessing."
        )


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values conservatively and log counts before filling."""

    imputed_df = df.copy()

    for column in FEATURE_NUMERIC_COLUMNS:
        missing_count = int(imputed_df[column].isna().sum())
        LOGGER.info("Missing values before imputation for %s: %s", column, missing_count)
        if missing_count == 0:
            continue

        median_value = imputed_df[column].median()
        if pd.isna(median_value):
            raise ValueError(
                f"Column '{column}' cannot be median-imputed because it contains only missing values."
            )
        imputed_df[column] = imputed_df[column].fillna(median_value)

    for column in FEATURE_CATEGORICAL_COLUMNS:
        missing_count = int(imputed_df[column].isna().sum())
        LOGGER.info("Missing values before imputation for %s: %s", column, missing_count)
        if missing_count:
            imputed_df[column] = imputed_df[column].fillna("Unknown")

    imputed_df["customer_id"] = imputed_df["customer_id"].astype("int64")
    imputed_df[TARGET_COLUMN] = imputed_df[TARGET_COLUMN].round().astype("int64")

    for column in FEATURE_CATEGORICAL_COLUMNS:
        imputed_df[column] = imputed_df[column].astype("string")

    return imputed_df


def process_dataframe(
    df: pd.DataFrame,
    id_column: str = "CustomerID",
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Transform a raw churn dataframe into the normalized processed schema."""

    validate_raw_columns(df, id_column=id_column)
    processed_df = rename_columns(df, id_column=id_column)
    processed_df = clean_dataframe(processed_df, drop_duplicates=drop_duplicates)
    processed_df = cast_column_types(processed_df)
    validate_required_entity_and_target(processed_df)
    processed_df = impute_missing_values(processed_df)
    return processed_df.loc[:, NORMALIZED_COLUMN_ORDER].copy()


def _ensure_parent_directory(output_path: Path) -> None:
    """Create the parent directory for an output file when needed."""

    output_path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Normalize and clean raw customer churn CSV data."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to the raw input CSV file.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path where the processed CSV will be written.",
    )
    parser.add_argument(
        "--drop-duplicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to drop exact duplicate rows and duplicate customer_id rows.",
    )
    parser.add_argument(
        "--id-column",
        default="CustomerID",
        help="Raw identifier column to map onto customer_id.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for local preprocessing."""

    setup_logging()
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    LOGGER.info("Reading raw CSV from %s", input_path)
    raw_df = pd.read_csv(input_path)
    LOGGER.info("Input row count: %s", len(raw_df))

    processed_df = process_dataframe(
        raw_df,
        id_column=args.id_column,
        drop_duplicates=args.drop_duplicates,
    )

    _ensure_parent_directory(output_path)
    processed_df.to_csv(output_path, index=False)

    LOGGER.info("Output row count: %s", len(processed_df))
    LOGGER.info("Final columns: %s", list(processed_df.columns))
    LOGGER.info("Processed CSV written to %s", output_path)


if __name__ == "__main__":
    main()
