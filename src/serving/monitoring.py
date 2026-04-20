"""Local-first drift monitoring utilities using Evidently."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_REFERENCE_DATA_PATH = Path("data/processed/df_processed.csv")
DEFAULT_CURRENT_DATA_PATH = Path("data/processed/processed_churn_data.parquet")
DEFAULT_REPORT_DIR = Path("reports/drift")

COMPARABLE_COLUMNS: list[str] = [
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
]

OPTIONAL_COMPARABLE_COLUMNS: list[str] = ["churned"]


@dataclass(frozen=True)
class DriftReportResult:
    """Result metadata for a generated drift report."""

    status: str
    reference_path: str
    current_path: str
    report_path: str
    compared_columns: list[str]


def _load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV or parquet dataset for drift reporting."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported dataset format '{suffix}'. Use .csv or .parquet.")


def _resolve_output_path(output_path: str | None) -> Path:
    """Resolve the drift report output path."""

    if output_path:
        return Path(output_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_REPORT_DIR / f"drift_report_{timestamp}.html"


def _select_comparable_columns(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> list[str]:
    """Validate and return the comparable columns shared by both datasets."""

    missing_reference = [column for column in COMPARABLE_COLUMNS if column not in reference_df.columns]
    missing_current = [column for column in COMPARABLE_COLUMNS if column not in current_df.columns]
    if missing_reference or missing_current:
        raise ValueError(
            "Reference and current datasets do not share the required comparable columns. "
            f"missing_in_reference={missing_reference}, missing_in_current={missing_current}"
        )

    selected_columns = list(COMPARABLE_COLUMNS)
    for column in OPTIONAL_COMPARABLE_COLUMNS:
        if column in reference_df.columns and column in current_df.columns:
            selected_columns.append(column)

    return selected_columns


def generate_drift_report(
    reference_path: str | None = None,
    current_path: str | None = None,
    output_path: str | None = None,
) -> DriftReportResult:
    """Generate and save an Evidently HTML drift report."""

    resolved_reference_path = Path(reference_path) if reference_path else DEFAULT_REFERENCE_DATA_PATH
    resolved_current_path = Path(current_path) if current_path else DEFAULT_CURRENT_DATA_PATH
    resolved_output_path = _resolve_output_path(output_path)

    reference_df = _load_dataset(resolved_reference_path)
    current_df = _load_dataset(resolved_current_path)
    comparable_columns = _select_comparable_columns(reference_df, current_df)

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise RuntimeError(
            "Evidently is not installed in the active environment. "
            "Install the project dependencies and retry."
        ) from exc

    report = Report([DataDriftPreset()])

    try:
        evaluation = report.run(
            current_data=current_df.loc[:, comparable_columns],
            reference_data=reference_df.loc[:, comparable_columns],
        )
    except Exception as exc:
        raise RuntimeError("Failed to generate the Evidently drift report.") from exc

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.save_html(str(resolved_output_path))

    return DriftReportResult(
        status="ok",
        reference_path=str(resolved_reference_path),
        current_path=str(resolved_current_path),
        report_path=str(resolved_output_path),
        compared_columns=comparable_columns,
    )
