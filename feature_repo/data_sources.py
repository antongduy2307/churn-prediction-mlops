"""Feast data source definitions for the local churn feature repository."""

from feast import FileSource

customer_stats_source = FileSource(
    name="customer_stats_source",
    path="../data/processed/processed_churn_data.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)
