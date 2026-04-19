"""Feast entity definitions for the local churn feature repository."""

from feast import Entity

customer = Entity(
    name="customer",
    join_keys=["customer_id"],
)
