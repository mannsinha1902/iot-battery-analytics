"""SQL-backed summaries and observed battery-discharge calculations."""

from __future__ import annotations

import sqlite3

import pandas as pd


DEVICE_TYPE_SUMMARY_SQL = """
SELECT
    device_type,
    COUNT(DISTINCT device_id) AS device_count,
    ROUND(AVG(battery_pct), 1) AS avg_battery_pct,
    ROUND(AVG(remaining_life_hours), 1) AS avg_remaining_life_hours,
    SUM(CASE WHEN battery_pct < 20 THEN 1 ELSE 0 END) AS low_battery_readings,
    ROUND(AVG(temperature_c), 1) AS avg_temperature_c
FROM telemetry
GROUP BY device_type
ORDER BY avg_remaining_life_hours ASC
"""


def query_device_type_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Run an aggregate query against a temporary SQLite telemetry table."""
    required = {
        "device_id",
        "device_type",
        "battery_pct",
        "remaining_life_hours",
        "temperature_c",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required summary columns: {', '.join(sorted(missing))}")
    with sqlite3.connect(":memory:") as connection:
        data[list(required)].to_sql("telemetry", connection, index=False)
        return pd.read_sql_query(DEVICE_TYPE_SUMMARY_SQL, connection)


def calculate_observed_discharge_rates(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate discharge from successive non-charging telemetry samples."""
    required = {"device_id", "timestamp", "battery_pct", "is_charging"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required discharge columns: {', '.join(sorted(missing))}")

    ordered = data.sort_values(["device_id", "timestamp"]).copy()
    grouped = ordered.groupby("device_id", sort=False)
    ordered["previous_battery_pct"] = grouped["battery_pct"].shift()
    ordered["previous_timestamp"] = grouped["timestamp"].shift()
    ordered["previous_is_charging"] = grouped["is_charging"].shift()
    elapsed_hours = (
        ordered["timestamp"] - ordered["previous_timestamp"]
    ).dt.total_seconds() / 3600
    ordered["discharge_rate_pct_per_hour"] = (
        ordered["previous_battery_pct"] - ordered["battery_pct"]
    ) / elapsed_hours
    valid = (
        ordered["is_charging"].eq(False)
        & ordered["previous_is_charging"].fillna(True).eq(False)
        & (ordered["discharge_rate_pct_per_hour"] > 0)
        & (elapsed_hours > 0)
    )
    return ordered.loc[
        valid,
        ["device_id", "device_type", "timestamp", "discharge_rate_pct_per_hour"],
    ].reset_index(drop=True)