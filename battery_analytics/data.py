"""Reproducible synthetic telemetry generation for battery analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEVICE_DRAIN_RATES = {
    "environment_sensor": 0.065,
    "fitness_tracker": 0.14,
    "smart_meter": 0.085,
    "security_camera": 0.42,
    "asset_tracker": 0.19,
}

TELEMETRY_COLUMNS = [
    "device_id",
    "device_type",
    "timestamp",
    "battery_pct",
    "voltage_v",
    "temperature_c",
    "signal_strength_dbm",
    "active_hours_24h",
    "tx_packets_24h",
    "battery_age_days",
    "is_charging",
    "simulated_drain_rate_pct_per_hour",
    "remaining_life_hours",
]


def generate_sensor_data(
    n_devices: int = 120,
    days: int = 14,
    interval_hours: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic device telemetry and known remaining-life targets."""
    if n_devices < 2:
        raise ValueError("n_devices must be at least 2")
    if days < 1:
        raise ValueError("days must be at least 1")
    if interval_hours < 1 or interval_hours > 24:
        raise ValueError("interval_hours must be between 1 and 24")

    rng = np.random.default_rng(seed)
    interval_count = int(np.ceil(days * 24 / interval_hours))
    timestamps = pd.date_range(
        "2025-01-01", periods=interval_count, freq=pd.Timedelta(hours=interval_hours)
    )
    device_types = list(DEVICE_DRAIN_RATES)
    records: list[dict[str, object]] = []

    for device_number in range(n_devices):
        device_type = str(rng.choice(device_types))
        base_drain_rate = DEVICE_DRAIN_RATES[device_type]
        battery_age_days = int(rng.integers(5, 1100))
        usage_baseline = float(rng.uniform(0.15, 0.9))
        battery_pct = float(rng.uniform(72, 100))
        device_id = f"IOT-{device_number + 1:04d}"

        for timestamp in timestamps:
            usage_intensity = float(
                np.clip(rng.normal(usage_baseline, 0.16), 0.05, 1.0)
            )
            active_hours = float(
                np.clip(usage_intensity * 17 + rng.normal(0, 1.5), 0, 24)
            )
            packets = int(rng.poisson(60 + usage_intensity * 980))
            temperature = float(rng.normal(22, 8))
            signal_strength = float(np.clip(rng.normal(-69, 13), -110, -25))
            temperature_factor = 1 + max(temperature - 25, 0) * 0.012
            temperature_factor += max(10 - temperature, 0) * 0.008
            signal_factor = 1 + max(-82 - signal_strength, 0) * 0.003
            age_factor = 1 + battery_age_days / 1800
            drain_rate = base_drain_rate * (0.48 + usage_intensity * 1.05)
            drain_rate *= temperature_factor * signal_factor * age_factor
            drain_rate *= float(rng.lognormal(mean=0, sigma=0.07))
            is_charging = battery_pct <= 20 or rng.random() < 0.012
            remaining_life = float(np.clip(battery_pct / drain_rate, 0, 24 * 30))
            voltage = float(3.0 + battery_pct / 100 * 1.2 + rng.normal(0, 0.025))

            records.append(
                {
                    "device_id": device_id,
                    "device_type": device_type,
                    "timestamp": timestamp,
                    "battery_pct": round(battery_pct, 2),
                    "voltage_v": round(voltage, 3),
                    "temperature_c": round(temperature, 2),
                    "signal_strength_dbm": round(signal_strength, 2),
                    "active_hours_24h": round(active_hours, 2),
                    "tx_packets_24h": packets,
                    "battery_age_days": battery_age_days,
                    "is_charging": bool(is_charging),
                    "simulated_drain_rate_pct_per_hour": round(drain_rate, 5),
                    "remaining_life_hours": round(remaining_life, 2),
                }
            )

            if is_charging:
                battery_pct = min(100.0, battery_pct + 5.5 * interval_hours)
            else:
                battery_pct = max(0.0, battery_pct - drain_rate * interval_hours)

    return pd.DataFrame.from_records(records, columns=TELEMETRY_COLUMNS)