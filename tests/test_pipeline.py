import numpy as np
import pytest

from battery_analytics.analytics import (
    calculate_observed_discharge_rates,
    query_device_type_summary,
)
from battery_analytics.data import generate_sensor_data
from battery_analytics.model import predict_remaining_hours, train_life_model


def test_synthetic_data_is_reproducible_and_well_formed():
    first = generate_sensor_data(n_devices=8, days=3, seed=11)
    second = generate_sensor_data(n_devices=8, days=3, seed=11)

    assert first.equals(second)
    assert first["device_id"].nunique() == 8
    assert first["battery_pct"].between(0, 100).all()
    assert (first["remaining_life_hours"] >= 0).all()
    assert first["is_charging"].dtype == bool


def test_generator_rejects_invalid_device_count():
    with pytest.raises(ValueError, match="n_devices"):
        generate_sensor_data(n_devices=1)


def test_model_trains_and_predicts_for_held_out_devices():
    data = generate_sensor_data(n_devices=30, days=5, seed=7)
    model = train_life_model(data)
    predictions = predict_remaining_hours(model, data.head(12))

    assert predictions.shape == (12,)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
    assert model.metrics["mae_hours"] >= 0
    assert np.isfinite(model.metrics["r2"])


def test_sql_summary_and_observed_discharge_rates():
    data = generate_sensor_data(n_devices=12, days=5, seed=19)
    summary = query_device_type_summary(data)
    discharge = calculate_observed_discharge_rates(data)

    assert not summary.empty
    assert summary["device_count"].sum() == data["device_id"].nunique()
    assert not discharge.empty
    assert (discharge["discharge_rate_pct_per_hour"] > 0).all()