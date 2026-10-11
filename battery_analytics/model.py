"""Train and apply a device-grouped battery-life prediction model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "battery_pct",
    "temperature_c",
    "signal_strength_dbm",
    "active_hours_24h",
    "tx_packets_24h",
    "battery_age_days",
]
CATEGORICAL_FEATURES = ["device_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "remaining_life_hours"


@dataclass
class BatteryLifeModel:
    estimator: Pipeline
    metrics: dict[str, float]


def train_life_model(data: pd.DataFrame, seed: int = 42) -> BatteryLifeModel:
    """Fit a random-forest regressor with held-out devices for evaluation."""
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, "device_id"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required model columns: {', '.join(sorted(missing))}")
    if data["device_id"].nunique() < 5:
        raise ValueError("At least 5 unique devices are required to train the model")

    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, test_indices = next(
        splitter.split(features, target, groups=data["device_id"])
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    estimator = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=120,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    estimator.fit(features.iloc[train_indices], target.iloc[train_indices])
    predictions = estimator.predict(features.iloc[test_indices])
    actual = target.iloc[test_indices]
    metrics = {
        "mae_hours": float(mean_absolute_error(actual, predictions)),
        "rmse_hours": float(np.sqrt(mean_squared_error(actual, predictions))),
        "r2": float(r2_score(actual, predictions)),
    }
    return BatteryLifeModel(estimator=estimator, metrics=metrics)


def predict_remaining_hours(
    model: BatteryLifeModel, data: pd.DataFrame
) -> np.ndarray:
    """Return non-negative remaining-life estimates for telemetry rows."""
    missing = set(FEATURE_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(sorted(missing))}")
    return np.maximum(model.estimator.predict(data[FEATURE_COLUMNS]), 0)