# IoT Battery Performance Analytics

A portfolio-ready Python project for synthetic IoT device battery monitoring. It generates sensor telemetry, runs SQL-style analysis, trains a machine-learning model for remaining battery life, and exposes an interactive Streamlit dashboard.

## Features

- Synthetic telemetry generation for multiple device families and charging states
- Battery discharge analysis using timestamp differencing and SQL aggregation
- A leakage-aware remaining-life prediction model built with scikit-learn
- Interactive dashboard for health, trend, and forecasting insights
- Unit tests covering data generation, model behavior, and summary calculations
- GitHub Actions CI workflow for automated validation

## Project structure

- `battery_analytics/data.py` — synthetic telemetry generation
- `battery_analytics/analytics.py` — SQLite-based summaries and discharge analysis
- `battery_analytics/model.py` — feature engineering and battery-life modeling
- `app.py` — interactive Streamlit dashboard
- `tests/test_pipeline.py` — regression and behavior tests
- `.github/workflows/python-tests.yml` — CI workflow

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate  # macOS/Linux
# or .\.venv\Scripts\activate  # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

## How it works

1. Synthetic telemetry is created for many devices over time with realistic usage variability, temperature, signal, charging states, and battery age.
2. Battery discharge is estimated by comparing successive non-charging readings over time.
3. A grouped train/test split keeps all readings from the same device in one fold, which helps avoid leakage in a realistic monitoring scenario.
4. A random-forest regressor predicts remaining life in hours from current battery, age, usage, temperature, and type.
5. The Streamlit app shows summary metrics, time-series trends, discharge distributions, and model performance.

## Notable notes

- No private employer data or customer telemetry is used.
- The app is fully synthetic and safe for demonstration, interviews, and portfolio projects.
- The model is designed to provide a realistic example rather than a production-ready field deployment.

## License

This project is intended for learning, portfolio, and demonstration use.
