# Demand Forecasting LightGBM Demo

Synthetic store-SKU demand forecasting pipeline using LightGBM.

This repository is a public, reproducible demo inspired by a real production planning forecasting project.

It does **not** contain proprietary data, proprietary business rules, internal queries, or company-specific logic.

The purpose is to demonstrate how temporal feature engineering changes model performance when forecasting demand at store-SKU granularity.

---

## Objective

The project compares three approaches:

1. **Rule-based baseline**
   - Forecast demand using recent historical averages.
   - Represents a simple operational heuristic.

2. **LightGBM without lag features**
   - Uses store, SKU, calendar, weather and operational variables.
   - Does not include recent demand history as model inputs.

3. **LightGBM with lag and rolling features**
   - Uses the same base variables.
   - Adds lagged demand, rolling averages and rolling volatility.
   - Tests whether explicitly encoding temporal dependency improves forecast performance.

The key question is:

> How much does model performance improve when temporal demand history is encoded directly into the feature set?

---

## Why This Matters

Tree-based tabular models such as LightGBM do not automatically understand row order.

If each row is treated as an independent observation, the model may fail to learn that recent sales influence near-future demand.

This demo shows that adding features such as:

- previous 7-day demand
- previous 14-day demand
- rolling averages
- rolling standard deviations
- recent demand momentum

can materially improve forecasting quality.

---

## Repository Structure

```text
demand-forecasting-lightgbm-demo/
├── README.md
├── requirements.txt
├── run_demo.py
├── src/
│   ├── synthetic_data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── plots.py
├── data/
│   └── sample/
├── outputs/
│   ├── figures/
│   └── metrics/
└── notebooks/
    └── README.md
```

---

## Quick Start

```bash
git clone https://github.com/sspratz96/demand-forecasting-lightgbm-demo.git
cd demand-forecasting-lightgbm-demo

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

python run_demo.py
```

The demo will:

1. Generate synthetic store-SKU daily demand data.
2. Build a 7-day-ahead forecasting dataset.
3. Train LightGBM without lag features.
4. Train LightGBM with lag and rolling features.
5. Compare both models against a rule-based baseline.
6. Save metrics and plots in `outputs/`.

---

## Generated Outputs

```text
outputs/
├── metrics/
│   ├── metrics_summary.csv
│   └── metrics_by_horizon.csv
└── figures/
    ├── metric_comparison.png
    ├── error_by_horizon.png
    └── sample_forecast_comparison.png
```

---

## Synthetic Data Design

The generated dataset simulates:

- multiple stores
- multiple SKUs
- daily sales
- store categories
- SKU categories
- product availability
- weekly seasonality
- yearly seasonality
- promotions
- holidays
- weather effects
- demand persistence
- store-SKU-specific behavior
- random noise

The data is intentionally synthetic, but structured to behave like a real operational demand planning dataset.

---

## Experiment Design

The target is daily demand at store-SKU level for the next 7 days.

Each training row represents a forecast origin date and a forecast horizon.

For example:

```text
forecast_origin_date = 2024-05-01
horizon = 3
target_date = 2024-05-04
target = demand on 2024-05-04
```

The model is trained to forecast demand using information available at the forecast origin date.

---

## Feature Sets

### Base features

Used by both LightGBM models:

- store ID
- SKU ID
- store type
- SKU category
- target day of week
- target month
- target weekend flag
- target holiday flag
- promotion flag
- weather variables
- product availability
- forecast horizon

### Lag and rolling features

Used only by the second LightGBM model:

- lag 1
- lag 7
- lag 14
- lag 21
- lag 28
- rolling mean 7
- rolling mean 14
- rolling mean 28
- rolling standard deviation 7
- rolling standard deviation 14
- rolling minimum 7
- rolling maximum 7
- recent demand momentum

---

## Expected Result

The LightGBM model without lag features should learn broad patterns such as store effects, SKU effects, calendar effects, weather effects and promotion effects.

However, it should struggle to capture recent demand changes.

The LightGBM model with lag and rolling features should generally perform better because the model receives explicit information about recent demand behavior.

---

## Disclaimer

This repository uses synthetic data only.

It is designed for portfolio and educational purposes.

It does not reproduce proprietary datasets, confidential queries, internal business logic, or company-specific forecasting rules.
