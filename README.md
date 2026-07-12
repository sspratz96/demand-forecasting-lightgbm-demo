# Demand Forecasting LightGBM Demo

Synthetic store-SKU demand forecasting pipeline using LightGBM.

This repository is a public, reproducible demo inspired by a real production planning forecasting project.

It does **not** contain proprietary data, proprietary business rules, internal queries, or company-specific logic.

The purpose is to demonstrate how temporal feature engineering changes model performance when forecasting demand at store-SKU granularity.

---

## Objective

The project compares six approaches:

1. **Rule-based baseline**
   - Forecast demand using recent historical averages.
   - Represents a simple operational heuristic.

2. **LightGBM without lag features**
   - Uses store, SKU, calendar, weather and operational variables.
   - Does not include recent demand history as model inputs.

3. **LightGBM with simple lag and rolling features**
   - Uses the same base variables.
   - Adds a compact set of lagged demand and rolling statistics.

4. **LightGBM with enriched lag features**
   - Adds a broader family of lag-based features, including lag differences, percentage changes, rolling sums, rolling standard deviations, high-demand flags, availability lags, campaign lags and pack lags.

5. **LightGBM with advanced temporal features**
   - Extends the enriched feature set with additional temporal signals such as exponentially weighted moving averages, zero-demand counts, volatility ratios, recent-vs-monthly demand ratios and sales acceleration.

6. **LightGBM advanced bagged ensemble**
   - Trains three LightGBM configurations over the advanced temporal feature set and averages their predictions to improve stability.

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

## Realistic Synthetic Scenario

The default generator is configured to approximate a realistic store-SKU planning context:

- 3 years of daily sales data
- 26 stores
- 200 SKUs
- store-SKU availability restrictions
- stores with different opening schedules
- only some stores closed on weekends
- some stores closed only on Sundays
- some stores open on weekends with reduced demand due to shorter schedules
- low-velocity SKUs selling 0-1 units on many days
- low/medium SKUs selling around 3-5 units depending on store and season
- medium/high SKUs selling from 5 to 35+ units depending on store, season and events
- Christmas, New Year, Mother's Day, Halloween and Chilean Independence Day effects
- winter vacation effects
- rain and winter sensitivity for selected SKUs
- store and SKU category effects
- temporal persistence in demand

Because the default scenario is larger than a toy dataset, the generated forecasting dataset can contain several million rows after expanding each observation into a 7-day forecast horizon.

For a lighter local run, use:

```bash
python run_demo.py --scale demo
```

For the larger realistic scenario, use:

```bash
python run_demo.py --scale realistic
```

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
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

python run_demo.py --scale demo
```

The demo will:

1. Generate synthetic store-SKU daily demand data.
2. Build a 7-day-ahead forecasting dataset.
3. Train a rule-based baseline.
4. Train LightGBM without lag features.
5. Train LightGBM with simple lag features.
6. Train LightGBM with enriched lag features.
7. Train LightGBM with advanced temporal features.
8. Train an advanced LightGBM bagged ensemble.
9. Evaluate model accuracy, runtime, horizon-weighted performance, error breakdowns and feature importance.
10. Save metrics and plots in `outputs/`.

---

## Generated Outputs

```text
outputs/
├── metrics/
│   ├── metrics_summary.csv
│   ├── metrics_by_horizon.csv
│   ├── metrics_horizon_weighted_final.csv
│   ├── runtime_summary.csv
│   ├── error_breakdown.csv
│   └── feature_importance.csv
└── figures/
    ├── metric_comparison.png
    ├── error_by_horizon.png
    ├── weighted_rmse_by_horizon.png
    ├── sample_forecast_comparison.png
    ├── actual_vs_predicted_*.png
    └── feature_importance_pred_lgbm_advanced_bagged.png
```

---

## Synthetic Data Design

The generated dataset simulates:

- 26 stores and 200 SKUs in realistic mode
- daily sales
- store categories
- SKU categories
- product availability restrictions
- store opening schedules
- weekly seasonality
- yearly seasonality
- promotions
- Chilean commercial/event dates
- winter vacation effects
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
- SKU velocity segment
- store schedule type
- target day of week
- target month
- target weekend flag
- target store open flag
- target event flags
- target promotion flag
- target weather variables
- SKU sensitivity flags
- forecast horizon

### Simple lag and rolling features

Used by the simple-lag LightGBM model:

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

### Enriched lag features

Used by the enriched-lag LightGBM model.

For each lag window `[7, 14, 21, 30]`, the model adds:

- lagged demand
- lag difference
- lag percentage change
- rolling average
- rolling sum
- rolling standard deviation
- high-demand flag
- lagged store-open / availability signal
- lagged campaign signal
- lagged pack signal

### Advanced temporal features

Used by the advanced temporal LightGBM model:

- exponential weighted moving averages
- zero-demand counts
- nonzero-demand counts
- rolling coefficients of variation
- recent-vs-monthly demand ratios
- sales acceleration between short and longer windows

### Advanced bagged ensemble

Used by the final LightGBM ensemble.

The ensemble trains three versions of the advanced temporal model:

- shallow and fast
- medium and balanced
- deeper and more detailed

The final prediction is the average of the three model outputs.

---

## Expected Result

The LightGBM model without lag features should learn broad patterns such as store effects, SKU effects, calendar effects, weather effects and promotion effects.

However, it should struggle to capture recent demand changes.

The largest performance gain is expected to come from adding temporal memory through lag and rolling features.

More advanced feature engineering and bagging may continue improving performance, but with diminishing returns and higher runtime cost.

In practice, the advanced temporal model is expected to offer the best balance between accuracy and computational cost, while the bagged model is expected to achieve the best overall accuracy.

---


## Results

The demo run produced the following horizon-weighted final forecast metrics:

| Model | WMAPE | R² | Interpretation |
|---|---:|---:|---|
| Rule-based baseline | 0.4720 | -0.132 | Simple and fast, but limited |
| LightGBM without lags | 0.4473 | 0.045 | Learns broad patterns, but lacks temporal memory |
| LightGBM with simple lags | 0.3816 | 0.274 | Largest improvement from adding recent demand history |
| LightGBM with enriched lags | 0.3735 | 0.306 | Better accuracy from richer temporal feature engineering |
| LightGBM advanced temporal | 0.3695 | 0.321 | Best balance between accuracy and runtime |
| LightGBM advanced bagged | 0.3686 | 0.324 | Best overall accuracy, but highest training cost |

### Accuracy vs Complexity Trade-off

The largest performance gain came from adding temporal memory through lag and rolling features.

More advanced feature engineering and bagging continued improving performance, but with diminishing returns and higher runtime cost.

The advanced temporal model offered the best balance between accuracy and computational cost, while the bagged model achieved the best overall accuracy.

### Runtime Benchmark

In one local demo run, total runtime was approximately 20.9 minutes.

| Stage | Runtime |
|---|---:|
| Synthetic data generation | 8.4 seconds |
| Feature engineering | 267.4 seconds |
| LightGBM without lags | 46.2 seconds |
| LightGBM simple lags | 70.5 seconds |
| LightGBM enriched lags | 135.7 seconds |
| LightGBM advanced temporal | 156.8 seconds |
| LightGBM advanced bagged | 496.7 seconds |
| Evaluation and diagnostics | 43.8 seconds |
| Plots | 4.6 seconds |

The results show that model complexity should be evaluated not only by accuracy, but also by runtime, maintainability and operational cost.

## Disclaimer

This repository uses synthetic data only.

It is designed for portfolio and educational purposes.

It does not reproduce proprietary datasets, confidential queries, internal business logic, or company-specific forecasting rules.
