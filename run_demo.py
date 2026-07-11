from pathlib import Path
import argparse
import pandas as pd

from src.synthetic_data import generate_synthetic_demand
from src.features import build_forecasting_dataset, get_feature_columns
from src.models import train_model, predict_model, rule_based_baseline
from src.evaluation import build_metrics_summary, build_metrics_by_horizon
from src.plots import plot_metric_comparison, plot_error_by_horizon, plot_sample_forecast


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "sample"
METRICS_DIR = ROOT / "outputs" / "metrics"
FIGURES_DIR = ROOT / "outputs" / "figures"

for path in [DATA_DIR, METRICS_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic LightGBM demand forecasting demo.")
    parser.add_argument(
        "--scale",
        choices=["demo", "realistic"],
        default="demo",
        help="demo is lighter for local laptops; realistic uses 26 stores and 200 SKUs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.scale == "realistic":
        generator_kwargs = {
            "start_date": "2022-01-01",
            "end_date": "2024-12-31",
            "n_stores": 26,
            "n_skus": 200,
            "availability_rate": 0.62,
        }
    else:
        generator_kwargs = {
            "start_date": "2022-01-01",
            "end_date": "2024-12-31",
            "n_stores": 12,
            "n_skus": 80,
            "availability_rate": 0.62,
        }

    print(f"Generating synthetic demand data with scale={args.scale}...")
    raw = generate_synthetic_demand(**generator_kwargs)
    raw.to_csv(DATA_DIR / "synthetic_store_sku_demand.csv", index=False)

    print(f"Raw rows: {len(raw):,}")
    print(f"Stores: {raw['store_id'].nunique():,}")
    print(f"SKUs: {raw['sku_id'].nunique():,}")
    print(f"Available store-SKU combinations: {raw[['store_id', 'sku_id']].drop_duplicates().shape[0]:,}")

    print("Building forecasting dataset...")
    model_df = build_forecasting_dataset(raw, max_horizon=7)
    model_df.to_csv(DATA_DIR / "forecasting_model_dataset.csv", index=False)

    split_date = pd.Timestamp("2024-07-01")
    train_df = model_df[model_df["target_date"] < split_date].copy()
    test_df = model_df[model_df["target_date"] >= split_date].copy()

    print(f"Modeling rows: {len(model_df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")

    print("Training LightGBM without lag features...")
    features_no_lags = get_feature_columns(use_lag_features=False)
    model_no_lags = train_model(train_df, features_no_lags)
    test_df["pred_lgbm_no_lags"] = predict_model(model_no_lags, test_df, features_no_lags)

    print("Training LightGBM with lag and rolling features...")
    features_with_lags = get_feature_columns(use_lag_features=True)
    model_with_lags = train_model(train_df, features_with_lags)
    test_df["pred_lgbm_with_lags"] = predict_model(model_with_lags, test_df, features_with_lags)

    print("Creating rule-based baseline...")
    test_df["pred_rule_based"] = rule_based_baseline(test_df)

    prediction_cols = ["pred_rule_based", "pred_lgbm_no_lags", "pred_lgbm_with_lags"]

    print("Evaluating models...")
    metrics = build_metrics_summary(test_df, prediction_cols)
    metrics_by_horizon = build_metrics_by_horizon(test_df, prediction_cols)

    metrics.to_csv(METRICS_DIR / "metrics_summary.csv", index=False)
    metrics_by_horizon.to_csv(METRICS_DIR / "metrics_by_horizon.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_predictions.csv", index=False)

    print("Creating plots...")
    plot_metric_comparison(metrics, FIGURES_DIR / "metric_comparison.png")
    plot_error_by_horizon(metrics_by_horizon, FIGURES_DIR / "error_by_horizon.png")
    plot_sample_forecast(test_df, FIGURES_DIR / "sample_forecast_comparison.png")

    print("\nMetrics summary:")
    print(metrics.to_string(index=False))

    print("\nDone.")
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
