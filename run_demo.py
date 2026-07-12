from pathlib import Path
import argparse
import time
import pandas as pd

from src.synthetic_data import generate_synthetic_demand
from src.features import build_forecasting_dataset, get_feature_columns
from src.models import (
    train_model,
    predict_model,
    train_bagged_models,
    predict_bagged_models,
    extract_feature_importance,
    extract_bagged_feature_importance,
    rule_based_baseline,
)
from src.evaluation import (
    build_metrics_summary,
    build_metrics_by_horizon,
    build_horizon_weighted_final_metrics,
    build_error_breakdown,
)
from src.plots import (
    plot_metric_comparison,
    plot_error_by_horizon,
    plot_weighted_rmse_by_horizon,
    plot_sample_forecast,
    plot_actual_vs_predicted_for_models,
    plot_feature_importance,
)


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


def _record_runtime(runtime_rows: list[dict], stage: str, start_time: float) -> None:
    runtime_rows.append({
        "stage": stage,
        "seconds": round(time.perf_counter() - start_time, 3),
    })


def main() -> None:
    total_start = time.perf_counter()
    runtime_rows = []

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
    stage_start = time.perf_counter()
    raw = generate_synthetic_demand(**generator_kwargs)
    raw.to_csv(DATA_DIR / "synthetic_store_sku_demand.csv", index=False)
    _record_runtime(runtime_rows, "01_generate_synthetic_data", stage_start)

    print(f"Raw rows: {len(raw):,}")
    print(f"Stores: {raw['store_id'].nunique():,}")
    print(f"SKUs: {raw['sku_id'].nunique():,}")
    print(f"Available store-SKU combinations: {raw[['store_id', 'sku_id']].drop_duplicates().shape[0]:,}")

    print("Building forecasting dataset...")
    stage_start = time.perf_counter()
    model_df = build_forecasting_dataset(raw, max_horizon=7)
    model_df.to_csv(DATA_DIR / "forecasting_model_dataset.csv", index=False)
    _record_runtime(runtime_rows, "02_feature_engineering", stage_start)

    split_date = pd.Timestamp("2024-07-01")
    train_df = model_df[model_df["target_date"] < split_date].copy()
    test_df = model_df[model_df["target_date"] >= split_date].copy()

    print(f"Modeling rows: {len(model_df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")

    fitted_models = {}

    print("Training LightGBM without lag features...")
    stage_start = time.perf_counter()
    features_no_lags = get_feature_columns("none")
    model_no_lags = train_model(train_df, features_no_lags)
    test_df["pred_lgbm_no_lags"] = predict_model(model_no_lags, test_df, features_no_lags)
    fitted_models["pred_lgbm_no_lags"] = model_no_lags
    _record_runtime(runtime_rows, "03_train_predict_lgbm_no_lags", stage_start)

    print("Training LightGBM with simple lag and rolling features...")
    stage_start = time.perf_counter()
    features_simple_lags = get_feature_columns("simple")
    model_simple_lags = train_model(train_df, features_simple_lags)
    test_df["pred_lgbm_simple_lags"] = predict_model(model_simple_lags, test_df, features_simple_lags)
    fitted_models["pred_lgbm_simple_lags"] = model_simple_lags
    _record_runtime(runtime_rows, "04_train_predict_lgbm_simple_lags", stage_start)

    print("Training LightGBM with enriched lag features...")
    stage_start = time.perf_counter()
    features_enriched_lags = get_feature_columns("enriched")
    model_enriched_lags = train_model(train_df, features_enriched_lags)
    test_df["pred_lgbm_enriched_lags"] = predict_model(model_enriched_lags, test_df, features_enriched_lags)
    fitted_models["pred_lgbm_enriched_lags"] = model_enriched_lags
    _record_runtime(runtime_rows, "05_train_predict_lgbm_enriched_lags", stage_start)

    print("Training LightGBM with advanced temporal features...")
    stage_start = time.perf_counter()
    features_advanced_temporal = get_feature_columns("advanced")
    model_advanced_temporal = train_model(train_df, features_advanced_temporal)
    test_df["pred_lgbm_advanced_temporal"] = predict_model(model_advanced_temporal, test_df, features_advanced_temporal)
    fitted_models["pred_lgbm_advanced_temporal"] = model_advanced_temporal
    _record_runtime(runtime_rows, "06_train_predict_lgbm_advanced_temporal", stage_start)

    print("Training advanced temporal LightGBM bagging ensemble...")
    stage_start = time.perf_counter()
    bagged_models = train_bagged_models(train_df, features_advanced_temporal)
    test_df["pred_lgbm_advanced_bagged"] = predict_bagged_models(
        bagged_models,
        test_df,
        features_advanced_temporal,
    )
    _record_runtime(runtime_rows, "07_train_predict_lgbm_advanced_bagged", stage_start)

    print("Creating rule-based baseline...")
    stage_start = time.perf_counter()
    test_df["pred_rule_based"] = rule_based_baseline(test_df)
    _record_runtime(runtime_rows, "08_rule_based_baseline", stage_start)

    prediction_cols = [
        "pred_rule_based",
        "pred_lgbm_no_lags",
        "pred_lgbm_simple_lags",
        "pred_lgbm_enriched_lags",
        "pred_lgbm_advanced_temporal",
        "pred_lgbm_advanced_bagged",
    ]

    print("Evaluating models...")
    stage_start = time.perf_counter()
    metrics = build_metrics_summary(test_df, prediction_cols)
    metrics_by_horizon = build_metrics_by_horizon(test_df, prediction_cols)
    horizon_weighted_metrics = build_horizon_weighted_final_metrics(test_df, prediction_cols)
    error_breakdown = build_error_breakdown(test_df, prediction_cols)

    metrics.to_csv(METRICS_DIR / "metrics_summary.csv", index=False)
    metrics_by_horizon.to_csv(METRICS_DIR / "metrics_by_horizon.csv", index=False)
    horizon_weighted_metrics.to_csv(METRICS_DIR / "metrics_horizon_weighted_final.csv", index=False)
    error_breakdown.to_csv(METRICS_DIR / "error_breakdown.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_predictions.csv", index=False)
    _record_runtime(runtime_rows, "09_evaluation_and_error_breakdown", stage_start)

    print("Extracting feature importance...")
    stage_start = time.perf_counter()
    feature_importance_frames = []
    for model_name, model in fitted_models.items():
        feature_importance_frames.append(extract_feature_importance(model, model_name))

    feature_importance_frames.append(
        extract_bagged_feature_importance(bagged_models, "pred_lgbm_advanced_bagged")
    )

    feature_importance = pd.concat(feature_importance_frames, ignore_index=True)
    feature_importance.to_csv(METRICS_DIR / "feature_importance.csv", index=False)
    _record_runtime(runtime_rows, "10_feature_importance", stage_start)

    print("Creating plots...")
    stage_start = time.perf_counter()
    plot_metric_comparison(metrics, FIGURES_DIR / "metric_comparison.png")
    plot_error_by_horizon(metrics_by_horizon, FIGURES_DIR / "error_by_horizon.png")
    plot_weighted_rmse_by_horizon(metrics_by_horizon, FIGURES_DIR / "weighted_rmse_by_horizon.png")
    plot_sample_forecast(test_df, FIGURES_DIR / "sample_forecast_comparison.png")
    plot_actual_vs_predicted_for_models(test_df, prediction_cols, FIGURES_DIR)
    plot_feature_importance(
        feature_importance,
        model_name="pred_lgbm_advanced_bagged",
        output_path=FIGURES_DIR / "feature_importance_pred_lgbm_advanced_bagged.png",
    )
    _record_runtime(runtime_rows, "11_plots", stage_start)

    _record_runtime(runtime_rows, "12_total_runtime", total_start)
    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(METRICS_DIR / "runtime_summary.csv", index=False)

    print("\nMetrics summary:")
    print(metrics.to_string(index=False))

    print("\nHorizon-weighted final forecast metrics:")
    print(horizon_weighted_metrics.to_string(index=False))

    print("\nRuntime summary:")
    print(runtime.to_string(index=False))

    print("\nDone.")
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
