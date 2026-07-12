from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def plot_metric_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    metric = "wmape"
    ordered = metrics.sort_values(metric)

    plt.figure(figsize=(10, 5))
    plt.bar(ordered["model"], ordered[metric])
    plt.title("Forecast Error Comparison (WMAPE)")
    plt.ylabel("WMAPE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_error_by_horizon(metrics_by_horizon: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for model, group in metrics_by_horizon.groupby("model"):
        group = group.sort_values("horizon")
        plt.plot(group["horizon"], group["wmape"], marker="o", label=model)
    plt.title("WMAPE by Forecast Horizon")
    plt.xlabel("Forecast horizon")
    plt.ylabel("WMAPE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_weighted_rmse_by_horizon(metrics_by_horizon: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for model, group in metrics_by_horizon.groupby("model"):
        group = group.sort_values("horizon")
        plt.plot(group["horizon"], group["weighted_rmse"], marker="o", label=model)
    plt.title("Weighted RMSE by Forecast Horizon")
    plt.xlabel("Forecast horizon")
    plt.ylabel("Weighted RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_sample_forecast(test_df: pd.DataFrame, output_path: Path) -> None:
    sample_keys = (
        test_df.groupby(["store_id", "sku_id"])["target"]
        .sum()
        .sort_values(ascending=False)
        .head(1)
        .index[0]
    )

    sample = test_df[
        (test_df["store_id"] == sample_keys[0])
        & (test_df["sku_id"] == sample_keys[1])
        & (test_df["horizon"] == 1)
    ].sort_values("target_date").tail(90)

    plt.figure(figsize=(12, 5))
    plt.plot(sample["target_date"], sample["target"], label="actual")
    plt.plot(sample["target_date"], sample["pred_rule_based"], label="rule_based")
    plt.plot(sample["target_date"], sample["pred_lgbm_no_lags"], label="lgbm_no_lags")
    plt.plot(sample["target_date"], sample["pred_lgbm_simple_lags"], label="lgbm_simple_lags")
    plt.plot(sample["target_date"], sample["pred_lgbm_enriched_lags"], label="lgbm_enriched_lags")
    plt.plot(sample["target_date"], sample["pred_lgbm_advanced_temporal"], label="lgbm_advanced_temporal")
    if "pred_lgbm_advanced_bagged" in sample.columns:
        plt.plot(sample["target_date"], sample["pred_lgbm_advanced_bagged"], label="lgbm_advanced_bagged")
    plt.title(f"Sample 1-Day Forecast: {sample_keys[0]} / {sample_keys[1]}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_actual_vs_predicted(
    test_df: pd.DataFrame,
    prediction_col: str,
    output_path: Path,
    max_points: int = 60000,
    random_state: int = 42,
) -> None:
    """Create actual-vs-predicted scatter plot with x=y reference line."""
    plot_df = test_df[["target", prediction_col]].dropna().copy()

    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=random_state)

    x = plot_df["target"].to_numpy()
    y = plot_df[prediction_col].to_numpy()

    max_value = float(np.nanmax([x.max(), y.max()]))
    max_value = max(max_value, 1.0)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.18, s=8)
    plt.plot([0, max_value], [0, max_value], linestyle="--", linewidth=1.5, label="x = y")
    plt.title(f"Actual vs Predicted: {prediction_col}")
    plt.xlabel("Actual demand")
    plt.ylabel("Predicted demand")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_actual_vs_predicted_for_models(
    test_df: pd.DataFrame,
    prediction_cols: list[str],
    output_dir: Path,
) -> None:
    for col in prediction_cols:
        plot_actual_vs_predicted(
            test_df=test_df,
            prediction_col=col,
            output_path=output_dir / f"actual_vs_predicted_{_safe_filename(col)}.png",
        )


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    model_name: str,
    output_path: Path,
    top_n: int = 25,
) -> None:
    model_importance = (
        feature_importance[feature_importance["model"] == model_name]
        .sort_values("importance_normalized", ascending=False)
        .head(top_n)
        .sort_values("importance_normalized", ascending=True)
    )

    plt.figure(figsize=(9, 7))
    plt.barh(model_importance["feature"], model_importance["importance_normalized"])
    plt.title(f"Top Feature Importance: {model_name}")
    plt.xlabel("Normalized importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
