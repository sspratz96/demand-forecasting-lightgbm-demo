from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    metric = "wmape"
    ordered = metrics.sort_values(metric)

    plt.figure(figsize=(8, 5))
    plt.bar(ordered["model"], ordered[metric])
    plt.title("Forecast Error Comparison (WMAPE)")
    plt.ylabel("WMAPE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_error_by_horizon(metrics_by_horizon: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
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


def plot_sample_forecast(test_df: pd.DataFrame, output_path: Path) -> None:
    sample_keys = (
        test_df.groupby(["store_id", "sku_id"])["target"].sum().sort_values(ascending=False).head(1).index[0]
    )

    sample = test_df[
        (test_df["store_id"] == sample_keys[0])
        & (test_df["sku_id"] == sample_keys[1])
        & (test_df["horizon"] == 1)
    ].sort_values("target_date").tail(90)

    plt.figure(figsize=(11, 5))
    plt.plot(sample["target_date"], sample["target"], label="actual")
    plt.plot(sample["target_date"], sample["pred_rule_based"], label="rule_based")
    plt.plot(sample["target_date"], sample["pred_lgbm_no_lags"], label="lgbm_no_lags")
    plt.plot(sample["target_date"], sample["pred_lgbm_with_lags"], label="lgbm_with_lags")
    plt.title(f"Sample 1-Day Forecast: {sample_keys[0]} / {sample_keys[1]}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
