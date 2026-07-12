import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def wmape(y_true, y_pred) -> float:
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denominator


def weighted_rmse(y_true, y_pred, weights) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 1.0)
    weights = np.clip(weights, 1.0, None)
    return np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights))


def evaluate_predictions(df: pd.DataFrame, prediction_col: str) -> dict:
    y_true = df["target"].to_numpy()
    y_pred = df[prediction_col].to_numpy()
    weights = df.get("series_weight", pd.Series(np.ones(len(df)), index=df.index)).to_numpy()

    mse = mean_squared_error(y_true, y_pred)

    try:
        r2 = r2_score(y_true, y_pred)
    except ValueError:
        r2 = np.nan

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mse),
        "weighted_rmse": weighted_rmse(y_true, y_pred, weights),
        "wmape": wmape(y_true, y_pred),
        "r2": r2,
    }


def build_metrics_summary(df: pd.DataFrame, prediction_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in prediction_cols:
        metrics = evaluate_predictions(df, col)
        metrics["model"] = col
        rows.append(metrics)
    return pd.DataFrame(rows)[["model", "mae", "rmse", "weighted_rmse", "wmape", "r2"]]


def build_metrics_by_horizon(df: pd.DataFrame, prediction_cols: list[str]) -> pd.DataFrame:
    rows = []
    for horizon, group in df.groupby("horizon"):
        for col in prediction_cols:
            metrics = evaluate_predictions(group, col)
            metrics["horizon"] = horizon
            metrics["model"] = col
            rows.append(metrics)
    return pd.DataFrame(rows)[["horizon", "model", "mae", "rmse", "weighted_rmse", "wmape", "r2"]]


def build_horizon_weighted_final_metrics(df: pd.DataFrame, prediction_cols: list[str]) -> pd.DataFrame:
    """Evaluate a horizon-weighted 7-day planning forecast.

    The metric compresses horizons 1..7 into one weighted planning signal per
    store/SKU/forecast origin. Nearer horizons receive higher weight.
    """
    working = df.copy()

    max_horizon = working["horizon"].max()
    working["horizon_weight"] = (max_horizon + 1 - working["horizon"]).astype(float)

    group_cols = ["store_id", "sku_id", "date"]

    rows = []
    for col in prediction_cols:
        weighted = working[group_cols + ["target", col, "horizon_weight", "series_weight"]].copy()
        weighted["weighted_target_component"] = weighted["target"] * weighted["horizon_weight"]
        weighted["weighted_pred_component"] = weighted[col] * weighted["horizon_weight"]

        aggregated = (
            weighted.groupby(group_cols, as_index=False)
            .agg(
                weighted_target_component=("weighted_target_component", "sum"),
                weighted_pred_component=("weighted_pred_component", "sum"),
                horizon_weight=("horizon_weight", "sum"),
                series_weight=("series_weight", "mean"),
            )
        )

        aggregated["target"] = aggregated["weighted_target_component"] / aggregated["horizon_weight"]
        aggregated[col] = aggregated["weighted_pred_component"] / aggregated["horizon_weight"]

        metrics = evaluate_predictions(aggregated, col)
        metrics["model"] = col
        metrics["evaluation_view"] = "horizon_weighted_7_day_final"
        rows.append(metrics)

    return pd.DataFrame(rows)[
        ["evaluation_view", "model", "mae", "rmse", "weighted_rmse", "wmape", "r2"]
    ]


def build_error_breakdown(
    df: pd.DataFrame,
    prediction_cols: list[str],
    dimensions: list[str] | None = None,
    min_rows: int = 100,
) -> pd.DataFrame:
    if dimensions is None:
        dimensions = [
            "horizon",
            "sku_velocity",
            "sku_category",
            "store_type",
            "schedule_type",
            "store_id",
        ]

    rows = []

    for dimension in dimensions:
        if dimension not in df.columns:
            continue

        for segment_value, group in df.groupby(dimension, dropna=False):
            if len(group) < min_rows:
                continue

            for col in prediction_cols:
                metrics = evaluate_predictions(group, col)
                metrics["dimension"] = dimension
                metrics["segment"] = str(segment_value)
                metrics["model"] = col
                metrics["rows"] = len(group)
                metrics["target_sum"] = group["target"].sum()
                rows.append(metrics)

    if not rows:
        return pd.DataFrame(
            columns=[
                "dimension",
                "segment",
                "model",
                "rows",
                "target_sum",
                "mae",
                "rmse",
                "weighted_rmse",
                "wmape",
                "r2",
            ]
        )

    return pd.DataFrame(rows)[
        [
            "dimension",
            "segment",
            "model",
            "rows",
            "target_sum",
            "mae",
            "rmse",
            "weighted_rmse",
            "wmape",
            "r2",
        ]
    ]
