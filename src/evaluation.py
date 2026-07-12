import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def wmape(y_true, y_pred) -> float:
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denominator


def evaluate_predictions(df: pd.DataFrame, prediction_col: str) -> dict:
    y_true = df["target"].to_numpy()
    y_pred = df[prediction_col].to_numpy()

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mse),
        "wmape": wmape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def build_metrics_summary(df: pd.DataFrame, prediction_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in prediction_cols:
        metrics = evaluate_predictions(df, col)
        metrics["model"] = col
        rows.append(metrics)
    return pd.DataFrame(rows)[["model", "mae", "rmse", "wmape", "r2"]]


def build_metrics_by_horizon(df: pd.DataFrame, prediction_cols: list[str]) -> pd.DataFrame:
    rows = []
    for horizon, group in df.groupby("horizon"):
        for col in prediction_cols:
            metrics = evaluate_predictions(group, col)
            metrics["horizon"] = horizon
            metrics["model"] = col
            rows.append(metrics)
    return pd.DataFrame(rows)[["horizon", "model", "mae", "rmse", "wmape", "r2"]]