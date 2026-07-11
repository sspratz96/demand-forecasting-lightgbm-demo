import pandas as pd


BASE_FEATURES = [
    "store_id",
    "sku_id",
    "store_type",
    "sku_category",
    "sku_velocity",
    "schedule_type",
    "horizon",
    "target_day_of_week",
    "target_month",
    "target_is_weekend",
    "target_is_store_open",
    "target_is_new_year",
    "target_is_christmas",
    "target_is_mothers_day",
    "target_is_halloween",
    "target_is_chile_independence",
    "target_is_winter_break",
    "target_is_special_event",
    "target_is_promo",
    "target_temperature",
    "target_rain_mm",
    "target_is_rainy",
    "rain_sensitive",
    "winter_sensitive",
    "holiday_sensitive",
    "september_sensitive",
]

LAG_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_min_7",
    "rolling_max_7",
    "demand_momentum_7_28",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features using only past demand."""
    df = df.sort_values(["store_id", "sku_id", "date"]).copy()
    group = df.groupby(["store_id", "sku_id"], sort=False)["demand"]

    for lag in [1, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = group.shift(lag)

    shifted = group.shift(1)
    grouped_shifted = shifted.groupby([df["store_id"], df["sku_id"]])

    df["rolling_mean_7"] = grouped_shifted.rolling(7).mean().reset_index(level=[0, 1], drop=True)
    df["rolling_mean_14"] = grouped_shifted.rolling(14).mean().reset_index(level=[0, 1], drop=True)
    df["rolling_mean_28"] = grouped_shifted.rolling(28).mean().reset_index(level=[0, 1], drop=True)
    df["rolling_std_7"] = grouped_shifted.rolling(7).std().reset_index(level=[0, 1], drop=True)
    df["rolling_std_14"] = grouped_shifted.rolling(14).std().reset_index(level=[0, 1], drop=True)
    df["rolling_min_7"] = grouped_shifted.rolling(7).min().reset_index(level=[0, 1], drop=True)
    df["rolling_max_7"] = grouped_shifted.rolling(7).max().reset_index(level=[0, 1], drop=True)
    df["demand_momentum_7_28"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + 1e-6)

    return df


def build_forecasting_dataset(
    raw_df: pd.DataFrame,
    max_horizon: int = 7,
    min_history_days: int = 35,
) -> pd.DataFrame:
    """Build a supervised forecasting dataset.

    Each row is a forecast origin date plus a forecast horizon.
    The target is demand observed on origin_date + horizon.
    """
    df = add_temporal_features(raw_df)
    frames = []

    target_covariates = raw_df[[
        "store_id",
        "sku_id",
        "date",
        "is_store_open",
        "is_new_year",
        "is_christmas",
        "is_mothers_day",
        "is_halloween",
        "is_chile_independence",
        "is_winter_break",
        "is_special_event",
        "is_promo",
        "temperature",
        "rain_mm",
        "is_rainy",
    ]].rename(columns={
        "date": "target_date",
        "is_store_open": "target_is_store_open",
        "is_new_year": "target_is_new_year",
        "is_christmas": "target_is_christmas",
        "is_mothers_day": "target_is_mothers_day",
        "is_halloween": "target_is_halloween",
        "is_chile_independence": "target_is_chile_independence",
        "is_winter_break": "target_is_winter_break",
        "is_special_event": "target_is_special_event",
        "is_promo": "target_is_promo",
        "temperature": "target_temperature",
        "rain_mm": "target_rain_mm",
        "is_rainy": "target_is_rainy",
    })

    for horizon in range(1, max_horizon + 1):
        origin = df.copy()
        target = df[["store_id", "sku_id", "date", "demand"]].copy()
        target["forecast_origin_date"] = target["date"] - pd.to_timedelta(horizon, unit="D")
        target = target.rename(columns={"date": "target_date", "demand": "target"})

        merged = origin.merge(
            target,
            left_on=["store_id", "sku_id", "date"],
            right_on=["store_id", "sku_id", "forecast_origin_date"],
            how="inner",
        )

        merged["horizon"] = horizon
        merged["target_day_of_week"] = merged["target_date"].dt.dayofweek
        merged["target_month"] = merged["target_date"].dt.month
        merged["target_is_weekend"] = merged["target_day_of_week"].isin([5, 6]).astype(int)

        merged = merged.merge(target_covariates, on=["store_id", "sku_id", "target_date"], how="left")
        frames.append(merged)

    model_df = pd.concat(frames, ignore_index=True)
    model_df["history_days"] = (
        model_df["date"] - model_df.groupby(["store_id", "sku_id"])["date"].transform("min")
    ).dt.days
    model_df = model_df[model_df["history_days"] >= min_history_days].copy()

    return model_df


def get_feature_columns(use_lag_features: bool) -> list[str]:
    if use_lag_features:
        return BASE_FEATURES + LAG_FEATURES
    return BASE_FEATURES
