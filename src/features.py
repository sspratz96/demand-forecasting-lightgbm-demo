import pandas as pd


ORIGINAL_LAG_WINDOWS = [7, 14, 21, 30]

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
    "target_is_campaign",
    "target_temperature",
    "target_rain_mm",
    "target_is_rainy",
    "rain_sensitive",
    "winter_sensitive",
    "holiday_sensitive",
    "september_sensitive",
    "campaign_sensitive",
    "is_pack_sku",
]

SIMPLE_LAG_FEATURES = [
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

ENRICHED_LAG_FEATURES = []
for lag in ORIGINAL_LAG_WINDOWS:
    ENRICHED_LAG_FEATURES.extend([
        f"lagged_demand_{lag}",
        f"diff_lag_{lag}",
        f"pct_change_lag_{lag}",
        f"rolling_avg_lag_{lag}",
        f"rolling_sum_lag_{lag}",
        f"rolling_std_lag_{lag}",
        f"high_demand_lag_{lag}",
        f"lag_disp_{lag}",
        f"campaign_lag_{lag}",
        f"pack_lag_{lag}",
    ])

ADVANCED_TEMPORAL_FEATURES = [
    "ewm_mean_7",
    "ewm_mean_14",
    "ewm_mean_30",
    "rolling_zero_count_7",
    "rolling_zero_count_30",
    "rolling_nonzero_count_30",
    "rolling_cv_14",
    "rolling_cv_30",
    "sales_acceleration_7_30",
    "recent_vs_monthly_ratio",
]

LAG_FEATURE_SETS = {
    "none": [],
    "simple": SIMPLE_LAG_FEATURES,
    "enriched": ENRICHED_LAG_FEATURES,
    "advanced": ENRICHED_LAG_FEATURES + ADVANCED_TEMPORAL_FEATURES,
}


def _safe_pct_change(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / (denominator.abs() + 1e-6)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features using only information available before forecast origin.

    No backward fill is used. Rolling windows are calculated after shifting demand.
    """
    df = df.sort_values(["store_id", "sku_id", "date"]).copy()
    keys = [df["store_id"], df["sku_id"]]

    demand_group = df.groupby(["store_id", "sku_id"], sort=False)["demand"]
    open_group = df.groupby(["store_id", "sku_id"], sort=False)["is_store_open"]
    campaign_group = df.groupby(["store_id", "sku_id"], sort=False)["is_campaign"]
    pack_group = df.groupby(["store_id", "sku_id"], sort=False)["is_pack_sku"]

    shifted_demand = demand_group.shift(1)
    shifted_group = shifted_demand.groupby(keys)

    for lag in [1, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = demand_group.shift(lag)

    df["rolling_mean_7"] = shifted_group.rolling(7).mean().reset_index(level=[0, 1], drop=True)
    df["rolling_mean_14"] = shifted_group.rolling(14).mean().reset_index(level=[0, 1], drop=True)
    df["rolling_mean_28"] = shifted_group.rolling(28).mean().reset_index(level=[0, 1], drop=True)
    df["rolling_std_7"] = shifted_group.rolling(7).std().reset_index(level=[0, 1], drop=True)
    df["rolling_std_14"] = shifted_group.rolling(14).std().reset_index(level=[0, 1], drop=True)
    df["rolling_min_7"] = shifted_group.rolling(7).min().reset_index(level=[0, 1], drop=True)
    df["rolling_max_7"] = shifted_group.rolling(7).max().reset_index(level=[0, 1], drop=True)
    df["demand_momentum_7_28"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + 1e-6)

    reference_recent = demand_group.shift(1)

    for lag in ORIGINAL_LAG_WINDOWS:
        lagged = demand_group.shift(lag)
        rolling_avg = shifted_group.rolling(lag).mean().reset_index(level=[0, 1], drop=True)
        rolling_sum = shifted_group.rolling(lag).sum().reset_index(level=[0, 1], drop=True)
        rolling_std = shifted_group.rolling(lag).std().reset_index(level=[0, 1], drop=True)

        df[f"lagged_demand_{lag}"] = lagged
        df[f"diff_lag_{lag}"] = reference_recent - lagged
        df[f"pct_change_lag_{lag}"] = _safe_pct_change(df[f"diff_lag_{lag}"], lagged)
        df[f"rolling_avg_lag_{lag}"] = rolling_avg
        df[f"rolling_sum_lag_{lag}"] = rolling_sum
        df[f"rolling_std_lag_{lag}"] = rolling_std
        df[f"high_demand_lag_{lag}"] = (lagged > (rolling_avg + rolling_std)).astype(float)
        df[f"lag_disp_{lag}"] = open_group.shift(lag)
        df[f"campaign_lag_{lag}"] = campaign_group.shift(lag)
        df[f"pack_lag_{lag}"] = pack_group.shift(lag)

    df["ewm_mean_7"] = shifted_demand.groupby(keys).ewm(span=7, adjust=False).mean().reset_index(level=[0, 1], drop=True)
    df["ewm_mean_14"] = shifted_demand.groupby(keys).ewm(span=14, adjust=False).mean().reset_index(level=[0, 1], drop=True)
    df["ewm_mean_30"] = shifted_demand.groupby(keys).ewm(span=30, adjust=False).mean().reset_index(level=[0, 1], drop=True)

    zero_shifted = (shifted_demand == 0).astype(float)
    zero_group = zero_shifted.groupby(keys)
    df["rolling_zero_count_7"] = zero_group.rolling(7).sum().reset_index(level=[0, 1], drop=True)
    df["rolling_zero_count_30"] = zero_group.rolling(30).sum().reset_index(level=[0, 1], drop=True)
    df["rolling_nonzero_count_30"] = 30 - df["rolling_zero_count_30"]

    df["rolling_cv_14"] = df["rolling_std_lag_14"] / (df["rolling_avg_lag_14"].abs() + 1e-6)
    df["rolling_cv_30"] = df["rolling_std_lag_30"] / (df["rolling_avg_lag_30"].abs() + 1e-6)
    df["sales_acceleration_7_30"] = df["rolling_avg_lag_7"] - df["rolling_avg_lag_30"]
    df["recent_vs_monthly_ratio"] = df["rolling_avg_lag_7"] / (df["rolling_avg_lag_30"] + 1e-6)

    return df


def build_forecasting_dataset(raw_df: pd.DataFrame, max_horizon: int = 7, min_history_days: int = 35) -> pd.DataFrame:
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
        "is_campaign",
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
        "is_campaign": "target_is_campaign",
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
    model_df["series_weight"] = model_df["rolling_avg_lag_30"].fillna(model_df["target"].mean()).clip(lower=1.0)

    return model_df


def get_feature_columns(feature_set: str = "none") -> list[str]:
    if feature_set not in LAG_FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set}. Expected one of {list(LAG_FEATURE_SETS)}.")
    return BASE_FEATURES + LAG_FEATURE_SETS[feature_set]
