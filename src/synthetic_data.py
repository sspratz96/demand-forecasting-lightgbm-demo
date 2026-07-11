import numpy as np
import pandas as pd


def generate_synthetic_demand(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    n_stores: int = 16,
    n_skus: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily store-SKU demand data.

    The generator intentionally creates temporal dependence so that lag
    and rolling features should improve forecasting performance.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    store_ids = [f"store_{i:02d}" for i in range(1, n_stores + 1)]
    sku_ids = [f"sku_{i:03d}" for i in range(1, n_skus + 1)]

    store_types = ["mall", "street", "office", "tourist"]
    sku_categories = ["fresh", "bakery", "beverage", "prepared", "snack"]

    stores = pd.DataFrame({
        "store_id": store_ids,
        "store_type": rng.choice(store_types, size=n_stores, replace=True),
        "store_base_traffic": rng.uniform(0.7, 1.4, size=n_stores),
    })

    skus = pd.DataFrame({
        "sku_id": sku_ids,
        "sku_category": rng.choice(sku_categories, size=n_skus, replace=True),
        "sku_base_demand": rng.uniform(3, 35, size=n_skus),
    })

    availability = (
        pd.MultiIndex.from_product([store_ids, sku_ids], names=["store_id", "sku_id"])
        .to_frame(index=False)
    )
    availability["available"] = rng.binomial(1, 0.68, size=len(availability))
    availability = availability[availability["available"] == 1].drop(columns="available")

    grid = (
        pd.MultiIndex.from_product([dates, store_ids, sku_ids], names=["date", "store_id", "sku_id"])
        .to_frame(index=False)
        .merge(availability, on=["store_id", "sku_id"], how="inner")
        .merge(stores, on="store_id", how="left")
        .merge(skus, on="sku_id", how="left")
    )

    grid["day_of_week"] = grid["date"].dt.dayofweek
    grid["month"] = grid["date"].dt.month
    grid["is_weekend"] = grid["day_of_week"].isin([5, 6]).astype(int)

    day_index = (grid["date"] - grid["date"].min()).dt.days
    yearly_temp = 18 + 9 * np.sin(2 * np.pi * day_index / 365.25)
    grid["temperature"] = yearly_temp + rng.normal(0, 3, len(grid))
    grid["rain_mm"] = np.maximum(0, rng.gamma(1.2, 2.0, len(grid)) - 1.8)
    grid["is_rainy"] = (grid["rain_mm"] > 1.5).astype(int)

    grid["is_holiday"] = rng.binomial(1, 0.025, len(grid))
    grid["is_promo"] = rng.binomial(1, 0.045, len(grid))

    weekday_effect = grid["day_of_week"].map({
        0: 0.90, 1: 0.95, 2: 1.00, 3: 1.05, 4: 1.15, 5: 1.30, 6: 1.10,
    }).astype(float)

    category_effect = grid["sku_category"].map({
        "fresh": 1.20, "bakery": 1.10, "beverage": 1.00, "prepared": 1.35, "snack": 0.85,
    }).astype(float)

    store_type_effect = grid["store_type"].map({
        "mall": 1.15, "street": 1.00, "office": 0.95, "tourist": 1.25,
    }).astype(float)

    weather_effect = 1 + 0.015 * (grid["temperature"] - grid["temperature"].mean()) - 0.04 * grid["is_rainy"]
    promo_effect = 1 + 0.35 * grid["is_promo"]
    holiday_effect = 1 + 0.20 * grid["is_holiday"]

    base_mu = (
        grid["sku_base_demand"]
        * grid["store_base_traffic"]
        * weekday_effect
        * category_effect
        * store_type_effect
        * weather_effect
        * promo_effect
        * holiday_effect
    )

    grid = grid.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    grid["expected_demand"] = np.maximum(base_mu, 0.1)

    demand_values = []
    for _, group in grid.groupby(["store_id", "sku_id"], sort=False):
        previous = None
        rolling_memory = None
        for mu in group["expected_demand"].to_numpy():
            if previous is None:
                temporal_mu = mu
                rolling_memory = mu
            else:
                temporal_mu = 0.62 * mu + 0.26 * previous + 0.12 * rolling_memory
                rolling_memory = 0.85 * rolling_memory + 0.15 * previous

            noisy_mu = max(0.1, temporal_mu * rng.lognormal(mean=0, sigma=0.16))
            demand = rng.poisson(noisy_mu)
            demand_values.append(demand)
            previous = demand

    grid["demand"] = demand_values
    return grid
