import numpy as np
import pandas as pd


def _second_sunday_of_may(year: int) -> pd.Timestamp:
    may = pd.date_range(f"{year}-05-01", f"{year}-05-31", freq="D")
    sundays = may[may.dayofweek == 6]
    return sundays[1]


def _add_chile_retail_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_new_year"] = ((df["date"].dt.month == 1) & (df["date"].dt.day == 1)).astype(int)
    df["is_christmas"] = ((df["date"].dt.month == 12) & (df["date"].dt.day.isin([24, 25]))).astype(int)
    df["is_halloween"] = ((df["date"].dt.month == 10) & (df["date"].dt.day == 31)).astype(int)
    df["is_chile_independence"] = (
        (df["date"].dt.month == 9) & (df["date"].dt.day.isin([17, 18, 19]))
    ).astype(int)

    mothers_days = {_second_sunday_of_may(int(y)) for y in df["date"].dt.year.unique()}
    df["is_mothers_day"] = df["date"].isin(mothers_days).astype(int)

    df["is_winter_break"] = (
        (df["date"].dt.month == 7) & (df["date"].dt.day.between(8, 24))
    ).astype(int)

    df["is_special_event"] = (
        df[
            [
                "is_new_year",
                "is_christmas",
                "is_halloween",
                "is_chile_independence",
                "is_mothers_day",
                "is_winter_break",
            ]
        ].sum(axis=1)
        > 0
    ).astype(int)

    return df


def generate_synthetic_demand(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    n_stores: int = 26,
    n_skus: int = 200,
    seed: int = 42,
    availability_rate: float = 0.62,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    store_ids = [f"store_{i:02d}" for i in range(1, n_stores + 1)]
    sku_ids = [f"sku_{i:03d}" for i in range(1, n_skus + 1)]

    store_types = ["mall", "street", "office", "tourist", "residential"]
    sku_categories = [
        "fresh",
        "bakery",
        "beverage",
        "prepared",
        "snack",
        "seasonal_winter",
        "seasonal_holiday",
        "low_velocity",
    ]

    stores = pd.DataFrame({
        "store_id": store_ids,
        "store_type": rng.choice(
            store_types,
            size=n_stores,
            replace=True,
            p=[0.28, 0.22, 0.18, 0.14, 0.18],
        ),
        "store_base_traffic": rng.uniform(0.65, 1.45, size=n_stores),
    })

    stores["schedule_type"] = rng.choice(
        ["all_week", "closed_weekends", "closed_sundays", "short_weekends"],
        size=n_stores,
        replace=True,
        p=[0.48, 0.16, 0.22, 0.14],
    )

    # Some store types have stronger seasonal sensitivity than others.
    stores["store_seasonality_strength"] = stores["store_type"].map({
        "mall": 1.10,
        "street": 0.95,
        "office": 0.75,
        "tourist": 1.30,
        "residential": 0.90,
    }).astype(float)

    sku_velocity = rng.choice(
        ["very_low", "low", "medium", "high"],
        size=n_skus,
        replace=True,
        p=[0.18, 0.32, 0.35, 0.15],
    )

    base_by_velocity = {
        "very_low": (0.15, 1.2),
        "low": (1.0, 5.0),
        "medium": (5.0, 35.0),
        "high": (35.0, 90.0),
    }

    skus = pd.DataFrame({
        "sku_id": sku_ids,
        "sku_category": rng.choice(sku_categories, size=n_skus, replace=True),
        "sku_velocity": sku_velocity,
        "sku_base_demand": [rng.uniform(*base_by_velocity[v]) for v in sku_velocity],
        "rain_sensitive": rng.binomial(1, 0.22, size=n_skus),
        "winter_sensitive": rng.binomial(1, 0.24, size=n_skus),
        "holiday_sensitive": rng.binomial(1, 0.18, size=n_skus),
        "september_sensitive": rng.binomial(1, 0.15, size=n_skus),
        "campaign_sensitive": rng.binomial(1, 0.20, size=n_skus),
        "is_pack_sku": rng.binomial(1, 0.16, size=n_skus),
    })

    # Category-specific seasonal amplitude and peak timing.
    # Peak day is approximate day-of-year where each category tends to sell more.
    skus["seasonal_amplitude"] = skus["sku_category"].map({
        "fresh": 0.08,
        "bakery": 0.10,
        "beverage": 0.22,
        "prepared": 0.14,
        "snack": 0.12,
        "seasonal_winter": 0.34,
        "seasonal_holiday": 0.38,
        "low_velocity": 0.06,
    }).astype(float)

    skus["seasonal_peak_day"] = skus["sku_category"].map({
        "fresh": 80,              # early autumn
        "bakery": 170,            # winter
        "beverage": 20,           # summer
        "prepared": 190,          # winter/office season
        "snack": 300,             # spring/events
        "seasonal_winter": 190,   # July
        "seasonal_holiday": 355,  # Christmas
        "low_velocity": 180,
    }).astype(float)

    skus["weekly_wave_amplitude"] = skus["sku_category"].map({
        "fresh": 0.06,
        "bakery": 0.08,
        "beverage": 0.10,
        "prepared": 0.07,
        "snack": 0.12,
        "seasonal_winter": 0.07,
        "seasonal_holiday": 0.08,
        "low_velocity": 0.04,
    }).astype(float)

    availability = (
        pd.MultiIndex.from_product([store_ids, sku_ids], names=["store_id", "sku_id"])
        .to_frame(index=False)
    )
    availability["available"] = rng.binomial(1, availability_rate, size=len(availability))
    availability = availability[availability["available"] == 1].drop(columns="available")

    grid = (
        pd.MultiIndex.from_product([dates, store_ids, sku_ids], names=["date", "store_id", "sku_id"])
        .to_frame(index=False)
        .merge(availability, on=["store_id", "sku_id"], how="inner")
        .merge(stores, on="store_id", how="left")
        .merge(skus, on="sku_id", how="left")
    )

    grid["day_of_week"] = grid["date"].dt.dayofweek
    grid["day_of_year"] = grid["date"].dt.dayofyear
    grid["month"] = grid["date"].dt.month
    grid["is_weekend"] = grid["day_of_week"].isin([5, 6]).astype(int)

    grid["is_store_open"] = 1
    grid.loc[(grid["schedule_type"] == "closed_weekends") & (grid["day_of_week"].isin([5, 6])), "is_store_open"] = 0
    grid.loc[(grid["schedule_type"] == "closed_sundays") & (grid["day_of_week"] == 6), "is_store_open"] = 0

    day_index = (grid["date"] - grid["date"].min()).dt.days

    # Southern Hemisphere style temperature pattern: colder around mid-year.
    yearly_temp = 17 - 8 * np.cos(2 * np.pi * day_index / 365.25)
    grid["temperature"] = yearly_temp + rng.normal(0, 3, len(grid))

    winter_month = grid["month"].isin([5, 6, 7, 8]).astype(int)
    rain_base = rng.gamma(1.2, 2.0, len(grid))
    grid["rain_mm"] = np.maximum(0, rain_base + winter_month * rng.gamma(1.8, 2.5, len(grid)) - 1.6)
    grid["is_rainy"] = (grid["rain_mm"] > 1.5).astype(int)

    grid = _add_chile_retail_calendar(grid)

    grid["is_promo"] = rng.binomial(1, 0.04, len(grid))
    grid["is_campaign"] = (
        (rng.binomial(1, 0.025, len(grid)) == 1)
        | ((grid["is_christmas"] == 1) & (rng.binomial(1, 0.35, len(grid)) == 1))
        | ((grid["is_chile_independence"] == 1) & (rng.binomial(1, 0.25, len(grid)) == 1))
    ).astype(int)

    # Discrete weekly behavior remains because real operations often have strong weekday-specific effects.
    weekday_effect = grid["day_of_week"].map({
        0: 0.88,
        1: 0.92,
        2: 0.98,
        3: 1.02,
        4: 1.12,
        5: 1.28,
        6: 1.08,
    }).astype(float)

    # Smooth weekly wave adds a more natural wave-like demand pattern on top of discrete weekday effects.
    # Peak is around Saturday.
    weekly_wave = 1 + grid["weekly_wave_amplitude"] * np.sin(
        2 * np.pi * (grid["day_of_week"] - 3.5) / 7
    )

    short_weekend_effect = np.where((grid["schedule_type"] == "short_weekends") & (grid["is_weekend"] == 1), 0.55, 1.0)

    category_effect = grid["sku_category"].map({
        "fresh": 1.15,
        "bakery": 1.10,
        "beverage": 1.00,
        "prepared": 1.30,
        "snack": 0.85,
        "seasonal_winter": 0.90,
        "seasonal_holiday": 0.80,
        "low_velocity": 0.55,
    }).astype(float)

    store_type_effect = grid["store_type"].map({
        "mall": 1.20,
        "street": 1.00,
        "office": 0.82,
        "tourist": 1.22,
        "residential": 0.95,
    }).astype(float)

    # Explicit smooth annual demand wave.
    # Different categories peak at different points in the year.
    annual_wave = 1 + (
        grid["seasonal_amplitude"]
        * grid["store_seasonality_strength"]
        * np.cos(2 * np.pi * (grid["day_of_year"] - grid["seasonal_peak_day"]) / 365.25)
    )
    annual_wave = annual_wave.clip(lower=0.55, upper=1.65)

    rain_effect = 1 + 0.22 * grid["rain_sensitive"] * grid["is_rainy"]
    winter_effect = 1 + 0.30 * grid["winter_sensitive"] * grid["is_winter_break"]
    christmas_effect = 1 + 0.45 * grid["holiday_sensitive"] * grid["is_christmas"]
    halloween_effect = 1 + 0.35 * grid["holiday_sensitive"] * grid["is_halloween"]
    mothers_day_effect = 1 + 0.30 * grid["holiday_sensitive"] * grid["is_mothers_day"]
    september_effect = 1 + 0.38 * grid["september_sensitive"] * grid["is_chile_independence"]
    promo_effect = 1 + 0.28 * grid["is_promo"]
    campaign_effect = 1 + 0.32 * grid["campaign_sensitive"] * grid["is_campaign"]
    pack_effect = 1 + 0.18 * grid["is_pack_sku"] * grid["is_campaign"]
    temperature_effect = 1 + 0.008 * (grid["temperature"] - grid["temperature"].mean())

    base_mu = (
        grid["sku_base_demand"]
        * grid["store_base_traffic"]
        * weekday_effect
        * weekly_wave
        * annual_wave
        * category_effect
        * store_type_effect
        * short_weekend_effect
        * temperature_effect
        * rain_effect
        * winter_effect
        * christmas_effect
        * halloween_effect
        * mothers_day_effect
        * september_effect
        * promo_effect
        * campaign_effect
        * pack_effect
        * grid["is_store_open"]
    )

    grid = grid.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    grid["expected_demand"] = np.maximum(base_mu, 0.0)

    demand_values = []
    for _, group in grid.groupby(["store_id", "sku_id"], sort=False):
        previous = None
        rolling_memory = None
        for mu, is_open in zip(group["expected_demand"].to_numpy(), group["is_store_open"].to_numpy()):
            if is_open == 0:
                demand_values.append(0)
                previous = 0
                rolling_memory = 0 if rolling_memory is None else rolling_memory * 0.8
                continue

            if previous is None:
                temporal_mu = mu
                rolling_memory = mu
            else:
                temporal_mu = 0.60 * mu + 0.28 * previous + 0.12 * rolling_memory
                rolling_memory = 0.85 * rolling_memory + 0.15 * previous

            noisy_mu = max(0.02, temporal_mu * rng.lognormal(mean=0, sigma=0.22))
            demand_values.append(rng.poisson(noisy_mu))
            previous = demand_values[-1]

    grid["demand"] = demand_values
    return grid
