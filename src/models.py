import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError("LightGBM is required. Install dependencies with: pip install -r requirements.txt") from exc


CATEGORICAL_FEATURES = [
    "store_id",
    "sku_id",
    "store_type",
    "sku_category",
    "sku_velocity",
    "schedule_type",
]


MODEL_CONFIGS = {
    "default": {
        "n_estimators": 650,
        "learning_rate": 0.035,
        "num_leaves": 48,
        "max_depth": -1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 30,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 42,
    },
    "shallow_fast": {
        "n_estimators": 350,
        "learning_rate": 0.055,
        "num_leaves": 24,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 60,
        "reg_alpha": 0.05,
        "reg_lambda": 0.10,
        "random_state": 11,
    },
    "medium_balanced": {
        "n_estimators": 650,
        "learning_rate": 0.035,
        "num_leaves": 48,
        "max_depth": -1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 30,
        "reg_alpha": 0.02,
        "reg_lambda": 0.05,
        "random_state": 42,
    },
    "deep_detailed": {
        "n_estimators": 950,
        "learning_rate": 0.025,
        "num_leaves": 96,
        "max_depth": -1,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_samples": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 0.02,
        "random_state": 77,
    },
}


def build_lightgbm_pipeline(feature_columns: list[str], config_name: str = "default") -> Pipeline:
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config_name={config_name}. Expected one of {list(MODEL_CONFIGS)}.")

    categorical = [c for c in feature_columns if c in CATEGORICAL_FEATURES]
    numeric = [c for c in feature_columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ],
        remainder="drop",
    )

    model = LGBMRegressor(
        objective="regression",
        n_jobs=-1,
        verbose=-1,
        **MODEL_CONFIGS[config_name],
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_model(train_df: pd.DataFrame, feature_columns: list[str], config_name: str = "default") -> Pipeline:
    pipeline = build_lightgbm_pipeline(feature_columns, config_name=config_name)
    pipeline.fit(train_df[feature_columns], train_df["target"])
    return pipeline


def predict_model(model: Pipeline, df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    preds = model.predict(df[feature_columns])
    return pd.Series(preds, index=df.index).clip(lower=0)


def train_bagged_models(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    config_names: list[str] | None = None,
) -> list[Pipeline]:
    """Train multiple LightGBM configurations for a simple bagging-style ensemble."""
    if config_names is None:
        config_names = ["shallow_fast", "medium_balanced", "deep_detailed"]

    models = []
    for config_name in config_names:
        print(f"  - Training ensemble member: {config_name}")
        models.append(train_model(train_df, feature_columns, config_name=config_name))

    return models


def predict_bagged_models(
    models: list[Pipeline],
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """Average predictions from multiple fitted models."""
    predictions = []
    for model in models:
        predictions.append(predict_model(model, df, feature_columns))

    prediction_frame = pd.concat(predictions, axis=1)
    return prediction_frame.mean(axis=1).clip(lower=0)


def _map_transformed_feature_to_original(transformed_name: str) -> str:
    if "__" in transformed_name:
        transformed_name = transformed_name.split("__", 1)[1]

    for categorical in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if transformed_name == categorical or transformed_name.startswith(f"{categorical}_"):
            return categorical

    return transformed_name


def extract_feature_importance(
    model: Pipeline,
    model_name: str,
    normalize: bool = True,
) -> pd.DataFrame:
    """Extract LightGBM feature importance aggregated back to original feature names."""
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]

    try:
        transformed_names = preprocessor.get_feature_names_out()
    except AttributeError:
        transformed_names = [f"feature_{i}" for i in range(len(estimator.feature_importances_))]

    raw = pd.DataFrame({
        "transformed_feature": transformed_names,
        "importance": estimator.feature_importances_,
    })

    raw["feature"] = raw["transformed_feature"].map(_map_transformed_feature_to_original)

    grouped = (
        raw.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    if normalize:
        total = grouped["importance"].sum()
        grouped["importance_normalized"] = grouped["importance"] / total if total else 0.0

    grouped["model"] = model_name
    return grouped[["model", "feature", "importance", "importance_normalized"]]


def extract_bagged_feature_importance(
    models: list[Pipeline],
    model_name: str,
) -> pd.DataFrame:
    frames = []
    for idx, model in enumerate(models, start=1):
        member = extract_feature_importance(model, f"{model_name}_member_{idx}", normalize=True)
        frames.append(member)

    combined = pd.concat(frames, ignore_index=True)

    aggregated = (
        combined.groupby("feature", as_index=False)
        .agg(
            importance=("importance", "mean"),
            importance_normalized=("importance_normalized", "mean"),
        )
        .sort_values("importance_normalized", ascending=False)
    )

    aggregated["model"] = model_name
    return aggregated[["model", "feature", "importance", "importance_normalized"]]


def rule_based_baseline(df: pd.DataFrame) -> pd.Series:
    pred = 0.50 * df["rolling_mean_7"] + 0.30 * df["rolling_mean_14"] + 0.20 * df["rolling_mean_28"]
    return pred.fillna(df["target"].mean()).clip(lower=0)
