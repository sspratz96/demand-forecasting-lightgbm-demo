import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError("LightGBM is required. Install dependencies with: pip install -r requirements.txt") from exc


CATEGORICAL_FEATURES = ["store_id", "sku_id", "store_type", "sku_category"]


def build_lightgbm_pipeline(feature_columns: list[str]) -> Pipeline:
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
        n_estimators=650,
        learning_rate=0.035,
        num_leaves=48,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_model(train_df: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    pipeline = build_lightgbm_pipeline(feature_columns)
    pipeline.fit(train_df[feature_columns], train_df["target"])
    return pipeline


def predict_model(model: Pipeline, df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    preds = model.predict(df[feature_columns])
    return pd.Series(preds, index=df.index).clip(lower=0)


def rule_based_baseline(df: pd.DataFrame) -> pd.Series:
    pred = 0.50 * df["rolling_mean_7"] + 0.30 * df["rolling_mean_14"] + 0.20 * df["rolling_mean_28"]
    return pred.fillna(df["target"].mean()).clip(lower=0)
