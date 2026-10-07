from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


REQUIRED_COLUMNS = [
    "Date",
    "App",
    "Usage (minutes)",
    "Notifications",
    "Times Opened",
]

FEATURE_COLUMNS = [
    "App",
    "Notifications",
    "Times Opened",
    "DayOfWeek",
    "DayOfMonth",
    "IsWeekend",
    "Notifications_x_TimesOpened",
]

TARGET = "Usage (minutes)"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load raw screentime data."""
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> dict:
    """Run lightweight schema and data-quality checks."""
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    missing_values = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = ["Usage (minutes)", "Notifications", "Times Opened"]
    negative_values = int((df[numeric_cols] < 0).sum().sum())

    if missing_values > 0:
        raise ValueError(f"Dataset contains {missing_values} missing values.")
    if negative_values > 0:
        raise ValueError(f"Dataset contains {negative_values} negative numeric values.")

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "date_app_repeats": int(df.duplicated(["Date", "App"]).sum()),
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create transparent, production-friendly features."""
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    data = data.sort_values(["Date", "App"]).reset_index(drop=True)

    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["DayOfMonth"] = data["Date"].dt.day
    data["IsWeekend"] = (data["DayOfWeek"] >= 5).astype(int)
    data["Notifications_x_TimesOpened"] = (
        data["Notifications"] * data["Times Opened"]
    )
    return data


def split_by_date(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Split by Date groups so the same date cannot appear in train and test.
    This better represents prediction on unseen days than a random row split.
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    X = data[FEATURE_COLUMNS]
    y = data[TARGET]
    groups = data["Date"]

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    return X_train, X_test, y_train, y_test


def build_model(random_state: int = 42) -> Pipeline:
    """Build preprocessing + Random Forest as one reproducible sklearn Pipeline."""
    numeric_features = [
        "Notifications",
        "Times Opened",
        "DayOfWeek",
        "DayOfMonth",
        "IsWeekend",
        "Notifications_x_TimesOpened",
    ]
    categorical_features = ["App"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
    }
    return metrics


def train_and_save(
    raw_path: str | Path,
    artifact_dir: str | Path,
    processed_path: str | Path | None = None,
) -> dict:
    """End-to-end training function used by both notebook and Airflow."""
    raw_path = Path(raw_path)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(raw_path)
    validation = validate_data(df)
    data = engineer_features(df)

    if processed_path is not None:
        processed_path = Path(processed_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(processed_path, index=False)

    X_train, X_test, y_train, y_test = split_by_date(data)

    model = build_model()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    joblib.dump(model, artifact_dir / "screentime_random_forest.joblib")

    prediction_df = X_test.copy()
    prediction_df["Actual_Usage"] = y_test.values
    prediction_df["Predicted_Usage"] = model.predict(X_test)
    prediction_df.to_csv(artifact_dir / "test_predictions.csv", index=False)

    metadata = {
        "model": "RandomForestRegressor",
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "validation": validation,
        "metrics": metrics,
    }

    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
