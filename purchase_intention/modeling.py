from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42
TARGET_COLUMN = "Revenue"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
MODEL_CARD_PATH = ARTIFACTS_DIR / "model_card.json"

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

CATEGORICAL_FEATURES = ["Month", "VisitorType", "Weekend"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class TrainingResult:
    best_model_name: str
    metrics: dict[str, dict[str, Any]]
    best_metrics: dict[str, Any]
    feature_importance: pd.DataFrame


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _normalize_categorical_values(values: Any) -> pd.DataFrame:
    frame = pd.DataFrame(values).copy()
    return frame.where(frame.isna(), frame.astype(str))


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = expected.difference(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing_str}")
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "normalizer",
                FunctionTransformer(
                    _normalize_categorical_values,
                    feature_names_out="one-to-one",
                    validate=False,
                ),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    GradientBoostingClassifier(
                        learning_rate=0.05,
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)
    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return metrics


def _extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(feature_names))

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": values}
    ).sort_values("importance", ascending=False)
    return importance_df.reset_index(drop=True)


def _round_floats(obj: Any, digits: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {key: _round_floats(value, digits) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(value, digits) for value in obj]
    return obj


def train_and_select_model(data_path: str | Path) -> TrainingResult:
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    metrics_by_model: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, pipeline in build_models().items():
        if name == "gradient_boosting":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
        else:
            pipeline.fit(X_train, y_train)
        metrics_by_model[name] = evaluate_model(pipeline, X_test, y_test)
        fitted_models[name] = pipeline

    best_model_name = max(
        metrics_by_model,
        key=lambda model_name: (
            metrics_by_model[model_name]["roc_auc"],
            metrics_by_model[model_name]["recall"],
        ),
    )
    best_model = fitted_models[best_model_name]
    feature_importance = _extract_feature_importance(best_model)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    rounded_metrics = _round_floats(metrics_by_model)
    METRICS_PATH.write_text(json.dumps(rounded_metrics, indent=2))
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    MODEL_CARD_PATH.write_text(
        json.dumps(
            {
                "best_model_name": best_model_name,
                "features": FEATURE_COLUMNS,
                "target": TARGET_COLUMN,
                "decision_thresholds": {
                    "high_intent": "> 0.7",
                    "medium_intent": "0.4 - 0.7",
                    "low_intent": "< 0.4",
                },
            },
            indent=2,
        )
    )

    return TrainingResult(
        best_model_name=best_model_name,
        metrics=rounded_metrics,
        best_metrics=rounded_metrics[best_model_name],
        feature_importance=feature_importance,
    )


def load_trained_model(model_path: str | Path = MODEL_PATH) -> Pipeline:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_file}. Run scripts/train.py first."
        )
    return joblib.load(model_file)
