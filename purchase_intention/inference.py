from __future__ import annotations

from typing import Any

import pandas as pd

from purchase_intention.modeling import FEATURE_COLUMNS, load_trained_model


def probability_to_intent_level(probability: float) -> str:
    if probability > 0.7:
        return "High intent"
    if probability >= 0.4:
        return "Medium intent"
    return "Low intent"


def predict_purchase(session_features: dict[str, Any]) -> dict[str, Any]:
    missing = [column for column in FEATURE_COLUMNS if column not in session_features]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required features: {missing_str}")

    model = load_trained_model()
    input_frame = pd.DataFrame([{column: session_features[column] for column in FEATURE_COLUMNS}])
    probability = float(model.predict_proba(input_frame)[:, 1][0])
    predicted_class = probability >= 0.5
    return {
        "purchase_probability": round(probability, 4),
        "predicted_purchase": predicted_class,
        "intent_level": probability_to_intent_level(probability),
    }

