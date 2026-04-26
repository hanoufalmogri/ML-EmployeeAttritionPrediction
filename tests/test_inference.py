from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from purchase_intention.inference import predict_purchase, probability_to_intent_level
from purchase_intention.modeling import FEATURE_COLUMNS


class DummyModel:
    def predict_proba(self, _input_frame):
        return np.array([[0.18, 0.82]])


class IntentLevelTests(unittest.TestCase):
    def test_high_intent_threshold(self) -> None:
        self.assertEqual(probability_to_intent_level(0.91), "High intent")

    def test_medium_intent_threshold(self) -> None:
        self.assertEqual(probability_to_intent_level(0.55), "Medium intent")

    def test_low_intent_threshold(self) -> None:
        self.assertEqual(probability_to_intent_level(0.12), "Low intent")


class PredictPurchaseTests(unittest.TestCase):
    def test_prediction_response_shape(self) -> None:
        payload = {
            "Administrative": 0,
            "Administrative_Duration": 0.0,
            "Informational": 0,
            "Informational_Duration": 0.0,
            "ProductRelated": 12,
            "ProductRelated_Duration": 320.0,
            "BounceRates": 0.02,
            "ExitRates": 0.04,
            "PageValues": 10.0,
            "SpecialDay": 0.0,
            "OperatingSystems": 2,
            "Browser": 2,
            "Region": 1,
            "TrafficType": 2,
            "Month": "Nov",
            "VisitorType": "Returning_Visitor",
            "Weekend": False,
        }

        with patch("purchase_intention.inference.load_trained_model", return_value=DummyModel()):
            result = predict_purchase(payload)

        self.assertEqual(
            set(result),
            {"purchase_probability", "predicted_purchase", "intent_level"},
        )
        self.assertEqual(result["purchase_probability"], 0.82)
        self.assertTrue(result["predicted_purchase"])
        self.assertEqual(result["intent_level"], "High intent")

    def test_missing_features_raise_clear_error(self) -> None:
        payload = {column: 0 for column in FEATURE_COLUMNS}
        del payload["Month"]

        with self.assertRaisesRegex(ValueError, "Missing required features: Month"):
            predict_purchase(payload)


if __name__ == "__main__":
    unittest.main()
