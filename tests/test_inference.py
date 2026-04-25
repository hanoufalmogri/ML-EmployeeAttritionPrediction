from __future__ import annotations

import unittest

from purchase_intention.inference import probability_to_intent_level


class IntentLevelTests(unittest.TestCase):
    def test_high_intent_threshold(self) -> None:
        self.assertEqual(probability_to_intent_level(0.91), "High intent")

    def test_medium_intent_threshold(self) -> None:
        self.assertEqual(probability_to_intent_level(0.55), "Medium intent")

    def test_low_intent_threshold(self) -> None:
        self.assertEqual(probability_to_intent_level(0.12), "Low intent")


if __name__ == "__main__":
    unittest.main()

