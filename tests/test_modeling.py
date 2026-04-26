from __future__ import annotations

import unittest

import pandas as pd

from purchase_intention.modeling import FEATURE_COLUMNS, build_preprocessor


class PreprocessorTests(unittest.TestCase):
    def test_categorical_values_with_strings_and_booleans_transform(self) -> None:
        rows = []
        for month, visitor_type, weekend in [
            ("May", "Returning_Visitor", False),
            ("Nov", "New_Visitor", True),
        ]:
            row = {column: 0 for column in FEATURE_COLUMNS}
            row.update(
                {
                    "Month": month,
                    "VisitorType": visitor_type,
                    "Weekend": weekend,
                    "ProductRelated": 1,
                    "ProductRelated_Duration": 10.0,
                }
            )
            rows.append(row)

        transformed = build_preprocessor().fit_transform(pd.DataFrame(rows))

        self.assertEqual(transformed.shape[0], 2)
        self.assertGreater(transformed.shape[1], len(FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
