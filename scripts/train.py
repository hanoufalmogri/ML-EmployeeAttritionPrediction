from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from purchase_intention.modeling import train_and_select_model


def main() -> None:
    data_path = Path("Data/Raw/online_shoppers_intention.csv")
    result = train_and_select_model(data_path)

    print(f"Best model: {result.best_model_name}")
    print("Metrics:")
    for model_name, metrics in result.metrics.items():
        print(f"  {model_name}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value}")

    print("\nTop features:")
    for row in result.feature_importance.head(10).itertuples(index=False):
        print(f"  {row.feature}: {row.importance:.4f}")


if __name__ == "__main__":
    main()
