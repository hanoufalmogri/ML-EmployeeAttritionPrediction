from __future__ import annotations

import importlib.util
from pathlib import Path

from purchase_intention.modeling import MODEL_PATH, train_and_select_model


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Data" / "Raw" / "online_shoppers_intention.csv"
GRADIO_APP_PATH = Path(__file__).resolve().parent / "app" / "gradio_app.py"


if not MODEL_PATH.exists():
    train_and_select_model(DATA_PATH)

spec = importlib.util.spec_from_file_location("purchase_intention_gradio_app", GRADIO_APP_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load Gradio app from {GRADIO_APP_PATH}")

gradio_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gradio_app)

APP_CSS = gradio_app.APP_CSS
demo = gradio_app.demo


if __name__ == "__main__":
    demo.launch(css=APP_CSS)
