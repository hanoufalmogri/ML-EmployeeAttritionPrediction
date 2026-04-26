from __future__ import annotations

import importlib.util
from pathlib import Path


GRADIO_APP_PATH = Path(__file__).resolve().parent / "app" / "gradio_app.py"
spec = importlib.util.spec_from_file_location("purchase_intention_gradio_app", GRADIO_APP_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load Gradio app from {GRADIO_APP_PATH}")

gradio_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gradio_app)

APP_CSS = gradio_app.APP_CSS
demo = gradio_app.demo


if __name__ == "__main__":
    demo.launch(css=APP_CSS)
