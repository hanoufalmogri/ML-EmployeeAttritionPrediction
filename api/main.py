from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from purchase_intention.inference import predict_purchase


app = FastAPI(
    title="Purchase Intention Prediction API",
    description="Predict whether an online shopping session will end in a purchase.",
    version="1.0.0",
)


class SessionFeatures(BaseModel):
    Administrative: int = Field(0, ge=0)
    Administrative_Duration: float = Field(0.0, ge=0.0)
    Informational: int = Field(0, ge=0)
    Informational_Duration: float = Field(0.0, ge=0.0)
    ProductRelated: int = Field(..., ge=0)
    ProductRelated_Duration: float = Field(..., ge=0.0)
    BounceRates: float = Field(..., ge=0.0)
    ExitRates: float = Field(..., ge=0.0)
    PageValues: float = Field(0.0, ge=0.0)
    SpecialDay: float = Field(0.0, ge=0.0)
    Month: str
    OperatingSystems: int = Field(..., ge=1)
    Browser: int = Field(..., ge=1)
    Region: int = Field(..., ge=1)
    TrafficType: int = Field(..., ge=1)
    VisitorType: str
    Weekend: bool


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(features: SessionFeatures) -> dict[str, object]:
    try:
        return predict_purchase(features.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
