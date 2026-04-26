---
title: Purchase Intention Prediction
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
python_version: 3.11
short_description: Predict purchase intent with an interactive ML demo.
---

# User Behavior Analysis for Purchase Intention Prediction

## Overview
This project predicts whether an online shopping session will end in a purchase using the Online Shoppers Purchasing Intention dataset. The repository now includes a full machine learning workflow: preprocessing, model comparison, artifact generation, an API, a Gradio demo, and Docker support.

## Problem Statement
The goal is to classify session-level purchase intent so teams can identify likely buyers in real time and support conversion optimization, personalization, and targeted marketing.

## Dataset Overview
- Dataset: `Data/Raw/online_shoppers_intention.csv`
- Task: binary classification
- Target: `Revenue`
- Shape: 12,330 rows x 18 columns
- Class balance: 10,422 non-purchase sessions, 1,908 purchase sessions

### Feature Groups
- Page interaction: `Administrative`, `Informational`, `ProductRelated`
- Duration: `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`
- Engagement: `BounceRates`, `ExitRates`, `PageValues`
- Session context: `Month`, `VisitorType`, `Weekend`, `SpecialDay`
- Technical context: `OperatingSystems`, `Browser`, `Region`, `TrafficType`

## Machine Learning Framing
- ML task: binary classification
- Why ML: purchase behavior depends on nonlinear interactions and mixed feature types
- ART criteria:
  - Available: session features are known before checkout completes
  - Relevant: browsing depth, page value, and timing behavior are predictive
  - Timely: predictions can be served during the active session

## Data Quality and Preprocessing
- No missing values in the provided dataset
- Numerical features are median-imputed and scaled
- Categorical features are mode-imputed and one-hot encoded
- Class imbalance is handled with `class_weight="balanced"` where supported and sample weights for gradient boosting

## Models Compared
- Logistic Regression
- Random Forest
- Gradient Boosting

Model selection is based primarily on ROC-AUC, with recall used as a secondary tie-breaker because missing a likely purchase session is often more costly than a false positive.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Results
Training was run locally with an 80/20 stratified train-test split.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8500 | 0.5107 | 0.7487 | 0.6072 | 0.8962 |
| Random Forest | 0.8982 | 0.6814 | 0.6440 | 0.6622 | 0.9243 |
| Gradient Boosting | 0.8662 | 0.5439 | 0.8429 | 0.6612 | 0.9289 |

Selected model: `GradientBoostingClassifier`

Why it won:
- Highest ROC-AUC
- Strongest recall
- Best balance for identifying likely purchasers

## Feature Importance
Top model signals from the trained gradient boosting pipeline:

1. `PageValues`
2. `Month_Nov`
3. `ProductRelated_Duration`
4. `ProductRelated`
5. `Month_May`
6. `ExitRates`
7. `Administrative`
8. `Administrative_Duration`
9. `Month_Mar`
10. `BounceRates`

Business interpretation:
- Higher `PageValues` strongly indicates buying intent
- Longer and deeper product page exploration is associated with conversion
- Seasonal effects matter, especially around high-activity months like November
- Exit and bounce behavior still provide useful friction signals

## Model Output and Decision Logic
The model returns a purchase probability and maps it to an intent segment:

- `> 0.7`: High intent
- `0.4 - 0.7`: Medium intent
- `< 0.4`: Low intent

## Project Structure
```text
.
├── Data/Raw/online_shoppers_intention.csv
├── api/main.py
├── app/gradio_app.py
├── artifacts/                     # generated after training
├── purchase_intention/
│   ├── inference.py
│   └── modeling.py
├── scripts/train.py
├── tests/test_inference.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## How to Run
### 1. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the model
```bash
python3 scripts/train.py
```

This creates:
- `artifacts/best_model.joblib`
- `artifacts/metrics.json`
- `artifacts/feature_importance.csv`
- `artifacts/model_card.json`

### 3. Start the API
```bash
uvicorn api.main:app --reload
```

Health check:
```bash
curl http://127.0.0.1:8000/health
```

Prediction example:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Administrative": 0,
    "Administrative_Duration": 0.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 1,
    "ProductRelated_Duration": 0.0,
    "BounceRates": 0.2,
    "ExitRates": 0.2,
    "PageValues": 0.0,
    "SpecialDay": 0.0,
    "Month": "Feb",
    "OperatingSystems": 1,
    "Browser": 1,
    "Region": 1,
    "TrafficType": 1,
    "VisitorType": "Returning_Visitor",
    "Weekend": false
  }'
```

Expected response shape:
```json
{
  "purchase_probability": 0.0202,
  "predicted_purchase": false,
  "intent_level": "Low intent"
}
```

### 4. Start the Gradio app
```bash
python3 app/gradio_app.py
```

The demo includes a purchase probability panel, intent-specific business recommendation, explanation bullets, scenario comparison, and the model's top feature signals.

## Docker
Build and run:

```bash
docker build -t purchase-intention-app .
docker run -p 8000:8000 purchase-intention-app
```

The container trains the model during image build and serves the FastAPI app on port `8000`.

## Testing
Run the smoke tests:

```bash
python3 -m unittest discover -s tests
```

## Notes
- The current repository still contains a notebook stub in `Data/Raw/Data.ipynb`, but the working implementation lives in the Python modules and app entry points.
- FastAPI, Uvicorn, and Gradio are listed in `requirements.txt` and need to be installed before serving the API or UI.
