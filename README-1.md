# Credit Card Fraud Detection Using Machine Learning

A complete, end-to-end machine learning project detecting fraudulent credit card
transactions on a severely imbalanced dataset (0.17% fraud). Built as a Data Science
portfolio project demonstrating real-world fraud-detection practices: leakage-safe
preprocessing, imbalanced classification, business-driven threshold optimization, and
production-ready model packaging.

## Problem

Binary classification — `Class 0` (genuine) vs. `Class 1` (fraud) — on 284,807
anonymized (PCA-transformed) European credit card transactions from September 2013.
Fraud detection is a textbook extreme-imbalance problem: accuracy is meaningless, and
the real cost of a mistake depends heavily on *which kind* of mistake it is (a missed
fraud vs. a false alarm).

## Dataset

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(`creditcard.csv`, ~284,807 rows, 31 columns: `Time`, `V1`-`V28`, `Amount`, `Class`).
Download it from Kaggle and place `creditcard.csv` in the project's working directory
before running the notebook — it is not included in this repository due to size/license.

## Project structure

```
.
├── YourName_CreditCardFraudDetection.ipynb   # full analysis notebook (all 8 phases)
├── YourName_ProjectReport.docx               # written project report
├── predict.py                                # reusable prediction function
├── requirements.txt
├── README.md
└── artifacts/
    ├── fraud_model_bundle.joblib             # model + scaler + threshold + feature order
    ├── final_results.json                    # final metrics, thresholds, top features
    ├── model_comparison_validation.csv
    ├── threshold_sweep.csv
    └── feature_importance.csv
```

## Approach

1. **Data understanding** — structure, missing values, 1,081 duplicate rows,
   class imbalance (492 fraud / 284,807 total), leakage risks.
2. **EDA** — class distribution, amount/time distributions, feature separability,
   correlation, fraud rate by amount bucket.
3. **Preprocessing** — stratified 60/20/20 train/val/test split; scaler fit on
   train only; duplicates dropped before splitting.
4. **Imbalanced classification** — four models compared: class-weighted Logistic
   Regression, class-weighted Random Forest, HistGradientBoosting, and SMOTE +
   Logistic Regression (SMOTE applied to training data only).
5. **Evaluation** — precision, recall, F1, ROC-AUC, PR-AUC; no model selected on
   accuracy alone.
6. **Threshold optimization** — swept 0.01-0.99; final threshold chosen to minimize
   a documented cost function (`25 × false negatives + 1 × false positives`), not the
   default 0.5.
7. **Advanced ML** — 3-fold cross-validation, isotonic probability calibration,
   permutation feature importance, and a time-based stability check.
8. **Model saving & deployment** — model, scaler, threshold, and feature order
   bundled with `joblib`; reusable `predict_fraud()` function; FastAPI/Streamlit
   deployment sketches and production considerations (drift monitoring, retraining
   cadence, logging, data privacy) in the notebook and report.

## Results (held-out test set)

| Metric | Value |
|---|---|
| Model | Random Forest (class-weighted, threshold=0.25) |
| Precision | 0.626 |
| Recall | 0.811 |
| F1 | 0.706 |
| ROC-AUC | 0.973 |
| PR-AUC (Average Precision) | 0.822 |
| Confusion matrix | TN=56,605  FP=46  FN=18  TP=77 |

Random Forest was selected over Logistic Regression, HistGradientBoosting, and
SMOTE+Logistic Regression by validation-set PR-AUC — the most trustworthy single
metric under this level of class imbalance. See the notebook for the full comparison
table and the reasoning behind every modeling decision.

## Top predictive features

`V12`, `V3`, `V4`, `V9`, `V10` (by permutation importance). These are anonymized PCA
components with no direct business interpretation, but they are consistently the
strongest fraud signal across both linear correlation and model-based importance.

## Setup

```bash
pip install -r requirements.txt
# place creditcard.csv (downloaded from Kaggle) in this directory
jupyter notebook YourName_CreditCardFraudDetection.ipynb
```

## Using the saved model

```python
from predict import predict_fraud

transaction = {"Time": 0, "V1": -1.36, "V2": -0.07, ..., "V28": -0.02, "Amount": 149.62}
result = predict_fraud(transaction)
# {'fraud_probability': 0.0025, 'is_fraud': False, 'threshold_used': 0.25, 'model_name': 'RandomForest_balanced'}
```

## Known limitations

- Features `V1`-`V28` are anonymized PCA components — no business/domain narrative is
  possible for individual features, only statistical importance.
- The dataset spans only ~48 hours, so a genuine time-based train/test split (e.g.
  train on month 1, test on month 2) is not possible here; production systems should
  validate with real time-based backtesting instead.
- The cost ratio used for threshold selection (25:1) is illustrative — it should be
  replaced with real figures from finance/risk stakeholders.
- `imbalanced-learn` was unavailable in the environment this project was built in, so
  SMOTE is implemented manually with `numpy`/`scikit-learn`; swap in
  `from imblearn.over_sampling import SMOTE` if the package is available to you.

## Future improvements

- Hyperparameter tuning via `RandomizedSearchCV`/`Optuna` for the winning model.
- SHAP-based interpretation for individual transaction explanations.
- Ensemble/stacking across the four model families.
- Real time-based backtesting once multi-month data is available.
- Online/streaming inference and automated drift monitoring in production.
