"""
Reusable fraud prediction function.
Loads the saved model bundle (model + scaler + threshold + feature order)
and exposes predict_fraud() for scoring new transactions.
"""
import joblib
import numpy as np
import pandas as pd

BUNDLE_PATH = 'artifacts/fraud_model_bundle.joblib'
_bundle = None


def load_bundle(path: str = BUNDLE_PATH):
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


def predict_fraud(transaction: dict, bundle_path: str = BUNDLE_PATH) -> dict:
    """
    Score a single transaction dict for fraud probability.

    Parameters
    ----------
    transaction : dict
        Must contain keys: 'Time', 'V1'..'V28', 'Amount'.

    Returns
    -------
    dict with keys: fraud_probability, is_fraud (bool, using saved threshold),
    threshold_used, model_name.
    """
    bundle = load_bundle(bundle_path)
    feature_order = bundle['feature_order']

    missing = [c for c in feature_order if c not in transaction]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    row = pd.DataFrame([{c: transaction[c] for c in feature_order}])
    row[bundle['scale_columns']] = bundle['scaler'].transform(row[bundle['scale_columns']])

    proba = float(bundle['model'].predict_proba(row)[:, 1][0])
    return {
        'fraud_probability': proba,
        'is_fraud': bool(proba >= bundle['threshold']),
        'threshold_used': bundle['threshold'],
        'model_name': bundle['model_name'],
    }


if __name__ == '__main__':
    # Smoke test using one genuine and one fraudulent row pulled from the
    # original dataset (verifies the bundle round-trips correctly).
    df = pd.read_csv('creditcard.csv')
    genuine_row = df[df.Class == 0].iloc[0].to_dict()
    fraud_row = df[df.Class == 1].iloc[0].to_dict()

    print("Genuine example ->", predict_fraud(genuine_row))
    print("Fraud example   ->", predict_fraud(fraud_row))
