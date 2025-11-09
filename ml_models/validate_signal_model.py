"""
validate_signal_model.py
Rolling validation and diagnostics for signal_model.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve
from joblib import load
from datetime import datetime

# Paths
X_train_path = "data/processed/X_train.parquet"
X_test_path = "data/processed/X_test.parquet"
y_train_path = "data/processed/y_train.parquet"
y_test_path = "data/processed/y_test.parquet"

def load_data():
    X_train = pd.read_parquet(X_train_path)
    X_test = pd.read_parquet(X_test_path)
    y_train = pd.read_parquet(y_train_path).squeeze()
    y_test = pd.read_parquet(y_test_path).squeeze()
    return X_train, X_test, y_train, y_test

def rolling_validation(X, y, n_splits=5, min_train_size=250):
    """Walk-forward validation."""
    metrics = []
    step = len(y) // n_splits
    for i in range(n_splits):
        train_end = min_train_size + i * step
        if train_end >= len(y): break
        X_train, X_val = X.iloc[:train_end], X.iloc[train_end:train_end+step]
        y_train, y_val = y.iloc[:train_end], y.iloc[train_end:train_end+step]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        labels = (preds > 0.5).astype(int)

        metrics.append({
            "fold": i+1,
            "roc_auc": roc_auc_score(y_val, preds),
            "accuracy": accuracy_score(y_val, labels),
            "precision": precision_score(y_val, labels),
            "recall": recall_score(y_val, labels),
            "f1": f1_score(y_val, labels),
            "brier": brier_score_loss(y_val, preds)
        })
    return pd.DataFrame(metrics)

def main():
    X_train, X_test, y_train, y_test = load_data()
    df = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    print(f"🧮 Total samples: {len(df)}")

    # Perform rolling validation
    results = rolling_validation(df, y, n_splits=6, min_train_size=200)
    print("\n=== Rolling Validation Results ===")
    print(results.round(3))

    print("\nAverage AUC:", results["roc_auc"].mean().round(3))
    print("Average Accuracy:", results["accuracy"].mean().round(3))

    # Save results
    results.to_csv("models/validation_results.csv", index=False)
    print("\n💾 Saved to models/validation_results.csv")

if __name__ == "__main__":
    main()