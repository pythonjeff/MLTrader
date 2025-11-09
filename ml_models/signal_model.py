import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "signal_model.joblib"
PREPROCESS_ARTIFACTS_PATH = DATA_DIR / "preprocess_artifacts.joblib"
PREDICTIONS_PATH = DATA_DIR / "signal_predictions.parquet"


def _load_series(path: Path, column: Optional[str] = None) -> pd.Series:
    df = pd.read_parquet(path)
    if column and column in df.columns:
        return df[column]
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    raise ValueError(f"Expected a single column in {path}, found {df.columns.tolist()}")


def load_preprocessed_splits() -> Dict[str, pd.DataFrame | pd.Series]:
    logging.info("Loading preprocessed train/test splits.")
    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
    y_train = _load_series(DATA_DIR / "y_train.parquet", column="label")
    y_test = _load_series(DATA_DIR / "y_test.parquet", column="label")
    logging.info("Loaded splits: X_train=%s, X_test=%s", X_train.shape, X_test.shape)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


@dataclass
class EarningsSignalModel:
    """Binary classifier predicting 10-day direction for earnings events."""

    model: LogisticRegression = field(
        default_factory=lambda: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
    )
    feature_columns: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        logging.info("Fitting logistic regression on %s samples.", len(X))
        self.feature_columns = X.columns.tolist()
        self.model.fit(X, y)
        logging.info("Model fit complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X[self.feature_columns]
        return self.model.predict(X_aligned)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X[self.feature_columns]
        return self.model.predict_proba(X_aligned)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X[self.feature_columns]
        return self.model.decision_function(X_aligned)

    def save(self, path: Path) -> None:
        logging.info("Saving signal model to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_columns": self.feature_columns},
            path,
        )
        logging.info("Model saved.")

    @classmethod
    def load(cls, path: Path) -> "EarningsSignalModel":
        payload = joblib.load(path)
        instance = cls()
        instance.model = payload["model"]
        instance.feature_columns = payload["feature_columns"]
        logging.info("Loaded model from %s", path)
        return instance


def evaluate_model(model: EarningsSignalModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    logging.info("Evaluating model on test set (%s samples).", len(X_test))
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    z_scores = model.decision_function(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    logging.info(
        "Evaluation metrics — accuracy: %.3f, precision: %.3f, recall: %.3f, f1: %.3f, roc_auc: %s",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        "nan" if np.isnan(metrics["roc_auc"]) else f"{metrics['roc_auc']:.3f}",
    )

    predictions = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "prob_up_10d": y_proba,
            "decision_score": z_scores,
        },
        index=y_test.index,
    )
    predictions.to_parquet(PREDICTIONS_PATH)
    logging.info("Saved test predictions to %s", PREDICTIONS_PATH)

    return metrics, predictions


def cross_validate_baseline(X: pd.DataFrame, y: pd.Series, folds: int = 5) -> List[float]:
    logging.info("Running %s-fold time-series cross-validation (ROC-AUC).", folds)
    estimator = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    cv = TimeSeriesSplit(n_splits=folds)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    logging.info("Cross-validation ROC-AUC: %s", ", ".join(f"{s:.3f}" for s in scores))
    return scores.tolist()


def train_and_evaluate(
    *,
    cv_folds: int = 5,
) -> Dict[str, object]:
    splits = load_preprocessed_splits()
    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]

    cv_scores = cross_validate_baseline(X_train, y_train, folds=cv_folds)

    signal_model = EarningsSignalModel()
    signal_model.fit(X_train, y_train)

    metrics, predictions = evaluate_model(signal_model, X_test, y_test)
    signal_model.save(MODEL_PATH)

    summary = {
        "metrics": metrics,
        "cv_scores": cv_scores,
        "model_path": str(MODEL_PATH),
        "preprocess_artifacts_path": str(PREPROCESS_ARTIFACTS_PATH),
        "predictions_path": str(PREDICTIONS_PATH),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "predictions": predictions,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    persistable = {k: v for k, v in summary.items() if k != "predictions"}
    joblib.dump(persistable, MODEL_DIR / "signal_training_summary.joblib")
    logging.info("Training artifacts saved to %s", MODEL_DIR)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting earnings signal model training.")
    summary = train_and_evaluate()
    logging.info("Training complete. Metrics: %s", summary["metrics"])


if __name__ == "__main__":
    main()

