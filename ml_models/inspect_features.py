# ml_models/inspect_features.py
import pandas as pd
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_PATH = Path("models/signal_model.joblib")
X_TRAIN_PATH = Path("data/processed/X_train.parquet")

def inspect_features():
    logging.info("🔍 Loading model and training data...")
    model = joblib.load(MODEL_PATH)
    X_train = pd.read_parquet(X_TRAIN_PATH)
    features = X_train.columns.tolist()

    if hasattr(model, "coef_"):
        logging.info("📈 Detected linear model — inspecting coefficients.")
        coefs = model.coef_[0]
        importance_df = pd.DataFrame({
            "feature": features,
            "importance": coefs,
            "abs_importance": abs(coefs)
        }).sort_values("abs_importance", ascending=False)

    elif hasattr(model, "feature_importances_"):
        logging.info("🌲 Detected tree-based model — inspecting feature importances.")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": features,
            "importance": importances
        }).sort_values("importance", ascending=False)

    else:
        logging.warning("⚠️ Model type not recognized for feature inspection.")
        return

    logging.info(f"Top 10 most influential features:\n{importance_df.head(10)}")
    importance_df.to_csv("models/feature_importance.csv", index=False)
    logging.info("💾 Saved feature importance to models/feature_importance.csv")

if __name__ == "__main__":
    inspect_features()