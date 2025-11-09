import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from data.earnings_calendar import get_earnings_calendar
from data.historical_earnings import LOOKBACK_YEARS, build_historical_dataset
from data.macro_data import main as build_macro_layer
from data.merge_datasets import merge_all
from data.preprocess_ml import preprocess
from data.price_reaction import main as build_price_reactions
from data.sp500_universe import fetch_sp500_constituents
from ml_models.signal_model import train_and_evaluate
from services.trade_ideas import generate_trade_ideas


def _log_section(title: str) -> None:
    logging.info("\n" + "=" * 80)
    logging.info("▶ %s", title)
    logging.info("=" * 80)


def refresh_universe() -> pd.DataFrame:
    _log_section("Refreshing S&P 500 universe")
    return fetch_sp500_constituents()


def refresh_earnings_calendar() -> Dict[str, pd.DataFrame]:
    _log_section("Fetching upcoming earnings calendar")
    earnings_df, sp500_flags = get_earnings_calendar()
    return {"earnings": earnings_df, "sp500_with_flags": sp500_flags}


def refresh_historical_earnings(symbols: pd.Series) -> Optional[pd.DataFrame]:
    _log_section("Building historical earnings dataset")
    universe = sorted(set(symbols.dropna().astype(str)))
    if not universe:
        logging.warning("Universe is empty; skipping historical earnings build.")
        return None
    return build_historical_dataset(universe, lookback_years=LOOKBACK_YEARS)


def refresh_price_reactions() -> Optional[pd.DataFrame]:
    _log_section("Computing price reaction features")
    return build_price_reactions()


def refresh_macro_features() -> Optional[pd.DataFrame]:
    _log_section("Fetching macroeconomic context")
    return build_macro_layer()


def run_pipeline(test_size: float = 0.2) -> Dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    summary: Dict[str, object] = {}

    sp500_df = refresh_universe()
    summary["sp500_count"] = len(sp500_df)

    calendar_payload = refresh_earnings_calendar()
    summary["earnings_rows"] = len(calendar_payload["earnings"])

    refresh_historical_earnings(sp500_df["symbol"])
    refresh_price_reactions()
    refresh_macro_features()

    _log_section("Merging processed datasets")
    merged = merge_all()
    summary["merged_rows"] = len(merged)
    summary["merged_columns"] = merged.shape[1]

    _log_section("Preprocessing features")
    prep_outputs = preprocess(test_size=test_size, return_metadata=True)
    X_train = prep_outputs["X_train"]
    X_test = prep_outputs["X_test"]
    y_train = prep_outputs["y_train"]
    y_test = prep_outputs["y_test"]
    test_df = prep_outputs["test_df"]

    summary["train_shape"] = X_train.shape
    summary["test_shape"] = X_test.shape

    _log_section("Training signal model")
    model_summary = train_and_evaluate()
    summary.update(
        {
            "model_metrics": model_summary["metrics"],
            "cv_scores": model_summary["cv_scores"],
            "model_path": model_summary["model_path"],
            "predictions_path": model_summary["predictions_path"],
        }
    )

    predictions = model_summary["predictions"]
    predictions = predictions.reset_index().rename(columns={"index": "row_id"})
    test_context = test_df.reset_index().rename(columns={"index": "row_id"})
    analysis_df = pd.merge(test_context, predictions, on="row_id", how="inner")

    analysis_path = Path("data/processed/trade_analysis.parquet")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_df.to_parquet(analysis_path, index=False)
    summary["analysis_path"] = str(analysis_path)

    report_path = Path("models") / "latest_pipeline_summary.joblib"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = report_path.with_suffix(".json")

    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return str(obj)

    _log_section("Generating LLM trade ideas")
    trade_ideas = generate_trade_ideas(analysis_df)
    summary["trade_ideas"] = trade_ideas

    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    logging.info("Pipeline summary saved to %s", json_path)
    joblib.dump(summary, report_path)
    logging.info("Pipeline summary serialized to %s", report_path)

    return summary


def main() -> None:
    summary = run_pipeline()
    logging.info("Pipeline complete. Metrics: %s", summary.get("model_metrics"))
    trade_section = summary.get("trade_ideas", {})
    llm_summary = trade_section.get("summary")
    if llm_summary:
        print("\n=== LLM Trade Ideas ===\n")
        print(llm_summary)


if __name__ == "__main__":
    main()