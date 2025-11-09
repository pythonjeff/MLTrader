"""
merge_datasets.py
-----------------
Combines all processed feature tables (earnings fundamentals, price reactions,
macro context) into a single parquet that feeds `data/preprocess_ml.py`.

The merged dataset only contains historical events (strictly before today) so
that downstream training mimics the live inference setting where the model must
predict the reaction to the next earnings release.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_FILE = DATA_DIR / "ml_merged.parquet"

EARNINGS_CANDIDATES: tuple[str, ...] = (
    "historical_earnings_filtered.parquet",
    "historical_earnings.parquet",
)
PRICE_FILE = "price_reaction.parquet"
MACRO_FILE = "macro_features.parquet"


def _resolve_earnings_file() -> str:
    for candidate in EARNINGS_CANDIDATES:
        if (DATA_DIR / candidate).exists():
            return candidate
    raise FileNotFoundError(
        f"None of the expected earnings files are present in {DATA_DIR}: "
        f"{', '.join(EARNINGS_CANDIDATES)}"
    )


def _load_parquet(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required parquet not found: {path}")
    df = pd.read_parquet(path)
    for required in ("event_date", "symbol"):
        if required not in df.columns:
            raise ValueError(f"{path} is missing required column '{required}'")
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str)
    return df


def _keep_past_events(df: pd.DataFrame) -> pd.DataFrame:
    today = datetime.utcnow().date()
    past_mask = df["event_date"].dt.date < today
    return df.loc[past_mask].copy()


def _dedupe(df: pd.DataFrame, subset: Iterable[str]) -> pd.DataFrame:
    return df.drop_duplicates(list(subset))


def merge_all() -> pd.DataFrame:
    earnings_file = _resolve_earnings_file()
    print(f"🔹 Loading earnings data from {earnings_file}")
    earnings = _load_parquet(earnings_file)
    price = _load_parquet(PRICE_FILE)
    macro = _load_parquet(MACRO_FILE)

    # Ensure we only train on historical data (no current/future events).
    earnings = _keep_past_events(earnings)
    price = _keep_past_events(price)
    macro = _keep_past_events(macro)

    # Drop duplicate symbol/event rows prior to merging.
    earnings = _dedupe(earnings, ("symbol", "event_date"))
    price = _dedupe(price, ("symbol", "event_date"))
    macro = _dedupe(macro, ("symbol", "event_date"))

    if "label" not in price.columns:
        raise ValueError(
            f"{PRICE_FILE} must contain a 'label' column computed in price_reaction.py"
        )
    price = price.dropna(subset=["label"])
    price["label"] = price["label"].astype(int)

    # ---- Merge step-by-step ----
    print("🔹 Merging earnings fundamentals with price reactions...")
    merged = pd.merge(
        price,
        earnings,
        on=["symbol", "event_date"],
        how="inner",
        validate="one_to_one",
    )

    print("🔹 Adding macro context features...")
    merged = pd.merge(
        merged,
        macro,
        on=["symbol", "event_date"],
        how="left",
        validate="one_to_one",
    )

    if merged.empty:
        raise RuntimeError("Merged dataset is empty; check upstream processing steps.")

    # Final ordering and persistence.
    merged = merged.sort_values(["symbol", "event_date"]).reset_index(drop=True)
    merged.to_parquet(OUTPUT_FILE, index=False)
    print(
        f"💾 Saved merged dataset: {OUTPUT_FILE} "
        f"({merged.shape[0]} rows, {merged.shape[1]} columns)"
    )

    # Display a concise sample of the most relevant columns for quick inspection.
    sample_columns = [
        "symbol",
        "event_date",
        "label",
        "pre_5d_return",
        "pre_20d_return",
        "pre_volatility",
        "surprise_percent",
        "eps_direction",
    ]
    available_preview = [col for col in sample_columns if col in merged.columns]
    print("\n=== Sample preview ===")
    print(merged.head(10)[available_preview])

    return merged


if __name__ == "__main__":
    merge_all()
"""
merge_datasets.py
-----------------
Combines all preprocessed data sources (earnings, price, macro, sentiment)
into a single ML-ready dataset for downstream preprocessing and training.
"""
