# data/preprocess_earnings.py
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

RAW_PATH = "data/raw/earnings_next_week.parquet"
OUTPUT_PATH = "data/processed/earnings_preprocessed.parquet"

def load_raw_earnings(path: str = RAW_PATH) -> pd.DataFrame:
    """Load raw earnings data from parquet."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw file not found: {path}")
    df = pd.read_parquet(path)
    print(f"✅ Loaded {len(df)} raw earnings records.")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Perform light cleaning and normalization."""
    df = df.copy()

    # Normalize date column
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["hour"].fillna("unknown")

    # Sort upcoming earnings chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure numeric columns are floats
    num_cols = ["epsActual", "epsEstimate", "revenueActual", "revenueEstimate"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"🧹 Cleaned basic fields, remaining columns: {list(df.columns)}")
    return df


def filter_liquid_equities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter for highly traded, optionable U.S. equities.
    This is a heuristic filter to retain the most liquid names.
    """
    # Remove tickers with strange symbols (OTC, pink sheets, etc.)
    df = df[df["symbol"].str.len() <= 5]

    # Drop ADR-like tickers
    df = df[~df["symbol"].str.endswith((".Y", ".F", ".L"))]

    # Optionally, you can plug in a liquidity reference file (e.g., from Polygon or Finnhub)
    # For now, we'll heuristically assume top-cap tickers have large revenues
    df["liquid_flag"] = df["revenueEstimate"].fillna(0) > 1e9  # $1B+ estimate
    df = df[df["liquid_flag"]]

    print(f"💧 Filtered to {len(df)} liquid equities.")
    return df.drop(columns=["liquid_flag"])


def retain_recent_reporters(df: pd.DataFrame, window_days: int = 2) -> pd.DataFrame:
    """
    Retain companies that have reported in the last N days,
    since post-earnings volatility may create opportunities.
    """
    # Convert date column to timezone-naive UTC
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)

    # Get current UTC date (timezone-naive) for consistent comparisons
    today = pd.Timestamp.utcnow().normalize()
    cutoff_recent = today - timedelta(days=window_days)
    df_recent = df[df["date"] >= cutoff_recent]
    df_future = df[df["date"] >= today]
    df_combined = pd.concat([df_future, df_recent]).drop_duplicates("symbol")

    print(f"⚡ Retained {len(df_combined)} upcoming or recent reporters (within {window_days} days).")
    return df_combined


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple engineered features to assist the model."""
    df["eps_diff"] = df["epsActual"] - df["epsEstimate"]
    df["revenue_diff"] = df["revenueActual"] - df["revenueEstimate"]
    df["eps_surprise_pct"] = df["eps_diff"] / df["epsEstimate"].replace(0, pd.NA)
    df["revenue_surprise_pct"] = df["revenue_diff"] / df["revenueEstimate"].replace(0, pd.NA)

    print("🧠 Added engineered features: EPS surprise %, Revenue surprise %")
    return df


def preprocess_earnings():
    df = load_raw_earnings()
    df = basic_cleaning(df)
    df = filter_liquid_equities(df)
    df = retain_recent_reporters(df, window_days=2)
    df = enrich_features(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    print(f"💾 Saved preprocessed dataset with {len(df)} rows → {OUTPUT_PATH}")
    print(df.head(10))
    print("\n📅 Next 10 upcoming earnings:")
    print(df.sort_values("date").head(10)[["symbol", "date", "hour", "epsEstimate", "revenueEstimate", "eps_surprise_pct"]])


if __name__ == "__main__":
    preprocess_earnings()