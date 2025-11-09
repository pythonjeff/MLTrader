# data/historical_earnings.py

import os
from glob import glob
from time import sleep
from typing import Optional

import pandas as pd
import requests

from utils.env import safe_load_env

safe_load_env()
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    print("⚠️ Warning: FMP_API_KEY is not set in the environment.")

def _int_from_env(key: str, default: int) -> int:
    """Fetch a positive integer from the environment with validation."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except ValueError:
        pass
    print(f"⚠️ Invalid value for {key}={value!r}. Using default {default}.")
    return default


def _price_lookback_years(default: int) -> int:
    primary_key = "PRICE_LOOKBACK_YEARS"
    fallback_key = "EARNINGS_LOOKBACK_YEARS"

    value = os.getenv(primary_key)
    source_key = primary_key
    if value is None:
        value = os.getenv(fallback_key)
        source_key = fallback_key
        if value is not None:
            print(
                "ℹ️ Using deprecated environment variable EARNINGS_LOOKBACK_YEARS; "
                "please migrate to PRICE_LOOKBACK_YEARS."
            )

    if value is None:
        return default

    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except ValueError:
        pass

    print(f"⚠️ Invalid value for {source_key}={value!r}. Using default {default}.")
    return default


def normalize_event_dates(series: pd.Series) -> pd.Series:
    """Standardize datetime series to timezone-naive (UTC-based) for consistent comparisons.

    This handles both tz-aware and tz-naive inputs without raising, and always
    returns a plain datetime64[ns] series (no timezone).
    """
    # First, let pandas infer/parse whatever is there
    dates = pd.to_datetime(series, errors="coerce")

    # If the result is timezone-aware, convert to UTC then drop the tz info
    try:
        tz = dates.dt.tz
    except AttributeError:
        # Not a datetime-like series; just return as-is
        return dates

    if tz is not None:
        # Convert everything to UTC, then make it tz-naive
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)

    return dates


LOOKBACK_YEARS = _price_lookback_years(10)
EVENTS_PER_YEAR = _int_from_env("EARNINGS_EVENTS_PER_YEAR", 10)


def get_lookback_cutoff(years: int = LOOKBACK_YEARS) -> pd.Timestamp:
    """Return a timezone-naive cutoff timestamp based on the lookback horizon."""
    cutoff = pd.Timestamp.utcnow().normalize() - pd.DateOffset(years=years)
    # Drop timezone if any (ensure naive)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    return cutoff


def get_cache_dir(years: int = LOOKBACK_YEARS) -> str:
    """Return the symbol-level cache directory for the selected lookback horizon."""
    return os.path.join("data", "raw", f"historical_earnings_{years}y")


def fetch_historical_earnings(
    symbol: str,
    lookback_years: int = LOOKBACK_YEARS,
    max_events: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch past earnings events from Financial Modeling Prep for a given ticker."""
    if max_events is None:
        max_events = lookback_years * EVENTS_PER_YEAR

    if not FMP_API_KEY:
        print(f"⚠️ {symbol}: Missing FMP_API_KEY environment variable.")
        return pd.DataFrame()

    url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}?apikey={FMP_API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"⚠️ {symbol}: Error {r.status_code}")
        return pd.DataFrame()

    data = r.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Normalize columns to expected schema
    if "date" in df.columns:
        df["event_date"] = normalize_event_dates(df["date"])
    else:
        print(f"⚠️ {symbol}: No usable date field found.")
        return pd.DataFrame()

    df = df.dropna(subset=["event_date"])

    df["actual"] = df.get("eps")
    df["estimate"] = df.get("epsEstimated")

    df["surprise"] = pd.NA
    df["surprise_percent"] = pd.NA
    mask = df["actual"].notna() & df["estimate"].notna() & (df["estimate"] != 0)
    df.loc[mask, "surprise"] = df.loc[mask, "actual"] - df.loc[mask, "estimate"]
    df.loc[mask, "surprise_percent"] = (
        (df.loc[mask, "surprise"] / df.loc[mask, "estimate"].abs()) * 100
    )

    df["eps_direction"] = pd.NA
    df.loc[df["surprise"] > 0, "eps_direction"] = "beat"
    df.loc[df["surprise"] < 0, "eps_direction"] = "miss"
    df.loc[df["surprise"] == 0, "eps_direction"] = "meet"

    df["year"] = df["event_date"].dt.year
    df["symbol"] = symbol

    df = df.drop_duplicates(subset=["symbol", "event_date"])
    cutoff_date = get_lookback_cutoff(lookback_years)
    df = df.sort_values("event_date", ascending=True)
    df = df[df["event_date"] >= cutoff_date]

    if df.empty:
        return df

    # Removed truncation to max_events to allow full available history
    # if max_events:
    #     df = df.tail(max_events)

    df["surprise_pct"] = pd.NA
    mask_pct = df["actual"].notna() & df["estimate"].notna() & (df["estimate"] != 0)
    df.loc[mask_pct, "surprise_pct"] = (
        (df.loc[mask_pct, "actual"] - df.loc[mask_pct, "estimate"])
        / df.loc[mask_pct, "estimate"].abs()
    )

    df["eps_change_qoq"] = df["actual"].pct_change(fill_method=None).shift(1)
    df["eps_change_yoy"] = df["actual"].pct_change(periods=4, fill_method=None).shift(1)

    df["surprise_pct"] = df["surprise_pct"].shift(1)

    df = df[df["event_date"] <= pd.Timestamp.today()]

    numeric_cols = ["actual", "estimate", "surprise", "surprise_percent", "surprise_pct", "eps_change_qoq", "eps_change_yoy"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df = df.sort_values("event_date", ascending=False)
    cols = [
        "symbol",
        "event_date",
        "actual",
        "estimate",
        "surprise",
        "surprise_percent",
        "eps_direction",
        "year",
        "eps_change_qoq",
        "eps_change_yoy",
        "surprise_pct",
    ]
    return df[cols]


def build_historical_dataset(
    universe: list,
    lookback_years: int = LOOKBACK_YEARS,
    events_per_year: int = EVENTS_PER_YEAR,
    max_events: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch and combine historical earnings for the filtered universe."""
    if max_events is None:
        max_events = lookback_years * events_per_year

    cache_dir = get_cache_dir(lookback_years)
    os.makedirs(cache_dir, exist_ok=True)

    print(
        f"▶️ Building historical earnings for {len(universe)} symbols "
        f"(~{lookback_years}y window, up to {max_events} events each)"
    )

    for sym in universe:
        cache_path = os.path.join(cache_dir, f"{sym}.parquet")
        if os.path.exists(cache_path):
            print(f"Skipping {sym} (cache found).")
            continue

        print(f"Fetching {sym}...")
        df_sym = fetch_historical_earnings(sym, lookback_years, max_events)
        if not df_sym.empty:
            df_sym.to_parquet(cache_path)
            print(f"  ➕ {sym}: {len(df_sym)} events saved.")
        else:
            print(f"  ⚠️ {sym}: No data saved.")
        sleep(0.5)

    all_files = glob(os.path.join(cache_dir, "*.parquet"))
    if not all_files:
        print("⚠️ No cached data to combine.")
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in all_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "event_date"])
    combined = combined[combined["event_date"] >= get_lookback_cutoff(lookback_years)]
    combined = combined.sort_values("event_date", ascending=False).reset_index(drop=True)

    output_path = "data/processed/historical_earnings.parquet"
    os.makedirs("data/processed", exist_ok=True)
    combined.to_parquet(output_path)
    print(f"💾 Saved combined dataset: {output_path} (rows: {len(combined)})")

    # Filtering step: keep symbols with sufficient event history and equalize length
    coverage = (
        combined.groupby("symbol")["event_date"]
        .agg(event_count="count")
        .reset_index()
    )
    coverage = coverage[coverage["event_count"] > 4]

    if coverage.empty:
        print("\n⚠️ No symbols have more than 4 events; filtered dataset will be empty.")
        filtered = combined.head(0).copy()
        filtered_symbols = []
        min_events = 0
    else:
        filtered_symbols = coverage["symbol"].tolist()
        min_events = int(coverage["event_count"].min())

        print(
            f"\n✅ Keeping {len(filtered_symbols)} symbols with more than 4 events "
            f"(uniform length: {min_events})"
        )
        print(", ".join(filtered_symbols))

        filtered = (
            combined[combined["symbol"].isin(filtered_symbols)]
            .sort_values("event_date", ascending=False)
            .groupby("symbol", group_keys=False)
            .head(min_events)
            .reset_index(drop=True)
        )

    filtered_output_path = "data/processed/historical_earnings_filtered.parquet"
    filtered.to_parquet(filtered_output_path)
    print(f"💾 Saved filtered dataset: {filtered_output_path} (rows: {len(filtered)})")

    print("\n=== Events per Symbol (Filtered) ===")
    print(filtered.groupby("symbol")["event_date"].count().sort_values(ascending=False))

    print("=== Sample rows (Filtered) ===")
    print(filtered.head(10))
    return combined


if __name__ == "__main__":
    print("📂 Loading all S&P 500 tickers from data/processed/sp500_constituents.parquet...")
    df_symbols = pd.read_parquet("data/processed/sp500_constituents.parquet")
    universe = sorted(df_symbols["symbol"].dropna().astype(str).unique().tolist())

    if not universe:
        raise ValueError("❌ No symbols found in sp500_constituents.parquet")

    print(f"✅ Loaded {len(universe)} S&P 500 symbols.")
    build_historical_dataset(universe, lookback_years=LOOKBACK_YEARS)