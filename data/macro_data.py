import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from utils.env import safe_load_env

MODULE_DIR = Path(__file__).resolve().parent
safe_load_env(
    paths=[
        MODULE_DIR.parent / ".env",
        MODULE_DIR / ".env",
        Path(".env"),
    ]
)


def _int_from_env(key: str, default: int) -> int:
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
    dates = pd.to_datetime(series, errors="coerce", utc=True)
    return dates.dt.tz_convert("UTC").dt.tz_localize(None)


FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("❌ Missing FRED_API_KEY in .env file.")

LOOKBACK_YEARS = _price_lookback_years(4)
MACRO_BUFFER_YEARS = _int_from_env("MACRO_BUFFER_YEARS", 1)
EARNINGS_EVENTS_PATH = os.getenv("EARNINGS_EVENTS_PATH", "data/processed/historical_earnings.parquet")

FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "treasury_10y": "DGS10",
    "sp500": "SP500",
    "vix": "VIXCLS"
}
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def get_lookback_cutoff(years: int = LOOKBACK_YEARS) -> pd.Timestamp:
    return pd.Timestamp.utcnow().normalize() - pd.DateOffset(years=years)


def compute_observation_start(
    events_path: str = EARNINGS_EVENTS_PATH,
    lookback_years: int = LOOKBACK_YEARS,
    buffer_years: int = MACRO_BUFFER_YEARS,
    min_buffer_years: int = 2,
) -> str:
    """Determine the observation_start parameter for FRED series requests."""
    fallback_date = (pd.Timestamp.utcnow().normalize() - pd.DateOffset(years=lookback_years + buffer_years)).date()

    if events_path and os.path.exists(events_path):
        try:
            events_df = pd.read_parquet(events_path, columns=["event_date"])
            events_df["event_date"] = normalize_event_dates(events_df["event_date"])
            min_event = events_df["event_date"].min()
            if pd.notnull(min_event):
                # Ensure at least min_buffer_years before earliest event
                start_candidate = (min_event - pd.DateOffset(years=max(buffer_years, min_buffer_years))).date()
                start_date = min(start_candidate, fallback_date)
                return start_date.strftime("%Y-%m-%d")
        except Exception as exc:
            print(f"⚠️ Unable to derive macro observation start from {events_path}: {exc}")

    return fallback_date.strftime("%Y-%m-%d")


def fetch_fred_series(series_id, observation_start: str):
    """Fetch a single FRED series as a pandas DataFrame."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": observation_start
    }
    r = requests.get(FRED_BASE_URL, params=params)
    if r.status_code != 200:
        print(f"⚠️ FRED fetch failed for {series_id}: {r.text}")
        return None
    data = r.json()["observations"]
    df = pd.DataFrame(data)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().set_index("date").sort_index()
    return df.rename(columns={"value": series_id})

def build_macro_features():
    observation_start = compute_observation_start()
    print(f"📡 Fetching macroeconomic data (start: {observation_start}, lookback≥{LOOKBACK_YEARS}y)...")
    dfs = []
    for name, sid in FRED_SERIES.items():
        df = fetch_fred_series(sid, observation_start)
        if df is not None:
            dfs.append(df)
            print(f"✅ Retrieved {name} ({sid}), {len(df)} points.")
    if not dfs:
        raise RuntimeError("❌ No macro data retrieved.")

    macro = pd.concat(dfs, axis=1).ffill()

    # --- Inflation context ---
    macro["cpi_mom"] = macro["CPIAUCSL"].pct_change() * 100
    macro["cpi_yoy"] = macro["CPIAUCSL"].pct_change(12) * 100
    macro["cpi_3m_trend"] = macro["cpi_mom"].rolling(3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x.dropna()) == 3 else np.nan)

    # --- Rate context ---
    macro["rate_change_1m"] = macro["FEDFUNDS"].diff()
    macro["rate_change_3m"] = macro["FEDFUNDS"].diff(3)
    macro["real_rate"] = macro["FEDFUNDS"] - macro["cpi_yoy"]

    # --- Yield curve ---
    macro["yield_spread"] = macro["DGS10"] - macro["FEDFUNDS"]

    # --- Market context ---
    macro["sp500_30d_ret"] = macro["SP500"].pct_change(21) * 100
    macro["sp500_mom_3m"] = macro["SP500"].pct_change(63) * 100
    macro["vix_30d_zscore"] = (macro["VIXCLS"] - macro["VIXCLS"].rolling(30).mean()) / macro["VIXCLS"].rolling(30).std()

    macro = macro.ffill().dropna()
    macro.to_parquet("data/raw/macro_context_raw.parquet")
    print("💾 Saved contextual macro data: data/raw/macro_context_raw.parquet")
    return macro

def align_macro_to_events(macro_df, events_path="data/processed/historical_earnings.parquet"):
    print("🗓️ Aligning macro context to earnings events...")
    events_df = pd.read_parquet(events_path)
    if "event_date" not in events_df.columns:
        for c in ["date", "period", "period_end"]:
            if c in events_df.columns:
                events_df = events_df.rename(columns={c: "event_date"})
                break
    events_df["event_date"] = normalize_event_dates(events_df["event_date"])
    cutoff = get_lookback_cutoff(LOOKBACK_YEARS)
    cutoff = cutoff.tz_localize(None)
    events_df["event_date"] = events_df["event_date"].dt.tz_localize(None)
    macro_df.index = macro_df.index.tz_localize(None)
    events_df = events_df[events_df["event_date"] >= cutoff]
    events_df = events_df.sort_values("event_date").reset_index(drop=True)
    if events_df.empty:
        print("⚠️ No earnings events within the configured lookback window.")
        return pd.DataFrame()
    aligned_rows = []

    skipped_events = 0
    total_events = len(events_df)

    for _, row in events_df.iterrows():
        event_date = row["event_date"]
        eligible_macro = macro_df.loc[macro_df.index <= event_date]
        if eligible_macro.empty:
            skipped_events += 1
            continue
        macro_snapshot = eligible_macro.iloc[-1]
        aligned_rows.append({
            "symbol": row["symbol"],
            "event_date": event_date,
            **macro_snapshot.to_dict()
        })

    if not aligned_rows:
        print("⚠️ No aligned macro rows produced.")
        return pd.DataFrame()

    aligned_df = pd.DataFrame(aligned_rows)
    aligned_df.to_parquet("data/processed/macro_features.parquet")
    print(f"💾 Saved aligned macro features: data/processed/macro_features.parquet ({len(aligned_df)} rows)")
    print("=== Sample ===")
    print(aligned_df.head())
    covered_events = total_events - skipped_events
    print(f"📊 Macro data coverage: {covered_events} events covered, {skipped_events} events missing macro data")
    return aligned_df

def main():
    macro_df = build_macro_features()
    aligned_df = align_macro_to_events(macro_df)
    return aligned_df

if __name__ == "__main__":
    main()