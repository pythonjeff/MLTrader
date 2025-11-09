"""Fetch price bars and compute price reaction metrics around earnings events.

Output schema includes:
- symbol: Stock ticker symbol
- event_date: Event date (UTC ISO string)
- label: Binary label indicating positive (1) or non-positive (0) 10-day return
- pre_5d_return: Return over 5 days before the event
- pre_10d_return: Return over 10 days before the event
- pre_20d_return: Return over 20 days before the event
- pre_volatility: 20-day realized volatility before the event
"""

import os
import time
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

CACHE_DIR = Path("data/raw/price_reaction")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
    return dates.dt.tz_localize(None)


LOOKBACK_YEARS = _price_lookback_years(4)
PRICE_REACTION_BUFFER_DAYS = _int_from_env("PRICE_REACTION_BUFFER_DAYS", 30)
TRAIN_TEST_CUTOFF = os.getenv("TRAIN_TEST_CUTOFF", "2025-10-01")
EARNINGS_EVENTS_PATH = os.getenv(
    "EARNINGS_EVENTS_PATH",
    "data/processed/historical_earnings_filtered.parquet",
)
START_DATE_ENV = os.getenv("START_DATE")


def get_lookback_cutoff(years: int = LOOKBACK_YEARS) -> pd.Timestamp:
    if START_DATE_ENV:
        try:
            cutoff = pd.to_datetime(START_DATE_ENV).tz_localize(None).normalize()
            print(f"Using explicit START_DATE from environment: {cutoff}")
            return cutoff
        except Exception as e:
            print(f"⚠️ Invalid START_DATE={START_DATE_ENV!r}: {e}. Falling back to lookback years.")
    cutoff = pd.Timestamp.utcnow().replace(tzinfo=None).normalize() - pd.DateOffset(years=years)
    print(f"Using lookback cutoff based on {years} years: {cutoff}")
    return cutoff


def fetch_bars(symbol, start_date, end_date):
    fmp_api_key = os.getenv("FMP_API_KEY")
    if not fmp_api_key:
        print("❌ FMP_API_KEY is not set in environment variables.")
        return None

    cache_name = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if not cached.empty:
                cached["t"] = pd.to_datetime(cached["t"])
                cached = cached.set_index("t").sort_index()
                print(f"📦 Loaded cached bars for {symbol} ({start_date.date()} → {end_date.date()})")
                return cached
        except Exception as exc:
            print(f"⚠️ Failed to load cached bars for {symbol}: {exc}. Refetching...")

    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    params = {
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "apikey": fmp_api_key
    }
    try:
        response = requests.get(url, params=params)
    except Exception as e:
        print(f"Failed to fetch data for {symbol} due to exception: {e}")
        return None

    if response.status_code == 403:
        print(f"❌ Forbidden (403) for {symbol}. Check if your FMP API key has access.")
        return None
    elif response.status_code == 404:
        print(f"❌ Not found (404) for {symbol}. Symbol may be invalid.")
        return None
    elif response.status_code != 200:
        print(f"Failed to fetch data for {symbol}: {response.status_code} {response.text}")
        return None

    data = response.json()
    historical = data.get("historical")
    if not historical:
        print(f"No historical data returned for {symbol} between {start_date} and {end_date}")
        return None

    df = pd.DataFrame(historical)
    if df.empty:
        print(f"No data in DataFrame for {symbol}")
        return None
    df['t'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    df.set_index('t', inplace=True)
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
    df = df.sort_index()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['vol'] = df['log_ret'].rolling(20).std()

    try:
        df_to_store = df.reset_index()
        df_to_store.to_parquet(cache_path, index=False)
        print(f"💾 Cached price bars for {symbol} at {cache_path}")
    except Exception as exc:
        print(f"⚠️ Unable to cache bars for {symbol}: {exc}")

    return df

def compute_returns(df, today):
    today = pd.Timestamp(today).tz_localize(None)
    results = {}
    if today not in df.index:
        # Find the next available trading day after today
        future_dates = df.index[df.index >= today]
        if len(future_dates) == 0:
            print(f"No trading data on or after date {today}")
            return None
        today = future_dates[0]

    close_price = df.loc[today, "close"]

    # Compute volatility using data up to (but not including) today
    pre_today_data = df[df.index < today]
    if 'log_ret' in pre_today_data.columns and len(pre_today_data) >= 20:
        vol_20 = pre_today_data['log_ret'].rolling(20).std().iloc[-1]
        if vol_20 is None or np.isnan(vol_20) or vol_20 == 0:
            vol_20 = None
    else:
        vol_20 = None

    for days in [1, 3, 5, 10]:
        future_date = today + pd.Timedelta(days=days)
        future_dates = df.index[df.index >= future_date]
        if len(future_dates) == 0:
            results[f"return_{days}d"] = None
            results[f"return_{days}d_norm"] = None
        else:
            future_close = df.loc[future_dates[0], "close"]
            ret = (future_close - close_price) / close_price
            results[f"return_{days}d"] = ret
            if vol_20 is not None:
                results[f"return_{days}d_norm"] = ret / vol_20
            else:
                results[f"return_{days}d_norm"] = None

    # Overnight gap: next open vs previous close
    next_day = today + pd.Timedelta(days=1)
    next_days = df.index[df.index >= next_day]
    prev_days = df.index[df.index < today]
    if len(next_days) == 0 or len(prev_days) == 0:
        results["overnight_gap"] = None
    else:
        next_open = df.loc[next_days[0], "open"]
        prev_close = df.loc[prev_days[-1], "close"]
        results["overnight_gap"] = (next_open - prev_close) / prev_close

    # Compute pre-today returns and volatility
    # pre_5d_return, pre_10d_return, pre_20d_return, pre_volatility
    for pre_days in [5, 10, 20]:
        start_date = today - pd.Timedelta(days=pre_days)
        pre_period_dates = df.index[(df.index >= start_date) & (df.index < today)]
        if len(pre_period_dates) == 0:
            results[f"pre_{pre_days}d_return"] = None
        else:
            first_date = pre_period_dates[0]
            last_date = pre_period_dates[-1]
            start_price = df.loc[first_date, "close"]
            end_price = df.loc[last_date, "close"]
            pre_ret = (end_price - start_price) / start_price
            results[f"pre_{pre_days}d_return"] = pre_ret

    # pre_volatility: 20-day realized vol before today
    if len(pre_today_data) >= 20:
        pre_vol = pre_today_data['log_ret'].rolling(20).std().iloc[-1]
        if pre_vol is None or np.isnan(pre_vol):
            pre_vol = None
    else:
        pre_vol = None
    results["pre_volatility"] = pre_vol

    return results

def main():
    fmp_key = os.getenv("FMP_API_KEY") or ""
    masked_key = f"{fmp_key[:4]}****" if fmp_key else "missing"
    print(f"Using FMP API for price bars (key={masked_key})")
    print("Loading symbols for processing...")

    # Instead of earnings events, get symbols from cached files or environment
    # For demonstration, assume a list of symbols is provided via env or hardcoded
    symbols_env = os.getenv("SYMBOLS")
    if symbols_env:
        symbols = symbols_env.split(",")
    else:
        # Fallback: read symbols from cached data directory (extract unique symbols)
        symbols = []
        for file in CACHE_DIR.glob("*.parquet"):
            parts = file.stem.split("_")
            if parts:
                sym = parts[0]
                if sym not in symbols:
                    symbols.append(sym)
        if not symbols:
            print("⚠️ No symbols specified and no cached data found.")
            return

    cutoff_date = pd.to_datetime(TRAIN_TEST_CUTOFF).tz_localize(None)
    lookback_cutoff = get_lookback_cutoff(LOOKBACK_YEARS)
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()

    all_results = []

    for symbol in symbols:
        print(f"Fetching bars for symbol: {symbol}")
        start_date = lookback_cutoff - pd.Timedelta(days=PRICE_REACTION_BUFFER_DAYS)
        end_date = today + pd.Timedelta(days=PRICE_REACTION_BUFFER_DAYS)

        bars_df = fetch_bars(symbol, start_date, end_date)
        if bars_df is None or bars_df.empty:
            print(f"Skipping symbol {symbol} due to no data")
            continue

        bars_df = bars_df.sort_index()

        # We will iterate over all dates in bars_df index except last 20 days for forward returns
        max_index = len(bars_df) - 20
        if max_index <= 0:
            print(f"Not enough data for symbol {symbol} to compute forward returns.")
            continue

        for idx in range(max_index):
            current_date = bars_df.index[idx]
            returns = compute_returns(bars_df, current_date)
            if returns is None:
                continue
            result = {
                "symbol": symbol,
                "event_date": current_date,
                **returns
            }
            all_results.append(result)
        time.sleep(0.2)  # short sleep to avoid rate limits

    if not all_results:
        print("No results to save.")
        return

    print("Saving aggregated price reactions...")
    results_df = pd.DataFrame(all_results)
    if "return_5d" not in results_df.columns:
        raise ValueError("❌ Unable to compute labels: return_5d column missing from results.")

    if "return_10d" not in results_df.columns:
        raise ValueError("❌ Unable to compute labels: return_10d column missing from results.")

    results_df = results_df.dropna(subset=["return_10d"]).copy()
    results_df["label"] = (results_df["return_10d"] > 0).astype(int)

    leakage_cols = [
        "return_1d_norm",
        "return_3d_norm",
        "return_5d_norm",
        "return_10d_norm",
        "abs_return_1d",
        "momentum_flag",
        "vol_norm",
    ]
    drop_cols = [c for c in leakage_cols if c in results_df.columns]
    if drop_cols:
        results_df = results_df.drop(columns=drop_cols)

    results_df['event_date'] = results_df['event_date'].astype(str)

    train_df = results_df[results_df["event_date"] < cutoff_date.strftime("%Y-%m-%d")]
    test_df = results_df[results_df["event_date"] >= cutoff_date.strftime("%Y-%m-%d")]

    train_path = "data/processed/price_reaction_train.parquet"
    test_path = "data/processed/price_reaction_test.parquet"

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print(f"Saved train data to {train_path} ({len(train_df)} rows)")
    print(f"Saved test data to {test_path} ({len(test_df)} rows)")
    print("Done.")
    print("=== Sample of saved train data ===")
    print(train_df.head())
    print("=== Sample of saved test data ===")
    print(test_df.head())

    return results_df

if __name__ == "__main__":
    main()