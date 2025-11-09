# data/earnings_calendar.py
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

from utils.env import safe_load_env

safe_load_env()

FINNHUB_URL = "https://finnhub.io/api/v1/calendar/earnings"
API_KEY = os.getenv("FINNHUB_API_KEY")

SP500_PATH = "data/processed/sp500_constituents.parquet"

def get_earnings_calendar() -> pd.DataFrame:
    """
    Fetch upcoming earnings for the next 14 days from Finnhub,
    filtered to the S&P 500 tickers.
    """
    if os.path.exists(SP500_PATH):
        sp500_df = pd.read_parquet(SP500_PATH)
        sp500_tickers = sp500_df["symbol"].unique().tolist()
    else:
        sp500_tickers = []

    start = datetime.utcnow()
    end = start + timedelta(days=14)

    params = {
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "token": API_KEY
    }
    resp = requests.get(FINNHUB_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    if "earningsCalendar" not in data:
        print("⚠️ No data field returned, check your API key or range.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(data["earningsCalendar"])
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["hour"].fillna("unknown")

    df = df[df["symbol"].isin(sp500_tickers)].reset_index(drop=True)

    merged = sp500_df.copy()
    merged["has_upcoming_earnings"] = merged["symbol"].isin(df["symbol"])

    return df, merged

# data/earnings_calendar.py
if __name__ == "__main__":
    print(f"📅 Fetching earnings for the next 14 days...\n")
    df, merged = get_earnings_calendar()
    df.to_parquet("data/processed/earnings_next_week.parquet")
    merged.to_parquet("data/processed/sp500_with_earnings_flag.parquet")
    print(f"💾 Saved {len(df)} records to data/processed/earnings_next_week.parquet")
    print(f"✅ Saved {len(merged)} total S&P 500 symbols ({merged['has_upcoming_earnings'].sum()} with earnings) to data/processed/sp500_with_earnings_flag.parquet")
    print(merged.head(10))