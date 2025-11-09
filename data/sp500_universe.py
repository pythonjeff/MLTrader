

import os

import pandas as pd
import requests

from utils.env import safe_load_env

safe_load_env()

FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/api/v3/sp500_constituent"
OUTPUT_PATH = "data/processed/sp500_constituents.parquet"

def fetch_sp500_constituents():
    if not FMP_API_KEY:
        raise ValueError("❌ Missing FMP_API_KEY in environment variables.")

    url = f"{BASE_URL}?apikey={FMP_API_KEY}"
    print(f"🔗 Fetching S&P 500 constituents from FMP: {url}")
    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(f"❌ Failed to fetch data: {response.status_code} {response.text}")

    data = response.json()
    if not data:
        raise ValueError("❌ No data returned from FMP API.")

    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("❌ Empty dataframe returned from API response.")

    df = df[["symbol", "name", "sector"]]
    df = df.dropna().drop_duplicates()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"✅ Saved {len(df)} S&P 500 constituents to {OUTPUT_PATH}")
    print("=== Sample Records ===")
    print(df.head(10))
    return df

if __name__ == "__main__":
    try:
        fetch_sp500_constituents()
    except Exception as e:
        print(f"⚠️ Error: {e}")