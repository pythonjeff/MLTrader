# data/cb_data.py
import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
import pandas as pd
import time

load_dotenv()

def get_cb_client(api_key: str = None, api_secret: str = None) -> RESTClient:
    """Initialize Coinbase Advanced REST client."""
    api_key = api_key or os.getenv("COINBASE_API_KEY")
    api_secret = api_secret or os.getenv("COINBASE_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise RuntimeError("Coinbase API key/secret required.")
    
    client = RESTClient(api_key=api_key, api_secret=api_secret)
    return client

def fetch_candles(
    product_id: str = "BTC-PERP-INTX",
    granularity: str = "ONE_HOUR",
    start_ts: int = None,
    end_ts: int = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch recent candles from Coinbase Advanced Trade API.
    product_id: e.g., 'BTC-USD', or your perp symbol (if available).
    granularity: ONE_MINUTE, FIVE_MINUTE, ONE_HOUR, ONE_DAY, etc.
    start_ts, end_ts: UNIX epoch seconds defining the interval.
    limit: number of buckets (up to max ~350).  
    """
    if end_ts is None:
        end_ts = int(time.time())
    if start_ts is None:
        gran_to_seconds = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800,
            "ONE_HOUR": 3600,
            "TWO_HOUR": 7200,
            "FOUR_HOUR": 14400,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400
        }
        bucket_secs = gran_to_seconds.get(granularity, 3600)
        start_ts = end_ts - (bucket_secs * limit)

    params = {
        "granularity": granularity,
        "start": str(start_ts),
        "end": str(end_ts),
        "limit": str(limit)
    }

    client = get_cb_client()
    try:
        response = client.get(
            f"/api/v3/brokerage/market/products/{product_id}/candles",
            params=params
        )

        raw = response.to_dict() if hasattr(response, "to_dict") else response
        candles = raw.get("candles", [])
        if not candles:
            print(f"Warning: No candle data returned for {product_id}")
            return pd.DataFrame()

        df = pd.DataFrame(candles, columns=["start", "low", "high", "open", "close", "volume"])
        df["start"] = pd.to_datetime(df["start"].astype(int), unit="s")
        df = df.sort_values("start").reset_index(drop=True)

        for col in ["low", "high", "open", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except Exception as e:
        print(f"Error fetching candles: {e}")
        raise


def fetch_multi_candles(product_ids, granularities, limit=1000):
    """
    Fetch candles for multiple product_ids and granularities.
    Returns a dictionary where each key is a product_id and the value is a concatenated DataFrame
    of all granularities for that product, sorted by time.
    Each DataFrame has columns: start, low, high, open, close, volume, granularity.
    """
    result_dict = {}
    output_dir = "data/raw/"
    os.makedirs(output_dir, exist_ok=True)

    for product_id in product_ids:
        print(f"Fetching data for product_id: {product_id}")
        dfs = []
        for granularity in granularities:
            print(f"  Fetching granularity: {granularity}")
            df = fetch_candles(product_id=product_id, granularity=granularity, limit=limit)
            if not df.empty:
                df["granularity"] = granularity
                dfs.append(df)
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values("start").reset_index(drop=True)
            result_dict[product_id] = combined_df

            timestamp = int(time.time())
            filename = f"{product_id}_{timestamp}.parquet"
            filepath = os.path.join(output_dir, filename)
            combined_df.to_parquet(filepath)
            abs_path = os.path.abspath(filepath)
            print(f"Saved data for {product_id} to {abs_path}")
        else:
            result_dict[product_id] = pd.DataFrame()
    return result_dict


if __name__ == "__main__":
    # Example: fetch BTC and ETH perp candles for multiple granularities
    product_ids = ["BTC-PERP-INTX", "ETH-PERP-INTX"]
    granularities = ["ONE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR", "FOUR_HOUR", "ONE_DAY"]
    data = fetch_multi_candles(product_ids, granularities, limit=349)
    for product_id, df in data.items():
        print(f"\n=== Last 5 rows for {product_id} ===")
        print(df.tail())