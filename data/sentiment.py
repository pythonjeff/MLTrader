import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_NEWS_SENTIMENT_URL = "https://finnhub.io/api/v1/news-sentiment"

def fetch_finnhub_sentiment(symbol):
    """
    Fetch sentiment data for a symbol from Finnhub's news-sentiment endpoint.
    Returns a DataFrame with columns: symbol, bullishPercent, bearishPercent, sentiment_score
    sentiment_score is computed as bullishPercent - bearishPercent.
    """
    params = {
        "symbol": symbol,
        "token": FINNHUB_API_KEY
    }
    try:
        resp = requests.get(FINNHUB_NEWS_SENTIMENT_URL, params=params)
        if resp.status_code != 200:
            print(f"[WARN] HTTP error for {symbol}: {resp.status_code} {resp.text}")
            return pd.DataFrame(columns=['symbol', 'bullishPercent', 'bearishPercent', 'sentiment_score'])
        data = resp.json()
        bullishPercent = data.get("bullishPercent", None)
        bearishPercent = data.get("bearishPercent", None)
        if bullishPercent is None or bearishPercent is None:
            print(f"[INFO] No sentiment data found for {symbol}")
            return pd.DataFrame(columns=['symbol', 'bullishPercent', 'bearishPercent', 'sentiment_score'])
        sentiment_score = bullishPercent - bearishPercent
        df = pd.DataFrame([{
            "symbol": symbol,
            "bullishPercent": bullishPercent,
            "bearishPercent": bearishPercent,
            "sentiment_score": sentiment_score
        }])
        return df
    except Exception as e:
        print(f"[ERROR] Exception fetching sentiment for {symbol}: {e}")
        return pd.DataFrame(columns=['symbol', 'bullishPercent', 'bearishPercent', 'sentiment_score'])


def build_sentiment_features(
    events_path="data/processed/historical_earnings.parquet",
    output_path="data/processed/sentiment_features.parquet"
):
    """
    Build sentiment features for each unique symbol in the events file.
    Fetch sentiment once per symbol from Finnhub.
    Saves aggregated features to output_path.
    """
    print(f"[INFO] Loading events from {events_path}")
    events = pd.read_parquet(events_path)
    if "symbol" not in events.columns or "event_date" not in events.columns:
        raise ValueError("Events file must have columns: symbol, event_date")

    unique_symbols = events["symbol"].unique()
    features = []
    for symbol in unique_symbols:
        print(f"[INFO] Fetching sentiment for {symbol}")
        sentiment_df = fetch_finnhub_sentiment(symbol)
        if sentiment_df.empty:
            print(f"[WARN] No sentiment data for {symbol}, skipping.")
            continue
        event_count = events[events["symbol"] == symbol].shape[0]
        sentiment_row = sentiment_df.iloc[0].to_dict()
        sentiment_row["event_count"] = event_count
        features.append(sentiment_row)

    features_df = pd.DataFrame(features)
    print(f"[INFO] Saving sentiment features to {output_path}")
    features_df.to_parquet(output_path, index=False)
    print("[INFO] Sentiment features preview:")
    print(features_df.head(5))


if __name__ == "__main__":
    build_sentiment_features()