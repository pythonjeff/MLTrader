import pandas as pd
import pandas_ta_classic as ta
from loguru import logger
from data.cb_data import fetch_candles

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.error(f"Input DataFrame is missing required columns: {missing}")
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    if "granularity" in df.columns:
        logger.info("Detected 'granularity' column. Computing indicators per granularity group.")
        dfs = []
        for granularity, group_df in df.groupby("granularity"):
            logger.info(f"Starting computation of technical indicators for granularity: {granularity}")
            computed_df = compute_indicators(group_df.drop(columns=["granularity"]))
            computed_df["granularity"] = granularity
            dfs.append(computed_df)
            logger.info(f"Completed computation for granularity: {granularity}")
        result_df = pd.concat(dfs).sort_index()
        return result_df

    df = df.copy()
    logger.info("Starting computation of technical indicators.")

    # EMA
    df["ema_fast"] = ta.ema(df["close"], length=9)
    df["ema_slow"] = ta.ema(df["close"], length=26)
    logger.info("Computed EMA (fast=9, slow=26).")

    # RSI
    df["rsi"] = ta.rsi(df["close"], length=14)
    logger.info("Computed RSI (14).")

    # MACD
    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    logger.info("Computed MACD.")

    # ATR
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    logger.info("Computed ATR (14).")

    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=20)
    df["bb_bbm"] = bbands["BBM_20_2.0"]
    df["bb_bbh"] = bbands["BBU_20_2.0"]
    df["bb_bbl"] = bbands["BBL_20_2.0"]
    logger.info("Computed Bollinger Bands (20).")

    # Volume change
    df["volume_change"] = df["volume"].pct_change()
    logger.info("Computed volume change.")

    # Drop rows with NaN values created by indicator calculations
    df.dropna(inplace=True)
    logger.info("Dropped rows with NaN values after indicator computation.")

    return df


if __name__ == "__main__":
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting test of technical_features module...")

    # Fetch BTC-PERP-INTX hourly candles
    df_candles = fetch_candles("BTC-PERP-INTX", "ONE_HOUR")

    # Compute indicators
    df_indicators = compute_indicators(df_candles)

    # Print the tail of the resulting DataFrame
    print(df_indicators.tail())

    logger.info("Test completed.")
