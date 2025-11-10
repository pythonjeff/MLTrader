import asyncio
import os
import pandas as pd
from loguru import logger
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from dydx_v4_client.indexer.candles_resolution import CandlesResolution
from strategies import perp_strategies as ps

class PerpTrader:
    def __init__(self):
        self.network = os.getenv("DYDX_NETWORK", "testnet")
        # Use the appropriate indexer URL based on network
        indexer_url = "https://indexer.v4testnet.dydx.exchange" if self.network == "testnet" else "https://indexer.dydx.trade"
        self.client = IndexerClient(host=indexer_url)

    async def fetch_candles(self, market="ETH-USD", resolution=CandlesResolution.ONE_DAY, limit=100):
        """Fetch candles and return a sorted DataFrame."""
        # Convert resolution enum to its string value
        resolution_str = resolution.value if hasattr(resolution, 'value') else resolution
        candles = await self.client.markets.get_perpetual_market_candles(market=market, resolution=resolution_str, limit=limit)
        if not candles or "candles" not in candles:
            logger.error("No candle data returned.")
            return pd.DataFrame()
        df = pd.DataFrame(candles["candles"])
        df["startedAt"] = pd.to_datetime(df["startedAt"])
        df = df.sort_values('startedAt').reset_index(drop=True)
        return df

    async def run(self):
        logger.info(f"Connecting to dYdX {self.network} API, Cat...")
        df = await self.fetch_candles("ETH-USD", resolution=CandlesResolution.ONE_DAY, limit=300)
        logger.info(f"Fetched {len(df)} candles for ETH-USD, Cat.")

        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")

        df = ps.moving_average_crossover_strategy(df)

        print(df[['startedAt', 'close', 'ema_fast', 'ema_slow', 'signal']].tail())
        
if __name__ == "__main__":
    asyncio.run(PerpTrader().run())