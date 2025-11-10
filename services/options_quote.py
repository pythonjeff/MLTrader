from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from services.trade_builder import OptionTrade, generate_option_trades


def _get_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
except ImportError:  # pragma: no cover - optional dependency
    TradingClient = None
    GetOptionContractsRequest = None
    StockHistoricalDataClient = None
    StockLatestTradeRequest = None


@dataclass(slots=True)
class ResolvedOptionContract:
    """Concrete option contract ready for execution or further sizing."""

    symbol: str
    option_symbol: str
    option_type: str  # "call" or "put"
    action: str  # "buy" or "sell"
    expiry: pd.Timestamp
    strike: float
    quantity: int
    estimated_price: float
    confidence: float
    probability: float
    notes: Dict[str, object] = field(default_factory=dict)


class OptionContractResolver:
    """Base resolver interface."""

    def resolve(self, trades: Sequence[OptionTrade]) -> List[ResolvedOptionContract]:
        raise NotImplementedError


def _parse_percentage_bias(bias: str) -> Optional[float]:
    if "pct" in bias:
        direction = 1.0 if "down" not in bias else -1.0
        pct = float(bias.split("pct")[0]) / 100.0
        return pct * direction
    return None


def _compute_target_strike(spot: float, strike_bias: str, option_type: str) -> float:
    pct_shift = _parse_percentage_bias(strike_bias)
    if pct_shift is not None:
        strike = spot * (1 + pct_shift if option_type == "call" else 1 - pct_shift)
    elif strike_bias.startswith("delta"):
        # Placeholder: convert delta to simple OTM shift
        delta = float(strike_bias.split("_")[1])
        shift = (0.5 - delta) * 0.2  # heuristic
        strike = spot * (1 + shift if option_type == "call" else 1 - shift)
    else:
        strike = spot  # ATM fallback
    return max(round(strike, 2), 0.01)


class MockOptionContractResolver(OptionContractResolver):
    """Offline resolver for deterministic tests."""

    def __init__(
        self,
        price_lookup: Optional[Dict[str, float]] = None,
        base_expiry: Optional[pd.Timestamp] = None,
        default_quantity: int = 1,
    ) -> None:
        self.price_lookup = price_lookup or {}
        self.base_expiry = base_expiry or pd.Timestamp.utcnow().normalize()
        self.default_quantity = default_quantity

    def resolve(self, trades: Sequence[OptionTrade]) -> List[ResolvedOptionContract]:
        results: List[ResolvedOptionContract] = []
        for trade in trades:
            price = self.price_lookup.get(trade.symbol, 100.0)
            strike = _compute_target_strike(price, trade.strike_bias, trade.option_type)
            expiry = self.base_expiry + pd.Timedelta(days=trade.expiry_target_days)
            option_symbol = f"{trade.symbol.upper()}-{expiry:%Y%m%d}-{trade.option_type[0].upper()}-{strike:.2f}"
            results.append(
                ResolvedOptionContract(
                    symbol=trade.symbol,
                    option_symbol=option_symbol,
                    option_type=trade.option_type,
                    action=trade.action,
                    expiry=expiry,
                    strike=strike,
                    quantity=self.default_quantity,
                    estimated_price=round(price * 0.1, 2),
                    confidence=trade.confidence,
                    probability=trade.probability,
                    notes=dict(trade.notes) | {"resolver": "mock"},
                )
            )
        return results


class AlpacaOptionContractResolver(OptionContractResolver):
    """Resolve contracts using Alpaca's trading and market data APIs."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: Optional[bool] = None,
        default_quantity: int = 1,
    ) -> None:
        if TradingClient is None or StockHistoricalDataClient is None:
            raise ImportError("alpaca-py must be installed to use AlpacaOptionContractResolver.")

        self.api_key = api_key or _get_env(
            "ALPACA_API_KEY",
            "APCA_API_KEY_ID",
        )
        self.api_secret = api_secret or _get_env(
            "ALPACA_SECRET_KEY",
            "APCA_API_SECRET_KEY",
        )
        self.base_url = base_url or _get_env(
            "ALPACA_BASE_URL",
            "APCA_API_BASE_URL",
            "APCA_API_PAPER_URL",
            default="https://paper-api.alpaca.markets",
        )
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not provided.")

        if paper is None:
            paper = True if self.base_url and "paper" in self.base_url else False

        self.trading_client = TradingClient(
            self.api_key, self.api_secret, paper=paper
        )
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        self.default_quantity = default_quantity

    def _latest_price(self, symbol: str) -> float:
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        response = self.data_client.get_stock_latest_trade(req)
        trade = response[symbol]
        return float(trade.price)

    def _nearest_contract(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        target_expiry: datetime,
    ):
        if GetOptionContractsRequest is None:
            raise ImportError("alpaca-py trading requests unavailable.")

        request = GetOptionContractsRequest(
            underlying_symbol=symbol,
            expiration_date_gte=target_expiry.date(),
            limit=100,
            option_type=option_type,
            status="active",
        )
        contracts = self.trading_client.get_option_contracts(request)  # type: ignore[attr-defined]
        if not contracts:
            return None
        contracts = [
            c
            for c in contracts
            if c.expiration_date >= target_expiry.date()
        ]
        if not contracts:
            return None
        # Choose contract with strike closest to target
        return min(
            contracts,
            key=lambda c: (abs(float(c.strike_price) - strike), c.expiration_date),
        )

    def resolve(self, trades: Sequence[OptionTrade]) -> List[ResolvedOptionContract]:
        results: List[ResolvedOptionContract] = []
        now = datetime.utcnow()

        for trade in trades:
            spot = self._latest_price(trade.symbol)
            strike = _compute_target_strike(spot, trade.strike_bias, trade.option_type)
            target_expiry = now + timedelta(days=trade.expiry_target_days)

            contract = self._nearest_contract(
                symbol=trade.symbol,
                option_type=trade.option_type,
                strike=strike,
                target_expiry=target_expiry,
            )
            if contract is None:
                results.append(
                    ResolvedOptionContract(
                        symbol=trade.symbol,
                        option_symbol="unavailable",
                        option_type=trade.option_type,
                        action=trade.action,
                        expiry=pd.Timestamp(target_expiry.date()),
                        strike=strike,
                        quantity=0,
                        estimated_price=0.0,
                        confidence=trade.confidence,
                        probability=trade.probability,
                        notes=dict(trade.notes)
                        | {
                            "resolver": "alpaca",
                            "error": "No contract found",
                            "target_expiry": target_expiry.date().isoformat(),
                        },
                    )
                )
                continue

            est_price = float(contract.ask_price or contract.mark_price or 0.0)
            results.append(
                ResolvedOptionContract(
                    symbol=trade.symbol,
                    option_symbol=contract.symbol,
                    option_type=trade.option_type,
                    action=trade.action,
                    expiry=pd.Timestamp(contract.expiration_date),
                    strike=float(contract.strike_price),
                    quantity=self.default_quantity,
                    estimated_price=est_price,
                    confidence=trade.confidence,
                    probability=trade.probability,
                    notes=dict(trade.notes)
                    | {
                        "resolver": "alpaca",
                        "spot_price": spot,
                        "target_strike": strike,
                    },
                )
            )

        return results


def _load_trade_candidates(
    analysis_path: Path,
    *,
    min_confidence: float,
    max_trades: int,
) -> List[OptionTrade]:
    df = pd.read_parquet(analysis_path)
    plan = generate_option_trades(
        df,
        min_confidence=min_confidence,
        max_trades=max_trades,
    )
    return plan.trades


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Resolve option contracts for candidate trades."
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("data/processed/trade_analysis.parquet"),
        help="Path to trade_analysis parquet generated by the pipeline.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.15,
        help="Minimum confidence threshold for selecting trades.",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=5,
        help="Maximum number of trades to resolve.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock resolver instead of Alpaca API.",
    )
    parser.add_argument(
        "--default-quantity",
        type=int,
        default=1,
        help="Default contract quantity to assign.",
    )
    args = parser.parse_args()

    if not args.analysis.exists():
        raise FileNotFoundError(
            f"Analysis file not found at {args.analysis}. Run main.py first."
        )

    trades = _load_trade_candidates(
        args.analysis, min_confidence=args.min_confidence, max_trades=args.max_trades
    )
    if not trades:
        print("No trades passed the confidence filter.")
        return

    if args.mock:
        resolver: OptionContractResolver = MockOptionContractResolver(
            default_quantity=args.default_quantity
        )
    else:
        resolver = AlpacaOptionContractResolver(default_quantity=args.default_quantity)

    resolved = resolver.resolve(trades)
    df = pd.DataFrame([r.__dict__ | {"expiry": r.expiry.isoformat()} for r in resolved])
    pd.set_option("display.max_columns", None)
    print(df)


if __name__ == "__main__":
    main()

