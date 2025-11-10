from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Iterable, List

import pandas as pd

from portfolio.options_risk import OptionOrder

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import OptionOrderRequest
    from alpaca.trading.enums import OrderType, OrderSide, TimeInForce
except ImportError:  # pragma: no cover - optional dependency
    TradingClient = None
    OptionOrderRequest = None
    OrderType = None
    OrderSide = None
    TimeInForce = None


@dataclass(slots=True)
class OrderSubmissionResult:
    option_symbol: str
    quantity: int
    side: str
    limit_price: float
    status: str
    broker_response: dict

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


class AlpacaOptionsTrader:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        dry_run: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        self.api_secret = api_secret or os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL") or os.getenv("APCA_API_BASE_URL") or os.getenv("APCA_API_PAPER_URL")
        paper_flag = True
        if base_url:
            paper_flag = "paper" in base_url.lower()
        else:
            env_mode = os.getenv("ALPACA_ENV", "paper").lower()
            paper_flag = env_mode != "live"
        if dry_run is None:
            dry_run = os.getenv("ALPACA_DRY_RUN", "true").lower() in {"1", "true", "yes"}
        self.dry_run = dry_run

        if not self.dry_run:
            if TradingClient is None:
                raise ImportError("alpaca-py must be installed to submit orders.")
            if not self.api_key or not self.api_secret:
                raise ValueError("Alpaca API credentials not provided.")
            self.trading_client = TradingClient(
                self.api_key, self.api_secret, paper=paper_flag
            )
        else:
            self.trading_client = None

    def _build_order_request(self, order: OptionOrder) -> OptionOrderRequest:
        if OptionOrderRequest is None:
            raise ImportError("alpaca-py trading requests unavailable.")
        side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL
        request = OptionOrderRequest(
            symbol=order.option_symbol,
            qty=order.quantity,
            side=side,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=order.limit_price,
        )
        return request

    def submit_order(self, order: OptionOrder) -> OrderSubmissionResult:
        if self.dry_run:
            return OrderSubmissionResult(
                option_symbol=order.option_symbol,
                quantity=order.quantity,
                side=order.side,
                limit_price=order.limit_price,
                status="dry_run",
                broker_response={"message": "dry run, order not sent"},
            )

        request = self._build_order_request(order)
        response = self.trading_client.submit_order(request)  # type: ignore[attr-defined]
        return OrderSubmissionResult(
            option_symbol=order.option_symbol,
            quantity=order.quantity,
            side=order.side,
            limit_price=order.limit_price,
            status=getattr(response, "status", "submitted"),
            broker_response=response.__dict__,
        )

    def submit_orders(self, orders: Iterable[OptionOrder]) -> List[OrderSubmissionResult]:
        results: List[OrderSubmissionResult] = []
        for order in orders:
            result = self.submit_order(order)
            results.append(result)
        return results


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Submit option orders via Alpaca.")
    parser.add_argument(
        "--orders",
        type=str,
        required=True,
        help="Path to orders JSON/Parquet file produced by the pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run mode even if environment enables live submission.",
    )
    args = parser.parse_args()

    path = args.orders
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)  # type: ignore[name-defined]
    else:
        df = pd.read_json(path)
    orders = []
    for record in df.to_dict(orient="records"):
        orders.append(
            OptionOrder(
                underlying_symbol=record["underlying_symbol"],
                option_symbol=record["option_symbol"],
                side=record["side"],
                quantity=int(record["quantity"]),
                limit_price=float(record["limit_price"]),
                expiry=pd.to_datetime(record["expiry"]),  # type: ignore[name-defined]
                strike=float(record["strike"]),
                confidence=float(record["confidence"]),
                probability=float(record["probability"]),
                notional=float(record["notional"]),
                risk_amount=float(record["risk_amount"]),
                notes=record.get("notes", {}),
            )
        )

    trader = AlpacaOptionsTrader(dry_run=args.dry_run)
    results = trader.submit_orders(orders)
    print(json.dumps([r.to_dict() for r in results], indent=2))


if __name__ == "__main__":
    main()

