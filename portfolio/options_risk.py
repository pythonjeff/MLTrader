from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from services.options_quote import ResolvedOptionContract


CONTRACT_MULTIPLIER = int(os.getenv("OPTIONS_CONTRACT_MULTIPLIER", "100"))


@dataclass(slots=True)
class OptionOrder:
    """Executable option order with risk metadata."""

    underlying_symbol: str
    option_symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    limit_price: float
    expiry: pd.Timestamp
    strike: float
    confidence: float
    probability: float
    notional: float
    risk_amount: float
    notes: Dict[str, object]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["expiry"] = self.expiry.isoformat()
        return payload


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _round_price(price: float) -> float:
    return round(max(price, 0.01), 2)


def generate_option_orders(
    contracts: Sequence[ResolvedOptionContract],
    *,
    account_equity: float | None = None,
    max_risk_bps: float | None = None,
    max_premium: float | None = None,
    markup_pct: float | None = None,
    min_quantity: int | None = None,
    max_quantity: int | None = None,
) -> List[OptionOrder]:
    """Turn resolved option contracts into sized limit orders.

    Parameters are sourced from environment variables when not provided:
    - OPTIONS_ACCOUNT_EQUITY (default 100000)
    - OPTIONS_MAX_RISK_BPS (default 50 -> 0.5%)
    - OPTIONS_MAX_PREMIUM (default 2500 dollars)
    - OPTIONS_LIMIT_MARKUP_PCT (default 0.1 -> +10% for buys)
    - OPTIONS_MIN_QUANTITY (default 1)
    - OPTIONS_MAX_QUANTITY (default 10)
    """
    account_equity = account_equity or _env_float("OPTIONS_ACCOUNT_EQUITY", 100_000.0)
    max_risk_bps = max_risk_bps or _env_float("OPTIONS_MAX_RISK_BPS", 50.0)
    max_premium = max_premium or _env_float("OPTIONS_MAX_PREMIUM", 2_500.0)
    markup_pct = markup_pct if markup_pct is not None else _env_float("OPTIONS_LIMIT_MARKUP_PCT", 0.10)
    min_quantity = min_quantity or int(os.getenv("OPTIONS_MIN_QUANTITY", "1"))
    max_quantity = max_quantity or int(os.getenv("OPTIONS_MAX_QUANTITY", "10"))

    per_trade_risk = account_equity * (max_risk_bps / 10_000.0)
    cap_per_trade = min(per_trade_risk, max_premium)

    orders: List[OptionOrder] = []
    for contract in contracts:
        if contract.option_symbol == "unavailable":
            continue
        est_price = contract.estimated_price
        if est_price <= 0:
            continue

        premium_per_contract = est_price * CONTRACT_MULTIPLIER
        if premium_per_contract <= 0:
            continue

        quantity = int(cap_per_trade // premium_per_contract)
        quantity = max(quantity, min_quantity)
        quantity = min(quantity, max_quantity)
        if quantity <= 0:
            continue

        notional = premium_per_contract * quantity
        if notional > cap_per_trade * 1.5:
            # guardrail if estimated price drifted since sizing
            quantity = max(int(cap_per_trade // premium_per_contract), 0)
            if quantity == 0:
                continue
            notional = premium_per_contract * quantity

        limit_price = est_price * (1 + markup_pct if contract.action.lower() == "buy" else 1 - markup_pct)
        limit_price = _round_price(limit_price)

        notes = dict(contract.notes)
        notes.update(
            {
                "estimated_price": est_price,
                "premium_per_contract": premium_per_contract,
                "cap_per_trade": cap_per_trade,
            }
        )

        orders.append(
            OptionOrder(
                underlying_symbol=contract.symbol,
                option_symbol=contract.option_symbol,
                side=contract.action,
                quantity=quantity,
                limit_price=limit_price,
                expiry=contract.expiry,
                strike=contract.strike,
                confidence=contract.confidence,
                probability=contract.probability,
                notional=notional,
                risk_amount=min(notional, cap_per_trade),
                notes=notes,
            )
        )

    return orders


def orders_to_dataframe(orders: Iterable[OptionOrder]) -> pd.DataFrame:
    records = [order.to_dict() for order in orders]
    if not records:
        return pd.DataFrame(
            columns=[
                "underlying_symbol",
                "option_symbol",
                "side",
                "quantity",
                "limit_price",
                "expiry",
                "strike",
                "confidence",
                "probability",
                "notional",
                "risk_amount",
                "notes",
            ]
        )
    df = pd.DataFrame.from_records(records)
    return df

