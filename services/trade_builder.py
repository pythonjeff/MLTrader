from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Sequence

import pandas as pd


@dataclass(slots=True)
class OptionTrade:
    """Structured representation of a candidate single-leg option trade."""

    symbol: str
    option_type: str  # "call" or "put"
    action: str  # "buy" or "sell"
    expiry_target_days: int  # target days until expiration (approximate)
    strike_bias: str  # e.g. "atm", "5pct_otm", "delta_0.35"
    confidence: float
    probability: float
    event_date: str
    has_upcoming_earnings: bool
    rationale: str = ""
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["notes"] = dict(self.notes) if self.notes else {}
        return payload


@dataclass(slots=True)
class TradePlan:
    """Collection of option trades plus metadata about the selection process."""

    trades: List[OptionTrade]
    selection_notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trades": [trade.to_dict() for trade in self.trades],
            "selection_notes": dict(self.selection_notes),
        }

    def __iter__(self):
        return iter(self.trades)


def _ensure_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "prob_up_10d" not in df.columns:
        raise ValueError("analysis_df must contain a 'prob_up_10d' column.")
    if "confidence" not in df.columns:
        df["confidence"] = (df["prob_up_10d"] - 0.5).abs()
    return df


def _pick_option_type(prob_up: float, bullish_threshold: float) -> str:
    return "call" if prob_up >= bullish_threshold else "put"


def _strike_bias(option_type: str, confidence: float) -> str:
    """Derive a simple strike bias heuristic based on conviction level."""
    if confidence >= 0.25:
        return "5pct_otm" if option_type == "call" else "5pct_otm_down"
    if confidence >= 0.15:
        return "atm"
    return "0.10_otm"


def generate_option_trades(
    analysis_df: pd.DataFrame,
    *,
    min_confidence: float = 0.15,
    bullish_threshold: float = 0.55,
    max_trades: int = 5,
    default_expiry_days: int = 21,
    rationale_source: str = "model_confidence",
    include_columns: Sequence[str] = ("pre_10d_return", "pre_20d_return", "pre_volatility"),
) -> TradePlan:
    """Generate structured option trades from the analysis dataframe.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Must include columns: symbol, prob_up_10d, event_date, has_upcoming_earnings (optional).
    min_confidence : float
        Minimum absolute deviation from 0.5 probability to consider a trade.
    bullish_threshold : float
        Probability threshold above which we favour call structures.
    max_trades : int
        Maximum number of trades to return.
    default_expiry_days : int
        Target expiry horizon in days (within the LLM guidance band of 14-28).
    rationale_source : str
        String describing where the rationale comes from (logged in notes).
    include_columns : Sequence[str]
        Additional columns from analysis_df to embed in trade notes if present.
    """
    working = _ensure_confidence(analysis_df)
    working = working.sort_values("confidence", ascending=False)
    filtered = working[working["confidence"] >= min_confidence].head(max_trades)

    trades: List[OptionTrade] = []
    for row in filtered.itertuples(index=False):
        prob_up = float(getattr(row, "prob_up_10d"))
        confidence = float(getattr(row, "confidence"))
        option_type = _pick_option_type(prob_up, bullish_threshold)
        strike_hint = _strike_bias(option_type, confidence)

        notes = {"rationale_source": rationale_source}
        for col in include_columns:
            if hasattr(row, col):
                notes[col] = getattr(row, col)

        trades.append(
            OptionTrade(
                symbol=getattr(row, "symbol"),
                option_type=option_type,
                action="buy",
                expiry_target_days=default_expiry_days,
                strike_bias=strike_hint,
                confidence=confidence,
                probability=prob_up,
                event_date=str(getattr(row, "event_date")),
                has_upcoming_earnings=bool(getattr(row, "has_upcoming_earnings", False)),
                notes=notes,
            )
        )

    selection_notes = {
        "min_confidence": min_confidence,
        "bullish_threshold": bullish_threshold,
        "max_trades": max_trades,
        "target_expiry_days": default_expiry_days,
        "num_candidates": len(trades),
    }
    return TradePlan(trades=trades, selection_notes=selection_notes)


def trades_to_dataframe(trades: Iterable[OptionTrade]) -> pd.DataFrame:
    """Convert a collection of OptionTrade objects into a pandas DataFrame."""
    records = [trade.to_dict() for trade in trades]
    if not records:
        return pd.DataFrame(columns=OptionTrade.__annotations__.keys())
    df = pd.DataFrame.from_records(records)
    return df

