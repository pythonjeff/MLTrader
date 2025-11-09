import json
import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


DEFAULT_MODEL = os.getenv("TRADE_IDEA_MODEL", "gpt-4o-mini")
DEFAULT_TOP_N = int(os.getenv("TRADE_IDEA_TOP_N", "5"))
MIN_CONFIDENCE = float(os.getenv("TRADE_IDEA_MIN_CONFIDENCE", "0.15"))


@dataclass
class TradeIdea:
    symbol: str
    thesis: str
    options_structure: str
    sizing_notes: str
    probability: float
    direction: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "probability": self.probability,
            "thesis": self.thesis,
            "options_structure": self.options_structure,
            "sizing_notes": self.sizing_notes,
        }


def _format_event_context(row: pd.Series) -> str:
    if hasattr(row, "_asdict"):
        data = row._asdict()
    elif isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)

    fields = {
        "symbol": data.get("symbol"),
        "event_date": str(data.get("event_date")),
        "prob_up_10d": round(float(data.get("prob_up_10d", 0.5)), 4),
        "confidence": round(float(data.get("confidence", 0.0)), 4),
        "has_upcoming_earnings": bool(data.get("has_upcoming_earnings", False)),
        "pre_5d_return": data.get("pre_5d_return"),
        "pre_10d_return": data.get("pre_10d_return"),
        "pre_20d_return": data.get("pre_20d_return"),
        "pre_volatility": data.get("pre_volatility"),
        "surprise_percent": data.get("surprise_percent"),
        "eps_direction": data.get("eps_direction"),
    }
    return json.dumps(fields, default=lambda x: None)


def _build_prompt(selected_rows: pd.DataFrame) -> str:
    formatted_rows = "\n".join(
        f"- {row.symbol}: { _format_event_context(row) }"
        for row in selected_rows.itertuples(index=False)
    )
    instructions = """You are a trading strategist generating actionable equity and options ideas.

Produce a markdown-formatted summary with two sections:
1. Directional Stock Trades — concise thesis, catalyst, and entry/exit guidance.
2. Options Structures — suggest spreads/straddles suited to the scenario, include rationale and risk controls.

For each idea include:
- Direction (long/short)
- Suggested probability of success (use provided probabilities as priors)
- Key drivers (macro, momentum, earnings)
- Position sizing or risk notes (max loss, hedge suggestions)

Focus on the practical trade setup, not generic commentary."""
    return f"{instructions}\n\nCandidates:\n{formatted_rows}"


def _call_openai(prompt: str, *, model: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.4,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert equity derivatives strategist focusing on earnings-related trades.",
                },
                {"role": "user", "content": prompt},
            ],
        )
    except Exception:
        return None
    message = response.choices[0].message.content
    return message.strip() if message else None


def _fallback_summary(selected_rows: pd.DataFrame) -> str:
    bullets = []
    for row in selected_rows.itertuples(index=False):
        prob = getattr(row, "prob_up_10d", 0.5)
        confidence = getattr(row, "confidence", 0.0)
        has_earnings = getattr(row, "has_upcoming_earnings", getattr(row, "has_upcoming", False))
        direction = "Long" if prob >= 0.5 else "Short"
        bullets.append(
            f"- {direction} {row.symbol}: prob_up_10d={prob:.2f}, "
            f"confidence={confidence:.2f}, has_upcoming_earnings={has_earnings}"
        )
    return "LLM not available; top signals:\n" + "\n".join(bullets)


def generate_trade_ideas(
    analysis_df: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
    min_confidence: float = MIN_CONFIDENCE,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    if analysis_df.empty:
        return {"summary": "No candidates available.", "selected": []}

    working = analysis_df.copy()
    working["confidence"] = (working["prob_up_10d"] - 0.5).abs()
    filtered = working[working["confidence"] >= min_confidence]
    if filtered.empty:
        filtered = working.nlargest(top_n, "confidence")

    long_candidates = filtered.sort_values("prob_up_10d", ascending=False).head(top_n)
    short_candidates = filtered.sort_values("prob_up_10d", ascending=True).head(top_n)
    selected = pd.concat([long_candidates, short_candidates]).drop_duplicates(subset=["symbol"])
    selected = selected.sort_values("confidence", ascending=False).head(top_n)

    prompt = _build_prompt(selected)
    llm_summary = _call_openai(prompt, model=model_name)
    if not llm_summary:
        llm_summary = _fallback_summary(selected)

    ideas: List[TradeIdea] = []
    for row in selected.itertuples(index=False):
        direction = "long" if row.prob_up_10d >= 0.5 else "short"
        ideas.append(
            TradeIdea(
                symbol=row.symbol,
                probability=float(row.prob_up_10d),
                direction=direction,
                thesis="Generated by LLM",
                options_structure="See summary",
                sizing_notes="See summary",
            )
        )

    return {
        "summary": llm_summary,
        "selected": [idea.to_dict() for idea in ideas],
        "prompt": prompt,
    }

