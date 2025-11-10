import json
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

DEFAULT_MODEL = os.getenv("OPTIONS_IDEA_MODEL", os.getenv("TRADE_IDEA_MODEL", "gpt-4o-mini"))
DEFAULT_TOP_N = int(os.getenv("OPTIONS_IDEA_TOP_N", "3"))
MIN_CONFIDENCE = float(os.getenv("OPTIONS_IDEA_MIN_CONFIDENCE", "0.2"))
SUMMARY_KEYS = ("model_metrics", "cv_scores", "train_shape", "test_shape")


def _load_summary(summary_path: Path) -> Dict[str, object]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Pipeline summary not found at {summary_path}")
    with summary_path.open() as fh:
        return json.load(fh)


def _select_candidates(analysis_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if analysis_df.empty:
        return analysis_df
    working = analysis_df.copy()
    if "prob_up_10d" not in working.columns:
        raise ValueError("analysis_df must contain 'prob_up_10d' column.")
    working["confidence"] = (working["prob_up_10d"] - 0.5).abs()
    working = working.sort_values("confidence", ascending=False)
    return working.head(top_n)


def _build_prompt(summary: Dict[str, object], candidates: pd.DataFrame) -> str:
    metrics = summary.get("model_metrics", {})
    cv_scores = summary.get("cv_scores", [])
    headline = f"""You are an experienced equity options strategist. 
The machine learning model predicts 10-day directional moves for earnings events."""
    metrics_text = json.dumps(metrics, indent=2)
    cv_text = ", ".join(f"{float(score):.3f}" for score in cv_scores) if cv_scores else "n/a"

    formatted_rows = "\n".join(
        "- "
        + json.dumps(
            {
                "symbol": row.get("symbol"),
                "event_date": str(row.get("event_date")),
                "prob_up_10d": round(float(row.get("prob_up_10d", 0.5)), 3),
                "confidence": round(float(row.get("confidence", 0.0)), 3),
                "pre_5d_return": row.get("pre_5d_return"),
                "pre_10d_return": row.get("pre_10d_return"),
                "pre_20d_return": row.get("pre_20d_return"),
                "pre_volatility": row.get("pre_volatility"),
                "surprise_percent": row.get("surprise_percent"),
                "has_upcoming_earnings": bool(row.get("has_upcoming_earnings", False)),
                "label": row.get("label"),
            },
            default=lambda x: None,
        )
        for row in candidates.to_dict(orient="records")
    )

    instructions = f"""
Model metrics:
{metrics_text}

Cross-validation ROC-AUC scores: {cv_text}

Provide **clean and concise options trade ideas** (calls or puts only) with a 14-28 day time horizon.
For each recommended trade:
- Specify the option type (call/put), approximate strike selection logic, and expiration week.
- Justify the trade using model probability, pre-event returns, macro context if relevant, and earnings timing.
- Include risk management guidance (position size cue, max loss, exit criteria).
- Present ideas in markdown with clear headings and bullet structure.
- Do not suggest spreads or multi-leg trades; single-leg calls or puts only.

Candidate signals:
{formatted_rows}
"""
    return headline + "\n" + instructions


def _call_openai(prompt: str, *, model_name: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert equity options strategist who writes concise trade plans.",
                },
                {"role": "user", "content": prompt},
            ],
        )
    except Exception:
        return None
    message = resp.choices[0].message.content
    return message.strip() if message else None


def _fallback_response(candidates: pd.DataFrame) -> str:
    if candidates.empty:
        return "No high-confidence signals available for options recommendations."
    lines = []
    for row in candidates.to_dict(orient="records"):
        prob = float(row.get("prob_up_10d", 0.5))
        direction = "Call" if prob >= 0.5 else "Put"
        confidence = abs(prob - 0.5)
        lines.append(
            f"- {direction} on {row.get('symbol')} (prob_up_10d={prob:.2f}, "
            f"confidence={confidence:.2f}); target expiration 3-4 weeks out."
        )
    return "LLM unavailable. Suggested single-leg ideas:\n" + "\n".join(lines)


def generate_options_ideas(
    *,
    summary_path: Path,
    analysis_df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    top_n: int = DEFAULT_TOP_N,
    min_confidence: float = MIN_CONFIDENCE,
) -> Dict[str, object]:
    summary = _load_summary(summary_path)
    if analysis_df.empty:
        return {"summary": summary, "ideas_markdown": "No analysis rows available.", "selected": []}

    working = analysis_df.copy()
    if "prob_up_10d" not in working.columns:
        raise ValueError("analysis_df must contain 'prob_up_10d' column.")

    working["confidence"] = (working["prob_up_10d"] - 0.5).abs()
    filtered = working[working["confidence"] >= min_confidence]
    if filtered.empty:
        filtered = working.sort_values("confidence", ascending=False).head(top_n)
    else:
        filtered = filtered.sort_values("confidence", ascending=False).head(top_n)

    prompt = _build_prompt(summary, filtered)
    llm_response = _call_openai(prompt, model_name=model_name)
    if not llm_response:
        llm_response = _fallback_response(filtered)

    return {
        "ideas_markdown": llm_response,
        "selected": filtered.to_dict(orient="records"),
        "prompt": prompt,
        "summary": {key: summary.get(key) for key in SUMMARY_KEYS if key in summary},
    }

