# AI Hedge Fund Pipeline Overview

This repository contains an end-to-end earnings-trading research stack that ingests capital-markets data, builds engineered features, trains a directional signal model, and produces long/short options ideas with an LLM overlay. The sections below describe each layer, the data artifacts produced, and how the pieces fit together.

---

## 1. Data Ingestion Layer

| Module | Purpose | Key Outputs |
|--------|---------|-------------|
| `data/sp500_universe.py` | Pulls current S&P 500 constituents from Financial Modeling Prep (FMP). | `data/processed/sp500_constituents.parquet` |
| `data/earnings_calendar.py` | Fetches the next 14 days of earnings (Finnhub) and tags the S&P universe with a `has_upcoming_earnings` boolean. | `data/processed/earnings_next_week.parquet`, `data/processed/sp500_with_earnings_flag.parquet` |
| `data/historical_earnings.py` | Downloads historical EPS events per ticker from FMP, computes surprises/YOY/QOQ deltas, and enforces a lookback horizon. | Per-symbol cache: `data/raw/historical_earnings_<lookback>/*.parquet`<br>Combined tables: `data/processed/historical_earnings.parquet`, `historical_earnings_filtered.parquet` |
| `data/price_reaction.py` | Fetches and caches daily bars (FMP) around each earnings event, then computes pre/post returns, volatility, and the 10-day directional label. | Per-symbol cache: `data/raw/price_reaction/*.parquet`<br>Aggregated reactions: `data/processed/price_reaction.parquet` |
| `data/macro_data.py` | Pulls macroeconomic series from FRED (Fed Funds, CPI, 10Y Treasury, SP500, VIX), engineers slopes/spreads, and aligns the latest snapshot to each event. | Raw macro table: `data/raw/macro_context_raw.parquet`<br>Aligned features: `data/processed/macro_features.parquet` |

Each ingestion module can be run standalone (e.g., `python -m data.sp500_universe`). Most scripts are idempotent and cache aggressively so subsequent runs reuse previously downloaded data.

---

## 2. Feature Merging

`data/merge_datasets.py` combines all processed sources into a single machine-learning table:

1. Loads earnings fundamentals (`historical_earnings_filtered.parquet`), price reactions (`price_reaction.parquet`), and macro features (`macro_features.parquet`).
2. Filters to historical events only (no future leakage).
3. Deduplicates on (`symbol`, `event_date`).
4. Keeps rows with a valid 10-day direction label.
5. Outputs `data/processed/ml_merged.parquet`, sorted by event date.

Typical columns include:
- Price context: `pre_5d_return`, `pre_10d_return`, `pre_volatility`.
- Earnings fundamentals: `surprise_percent`, `eps_direction`.
- Macro snapshot: Fed Funds rate, CPI trends, yield spread, SPX momentum, VIX z-scores.
- Target: `label` (1 if 10-day return > 0).

---

## 3. Preprocessing & Train/Test Split

`data/preprocess_ml.py` handles feature selection and normalization:

1. Loads `ml_merged.parquet`.
2. Drops leakage columns (future returns, raw EPS fields, etc.).
3. Performs a chronological split (default 80/20) to avoid look-ahead bias.
4. Median-imputes numeric fields and applies `StandardScaler`; no categorical fields remain after filtration.
5. Persisted artifacts:
   - `data/processed/X_train.parquet`, `X_test.parquet`
   - `data/processed/y_train.parquet`, `y_test.parquet`
   - `data/processed/preprocess_artifacts.joblib` (imputer/scaler metadata)
   - Optional `trade_analysis.parquet` downstream for inference diagnostics.

Running:
```bash
python -m data.preprocess_ml
```

Returns Pandas DataFrames ready for modeling and ensures the same transformations can be replayed in live inference.

---

## 4. Signal Model

`ml_models/signal_model.py` trains a logistic regression classifier on the 10-day direction label:

- Model: `LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")`.
- TimeSeriesSplit cross-validation (default 5 folds) scored by ROC-AUC.
- Evaluation metrics include accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and a classification report.
- Artifacts:
  - `models/signal_model.joblib` (fitted estimator + feature order)
  - `data/processed/signal_predictions.parquet` (test predictions with probabilities)
  - `models/signal_training_summary.joblib` (metrics, CV scores, paths)

Execute:
```bash
python -m ml_models.signal_model
```

This stage assumes the preprocessing outputs are already written to `data/processed`.

---

## 5. LLM Trade-Idea Layer

Two lightweight services generate narrative trade insights:

### `services/trade_ideas.py`
- Consumes the merged test set joined with model predictions (`trade_analysis.parquet`).
- Ranks symbols by confidence (`abs(prob_up_10d - 0.5)`), selects long and short candidates.
- Calls OpenAI (via `OPENAI_API_KEY`) to draft earnings-oriented stock/option strategies.
- Fallback prints top signals if the LLM is unavailable.

### `services/options_ideas.py`
- Reads `models/latest_pipeline_summary.json` plus the candidate table.
- Summarizes model metrics and produces single-leg call/put ideas targeting a 14–28 day horizon.
- Includes strike selection rationale, expiration guidance, and risk notes.
- Fallback generates bullet recommendations when no API access is present.

Both modules rely on the general configuration variable `TRADE_IDEA_MODEL` (default `gpt-4o-mini`) and respect confidence thresholds.

### Option Trade Construction & Execution Modules

- `services/trade_builder.py` transforms the model analysis dataframe into structured `OptionTrade` objects with direction, strike bias, expiry target, and confidence metadata.
- `services/options_quote.py` resolves those trades into concrete option contracts. Two resolvers exist:
  - `MockOptionContractResolver` for offline validation.
  - `AlpacaOptionContractResolver` which calls Alpaca’s market data API. Example:
    ```bash
    export ALPACA_API_KEY=...
    export ALPACA_SECRET_KEY=...
    python -m services.options_quote --max-trades 5
    ```
    Append `--mock` to skip live API calls.
- `portfolio/options_risk.py` sizes contracts into limit orders using configurable risk constraints (per-trade risk bps, premium caps, limit markup).
- `trading/alpaca_broker.py` submits the sized orders to Alpaca. Defaults to dry-run:
  ```bash
  python -m trading.alpaca_broker --orders data/processed/options_orders.parquet --dry-run
  ```
  It respects either `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` or Alpaca’s standard `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` environment variables. Set `ALPACA_ENV=live` or point `ALPACA_BASE_URL` / `APCA_API_BASE_URL` at the live endpoint when you’re ready to switch out of paper mode.

---

## 6. Pipeline Orchestration

`main.py` stitches everything together:

1. Refresh S&P universe and upcoming earnings.
2. Rebuild historical earnings (skip cached symbols).
3. Compute price reactions with cached price bars.
4. Pull macro context and align to events.
5. Merge data, preprocess, and store train/test splits.
6. Train the signal model and save evaluation artifacts.
7. Save `data/processed/trade_analysis.parquet` (test context + predictions).
8. Generate narrative trade ideas and options ideas, embedding both in:
   - `models/latest_pipeline_summary.json`
   - `models/latest_pipeline_summary.joblib`
9. If `OPTIONS_BUILD_TRADES=true`, build option trades, resolve contracts (mock or Alpaca), size orders, and write them to `data/processed/options_orders.{parquet,json}`. If `OPTIONS_SUBMIT_ORDERS=true`, submit via Alpaca (respects `OPTIONS_DRY_RUN`, default true).
10. Print the options-layer markdown summary to stdout.

Run the entire workflow:
```bash
python main.py
```

All API keys (FMP, Finnhub, FRED, OpenAI) should be available as environment variables or in a readable `.env`. Helper `utils/env.py` provides `safe_load_env` to load `.env` without crashing if permissions are restricted.

---

## 7. Environment & Utilities

- Python dependencies are listed in `requirements.txt`; install via:
  ```bash
  python -m pip install -r requirements.txt
  ```
- `agents/test_openai.py` offers a quick connectivity check for the OpenAI client.
- Cached data lives under `data/raw/` to keep downloads incremental.
- Logs and model summaries land under `logs/` and `models/` respectively.

---

## 8. Typical Runbook

1. **Setup**
   ```bash
   python -m pip install -r requirements.txt
   set -a; source .env; set +a
   ```
2. **Incremental data refresh (optional)**
   ```bash
   python -m data.sp500_universe
   python -m data.earnings_calendar
   python -m data.historical_earnings
   python -m data.price_reaction
   python -m data.macro_data
   ```
3. **Model pipeline**
   ```bash
   python -m data.merge_datasets
   python -m data.preprocess_ml
   python -m ml_models.signal_model
   # Option trade flow (optional)
   python -m services.options_quote --mock --max-trades 3
   python -m trading.alpaca_broker --orders data/processed/options_orders.parquet --dry-run
   ```
4. **Full automation**
   ```bash
   python main.py
   ```
5. **Inspect outputs**
   - Quant metrics: `models/latest_pipeline_summary.json`
   - LLM summaries: printed at the end of `main.py`
   - Test predictions: `data/processed/signal_predictions.parquet`
   - Trade context: `data/processed/trade_analysis.parquet`
   - Structured option orders: `data/processed/options_orders.{parquet,json}`

---

## 9. Extensibility Notes

- **Additional Features**: Augment `merge_datasets.py` with new sources (e.g., sentiment, alternative data) once they produce parquet files keyed by `symbol` and `event_date`.
- **Different Models**: Replace or extend `signal_model.py` with gradient boosting, neural nets, or multi-task learners—ensure feature alignment via `preprocess_artifacts.joblib`.
- **Execution Layer**: The repository includes scaffolding for trade agents (`agents/`, `trading/`, `portfolio/`); integrate live signals after validating backtests.
- **Scheduling**: Cron or Airflow can call `main.py` daily once API keys are secured in the runtime environment.
- **Option Execution Configuration**:
  - `OPTIONS_BUILD_TRADES` — enable order construction in `main.py`.
  - `OPTIONS_MIN_CONFIDENCE`, `OPTIONS_MAX_TRADES` — filter trade candidates.
  - `OPTIONS_RESOLVER` — choose `alpaca` (default) or `mock`.
  - `OPTIONS_ACCOUNT_EQUITY`, `OPTIONS_MAX_RISK_BPS`, `OPTIONS_MAX_PREMIUM` — risk sizing parameters.
  - `OPTIONS_SUBMIT_ORDERS` — submit orders after sizing.
  - `OPTIONS_DRY_RUN` — keep submissions simulated (default `true`).
  - `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL` — required for live resolution/submission.

---

This documentation should help new contributors and reviewers understand how raw market data flows through preprocessing, machine learning, and LLM-based recommendation layers. For questions or enhancements, open an issue or contact the maintainer.*** End Patch

