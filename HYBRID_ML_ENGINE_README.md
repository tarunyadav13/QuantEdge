# QuantEdge — Deliverable 3: Hybrid ML Trading Engine

## What this delivers
A pooled, date-split ML trading experiment that combines:
- A five-session directional classifier trained across the six configured stocks.
- Technical confirmation using trend (20/50 SMA), MACD histogram, and RSI regime.
- A five-session, non-overlapping execution rule to match the five-session prediction horizon.
- Validation-only model/threshold selection.
- Same-test-window comparison against an RSI recovery baseline and equal-weight buy-and-hold.

This is a research backtest, not a live trading system or investment recommendation.

## Input data
Put these files in `data/raw/`:
- `RELIANCE_NS_1d.csv`
- `HDFCBANK_NS_1d.csv`
- `ICICIBANK_NS_1d.csv`
- `INFY_NS_1d.csv`
- `TCS_NS_1d.csv`
- `SBIN_NS_1d.csv`
- Optional `NIFTYBEES_NS_1d.csv` (used only for benchmark-relative features).

CSV requires Date, Open, High, Low, Close, Volume. `Adj Close` can be used as Close if Close is absent.

## Install
From the QuantEdge project root in PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
pip install numpy pandas scikit-learn
```
If the project already has an activated virtual environment, use it. Do not replace your existing requirements file just for this deliverable.

## Run
```powershell
$env:PYTHONPATH="src"
python scripts/run_hybrid_ml_engine.py
```

## Output
Created under `artifacts/hybrid_ml_engine/`:
- `test_strategy_metrics.csv` — final test comparison.
- `validation_model_selection.csv` — validation metrics and selected threshold for each candidate.
- `test_predictions.csv` — per-stock out-of-sample probabilities and hybrid signals.
- `test_portfolio_daily_returns.csv` — daily equal-weight portfolio return streams.
- `run_assumptions.txt` — explicit assumptions and limitations.

## Methodology
1. Features use only information available through each session's close.
2. Target is whether Close[t+5] / Close[t] - 1 is positive.
3. All stocks share the same chronological date boundaries. The split is pooled by date, not by individual rows or random sampling.
4. Five trading sessions are purged at the train/validation and validation/test boundaries.
5. Median imputation and model fitting are learned on training data only.
6. Random Forest and HistGradientBoosting are candidate models. The validation-period strategy Sharpe selects the candidate and probability threshold.
7. Test data is used only after model/threshold selection.
8. The model probability must clear the selected threshold and the technical confirmation score must be at least 0.60.
9. A signal is formed after the close, entered next session, held for five sessions, and then exited. A new trade in the same stock is not opened while the prior trade is active.
10. Costs are modelled as 0.15% per side (0.10% transaction charges plus 0.05% slippage). This is a simplified flat-cost assumption.

## Important interpretation notes
- Compare only the three strategies in `test_strategy_metrics.csv`: they share the same out-of-sample dates and equal-weight basket.
- Buy-and-hold is the six-stock basket, not the NIFTY index. NIFTYBEES is an optional feature proxy.
- The basket averages per-stock daily returns equally. This simplified implementation does not model portfolio-level rebalance costs, taxes, market impact, partial fills, liquidity constraints, or corporate-action cash flows.
- Five-session forward labels overlap in the dataset; the boundary purge reduces leakage across splits, but observations within each split are statistically dependent.
- A positive historical return does not establish a robust edge. Review AUC, balanced accuracy, drawdown, trade count, and stability across market regimes before drawing conclusions.
- Do not tune against the test results. If results are weak, report them honestly rather than changing the test period or repeatedly adjusting parameters.
