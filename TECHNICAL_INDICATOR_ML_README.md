# QuantEdge — Deliverable 2: Technical Indicator-Based ML

This add-on trains tabular ML models using causal technical-indicator features and evaluates trading signals on a chronological, purged holdout. It adds new files only; it does not overwrite Phase 1–12 modules or previous traditional-strategy reports.

## Models

- Random Forest (required; scikit-learn)
- XGBoost (used if installed)
- LightGBM (used if installed)
- HistGradientBoosting (scikit-learn comparator/fallback)

## Features

Returns (1/2/5/10/20/60 days), SMA ratios/spreads, RSI(14), MACD and signal/histogram, Bollinger width/%B, ATR percentage, rolling volatility, volume features, and optional NIFTYBEES benchmark returns/relative returns.

Target: whether the forward 5-trading-day close-to-close return is positive.

## Setup

From the QuantEdge root in PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m pip install -U pandas numpy scikit-learn
```

Optional model packages:

```powershell
python -m pip install xgboost lightgbm
```

## Run

```powershell
$env:PYTHONPATH="src"
python -m pytest tests/test_technical_ml.py -q
python scripts/run_technical_indicator_ml.py
```

The script expects the existing raw files under `data/raw`:
- RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, SBIN
- Optional benchmark: NIFTYBEES

Accepted file names: `{TICKER}_NS_1d.csv`, `{TICKER}.NS_1d.csv`, `{TICKER}_NS.csv`, `{TICKER}.NS.csv`.

## Outputs

`artifacts/technical_indicator_ml/`
- `model_test_metrics.csv`: untouched test classification metrics
- `strategy_test_metrics.csv`: per-stock test trading metrics
- `equal_weight_portfolio_test_metrics.csv`: cross-stock test comparison
- `predictions/`: test probabilities and labels
- `equity_curves/`: per-stock and portfolio test equity curves
- `assumptions.json`: split, target, costs and limitations

## Methodology

- Chronological 60% train / 20% validation / 20% test.
- Five-observation purge gap at train/validation and validation/test boundaries, matching the five-day forward label horizon.
- Median imputation is fitted only on training data within each model pipeline.
- Threshold is selected on validation Sharpe from a fixed grid and frozen before test evaluation.
- Probability is generated after close; position is applied from the next trading day.
- One-way transaction cost 0.10% and slippage 0.05%.

## Important interpretation notes

- Accuracy/AUC measure classification, not profitability.
- A model can have useful AUC and still lose money after costs.
- Do not select a model based on test results and then report that same test as unbiased performance.
- Equal-weight portfolio results average stock daily returns and do not include explicit periodic rebalancing costs.
- The five-day target labels overlap within each split; the boundary purge prevents label windows crossing between splits. This is a single holdout evaluation, not a guarantee of future returns.
- The feature builder is intentionally self-contained and transparent. It does not alter or depend on the existing Phase 10 feature pipeline, preventing accidental changes to prior artifacts.
