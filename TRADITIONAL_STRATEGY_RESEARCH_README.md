# QuantEdge — Traditional Strategy Research

## Strategies included
- SMA 9/21 crossover (the existing reference strategy)
- SMA 20/50 crossover
- SMA 50/200 crossover
- MACD 12/26/9 trend following
- RSI(14) mean reversion (enter below 30, exit above 55)
- Bollinger Bands(20, 2) mean reversion
- Donchian 20/10 breakout
- Buy-and-hold benchmark

## Methodology
- Signals use closing-price information and are executed from the next trading day.
- Long-only; target exposure is either 0% or 100%.
- One-way transaction cost defaults to 0.10%; one-way slippage defaults to 0.05%.
- ₹10,000 starting capital is used for each single-stock comparison.
- A cross-stock equal-weight portfolio is also reported, with ₹10,000 initial capital.
- Returns compound daily; Sharpe/Sortino and drawdown are calculated from daily returns.
- The script does not optimize strategy parameters on the reported period.

These are research assumptions, not a claim that the assumptions exactly match a broker's charges. Update transaction costs to reflect the actual execution venue and fee schedule before presenting results.

## Run
From the QuantEdge project root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_traditional_strategy_research.py
```

The script expects the existing raw data CSVs under `data/raw/`, with names such as:
- `RELIANCE_NS_1d.csv`
- `HDFCBANK_NS_1d.csv`
- `ICICIBANK_NS_1d.csv`
- `INFY_NS_1d.csv`
- `TCS_NS_1d.csv`
- `SBIN_NS_1d.csv`

The loader also accepts the corresponding `.NS_1d.csv` and `_NS.csv` variants.

## Outputs
`artifacts/traditional_strategy_research/`:
- `per_stock_strategy_metrics.csv`
- `equal_weight_portfolio_metrics.csv`
- `per_stock_equity_curves.csv`
- `per_stock_daily_returns.csv`
- `equal_weight_portfolio_equity_curves.csv`
- `backtest_assumptions.json`

## Important
This pack does not promise positive returns. The output must be reported as the measured result, including strategies that lose money. Do not choose a strategy based on the same period used to report its performance. For a stronger research result, use an earlier development/validation period to choose a strategy, freeze it, and report the later untouched test period separately.
