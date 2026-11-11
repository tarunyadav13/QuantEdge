# QuantEdge — Hybrid Traditional Strategy Research

This package adds four hybrid rule-based strategies without changing existing QuantEdge files.

## Included strategies

1. `HYBRID_RSI_MACD`: RSI oversold recovery confirmed by bullish MACD.
2. `HYBRID_SMA_RSI`: SMA 50 above SMA 200, with RSI recovery from below 40.
3. `HYBRID_BB_RSI`: Close re-enters above the lower Bollinger Band while RSI recovers from oversold.
4. `HYBRID_CONSENSUS`: score-based combination of trend, RSI recovery, MACD, Bollinger re-entry and a close-only directional-movement proxy.

All strategies are long-only. Signals are generated from closing prices and positions are applied from the next trading day.

## Install / run

From the QuantEdge project root in PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m pytest tests/test_hybrid_traditional.py -q
python scripts/run_hybrid_traditional_research.py
```

Input CSVs are read from `data/raw`. Accepted names:
- `{TICKER}_NS_1d.csv`
- `{TICKER}.NS_1d.csv`
- `{TICKER}_NS.csv`
- `{TICKER}.NS.csv`

Tickers: RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, SBIN.

## Outputs

Written to `artifacts/hybrid_traditional_research/`:
- `per_stock_strategy_metrics.csv`
- `equal_weight_portfolio_metrics.csv`
- `signals/{TICKER}_hybrid_signals.csv`
- `equity_curves/{TICKER}_{STRATEGY}.csv`
- `equity_curves/PORTFOLIO_{STRATEGY}.csv`
- `backtest_assumptions.json`

## Assumptions and limitations

- Initial capital for each individual stock comparison: ₹10,000.
- Transaction cost: 0.10% one-way; slippage: 0.05% one-way.
- Long-only positions; no leverage or short selling.
- No same-close execution: position is shifted by one trading day.
- The portfolio is a daily equal-weight average of stock returns. Explicit portfolio rebalancing turnover costs are not included.
- The ADX-like component is a close-only proxy, not a proper OHLC ADX calculation.
- No parameter optimization is performed. These are fixed, transparent starting rules.
- These are historical backtests, not evidence of future profitability. Evaluate on a chronological untouched test period before making performance claims.
