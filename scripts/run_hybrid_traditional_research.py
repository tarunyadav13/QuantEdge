from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantedge.strategies.hybrid_traditional import (  # noqa: E402
    backtest_hybrid_positions, generate_hybrid_signals,
)

TICKERS = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN"]
INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005
STRATEGIES = ["HYBRID_RSI_MACD", "HYBRID_SMA_RSI", "HYBRID_BB_RSI", "HYBRID_CONSENSUS"]


def find_file(ticker: str) -> Path:
    candidates = [
        ROOT / "data/raw" / f"{ticker}_NS_1d.csv",
        ROOT / "data/raw" / f"{ticker}.NS_1d.csv",
        ROOT / "data/raw" / f"{ticker}_NS.csv",
        ROOT / "data/raw" / f"{ticker}.NS.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No raw CSV found for {ticker}. Checked: " + ", ".join(map(str, candidates)))


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Flatten common yfinance multi-index-like column labels if serialized as strings.
    df.columns = [str(c).strip().replace("'", "").replace('"', "") for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime"}), None)
    if date_col is None:
        raise ValueError(f"{path.name}: could not identify Date column. Found {list(df.columns)}")
    close_col = next((c for c in df.columns if c.lower() == "adj close"), None)
    if close_col is None:
        close_col = next((c for c in df.columns if c.lower() == "close"), None)
    if close_col is None:
        raise ValueError(f"{path.name}: could not identify Close/Adj Close column.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[date_col, close_col]).sort_values(date_col)
    df = df.drop_duplicates(date_col, keep="last").set_index(date_col)
    return df[[close_col]].rename(columns={close_col: "Close"})


def main() -> None:
    output = ROOT / "artifacts/hybrid_traditional_research"
    (output / "equity_curves").mkdir(parents=True, exist_ok=True)
    (output / "signals").mkdir(parents=True, exist_ok=True)
    metric_rows = []
    curve_rows = []
    daily_rows = []

    for ticker in TICKERS:
        prices = load_prices(find_file(ticker))
        signals = generate_hybrid_signals(prices)
        signals.to_csv(output / "signals" / f"{ticker}_hybrid_signals.csv", index_label="date")
        close = prices["Close"]
        for strategy in STRATEGIES:
            curve, metrics = backtest_hybrid_positions(
                close, signals[strategy],
                initial_capital=INITIAL_CAPITAL,
                transaction_cost=TRANSACTION_COST,
                slippage=SLIPPAGE,
            )
            metric_rows.append({"ticker": ticker, "strategy": strategy, **metrics})
            temp = curve.reset_index().rename(columns={curve.index.name or "index": "date"})
            temp.insert(0, "ticker", ticker)
            temp.insert(1, "strategy", strategy)
            curve_rows.append(temp)
            daily = curve[["net_strategy_return", "held_position", "turnover"]].copy()
            daily["ticker"] = ticker
            daily["strategy"] = strategy
            daily_rows.append(daily.reset_index().rename(columns={curve.index.name or "index": "date"}))
            curve.to_csv(output / "equity_curves" / f"{ticker}_{strategy}.csv", index_label="date")
        print(f"{ticker}: {len(prices):,} rows | completed {len(STRATEGIES)} hybrid strategies")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output / "per_stock_strategy_metrics.csv", index=False)

    # Equal-weight daily portfolio: average the six stocks' net daily strategy returns.
    # This represents daily equal-weight exposure; no explicit periodic rebalancing
    # turnover charge is modelled, and that limitation is recorded in assumptions.
    portfolio_rows = []
    for strategy in STRATEGIES:
        parts = []
        for ticker in TICKERS:
            curve = pd.read_csv(output / "equity_curves" / f"{ticker}_{strategy}.csv", parse_dates=["date"])
            parts.append(curve.set_index("date")["net_strategy_return"].rename(ticker))
        returns = pd.concat(parts, axis=1).dropna()
        portfolio_return = returns.mean(axis=1)
        equity = INITIAL_CAPITAL * (1 + portfolio_return).cumprod()
        dd = equity / equity.cummax() - 1
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 252)
        std = portfolio_return.std(ddof=1)
        portfolio_rows.append({
            "strategy": strategy,
            "total_return": equity.iloc[-1] / INITIAL_CAPITAL - 1,
            "cagr": (equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1,
            "final_equity": equity.iloc[-1],
            "sharpe_ratio": portfolio_return.mean() / std * (252 ** 0.5) if std > 0 else 0.0,
            "max_drawdown": dd.min(),
            "observations": len(portfolio_return),
        })
        pd.DataFrame({"daily_return": portfolio_return, "equity": equity, "drawdown": dd}).to_csv(
            output / "equity_curves" / f"PORTFOLIO_{strategy}.csv", index_label="date"
        )

    portfolio_df = pd.DataFrame(portfolio_rows).sort_values("total_return", ascending=False)
    portfolio_df.to_csv(output / "equal_weight_portfolio_metrics.csv", index=False)
    assumptions = {
        "initial_capital_per_stock_comparison": INITIAL_CAPITAL,
        "transaction_cost_one_way": TRANSACTION_COST,
        "slippage_one_way": SLIPPAGE,
        "execution": "Close-derived signal applied from next trading day",
        "positioning": "Long-only; 0 or 1 exposure",
        "portfolio": "Daily equal-weight average of six stock net returns",
        "portfolio_rebalancing_costs": "Not explicitly modelled; portfolio is a daily equal-weight return approximation",
        "indicator_note": "Close-only ADX proxy is used for consensus; it is not full OHLC ADX",
        "strategies": STRATEGIES,
    }
    (output / "backtest_assumptions.json").write_text(json.dumps(assumptions, indent=2), encoding="utf-8")
    print("\n" + "=" * 76)
    print("EQUAL-WEIGHT HYBRID TRADITIONAL STRATEGY RESULTS")
    print("=" * 76)
    view = portfolio_df.copy()
    for col in ["total_return", "cagr", "max_drawdown"]:
        view[col] = view[col].map(lambda x: f"{x:+.2%}")
    view["final_equity"] = view["final_equity"].map(lambda x: f"₹{x:,.0f}")
    view["sharpe_ratio"] = view["sharpe_ratio"].map(lambda x: f"{x:.2f}")
    print(view.to_string(index=False))
    print(f"\nReports saved to: {output}")


if __name__ == "__main__":
    main()
