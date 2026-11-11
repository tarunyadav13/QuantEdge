"""Run a consistent traditional-strategy comparison on QuantEdge raw CSVs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantedge.strategies.traditional import generate_signals, backtest_positions


TICKERS = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN"]
INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.001       # 0.10% each time position changes
SLIPPAGE = 0.0005              # 0.05% each time position changes
PERIODS_PER_YEAR = 252

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "artifacts" / "traditional_strategy_research"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def locate_file(ticker: str) -> Path:
    candidates = [
        RAW_DIR / f"{ticker}_NS_1d.csv",
        RAW_DIR / f"{ticker}.NS_1d.csv",
        RAW_DIR / f"{ticker}_NS.csv",
        RAW_DIR / f"{ticker}.NS.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No raw data for {ticker}. Expected one of: "
        + ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in candidates)
    )


def load_data(ticker: str) -> pd.DataFrame:
    path = locate_file(ticker)
    df = pd.read_csv(path)

    date_col = next((c for c in df.columns if str(c).lower() in ("date", "datetime", "timestamp")), None)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
    else:
        first = df.columns[0]
        parsed = pd.to_datetime(df[first], errors="coerce")
        if parsed.notna().mean() < 0.9:
            raise ValueError(f"Cannot identify date column in {path}")
        df[first] = parsed
        df = df.dropna(subset=[first]).set_index(first)

    df.index = pd.DatetimeIndex(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.DatetimeIndex(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Normalize yfinance-style column names.
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in ("adj_close", "adjclose"):
            rename[col] = "Adj Close"
        elif key in ("close", "open", "high", "low", "volume"):
            rename[col] = key.title()
    df = df.rename(columns=rename)

    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if close_col not in df.columns:
        raise ValueError(f"No Close/Adj Close column in {path}")

    # Use adjusted close consistently for return calculations and indicators.
    if close_col == "Adj Close":
        factor = pd.to_numeric(df["Adj Close"], errors="coerce") / pd.to_numeric(df["Close"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
        for col in ("Open", "High", "Low"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * factor

    required = ["Close"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df


def main():
    print("=" * 76)
    print("QUANTEDGE | TRADITIONAL STRATEGY RESEARCH")
    print("=" * 76)
    print(f"Initial capital per comparison: ₹{INITIAL_CAPITAL:,.0f}")
    print(f"One-way transaction cost: {TRANSACTION_COST:.2%}")
    print(f"One-way slippage assumption: {SLIPPAGE:.2%}")
    print("Execution: close signal applied from next trading day.")

    per_stock_rows = []
    all_curves = []
    all_daily = []
    loaded = {}

    for ticker in TICKERS:
        try:
            data = load_data(ticker)
            if len(data) < 220:
                print(f"Skipping {ticker}: only {len(data)} rows; need at least 220.")
                continue
            loaded[ticker] = data
            signals = generate_signals(data)

            print(f"\n{ticker}: {len(data)} observations | {data.index.min().date()} to {data.index.max().date()}")

            for strategy_name, signal in signals.items():
                curve, metrics = backtest_positions(
                    data["Close"], signal,
                    initial_capital=INITIAL_CAPITAL,
                    transaction_cost=TRANSACTION_COST,
                    slippage=SLIPPAGE,
                    periods_per_year=PERIODS_PER_YEAR,
                )
                row = {"ticker": ticker, "strategy": strategy_name, **metrics}
                per_stock_rows.append(row)
                curve = curve.reset_index(names="date")
                curve.insert(1, "ticker", ticker)
                curve.insert(2, "strategy", strategy_name)
                all_curves.append(curve)
                daily = curve[["date", "ticker", "strategy", "strategy_return"]].copy()
                all_daily.append(daily)
                print(f"  {strategy_name:18s} return={metrics['total_return']:+7.2%} "
                      f"CAGR={metrics['CAGR']:+7.2%} Sharpe={metrics['sharpe_ratio']:6.2f} "
                      f"MaxDD={metrics['max_drawdown']:7.2%}")

            # Buy-and-hold baseline for each stock.
            buy_hold_signal = pd.Series(1.0, index=data.index)
            curve, metrics = backtest_positions(
                data["Close"], buy_hold_signal,
                initial_capital=INITIAL_CAPITAL,
                transaction_cost=TRANSACTION_COST,
                slippage=SLIPPAGE,
            )
            per_stock_rows.append({"ticker": ticker, "strategy": "BUY_AND_HOLD", **metrics})
            curve = curve.reset_index(names="date")
            curve.insert(1, "ticker", ticker)
            curve.insert(2, "strategy", "BUY_AND_HOLD")
            all_curves.append(curve)
            all_daily.append(curve[["date", "ticker", "strategy", "strategy_return"]].copy())

        except Exception as exc:
            print(f"WARNING: {ticker} not evaluated: {exc}")

    if not loaded:
        raise RuntimeError(
            f"No usable stock datasets found in {RAW_DIR}. "
            "Check that the six raw CSV files are present."
        )

    metrics_df = pd.DataFrame(per_stock_rows)
    curves_df = pd.concat(all_curves, ignore_index=True)
    daily_df = pd.concat(all_daily, ignore_index=True)

    # Cross-stock equal-weight portfolio: each stock contributes 1/N of capital;
    # cash in an individual strategy remains cash and earns zero in this study.
    aggregate_rows = []
    aggregate_equity_frames = []
    for strategy in sorted(daily_df["strategy"].unique()):
        subset = daily_df[daily_df["strategy"] == strategy]
        wide = subset.pivot(index="date", columns="ticker", values="strategy_return")
        # Only dates where every included stock has a return are used.
        wide = wide.dropna(how="any")
        if wide.empty:
            continue
        portfolio_return = wide.mean(axis=1)
        equity = INITIAL_CAPITAL * (1 + portfolio_return).cumprod()
        from quantedge.strategies.traditional import calculate_metrics
        m = calculate_metrics(portfolio_return, equity, INITIAL_CAPITAL)
        aggregate_rows.append({"ticker": "EQUAL_WEIGHT_6_STOCKS", "strategy": strategy, **m})
        aggregate_equity_frames.append(pd.DataFrame({
            "date": portfolio_return.index,
            "strategy": strategy,
            "portfolio_return": portfolio_return.values,
            "portfolio_equity": equity.values,
            "number_of_stocks": wide.shape[1],
        }))

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_curve_df = pd.concat(aggregate_equity_frames, ignore_index=True) if aggregate_equity_frames else pd.DataFrame()

    metrics_df.to_csv(OUT_DIR / "per_stock_strategy_metrics.csv", index=False)
    aggregate_df.to_csv(OUT_DIR / "equal_weight_portfolio_metrics.csv", index=False)
    curves_df.to_csv(OUT_DIR / "per_stock_equity_curves.csv", index=False)
    daily_df.to_csv(OUT_DIR / "per_stock_daily_returns.csv", index=False)
    if not aggregate_curve_df.empty:
        aggregate_curve_df.to_csv(OUT_DIR / "equal_weight_portfolio_equity_curves.csv", index=False)

    config = {
        "tickers_evaluated": list(loaded.keys()),
        "initial_capital": INITIAL_CAPITAL,
        "transaction_cost_one_way": TRANSACTION_COST,
        "slippage_one_way": SLIPPAGE,
        "execution": "Signals computed at close; position applied next trading day.",
        "portfolio_method": "Equal weight across included stocks; strategy cash allocation earns zero.",
        "strategies": sorted(daily_df["strategy"].unique().tolist()),
        "note": "Historical backtest, not a live-return guarantee. No parameters were optimized on test data."
    }
    (OUT_DIR / "backtest_assumptions.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print("EQUAL-WEIGHT PORTFOLIO RESULTS (₹10,000 INITIAL CAPITAL)")
    print("=" * 76)
    if not aggregate_df.empty:
        display_cols = ["strategy", "total_return", "CAGR", "final_equity", "sharpe_ratio", "max_drawdown", "average_exposure"] 
        # average_exposure is not defined for the aggregate; display only available fields.
        display_cols = [c for c in display_cols if c in aggregate_df.columns]
        print(aggregate_df[display_cols].sort_values("total_return", ascending=False).to_string(index=False, formatters={
            "total_return": "{:.2%}".format, "CAGR": "{:.2%}".format,
            "final_equity": "₹{:,.0f}".format, "sharpe_ratio": "{:.2f}".format,
            "max_drawdown": "{:.2%}".format
        }))
    print(f"\nReports saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
