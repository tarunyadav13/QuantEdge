"""Traditional, long-only technical strategies for QuantEdge.

Signals are computed using information available at the close. The backtest
applies the signal from the following trading day, avoiding same-bar lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_close(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce").astype(float).dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if len(close) < 5:
        raise ValueError("At least five valid closing prices are required.")
    if (close <= 0).any():
        raise ValueError("Closing prices must be positive.")
    return close


def _stateful_entry_exit(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Hold 1 after entry until exit; otherwise remain in cash."""
    position = np.zeros(len(entry), dtype=float)
    active = False
    for i, (go_long, go_flat) in enumerate(zip(entry.fillna(False), exit_.fillna(False))):
        if not active and bool(go_long):
            active = True
        elif active and bool(go_flat):
            active = False
        position[i] = float(active)
    return pd.Series(position, index=entry.index)


def generate_signals(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Return close-of-day target positions (0=cash, 1=fully invested)."""
    if "Close" not in data.columns:
        raise ValueError("Input data must contain a 'Close' column.")

    close = _validate_close(data["Close"])
    signals = {}

    # Moving-average trend-following strategies.
    for fast, slow in [(9, 21), (20, 50), (50, 200)]:
        fast_ma = close.rolling(fast, min_periods=fast).mean()
        slow_ma = close.rolling(slow, min_periods=slow).mean()
        signals[f"SMA_{fast}_{slow}"] = (fast_ma > slow_ma).astype(float)

    # MACD trend strategy.
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    signals["MACD_12_26_9"] = (macd > macd_signal).astype(float)

    # RSI(14) mean reversion: enter below 30, exit above 55.
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(loss.ne(0), 100.0)
    signals["RSI_14_MR"] = _stateful_entry_exit(rsi < 30, rsi > 55)

    # Bollinger mean reversion: enter below lower band, exit at/above middle.
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std()
    lower = mid - 2 * std
    signals["BOLLINGER_20_2"] = _stateful_entry_exit(close < lower, close >= mid)

    # Donchian breakout. Levels use prior bars only.
    if {"High", "Low"}.issubset(data.columns):
        high = pd.to_numeric(data["High"], errors="coerce").reindex(close.index)
        low = pd.to_numeric(data["Low"], errors="coerce").reindex(close.index)
        upper20 = high.shift(1).rolling(20, min_periods=20).max()
        lower10 = low.shift(1).rolling(10, min_periods=10).min()
        signals["DONCHIAN_20_10"] = _stateful_entry_exit(close > upper20, close < lower10)

    result = pd.DataFrame(signals, index=close.index).fillna(0.0)
    # Warm-up periods stay in cash until indicators are available.
    for name in ["SMA_9_21", "SMA_20_50", "SMA_50_200", "MACD_12_26_9"]:
        result[name] = result[name].where(result[name].notna(), 0.0)
    return {column: result[column].astype(float) for column in result.columns}


def backtest_positions(
    close: pd.Series,
    target_position: pd.Series,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, dict]:
    """Backtest close-generated signals with next-day execution and one-way costs.

    Costs are charged on changes in the executed position. A 0->1 entry and
    1->0 exit each pay transaction_cost + slippage.
    """
    close = _validate_close(close)
    position = target_position.reindex(close.index).fillna(0.0).clip(0.0, 1.0)

    asset_returns = close.pct_change().fillna(0.0)
    held_position = position.shift(1).fillna(0.0)
    turnover = held_position.diff().abs().fillna(held_position.abs())
    one_way_cost = float(transaction_cost) + float(slippage)
    costs = turnover * one_way_cost
    strategy_returns = held_position * asset_returns - costs

    equity = initial_capital * (1.0 + strategy_returns).cumprod()
    benchmark_equity = initial_capital * (1.0 + asset_returns).cumprod()

    curve = pd.DataFrame({
        "close": close,
        "target_position_at_close": position,
        "held_position": held_position,
        "asset_return": asset_returns,
        "turnover": turnover,
        "estimated_cost": costs,
        "strategy_return": strategy_returns,
        "strategy_equity": equity,
        "buy_hold_equity": benchmark_equity,
    })

    metrics = calculate_metrics(
        strategy_returns, equity, initial_capital, periods_per_year
    )
    metrics["buy_hold_total_return"] = float(benchmark_equity.iloc[-1] / initial_capital - 1)
    metrics["buy_hold_final_equity"] = float(benchmark_equity.iloc[-1])
    metrics["average_exposure"] = float(held_position.mean())
    metrics["total_turnover"] = float(turnover.sum())
    metrics["estimated_trades"] = int((turnover > 0).sum())
    metrics["total_estimated_cost"] = float(costs.sum())
    return curve, metrics


def calculate_metrics(
    returns: pd.Series,
    equity: pd.Series,
    initial_capital: float = 10_000.0,
    periods_per_year: int = 252,
) -> dict:
    returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    equity = pd.Series(equity).reindex(returns.index).dropna()
    if returns.empty or equity.empty:
        raise ValueError("Cannot calculate metrics on empty returns.")

    total_return = float(equity.iloc[-1] / initial_capital - 1)
    elapsed_days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = elapsed_days / 365.25
    cagr = float((equity.iloc[-1] / initial_capital) ** (1 / years) - 1) if equity.iloc[-1] > 0 else np.nan
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else np.nan
    annual_mean = float(returns.mean() * periods_per_year)
    sharpe = float(annual_mean / volatility) if volatility and np.isfinite(volatility) and volatility > 0 else np.nan
    downside = returns.clip(upper=0)
    downside_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year))
    sortino = float(annual_mean / downside_dev) if downside_dev > 0 else np.nan
    drawdown = equity / equity.cummax() - 1

    return {
        "total_return": total_return,
        "CAGR": cagr,
        "final_equity": float(equity.iloc[-1]),
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": float(drawdown.min()),
        "daily_win_rate": float((returns > 0).mean()),
        "observations": int(len(returns)),
    }
