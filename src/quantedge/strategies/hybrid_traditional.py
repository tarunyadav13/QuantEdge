"""Hybrid traditional trading strategies for QuantEdge.

Signals are calculated using information available at the close. The backtester
applies positions from the following trading day to avoid same-bar look-ahead.
All strategies are long-only (0/1 exposure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _close_series(data: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(data, pd.Series):
        close = data.copy()
    elif isinstance(data, pd.DataFrame) and "Close" in data.columns:
        close = data["Close"].copy()
    else:
        raise ValueError("Input must be a Close Series or DataFrame with a 'Close' column.")
    close = pd.to_numeric(close, errors="coerce").dropna().astype(float)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if len(close) < 220:
        raise ValueError(f"At least 220 valid close prices are required; received {len(close)}.")
    if (close <= 0).any():
        raise ValueError("Close prices must be positive.")
    return close.rename("Close")


def _stateful(condition: pd.Series, exit_condition: pd.Series) -> pd.Series:
    """Enter when condition is true; remain invested until exit condition is true."""
    condition = condition.fillna(False).astype(bool)
    exit_condition = exit_condition.fillna(False).astype(bool)
    values = np.zeros(len(condition), dtype=np.int8)
    invested = 0
    for i, (enter, leave) in enumerate(zip(condition.to_numpy(), exit_condition.to_numpy())):
        if invested == 0 and enter:
            invested = 1
        elif invested == 1 and leave:
            invested = 0
        values[i] = invested
    return pd.Series(values, index=condition.index, name="position")


def _indicators(close: pd.Series) -> pd.DataFrame:
    x = pd.DataFrame(index=close.index)
    x["sma_9"] = close.rolling(9, min_periods=9).mean()
    x["sma_21"] = close.rolling(21, min_periods=21).mean()
    x["sma_50"] = close.rolling(50, min_periods=50).mean()
    x["sma_200"] = close.rolling(200, min_periods=200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["rsi_14"] = 100 - (100 / (1 + rs))
    x.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]

    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    x["bb_mid"] = mid
    x["bb_upper"] = mid + 2 * std
    x["bb_lower"] = mid - 2 * std

    # Wilder-style directional movement / ADX approximation.
    high = close  # Close-only input: use close changes as a conservative proxy.
    low = close
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = close.diff().abs()
    atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["adx_proxy"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    x["plus_di_proxy"] = plus_di
    return x


def generate_hybrid_signals(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Return position signals for four hybrid strategies plus indicator columns."""
    close = _close_series(data)
    ind = _indicators(close)
    rsi_recovery = (ind["rsi_14"].shift(1) < 30) & (ind["rsi_14"] >= 30)

    # A: RSI oversold recovery, confirmed by bullish MACD.
    a = _stateful(rsi_recovery & (ind["macd"] > ind["macd_signal"]),
                  (ind["macd"] < ind["macd_signal"]) | (ind["rsi_14"] > 70))

    # B: Long-term bullish trend, enter on RSI recovery from below 40.
    b = _stateful((ind["sma_50"] > ind["sma_200"]) &
                  (ind["rsi_14"].shift(1) < 40) & (ind["rsi_14"] >= 40),
                  (ind["sma_50"] < ind["sma_200"]) | (ind["rsi_14"] > 75))

    # C: Close re-enters above lower Bollinger band while RSI recovers from oversold.
    c = _stateful((close.shift(1) < ind["bb_lower"].shift(1)) &
                  (close >= ind["bb_lower"]) &
                  (ind["rsi_14"].shift(1) < 35) & (ind["rsi_14"] >= 35),
                  (close >= ind["bb_mid"]) | (ind["rsi_14"] > 65))

    # D: Consensus score with trend/momentum/reversal confirmations.
    score = pd.DataFrame(index=close.index)
    score["trend"] = (ind["sma_50"] > ind["sma_200"]).astype(int)
    score["rsi_recovery"] = rsi_recovery.astype(int)
    score["macd_bullish"] = (ind["macd"] > ind["macd_signal"]).astype(int)
    score["bb_reentry"] = ((close.shift(1) < ind["bb_lower"].shift(1)) &
                           (close >= ind["bb_lower"])).astype(int)
    score["adx_trend"] = ((ind["adx_proxy"] > 25) &
                          (ind["plus_di_proxy"] > 0)).astype(int)
    consensus_score = score.sum(axis=1)
    d = _stateful(consensus_score >= 3, consensus_score <= 1)

    out = pd.DataFrame({
        "HYBRID_RSI_MACD": a,
        "HYBRID_SMA_RSI": b,
        "HYBRID_BB_RSI": c,
        "HYBRID_CONSENSUS": d,
        "consensus_score": consensus_score,
    }, index=close.index)
    for col in ind.columns:
        out[col] = ind[col]
    return out


def backtest_hybrid_positions(
    close: pd.Series,
    target_position: pd.Series,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, dict]:
    """Next-day long-only backtest; costs charged on each one-way position change."""
    close = pd.to_numeric(close, errors="coerce").dropna().astype(float)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    position = target_position.reindex(close.index).fillna(0).clip(0, 1)
    held = position.shift(1).fillna(0)
    asset_returns = close.pct_change().fillna(0)
    turnover = held.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(held.iloc[0])
    one_way_cost = transaction_cost + slippage
    net_returns = held * asset_returns - turnover.fillna(0) * one_way_cost
    equity = initial_capital * (1 + net_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / periods_per_year)
    total_return = float(equity.iloc[-1] / initial_capital - 1)
    vol = float(net_returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(net_returns) > 1 else 0.0
    sharpe = float(net_returns.mean() / net_returns.std(ddof=1) * np.sqrt(periods_per_year)) if net_returns.std(ddof=1) > 0 else 0.0
    downside = net_returns[net_returns < 0]
    downside_std = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    sortino = float(net_returns.mean() * periods_per_year / downside_std) if downside_std > 0 else 0.0
    metrics = {
        "total_return": total_return,
        "cagr": float((equity.iloc[-1] / initial_capital) ** (1 / years) - 1),
        "final_equity": float(equity.iloc[-1]),
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": float(drawdown.min()),
        "daily_win_rate": float((net_returns > 0).mean()),
        "average_exposure": float(held.mean()),
        "turnover": float(turnover.fillna(0).sum()),
        "position_changes": int((turnover.fillna(0) > 0).sum()),
        "estimated_cost_paid": float((turnover.fillna(0) * one_way_cost).sum()),
        "observations": int(len(close)),
    }
    curve = pd.DataFrame({
        "close": close, "target_position": position, "held_position": held,
        "daily_return": asset_returns, "turnover": turnover.fillna(0),
        "net_strategy_return": net_returns, "equity": equity, "drawdown": drawdown,
    })
    return curve, metrics
