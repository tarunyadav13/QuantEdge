import numpy as np
import pandas as pd

from quantedge.strategies.traditional import (
    generate_signals,
    backtest_positions,
)


def make_data(n=300):
    idx = pd.bdate_range("2020-01-01", periods=n)
    # Deterministic trending series with small oscillations.
    close = 100 * np.exp(np.linspace(0, 0.35, n)) * (1 + 0.015 * np.sin(np.arange(n) / 7))
    return pd.DataFrame({
        "Open": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": 100000,
    }, index=idx)


def test_signal_generators_return_valid_positions():
    data = make_data()
    signals = generate_signals(data)
    assert "SMA_9_21" in signals
    assert "SMA_20_50" in signals
    assert "RSI_14_MR" in signals
    for signal in signals.values():
        assert signal.index.equals(data.index)
        assert set(signal.dropna().unique()).issubset({0.0, 1.0})


def test_backtest_applies_signal_with_one_day_lag():
    idx = pd.bdate_range("2024-01-01", periods=5)
    close = pd.Series([100, 110, 110, 110, 110], index=idx)
    position = pd.Series([1, 1, 1, 1, 1], index=idx)
    curve, metrics = backtest_positions(
        close, position, initial_capital=10000,
        transaction_cost=0, slippage=0
    )
    # First day's return is not captured because the close signal is applied next day.
    assert curve["held_position"].iloc[0] == 0
    assert curve["held_position"].iloc[1] == 1
    assert np.isclose(metrics["total_return"], 0.10)


def test_costs_reduce_returns():
    data = make_data()
    signal = pd.Series(1.0, index=data.index)
    _, free = backtest_positions(data["Close"], signal, transaction_cost=0, slippage=0)
    _, charged = backtest_positions(data["Close"], signal, transaction_cost=0.001, slippage=0.0005)
    assert charged["total_return"] < free["total_return"]
