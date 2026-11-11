import numpy as np
import pandas as pd

from quantedge.ml.technical_ml import build_technical_features, chronological_purged_split


def sample_data(n=800):
    rng = np.random.default_rng(23)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    volume = rng.integers(100_000, 500_000, n)
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, .002, n)),
        "High": close * 1.01,
        "Low": close * .99,
        "Close": close,
        "Volume": volume,
    }, index=pd.bdate_range("2020-01-01", periods=n))


def test_features_and_future_target_created():
    frame = build_technical_features(sample_data(), horizon=5)
    assert "rsi_14" in frame
    assert "macd_hist_pct" in frame
    assert "target" in frame
    assert frame["target"].tail(5).isna().all()


def test_purged_chronological_split_has_no_overlap():
    frame = build_technical_features(sample_data(), horizon=5).dropna()
    train, val, test = chronological_purged_split(frame, purge=5)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    assert len(train) >= 50 and len(val) >= 50 and len(test) >= 50


def test_benchmark_features_are_added():
    data = sample_data()
    benchmark = pd.Series(data["Close"].values * 0.8, index=data.index)
    frame = build_technical_features(data, benchmark_close=benchmark)
    assert "benchmark_return_5d" in frame
    assert "relative_return_20d" in frame
