import numpy as np
import pandas as pd

from quantedge.ml.hybrid_ml_engine import (
    date_purged_split, make_features, normalize_ohlcv, summarize_returns,
)


def sample_ohlcv(n=500):
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(.0003, .01, n)))
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, .002, n)),
        "High": close * 1.01,
        "Low": close * .99,
        "Close": close,
        "Volume": rng.integers(10000, 100000, n),
    }, index=idx)


def test_feature_builder_has_expected_columns_and_no_infinite_values():
    f = make_features(sample_ohlcv())
    assert {"ret_1", "rsi_14", "target", "target_return", "Close"} <= set(f.columns)
    assert not np.isinf(f.select_dtypes("number").to_numpy()).any()


def test_pooled_split_has_disjoint_dates_and_purge():
    f1 = make_features(sample_ohlcv())
    f1["ticker"] = "AAA"
    f2 = make_features(sample_ohlcv() * 1.2)
    f2["ticker"] = "BBB"
    panel = pd.concat([f1, f2]).reset_index(names="Date").set_index(["Date", "ticker"])
    tr, va, te = date_purged_split(panel, purge=5)
    td = set(tr.index.get_level_values("Date"))
    vd = set(va.index.get_level_values("Date"))
    ed = set(te.index.get_level_values("Date"))
    assert not td & vd and not vd & ed and not td & ed
    assert max(td) < min(vd) and max(vd) < min(ed)


def test_return_summary_is_finite():
    stats = summarize_returns(pd.Series([.01, -.005, .004, 0.0]))
    assert np.isfinite(stats["total_return"])
    assert stats["final_equity"] > 0


def test_normalize_csv_style_columns():
    df = sample_ohlcv().reset_index(names="Date")
    normalized = normalize_ohlcv(df)
    assert list(normalized.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert normalized.index.is_monotonic_increasing
