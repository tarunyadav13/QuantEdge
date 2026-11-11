"""Leakage-conscious technical-indicator feature engineering and ML evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]) for c in out.columns]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def normalize_market_data(data: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(data)
    lower = {str(c).lower(): c for c in df.columns}
    date_col = next((lower[k] for k in ("date", "datetime") if k in lower), None)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    rename = {}
    for standard in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        found = next((c for c in df.columns if str(c).lower() == standard.lower()), None)
        if found is not None:
            rename[found] = standard
    df = df.rename(columns=rename)
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            raise ValueError("CSV must contain a Close or Adj Close column.")
    for col in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    if "Volume" not in df:
        df["Volume"] = np.nan
    return df


def build_technical_features(data: pd.DataFrame, benchmark_close: pd.Series | None = None,
                             horizon: int = 5) -> pd.DataFrame:
    """Build indicators from current/past bars and a future direction label."""
    df = normalize_market_data(data)
    close = df["Close"].astype(float)
    high = df["High"].astype(float) if "High" in df else close
    low = df["Low"].astype(float) if "Low" in df else close
    volume = df["Volume"].astype(float)

    f = pd.DataFrame(index=df.index)
    # Returns and trend
    for period in (1, 2, 5, 10, 20, 60):
        f[f"return_{period}d"] = close.pct_change(period)
    for period in (9, 21, 50, 100, 200):
        sma = close.rolling(period, min_periods=period).mean()
        f[f"close_sma_{period}_ratio"] = close / sma - 1
    f["sma_9_21_spread"] = close.rolling(9).mean() / close.rolling(21).mean() - 1
    f["sma_50_200_spread"] = close.rolling(50).mean() / close.rolling(200).mean() - 1

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    f["rsi_14"] = 100 - 100 / (1 + rs)
    f.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100

    # MACD
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd_pct"] = macd / close
    f["macd_signal_pct"] = signal / close
    f["macd_hist_pct"] = (macd - signal) / close

    # Bollinger bands
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    upper, lower_band = mid + 2 * std, mid - 2 * std
    f["bb_width"] = (upper - lower_band) / mid
    f["bb_percent_b"] = (close - lower_band) / (upper - lower_band).replace(0, np.nan)

    # Range/volatility
    prev_close = close.shift(1)
    true_range = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    f["atr_pct"] = atr / close
    daily_ret = close.pct_change()
    for period in (5, 10, 20, 60):
        f[f"volatility_{period}d"] = daily_ret.rolling(period, min_periods=period).std() * np.sqrt(252)

    # Volume features
    f["volume_change_1d"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    vol_mean = volume.rolling(20, min_periods=20).mean()
    f["volume_sma20_ratio"] = volume / vol_mean
    f["volume_price_corr20"] = daily_ret.rolling(20, min_periods=20).corr(volume.pct_change())

    # Benchmark context (aligned by date; only historical benchmark values)
    if benchmark_close is not None:
        bench = pd.to_numeric(benchmark_close, errors="coerce").copy()
        bench.index = pd.to_datetime(bench.index)
        bench = bench.sort_index().reindex(f.index).ffill()
        for period in (1, 5, 20):
            f[f"benchmark_return_{period}d"] = bench.pct_change(period)
        f["relative_return_5d"] = f["return_5d"] - f["benchmark_return_5d"]
        f["relative_return_20d"] = f["return_20d"] - f["benchmark_return_20d"]

    # Label: positive forward horizon return. This column is never a feature.
    f["forward_return"] = close.shift(-horizon) / close - 1
    f["target"] = (f["forward_return"] > 0).astype("float")
    f.loc[f["forward_return"].isna(), "target"] = np.nan
    return f.replace([np.inf, -np.inf], np.nan)


def chronological_purged_split(frame: pd.DataFrame, purge: int = 5,
                               train_fraction: float = 0.60,
                               validation_fraction: float = 0.20):
    """Chronological 60/20/20 split with horizon-sized gaps at boundaries."""
    n = len(frame)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))
    train = frame.iloc[:max(0, train_end-purge)].copy()
    validation = frame.iloc[train_end: max(train_end, val_end-purge)].copy()
    test = frame.iloc[val_end:].copy()
    if min(len(train), len(validation), len(test)) < 50:
        raise ValueError(
            f"Not enough samples after purging: train={len(train)}, "
            f"validation={len(validation)}, test={len(test)}. Need >=50 each."
        )
    return train, validation, test
