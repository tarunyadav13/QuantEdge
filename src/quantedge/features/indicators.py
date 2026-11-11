import numpy as np
import pandas as pd


def calculate_returns(df):

    df["daily_return"] = df["Close"].pct_change()

    df["log_return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    for period in [5, 10, 20, 60]:

        df[f"return_{period}d"] = (
            df["Close"].pct_change(period)
        )

    return df


def calculate_moving_averages(df):

    for period in [10, 20, 50, 100, 200]:

        df[f"sma_{period}"] = (
            df["Close"].rolling(period).mean()
        )

    for period in [9, 21, 50]:

        df[f"ema_{period}"] = (
            df["Close"].ewm(
                span=period,
                adjust=False
            ).mean()
        )

    df["price_sma20_ratio"] = (
        df["Close"] / df["sma_20"]
    )

    df["price_sma50_ratio"] = (
        df["Close"] / df["sma_50"]
    )

    return df


def calculate_rsi(df, period=14):

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def calculate_macd(df):

    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_histogram"] = (
        df["macd"] - df["macd_signal"]
    )

    return df


def calculate_bollinger_bands(df, period=20):

    rolling_mean = df["Close"].rolling(period).mean()

    rolling_std = df["Close"].rolling(period).std()

    df["bb_upper"] = rolling_mean + (2 * rolling_std)

    df["bb_lower"] = rolling_mean - (2 * rolling_std)

    df["bb_width"] = (
        (df["bb_upper"] - df["bb_lower"])
        / rolling_mean
    )

    df["bb_position"] = (
        (df["Close"] - df["bb_lower"])
        / (df["bb_upper"] - df["bb_lower"])
    )

    return df


def calculate_atr(df, period=14):

    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] - df["Close"].shift(1)
    ).abs()

    low_close = (
        df["Low"] - df["Close"].shift(1)
    ).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    df["atr_percentage"] = (
        df["atr"] / df["Close"]
    )

    return df


def calculate_adx(df, period=14):

    up_move = df["High"].diff()

    down_move = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0
        ),
        index=df.index
    )

    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] - df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"] - df["Close"].shift()
    ).abs()

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    plus_di = (
        100 * plus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean() / atr
    )

    minus_di = (
        100 * minus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean() / atr
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["plus_di"] = plus_di

    df["minus_di"] = minus_di

    df["adx"] = dx.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    return df


def calculate_stochastic(df, period=14):

    lowest_low = df["Low"].rolling(period).min()

    highest_high = df["High"].rolling(period).max()

    df["stoch_k"] = (
        100 * (df["Close"] - lowest_low)
        / (highest_high - lowest_low)
    )

    df["stoch_d"] = (
        df["stoch_k"].rolling(3).mean()
    )

    return df


def calculate_volume_indicators(df):

    df["volume_change"] = df["Volume"].pct_change()

    df["volume_sma20"] = (
        df["Volume"].rolling(20).mean()
    )

    df["relative_volume"] = (
        df["Volume"] / df["volume_sma20"]
    )

    direction = np.sign(df["Close"].diff()).fillna(0)

    df["obv"] = (
        direction * df["Volume"]
    ).cumsum()

    typical_price = (
        df["High"] + df["Low"] + df["Close"]
    ) / 3

    df["vwap_20"] = (
        (typical_price * df["Volume"]).rolling(20).sum()
        / df["Volume"].rolling(20).sum()
    )

    return df


def calculate_volatility(df):

    for period in [10, 20, 30, 60]:

        df[f"volatility_{period}d"] = (
            df["daily_return"].rolling(period).std()
            * np.sqrt(252)
        )

    df["rolling_max"] = (
        df["Close"].rolling(252, min_periods=1).max()
    )

    df["drawdown"] = (
        df["Close"] / df["rolling_max"] - 1
    )

    return df


def calculate_price_action(df):

    df["high_low_range"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )

    df["open_close_change"] = (
        (df["Close"] - df["Open"]) / df["Open"]
    )

    df["gap_percentage"] = (
        (df["Open"] - df["Close"].shift(1))
        / df["Close"].shift(1)
    )

    return df


def calculate_roc(df, period=10):

    df["roc"] = (
        df["Close"].pct_change(period) * 100
    )

    return df


def calculate_cci(df, period=20):

    typical_price = (
        df["High"] + df["Low"] + df["Close"]
    ) / 3

    sma = typical_price.rolling(period).mean()

    mean_deviation = typical_price.rolling(period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )

    df["cci"] = (
        (typical_price - sma)
        / (0.015 * mean_deviation)
    )

    return df


def calculate_williams_r(df, period=14):

    highest_high = df["High"].rolling(period).max()

    lowest_low = df["Low"].rolling(period).min()

    df["williams_r"] = (
        -100 * (highest_high - df["Close"])
        / (highest_high - lowest_low)
    )

    return df


def calculate_keltner_channels(df):

    ema = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    if "atr" not in df.columns:
        df = calculate_atr(df)

    df["keltner_upper"] = ema + 2 * df["atr"]

    df["keltner_lower"] = ema - 2 * df["atr"]

    return df


def calculate_all_indicators(df):

    df = df.copy()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.sort_index()

    df = calculate_returns(df)

    df = calculate_moving_averages(df)

    df = calculate_rsi(df)

    df = calculate_macd(df)

    df = calculate_bollinger_bands(df)

    df = calculate_atr(df)

    df = calculate_adx(df)

    df = calculate_stochastic(df)

    df = calculate_volume_indicators(df)

    df = calculate_volatility(df)

    df = calculate_price_action(df)

    df = calculate_roc(df)

    df = calculate_cci(df)

    df = calculate_williams_r(df)

    df = calculate_keltner_channels(df)

    return df