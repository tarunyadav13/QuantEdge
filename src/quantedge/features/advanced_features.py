import numpy as np
import pandas as pd
from requests.packages import target
from requests.packages import target


class AdvancedFeatureEngineer:

    def __init__(self):

        self.return_periods = [1, 2, 3, 5, 10, 20]

        self.momentum_periods = [5, 10, 20, 50]

        self.volatility_periods = [5, 10, 20, 50]

        self.ma_periods = [9, 20, 21, 50, 100, 200]

    # --------------------------------------------------
    # DATA NORMALIZATION
    # --------------------------------------------------

    def _normalize_columns(self, df):

        df = df.copy()

        df.columns = [
            str(col).lower().replace(" ", "_")
            for col in df.columns
        ]

        aliases = {
            "adj_close": "close",
            "date": "date"
        }

        df = df.rename(columns=aliases)

        if "date" in df.columns:

            df["date"] = pd.to_datetime(df["date"])

            df = df.set_index("date")

        df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        df = df[~df.index.duplicated(keep="last")]

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing required OHLCV columns: {missing}"
            )

        for col in required:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df[required]

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return df

    # --------------------------------------------------
    # RETURNS AND MOMENTUM
    # --------------------------------------------------

    def add_returns(self, df):

        close = df["close"]

        for period in self.return_periods:

            df[f"return_{period}d"] = (
                close.pct_change(period)
            )

            df[f"log_return_{period}d"] = (
                np.log(close / close.shift(period))
            )

        for period in self.momentum_periods:

            df[f"momentum_{period}d"] = (
                close / close.shift(period) - 1
            )

        # Rate of change

        df["roc_10"] = (
            close / close.shift(10) - 1
        ) * 100

        df["roc_20"] = (
            close / close.shift(20) - 1
        ) * 100

        # RSI using Wilder-style exponential smoothing

        delta = close.diff()

        gain = delta.clip(lower=0)

        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        df["rsi_14"] = (
            100 - (100 / (1 + rs))
        )

        # Handle RSI edge cases
        df.loc[
            (avg_loss == 0) & (avg_gain > 0),
            "rsi_14"
        ] = 100

        df.loc[
            (avg_gain == 0) & (avg_loss > 0),
        "rsi_14"
        ] = 0

        df.loc[
            (avg_gain == 0) & (avg_loss == 0),
            "rsi_14"
        ] = 50

        # Stochastic RSI-like price position

        low_14 = df["low"].rolling(14).min()

        high_14 = df["high"].rolling(14).max()

        df["stochastic_14"] = (
            (close - low_14) /
            (high_14 - low_14).replace(0, np.nan)
        ) * 100

        return df

    # --------------------------------------------------
    # TREND FEATURES
    # --------------------------------------------------

    def add_trend_features(self, df):

        close = df["close"]

        for period in self.ma_periods:

            sma = close.rolling(period).mean()

            ema = close.ewm(
                span=period,
                adjust=False,
                min_periods=period
            ).mean()

            df[f"sma_{period}"] = sma

            df[f"ema_{period}"] = ema

            # Normalize price distance

            df[f"price_sma_ratio_{period}"] = (
                close / sma - 1
            )

            df[f"price_ema_ratio_{period}"] = (
                close / ema - 1
            )

        # Moving average crossovers

        df["ema_9_21_spread"] = (
            df["ema_9"] / df["ema_21"] - 1
        )

        df["ema_21_50_spread"] = (
            df["ema_21"] / df["ema_50"] - 1
        )

        df["ema_50_200_spread"] = (
            df["ema_50"] / df["ema_200"] - 1
        )

        # MACD

        ema_12 = close.ewm(
            span=12,
            adjust=False
        ).mean()

        ema_26 = close.ewm(
            span=26,
            adjust=False
        ).mean()

        df["macd"] = ema_12 - ema_26

        df["macd_signal"] = (
            df["macd"].ewm(
                span=9,
                adjust=False
            ).mean()
        )

        df["macd_histogram"] = (
            df["macd"] - df["macd_signal"]
        )

        df["macd_normalized"] = (
            df["macd"] / close
        )

        return df

    # --------------------------------------------------
    # VOLATILITY FEATURES
    # --------------------------------------------------

    def add_volatility_features(self, df):

        close = df["close"]

        previous_close = close.shift(1)

        high_low = df["high"] - df["low"]

        high_close = (
            df["high"] - previous_close
        ).abs()

        low_close = (
            df["low"] - previous_close
        ).abs()

        true_range = pd.concat(
            [
                high_low,
                high_close,
                low_close
            ],
            axis=1
        ).max(axis=1)

        df["atr_14"] = true_range.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()

        df["atr_percentage"] = (
            df["atr_14"] / close
        )

        # Bollinger Bands

        rolling_mean = close.rolling(20).mean()

        rolling_std = close.rolling(20).std()

        upper = rolling_mean + 2 * rolling_std

        lower = rolling_mean - 2 * rolling_std

        df["bollinger_upper"] = upper

        df["bollinger_lower"] = lower

        df["bollinger_width"] = (
            (upper - lower) /
            rolling_mean.replace(0, np.nan)
        )

        df["bollinger_percent_b"] = (
            (close - lower) /
            (upper - lower).replace(0, np.nan)
        )

        # Rolling realized volatility

        for period in self.volatility_periods:

            df[f"volatility_{period}d"] = (
                df["return_1d"]
                .rolling(period)
                .std()
                * np.sqrt(252)
            )

        # Downside volatility

        negative_returns = df["return_1d"].clip(
            upper=0
        )

        df["downside_volatility_20d"] = (
            negative_returns
            .rolling(20)
            .std()
            * np.sqrt(252)
        )

        return df

    # --------------------------------------------------
    # VOLUME FEATURES
    # --------------------------------------------------

    def add_volume_features(self, df):

        volume = df["volume"]

        close = df["close"]

        for period in [5, 10, 20, 50]:

            average_volume = volume.rolling(
                period
            ).mean()

            df[f"relative_volume_{period}d"] = (
                volume /
                average_volume.replace(0, np.nan)
            )

        df["volume_change"] = volume.pct_change()

        df["volume_momentum_5d"] = (
            volume / volume.shift(5) - 1
        )

        df["volume_momentum_20d"] = (
            volume / volume.shift(20) - 1
        )

        # On Balance Volume

        direction = np.sign(close.diff()).fillna(0)

        df["obv"] = (
            direction * volume
        ).cumsum()

        df["obv_change_5d"] = (
            df["obv"].diff(5) /
            volume.rolling(20).mean().replace(0, np.nan)
        )

        # Money flow-style price-volume feature

        df["price_volume_change"] = (
            df["return_1d"] * df["volume_change"]
        )

        return df

    # --------------------------------------------------
    # RISK FEATURES
    # --------------------------------------------------

    def add_risk_features(self, df):

        close = df["close"]

        for period in [20, 50, 100]:

            rolling_high = close.rolling(
                period
            ).max()

            df[f"drawdown_{period}d"] = (
                close / rolling_high - 1
            )

        df["high_low_range_pct"] = (
            (df["high"] - df["low"]) / close
        )

        df["open_close_return"] = (
            df["close"] / df["open"] - 1
        )

        df["gap_return"] = (
            df["open"] / close.shift(1) - 1
        )

        df["distance_from_52w_high"] = (
            close / close.rolling(252).max() - 1
        )

        df["distance_from_52w_low"] = (
            close / close.rolling(252).min() - 1
        )

        return df

    # --------------------------------------------------
    # BENCHMARK FEATURES
    # --------------------------------------------------

    def add_benchmark_features(
        self,
        df,
        benchmark_df=None
    ):

        if benchmark_df is None:

            return df

        benchmark = self._normalize_columns(
            benchmark_df
        )

        benchmark_close = benchmark["close"]

        benchmark_returns = (
            benchmark_close.pct_change()
        )

        benchmark_features = pd.DataFrame(
            index=benchmark.index
        )

        for period in [1, 5, 10, 20]:

            benchmark_features[
                f"benchmark_return_{period}d"
            ] = benchmark_close.pct_change(period)

        benchmark_features[
            "benchmark_sma_50_ratio"
        ] = (
            benchmark_close /
            benchmark_close.rolling(50).mean() - 1
        )

        benchmark_features[
            "benchmark_volatility_20d"
        ] = (
            benchmark_returns.rolling(20).std()
            * np.sqrt(252)
        )

        # Align only on dates with available observations.
        # Forward-fill carries the last known benchmark observation.
        benchmark_features = benchmark_features.reindex(
            df.index
        ).ffill()

        df = df.join(
            benchmark_features,
            how="left"
        )

        # Relative performance

        for period in [5, 10, 20]:

            stock_return = df[f"return_{period}d"]

            market_return = df[
                f"benchmark_return_{period}d"
            ]

            df[f"relative_strength_{period}d"] = (
                stock_return - market_return
            )

        # Rolling beta and correlation

        aligned_market_returns = (
            benchmark_returns.reindex(df.index).ffill()
        )

        stock_returns = df["return_1d"]

        rolling_covariance = (
            stock_returns
            .rolling(60)
            .cov(aligned_market_returns)
        )

        rolling_market_variance = (
            aligned_market_returns
            .rolling(60)
            .var()
        )

        df["rolling_beta_60d"] = (
            rolling_covariance /
            rolling_market_variance.replace(0, np.nan)
        )

        df["rolling_market_correlation_60d"] = (
            stock_returns
            .rolling(60)
            .corr(aligned_market_returns)
        )

        return df

    # --------------------------------------------------
    # MARKET REGIME
    # --------------------------------------------------

    def add_market_regime(self, df):

        trend = df["price_sma_ratio_50"]

        long_trend = df["price_sma_ratio_200"]

        volatility = df["volatility_20d"]

        regime = pd.Series(
            "SIDEWAYS",
            index=df.index,
            dtype="object"
        )

        bullish = (
            (trend > 0) &
            (long_trend > 0)
        )

        bearish = (
            (trend < 0) &
            (long_trend < 0)
        )

        regime.loc[bullish] = "BULLISH"

        regime.loc[bearish] = "BEARISH"

        high_volatility = (
            volatility >
            volatility.rolling(252, min_periods=60).quantile(0.75)
        )

        regime.loc[high_volatility] = "HIGH_VOLATILITY"

        df["market_regime"] = regime

        # Numeric regime encoding for ML

        mapping = {
            "BEARISH": 0,
            "SIDEWAYS": 1,
            "BULLISH": 2,
            "HIGH_VOLATILITY": 3
        }

        df["market_regime_code"] = (
            df["market_regime"].map(mapping)
        )

        return df

    # --------------------------------------------------
    # FUTURE TARGETS
    # --------------------------------------------------

    def add_targets(self, df):

        df = df.copy()

        for horizon in [1, 5, 10]:

            future_return = (
                df["close"].shift(-horizon) /
                df["close"]
            ) - 1

            df[f"target_return_{horizon}d"] = (
                future_return
            )

            direction = (
                future_return > 0
            ).astype(float)

            # Preserve missing future observations.
            # Do not incorrectly label the final rows as DOWN.
            direction.loc[
                future_return.isna()
            ] = np.nan

            df[
                f"target_direction_{horizon}d"
            ] = direction

        return df

    # --------------------------------------------------
    # COMPLETE PIPELINE
    # --------------------------------------------------

    def build_features(
        self,
        stock_df,
        benchmark_df=None,
        include_targets=True
    ):

        df = self._normalize_columns(stock_df)

        df = self.add_returns(df)

        df = self.add_trend_features(df)

        df = self.add_volatility_features(df)

        df = self.add_volume_features(df)

        df = self.add_risk_features(df)

        df = self.add_benchmark_features(
            df,
            benchmark_df
        )

        df = self.add_market_regime(df)

        if include_targets:

            df = self.add_targets(df)

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return df

    # --------------------------------------------------
    # FEATURE COLUMN SELECTION
    # --------------------------------------------------

    @staticmethod
    def get_feature_columns(df):

        excluded = {
            "market_regime",
            "target_return_1d",
            "target_return_5d",
            "target_return_10d",
            "target_direction_1d",
            "target_direction_5d",
            "target_direction_10d"
        }

        return [
            col
            for col in df.columns
            if col not in excluded
            and pd.api.types.is_numeric_dtype(df[col])
            and not col.startswith("target_")
        ]