import numpy as np
import pandas as pd

from quantedge.features.indicators import (
    calculate_all_indicators
)


class FeaturePipeline:

    def __init__(self, neutral_threshold=0.002):

        self.neutral_threshold = neutral_threshold

    def add_benchmark_features(
        self,
        df,
        benchmark_df=None
    ):

        if benchmark_df is None:
            return df

        benchmark = benchmark_df.copy()

        benchmark = benchmark.sort_index()

        benchmark_returns = (
            benchmark["Close"].pct_change()
        )

        stock_returns = df["daily_return"]

        aligned_returns = pd.concat(
            [
                stock_returns.rename("stock"),
                benchmark_returns.rename("benchmark")
            ],
            axis=1
        ).sort_index()

        aligned_returns = aligned_returns.ffill()

        df["benchmark_return"] = (
            aligned_returns["benchmark"].reindex(df.index)
        )

        df["relative_return"] = (
            df["daily_return"] - df["benchmark_return"]
        )

        df["relative_strength_20d"] = (
            (1 + df["daily_return"])
            .rolling(20)
            .apply(np.prod, raw=True)
            /
            (1 + df["benchmark_return"])
            .rolling(20)
            .apply(np.prod, raw=True)
        )

        df["market_correlation_20d"] = (
            df["daily_return"]
            .rolling(20)
            .corr(df["benchmark_return"])
        )

        return df

    def add_prediction_targets(self, df):

        # Next trading day's return.
        df["target_return"] = (
            df["Close"].shift(-1) / df["Close"] - 1
        )

        # Binary classification.
        df["target_direction"] = (
            df["target_return"] > 0
        ).astype("float")

        # Three-class classification.
        threshold = self.neutral_threshold

        conditions = [
            df["target_return"] > threshold,
            df["target_return"] < -threshold
        ]

        choices = [1, -1]

        df["target_market_regime"] = np.select(
            conditions,
            choices,
            default=0
        )

        # Final row has no known next-day target.
        df.loc[
            df["target_return"].isna(),
            [
                "target_direction",
                "target_market_regime"
            ]
        ] = np.nan

        return df

    def build_features(
        self,
        df,
        benchmark_df=None,
        include_targets=True
    ):

        if df is None or df.empty:
            raise ValueError(
                "Input market dataset is empty."
            )

        df = df.copy()

        df = df.sort_index()

        df = calculate_all_indicators(df)

        df = self.add_benchmark_features(
            df,
            benchmark_df
        )

        if include_targets:
            df = self.add_prediction_targets(df)

        return df

    @staticmethod
    def get_feature_columns(df):

        excluded = {
            "target_return",
            "target_direction",
            "target_market_regime"
        }

        return [
            col for col in df.columns
            if col not in excluded
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    @staticmethod
    def clean_features(df):

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return df.dropna()