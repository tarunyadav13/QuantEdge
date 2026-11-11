import pandas as pd
import numpy as np


class PortfolioReturns:

    @staticmethod
    def calculate_returns(price_data):

        if price_data.empty:
            raise ValueError("Price data is empty.")

        returns = price_data.pct_change(fill_method=None)

        returns = returns.replace(
            [np.inf, -np.inf],
            np.nan
        )

        returns = returns.dropna(how="all")

        return returns

    @staticmethod
    def calculate_covariance(returns):

        covariance = returns.cov()

        return covariance

    @staticmethod
    def calculate_expected_returns(returns):

        expected_returns = returns.mean() * 252

        return expected_returns

    @staticmethod
    def calculate_volatility(returns):

        volatility = returns.std() * np.sqrt(252)

        return volatility
