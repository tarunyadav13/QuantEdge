
import numpy as np
import pandas as pd


class PortfolioRiskManager:

    def __init__(
        self,
        max_asset_weight=0.35,
        max_volatility=0.25
    ):

        self.max_asset_weight = max_asset_weight
        self.max_volatility = max_volatility

    @staticmethod
    def calculate_volatility(returns):

        return returns.std() * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(returns):

        wealth = (1 + returns).cumprod()

        running_peak = wealth.cummax()

        drawdown = (
            wealth / running_peak
        ) - 1

        return drawdown.min()

    @staticmethod
    def calculate_var(
        returns,
        confidence=0.95
    ):

        return returns.quantile(
            1 - confidence
        )

    @staticmethod
    def calculate_cvar(
        returns,
        confidence=0.95
    ):

        var = returns.quantile(
            1 - confidence
        )

        tail_returns = returns[
            returns <= var
        ]

        if tail_returns.empty:
            return np.nan

        return tail_returns.mean()

    @staticmethod
    def calculate_concentration(weights):

        return np.sum(weights.values ** 2)

    def check_position_limits(self, weights):

        return (
            weights.max() <= self.max_asset_weight
        )

    def risk_report(
        self,
        returns,
        weights
    ):

        return {
            "Annualized Volatility": self.calculate_volatility(
                returns
            ),

            "Maximum Drawdown": self.calculate_max_drawdown(
                returns
            ),

            "VaR 95%": self.calculate_var(
                returns
            ),

            "CVaR 95%": self.calculate_cvar(
                returns
            ),

            "Concentration HHI": self.calculate_concentration(
                weights
            ),

            "Maximum Asset Weight": weights.max()
        }
