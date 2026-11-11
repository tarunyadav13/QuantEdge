
import numpy as np
import pandas as pd


class PortfolioMetrics:

    @staticmethod
    def calculate(returns):

        returns = returns.dropna()

        if returns.empty:
            raise ValueError("Returns are empty.")

        wealth = (1 + returns).cumprod()

        total_return = wealth.iloc[-1] - 1

        years = len(returns) / 252

        if years > 0 and wealth.iloc[-1] > 0:

            cagr = (
                wealth.iloc[-1] ** (1 / years)
            ) - 1

        else:

            cagr = np.nan

        volatility = (
            returns.std() * np.sqrt(252)
        )

        sharpe = (
            returns.mean() * 252
        ) / volatility if volatility > 0 else np.nan

        drawdown = (
            wealth / wealth.cummax()
        ) - 1

        return {
            "Total Return": total_return,
            "CAGR": cagr,
            "Annualized Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Maximum Drawdown": drawdown.min(),
            "Positive Days": (returns > 0).mean()
        }
