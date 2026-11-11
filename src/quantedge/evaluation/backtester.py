import numpy as np
import pandas as pd


class Backtester:

    def __init__(self, transaction_cost=0.001, periods_per_year=252):
        self.transaction_cost = transaction_cost
        self.periods_per_year = periods_per_year

    def calculate_metrics(self, returns, trades):

        returns = returns.dropna()

        if len(returns) == 0:
            raise ValueError("No valid returns available.")

        equity = (1 + returns).cumprod()

        total_return = equity.iloc[-1] - 1

        years = len(returns) / self.periods_per_year

        cagr = (
            equity.iloc[-1] ** (1 / years) - 1
            if years > 0 and equity.iloc[-1] > 0
            else np.nan
        )

        volatility = (
            returns.std() * np.sqrt(self.periods_per_year)
        )

        sharpe = (
            returns.mean() / returns.std() * np.sqrt(self.periods_per_year)
            if returns.std() > 0 else 0
        )

        drawdown = equity / equity.cummax() - 1

        max_drawdown = drawdown.min()

        win_rate = (returns > 0).mean()

        return {
            "total_return": total_return,
            "cagr": cagr,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "number_of_trades": int(trades)
        }

    def run_strategy(self, predictions, actual_returns, threshold=0.50):

        data = predictions.copy()

        data["actual_return"] = actual_returns

        data = data.dropna(subset=["actual_return"]).copy()

        data["position"] = (
            data["probability_up"] >= threshold
        ).astype(int)

        # A trade occurs whenever the position changes.
        previous_position = data["position"].shift(1).fillna(0)

        data["turnover"] = (
            data["position"] - previous_position
        ).abs()

        data["transaction_cost"] = (
            data["turnover"] * self.transaction_cost
        )

        data["strategy_return"] = (
            data["position"] * data["actual_return"]
            - data["transaction_cost"]
        )

        data["equity"] = (
            1 + data["strategy_return"]
        ).cumprod()

        metrics = self.calculate_metrics(
            data["strategy_return"],
            data["turnover"].sum()
        )

        return data, metrics

    def run_buy_and_hold(self, actual_returns):

        returns = actual_returns.dropna()

        equity = (1 + returns).cumprod()

        metrics = self.calculate_metrics(
            returns,
            trades=1
        )

        return equity, metrics