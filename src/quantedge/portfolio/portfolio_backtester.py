
import pandas as pd
import numpy as np


class PortfolioBacktester:

    def __init__(
        self,
        transaction_cost=0.001,
        periods_per_year=252
    ):

        self.transaction_cost = transaction_cost
        self.periods_per_year = periods_per_year

    def run(
        self,
        asset_returns,
        weights,
        rebalance="monthly"
    ):

        asset_returns = asset_returns[
            weights.index
        ].dropna()

        if asset_returns.empty:
            raise ValueError("No aligned returns available.")

        if rebalance == "monthly":

            groups = (
                asset_returns.index
                .to_period("M")
            )

        elif rebalance == "quarterly":

            groups = (
                asset_returns.index
                .to_period("Q")
            )

        elif rebalance == "none":

            groups = pd.Series(
                0,
                index=asset_returns.index
            )

        else:

            raise ValueError(
                "Invalid rebalance frequency."
            )

        portfolio_returns = []

        previous_weights = None

        for _, group in asset_returns.groupby(groups):

            if previous_weights is None:

                current_weights = weights.copy()

            else:

                current_weights = previous_weights.copy()

            first_date = group.index[0]

            if previous_weights is not None:

                turnover = (
                    current_weights - previous_weights
                ).abs().sum() / 2

                cost = (
                    turnover * self.transaction_cost
                )

            else:

                turnover = 0
                cost = 0

            for date, row in group.iterrows():

                daily_return = np.dot(
                    current_weights.values,
                    row.values
                )

                if date == first_date:
                    daily_return -= cost

                portfolio_returns.append(
                    (date, daily_return)
                )

                # Drift weights based on realized returns.
                updated_values = (
                    current_weights.values *
                    (1 + row.values)
                )

                total_value = updated_values.sum()

                if total_value > 0:

                    current_weights = pd.Series(
                        updated_values / total_value,
                        index=weights.index
                    )

            previous_weights = current_weights.copy()

        result = pd.Series(
            dict(portfolio_returns),
            name="Portfolio Return"
        ).sort_index()

        return result
