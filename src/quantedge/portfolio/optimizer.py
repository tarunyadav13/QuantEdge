
import numpy as np
import pandas as pd

from scipy.optimize import minimize


class PortfolioOptimizer:

    def __init__(
        self,
        risk_free_rate=0.06,
        max_weight=0.35
    ):

        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight

    def equal_weight(self, assets):

        n = len(assets)

        if n == 0:
            raise ValueError("No assets supplied.")

        weights = np.ones(n) / n

        return pd.Series(
            weights,
            index=assets,
            name="Equal Weight"
        )

    def portfolio_return(
        self,
        weights,
        expected_returns
    ):

        return np.dot(
            weights,
            expected_returns
        )

    def portfolio_volatility(
        self,
        weights,
        covariance
    ):

        return np.sqrt(
            np.dot(
                weights.T,
                np.dot(covariance, weights)
            )
        )

    def optimize(
        self,
        expected_returns,
        covariance,
        objective="sharpe"
    ):

        assets = expected_returns.index

        n = len(assets)

        if n == 0:
            raise ValueError("No assets supplied.")

        if n * self.max_weight < 1:
            raise ValueError(
                "Maximum weight is too restrictive."
            )

        mu = expected_returns.values
        cov = covariance.loc[assets, assets].values

        initial_weights = np.ones(n) / n

        bounds = tuple(
            (0.0, self.max_weight)
            for _ in range(n)
        )

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1
            }
        ]

        def portfolio_volatility(w):

            variance = w.T @ cov @ w

            return np.sqrt(max(variance, 0))

        def negative_sharpe(w):

            returns = np.dot(w, mu)

            volatility = portfolio_volatility(w)

            if volatility <= 1e-12:
                return 1e6

            return -(
                returns - self.risk_free_rate
            ) / volatility

        def variance(w):

            return w.T @ cov @ w

        if objective == "sharpe":

            objective_function = negative_sharpe

        elif objective == "min_variance":

            objective_function = variance

        else:

            raise ValueError(
                "Objective must be sharpe or min_variance."
            )

        result = minimize(
            objective_function,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 1000,
                "ftol": 1e-10
            }
        )

        if not result.success:
            raise RuntimeError(
                f"Optimization failed: {result.message}"
            )

        weights = np.clip(result.x, 0, self.max_weight)

        weights = weights / weights.sum()

        return pd.Series(
            weights,
            index=assets,
            name=objective
        )

    def minimum_variance(
        self,
        expected_returns,
        covariance
    ):

        return self.optimize(
            expected_returns,
            covariance,
            objective="min_variance"
        )

    def maximum_sharpe(
        self,
        expected_returns,
        covariance
    ):

        return self.optimize(
            expected_returns,
            covariance,
            objective="sharpe"
        )
