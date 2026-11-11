import numpy as np


class EnsembleModel:

    def __init__(self, models, weights=None):

        self.models = models

        if weights is None:

            weights = np.ones(
                len(models)
            )

        weights = np.asarray(
            weights,
            dtype=float
        )

        if len(weights) != len(models):
            raise ValueError(
                "Number of weights must match models."
            )

        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError(
                "Weights must be non-negative with a positive sum."
            )

        self.weights = weights / weights.sum()

    def predict_proba(self, X):

        probabilities = []

        for model in self.models:

            probabilities.append(
                model.predict_proba(X)[:, 1]
            )

        probabilities = np.asarray(
            probabilities
        )

        positive_probability = np.average(
            probabilities,
            axis=0,
            weights=self.weights
        )

        return np.column_stack([
            1 - positive_probability,
            positive_probability
        ])

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (
            probabilities[:, 1] >= 0.5
        ).astype(int)