import numpy as np


class HybridEnsemble:
    """
    Equal-weight probability ensemble for Phase 11 models.

    Initial base models:
        - LSTM
        - GRU
        - RandomForest

    Each model must provide probabilities for the same observations
    and prediction horizon.
    """

    MODEL_NAMES = ["LSTM", "GRU", "RandomForest"]

    def __init__(self, weights=None):

        if weights is None:
            weights = {
                "LSTM": 1 / 3,
                "GRU": 1 / 3,
                "RandomForest": 1 / 3
            }

        if set(weights) != set(self.MODEL_NAMES):
            raise ValueError(
                f"Weights must contain exactly: {self.MODEL_NAMES}"
            )

        if any(weight < 0 for weight in weights.values()):
            raise ValueError("Weights cannot be negative.")

        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Ensemble weights must sum to 1.")

        self.weights = weights

    def predict_proba(self, predictions):
        """
        predictions:
        {
            "LSTM": numpy array,
            "GRU": numpy array,
            "RandomForest": numpy array
        }
        """

        if set(predictions) != set(self.MODEL_NAMES):
            raise ValueError(
                f"Expected predictions from {self.MODEL_NAMES}"
            )

        arrays = {}

        for model_name in self.MODEL_NAMES:

            values = np.asarray(
                predictions[model_name],
                dtype=float
            )

            if values.ndim != 1:
                raise ValueError(
                    f"{model_name} probabilities must be one-dimensional."
                )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{model_name} contains invalid probabilities."
                )

            if ((values < 0) | (values > 1)).any():
                raise ValueError(
                    f"{model_name} probabilities must be between 0 and 1."
                )

            arrays[model_name] = values

        lengths = {len(values) for values in arrays.values()}

        if len(lengths) != 1:
            raise ValueError(
                "All models must predict the same number of observations."
            )

        ensemble_probability = np.zeros(
            len(arrays["LSTM"]),
            dtype=float
        )

        for model_name in self.MODEL_NAMES:

            ensemble_probability += (
                self.weights[model_name]
                * arrays[model_name]
            )

        return np.clip(ensemble_probability, 0, 1)

    @staticmethod
    def predict_direction(probabilities, threshold=0.5):

        probabilities = np.asarray(probabilities)

        return (probabilities >= threshold).astype(int)