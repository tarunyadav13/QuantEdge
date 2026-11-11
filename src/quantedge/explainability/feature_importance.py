
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance


class FeatureImportanceAnalyzer:

    @staticmethod
    def native_importance(model, feature_names):

        if not hasattr(model, "feature_importances_"):
            raise ValueError(
                "Model does not support native feature importance."
            )

        result = pd.DataFrame({

            "Feature": feature_names,

            "Native Importance": (
                model.feature_importances_
            )

        })

        return result.sort_values(
            "Native Importance",
            ascending=False
        )

    @staticmethod
    def permutation_importance_analysis(
        model,
        X,
        y,
        feature_names,
        n_repeats=10
    ):

        result = permutation_importance(

            model,
            X,
            y,

            n_repeats=n_repeats,

            random_state=42,

            scoring="balanced_accuracy"

        )

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Permutation Importance": (
                result.importances_mean
            ),

            "Importance Std": (
                result.importances_std
            )

        })

        return importance.sort_values(
            "Permutation Importance",
            ascending=False
        )

    @staticmethod
    def combine_importance(
        shap_importance,
        native_importance,
        permutation_importance
    ):

        result = shap_importance.merge(
            native_importance,
            on="Feature",
            how="outer"
        )

        result = result.merge(
            permutation_importance,
            on="Feature",
            how="outer"
        )

        return result.fillna(0).sort_values(
            "Mean Absolute SHAP",
            ascending=False
        )
