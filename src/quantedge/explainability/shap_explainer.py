
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


class SHAPExplainer:

    def __init__(
        self,
        model,
        feature_names,
        output_dir="artifacts/explainability"
    ):

        self.model = model
        self.feature_names = list(feature_names)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.explainer = None
        self.shap_values = None
        self.explanation = None

    def _prepare_data(self, X):

        X = X.copy()

        X = X[self.feature_names]

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return X

    def calculate_shap_values(self, X):

        X = self._prepare_data(X)

        self.explainer = shap.TreeExplainer(
            self.model
        )

        self.explanation = self.explainer(
            X
        )

        self.shap_values = self.explanation.values

        # Binary classifiers may return explanations
        # for both classes. Explain upward movement.
        if self.shap_values.ndim == 3:

            self.shap_values = (
                self.shap_values[:, :, 1]
            )

        self.X_explained = X

        return self.shap_values

    def global_importance(self):

        if self.shap_values is None:
            raise RuntimeError(
                "Calculate SHAP values first."
            )

        importance = np.abs(
            self.shap_values
        ).mean(axis=0)

        result = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean Absolute SHAP": importance
        })

        return result.sort_values(
            "Mean Absolute SHAP",
            ascending=False
        )

    def local_explanation(self, index=0):

        if self.shap_values is None:
            raise RuntimeError(
                "Calculate SHAP values first."
            )

        if not 0 <= index < len(self.X_explained):
            raise IndexError(
                "Explanation index out of range."
            )

        values = self.shap_values[index]

        explanation = pd.DataFrame({

            "Feature": self.feature_names,

            "Feature Value": (
                self.X_explained
                .iloc[index]
                .values
            ),

            "SHAP Value": values

        })

        explanation["Absolute SHAP"] = (
            explanation["SHAP Value"].abs()
        )

        return explanation.sort_values(
            "Absolute SHAP",
            ascending=False
        )

    def save_summary_plot(self):

        shap.summary_plot(
            self.shap_values,
            self.X_explained,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "shap_summary.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

    def save_bar_plot(self):

        shap.summary_plot(
            self.shap_values,
            self.X_explained,
            plot_type="bar",
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            self.output_dir / "shap_bar.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

    def save_waterfall_plot(self, index=0):

        if self.explanation is None:
            raise RuntimeError(
                "Calculate SHAP values first."
            )

        values = self.explanation.values

        if values.ndim == 3:

            base_values = self.explanation.base_values

            explanation = shap.Explanation(
                values=values[index, :, 1],
                base_values=base_values[index, 1],
                data=self.X_explained.iloc[index].values,
                feature_names=self.feature_names
            )

        else:

            explanation = self.explanation[index]

        plt.figure(figsize=(12, 7))

        shap.plots.waterfall(
            explanation,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            self.output_dir / f"shap_waterfall_{index}.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

    def save_dependence_plot(self, feature):

        if feature not in self.feature_names:
            raise ValueError(
                f"Unknown feature: {feature}"
            )

        shap.dependence_plot(
            feature,
            self.shap_values,
            self.X_explained,
            show=False
        )

        plt.tight_layout()

        safe_name = feature.replace("/", "_")

        plt.savefig(
            self.output_dir /
            f"dependence_{safe_name}.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()
