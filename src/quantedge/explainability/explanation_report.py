
import pandas as pd


class ExplanationReport:

    @staticmethod
    def generate(
        ticker,
        probability_up,
        explanation_df
    ):

        prediction = (
            "BULLISH"
            if probability_up >= 0.5
            else "BEARISH"
        )

        positive = explanation_df[
            explanation_df["SHAP Value"] > 0
        ].head(5)

        negative = explanation_df[
            explanation_df["SHAP Value"] < 0
        ].head(5)

        report = {

            "Ticker": ticker,

            "Probability Up": float(
                probability_up
            ),

            "Prediction": prediction,

            "Positive Contributors": (
                positive[
                    ["Feature", "SHAP Value"]
                ].to_dict("records")
            ),

            "Negative Contributors": (
                negative[
                    ["Feature", "SHAP Value"]
                ].to_dict("records")
            )

        }

        return report
