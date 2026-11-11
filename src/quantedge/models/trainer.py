import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone

from quantedge.config import PROJECT_ROOT

from quantedge.models.baseline import (
    LogisticRegressionModel
)

from quantedge.models.random_forest import (
    RandomForestModel
)

from quantedge.models.xgboost_model import (
    XGBoostModel
)

from quantedge.models.lightgbm_model import (
    LightGBMModel
)

from quantedge.models.lstm import LSTMModel

from quantedge.models.gru import GRUModel

from quantedge.models.ensemble import EnsembleModel

from quantedge.evaluation.metrics import (
    calculate_classification_metrics,
    create_leaderboard
)

logger = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(
        self,
        test_size=0.20,
        lookback=60,
        random_state=42
    ):

        self.test_size = test_size

        self.lookback = lookback

        self.random_state = random_state

        self.artifact_dir = (
            PROJECT_ROOT / "artifacts" / "models"
        )

        self.artifact_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def prepare_dataset(self, df):

        df = df.copy()

        if "target_direction" not in df.columns:
            raise ValueError(
                "target_direction missing. "
                "Run FeaturePipeline first."
            )

        excluded = {
            "target_direction",
            "target_market_regime",
            "target_return"
        }

        feature_columns = [
            col for col in df.columns
            if col not in excluded
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        X = df[feature_columns].replace(
            [np.inf, -np.inf],
            np.nan
        )

        y = df["target_direction"]

        valid = y.notna()

        X = X.loc[valid]

        y = y.loc[valid].astype(int)

        valid_features = X.notna().any(axis=0)

        X = X.loc[:, valid_features]

        if X.shape[1] == 0:
            raise ValueError(
                "No usable features available."
            )

        return X, y

    def split_data(self, X, y):

        split_index = int(
            len(X) * (1 - self.test_size)
        )

        X_train = X.iloc[:split_index]

        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]

        y_test = y.iloc[split_index:]

        if len(X_train) <= self.lookback:
            raise ValueError(
                "Insufficient training data for sequence models."
            )

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

    def get_models(self):

        return {

            "LogisticRegression":
                LogisticRegressionModel(
                    self.random_state
                ),

            "RandomForest":
                RandomForestModel(
                    self.random_state
                ),

            "XGBoost":
                XGBoostModel(
                    self.random_state
                ),

            "LightGBM":
                LightGBMModel(
                    self.random_state
                ),

            "LSTM":
                LSTMModel(
                    lookback=self.lookback,
                    random_state=self.random_state
                ),

            "GRU":
                GRUModel(
                    lookback=self.lookback,
                    random_state=self.random_state
                )

        }

    def save_model(
        self,
        model_name,
        model,
        ticker
    ):

        ticker_dir = (
            self.artifact_dir /
            ticker.replace(".", "_")
        )

        ticker_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if model_name in ["LSTM", "GRU"]:

            model_path = (
                ticker_dir /
                f"{model_name.lower()}.keras"
            )

            model.save(model_path)

            joblib.dump(
                {
                    "imputer": model.imputer,
                    "scaler": model.scaler,
                    "lookback": model.lookback
                },
                ticker_dir /
                f"{model_name.lower()}_preprocessing.joblib"
            )

        else:

            model_path = (
                ticker_dir /
                f"{model_name.lower()}.joblib"
            )

            joblib.dump(
                model.get_model(),
                model_path
            )

        return model_path

    def train(self, df, ticker):

        logger.info(
            "Starting model training for %s",
            ticker
        )

        X, y = self.prepare_dataset(df)

        X_train, X_test, y_train, y_test = (
            self.split_data(X, y)
        )

        models = self.get_models()

        results = []

        trained_models = {}

        prediction_store = {}

        for name, model in models.items():

            logger.info(
                "Training model: %s",
                name
            )

            try:

                if name in ["LSTM", "GRU"]:

                    model.fit(
                        X_train,
                        y_train
                    )

                    # Include training context for the
                    # first test sequence.
                    prediction_input = pd.concat([
                        X_train.tail(self.lookback),
                        X_test
                    ])

                    probabilities = model.predict_proba(
                        prediction_input
                    )

                    # Sequence model skips the first
                    # lookback rows.
                    expected_length = len(X_test)

                    if len(probabilities) != expected_length:
                        raise ValueError(
                            f"{name} prediction alignment error."
                        )

                    predictions = (
                        probabilities[:, 1] >= 0.5
                    ).astype(int)

                    y_evaluation = y_test

                else:

                    model.fit(
                        X_train,
                        y_train
                    )

                    probabilities = model.predict_proba(
                        X_test
                    )

                    predictions = model.predict(
                        X_test
                    )

                    y_evaluation = y_test

                metrics = calculate_classification_metrics(
                    y_evaluation,
                    predictions,
                    probabilities
                )

                result = {
                    "model": name,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "confusion_matrix"
                    }
                }

                results.append(result)

                prediction_store[name] = probabilities[:, 1]

                trained_models[name] = model

                self.save_model(
                    name,
                    model,
                    ticker
                )

                logger.info(
                    "%s completed. Balanced Accuracy: %.4f",
                    name,
                    metrics["balanced_accuracy"]
                )

            except Exception as error:

                logger.exception(
                    "Model %s failed: %s",
                    name,
                    error
                )

        # Build ensemble from the four classical models.
        ensemble_names = [
            "LogisticRegression",
            "RandomForest",
            "XGBoost",
            "LightGBM"
        ]

        available = [
            trained_models[name]
            for name in ensemble_names
            if name in trained_models
        ]

        if len(available) >= 2:

            try:

                ensemble = EnsembleModel(
                    available
                )

                probabilities = (
                    ensemble.predict_proba(X_test)
                )

                predictions = ensemble.predict(
                    X_test
                )

                metrics = calculate_classification_metrics(
                    y_test,
                    predictions,
                    probabilities
                )

                results.append({
                    "model": "Ensemble",
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "confusion_matrix"
                    }
                })

                ticker_dir = (
                    self.artifact_dir /
                    ticker.replace(".", "_")
                )

                joblib.dump(
                    ensemble,
                    ticker_dir / "ensemble.joblib"
                )

            except Exception:

                logger.exception(
                    "Ensemble training failed."
                )

        leaderboard = create_leaderboard(
            results
        )

        output_dir = (
            self.artifact_dir /
            ticker.replace(".", "_")
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        leaderboard.to_csv(
            output_dir / "model_leaderboard.csv",
            index=False
        )

        report = {
            "ticker": ticker,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "features": list(X.columns),
            "feature_count": len(X.columns),
            "class_distribution": {
                str(key): int(value)
                for key, value in y.value_counts().items()
            },
            "leaderboard": results
        }

        with open(
            output_dir / "training_report.json",
            "w"
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                default=str
            )

        logger.info(
            "Training completed for %s",
            ticker
        )

        return leaderboard