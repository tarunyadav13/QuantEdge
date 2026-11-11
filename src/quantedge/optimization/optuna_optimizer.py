import optuna
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class OptunaOptimizer:

    def __init__(
        self,
        n_trials=50,
        n_splits=5,
        random_state=42
    ):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state

        self.best_params = {}
        self.study_results = {}

    def create_model(self, model_name, params):

        if model_name == "RandomForest":
            return RandomForestClassifier(
                **params,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced"
            )

        if model_name == "XGBoost":
            return XGBClassifier(
                **params,
                random_state=self.random_state,
                eval_metric="logloss",
                n_jobs=-1
            )

        if model_name == "LightGBM":
            return LGBMClassifier(
                **params,
                random_state=self.random_state,
                verbosity=-1,
                class_weight="balanced",
                n_jobs=-1
            )

        raise ValueError(f"Unknown model: {model_name}")

    def suggest_params(self, trial, model_name):

        if model_name == "RandomForest":

            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 200, 800
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 3, 20
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 20
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 10
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                )
            }

        if model_name == "XGBoost":

            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 200, 800
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 2, 8
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.2, log=True
                ),
                "subsample": trial.suggest_float(
                    "subsample", 0.6, 1.0
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-4, 10, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-3, 20, log=True
                )
            }

        if model_name == "LightGBM":

            return {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 200, 800
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 3, 12
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.2, log=True
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves", 10, 80
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 5, 50
                ),
                "subsample": trial.suggest_float(
                    "subsample", 0.6, 1.0
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-3, 20, log=True
                )
            }

        raise ValueError(model_name)

    def objective(self, trial, model_name, X, y):

        params = self.suggest_params(trial, model_name)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        scores = []

        for train_idx, val_idx in tscv.split(X):

            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # Replace infinite values.
            X_train = X_train.replace(
                [np.inf, -np.inf], np.nan
            )

            X_val = X_val.replace(
                [np.inf, -np.inf], np.nan
            )

            # Remove columns entirely missing in training.
            valid_columns = X_train.columns[
                X_train.notna().any(axis=0)
            ]

            X_train = X_train[valid_columns]
            X_val = X_val[valid_columns]

            # Fold-local imputation.
            imputer = SimpleImputer(strategy="median")

            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)

            model = self.create_model(
                model_name,
                params
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            score = balanced_accuracy_score(
                y_val,
                predictions
            )

            scores.append(score)

        return float(np.mean(scores))

    def optimize(self, model_name, X, y):

        print(f"\nOptimizing {model_name}...")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=self.random_state
            )
        )

        study.optimize(
            lambda trial: self.objective(
                trial,
                model_name,
                X,
                y
            ),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params[model_name] = study.best_params
        self.study_results[model_name] = study

        print(f"\n{model_name} optimization completed.")
        print("Best balanced accuracy:", study.best_value)
        print("Best parameters:", study.best_params)

        return study.best_params

    def optimize_all(self, X, y):

        models = [
            "RandomForest",
            "XGBoost",
            "LightGBM"
        ]

        for model_name in models:

            self.optimize(
                model_name,
                X,
                y
            )

        return self.best_params