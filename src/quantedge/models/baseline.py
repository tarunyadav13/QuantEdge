import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:

    def __init__(self, random_state=42):

        self.random_state = random_state

        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),

            ("scaler", StandardScaler()),

            ("classifier", LogisticRegression(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state
            ))
        ])

    def fit(self, X, y):

        self.model.fit(X, y)

        return self

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)

    def get_model(self):

        return self.model