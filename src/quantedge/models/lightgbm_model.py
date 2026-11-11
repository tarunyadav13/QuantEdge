from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from lightgbm import LGBMClassifier


class LightGBMModel:

    def __init__(self, random_state=42):

        self.random_state = random_state

        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),

            ("classifier", LGBMClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.03,
                num_leaves=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                class_weight="balanced",
                random_state=random_state,
                verbosity=-1,
                n_jobs=-1
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