from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


class XGBoostModel:

    def __init__(self, random_state=42):

        self.random_state = random_state

        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),

            ("classifier", XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
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