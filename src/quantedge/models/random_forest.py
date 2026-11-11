from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:

    def __init__(self, random_state=42):

        self.random_state = random_state

        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),

            ("classifier", RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
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