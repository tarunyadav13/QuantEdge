import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    GRU,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


class GRUModel:

    def __init__(
        self,
        lookback=60,
        epochs=40,
        batch_size=32,
        random_state=42
    ):

        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size

        self.random_state = random_state

        self.imputer = SimpleImputer(
            strategy="median"
        )

        self.scaler = StandardScaler()

        self.model = None

    def prepare_sequences(self, X, y):

        X_sequences = []
        y_sequences = []

        for i in range(self.lookback, len(X)):

            X_sequences.append(
                X[i - self.lookback:i]
            )

            y_sequences.append(y[i])

        return (
            np.asarray(X_sequences),
            np.asarray(y_sequences)
        )

    def fit(self, X, y):

        tf.keras.utils.set_random_seed(
            self.random_state
        )

        X = np.asarray(X, dtype=np.float32)

        y = np.asarray(y, dtype=np.float32)

        X = self.imputer.fit_transform(X)

        X = self.scaler.fit_transform(X)

        X_sequences, y_sequences = (
            self.prepare_sequences(X, y)
        )

        if len(X_sequences) == 0:
            raise ValueError(
                "Insufficient training data for GRU."
            )

        positive = np.sum(y_sequences == 1)
        negative = np.sum(y_sequences == 0)

        class_weights = {
            0: len(y_sequences) / (2 * max(negative, 1)),
            1: len(y_sequences) / (2 * max(positive, 1))
        }

        self.model = Sequential([

            Input(
                shape=(
                    self.lookback,
                    X.shape[1]
                )
            ),

            GRU(
                64,
                return_sequences=True
            ),

            BatchNormalization(),

            Dropout(0.2),

            GRU(32),

            Dropout(0.2),

            Dense(16, activation="relu"),

            Dense(1, activation="sigmoid")

        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = [

            EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True
            ),

            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )

        ]

        validation_split = 0.15 if len(X_sequences) >= 100 else 0.0

        self.model.fit(
            X_sequences,
            y_sequences,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            shuffle=False,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        return self

    def _prepare_prediction_sequences(self, X):

        X = np.asarray(X, dtype=np.float32)

        X = self.imputer.transform(X)

        X = self.scaler.transform(X)

        X_sequences = []

        for i in range(self.lookback, len(X)):

            X_sequences.append(
                X[i - self.lookback:i]
            )

        return np.asarray(X_sequences)

    def predict_proba(self, X):

        sequences = self._prepare_prediction_sequences(X)

        if len(sequences) == 0:
            return np.empty((0, 2))

        probabilities = self.model.predict(
            sequences,
            verbose=0
        ).reshape(-1)

        return np.column_stack([
            1 - probabilities,
            probabilities
        ])

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (
            probabilities[:, 1] >= 0.5
        ).astype(int)

    def save(self, path):

        self.model.save(path)