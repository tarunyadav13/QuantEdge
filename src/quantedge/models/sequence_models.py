import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


HORIZONS = [1, 5, 10]


def build_lstm_model(
    sequence_length,
    n_features,
    learning_rate=0.001
):
    """
    Multi-horizon LSTM classifier.

    Input:
        (batch, 60, features)

    Output:
        Probability of upward movement over
        1, 5 and 10 trading days.
    """

    inputs = layers.Input(
        shape=(sequence_length, n_features)
    )

    x = layers.LSTM(
        64,
        return_sequences=True
    )(inputs)

    x = layers.Dropout(0.25)(x)

    x = layers.LSTM(
        32,
        return_sequences=False
    )(x)

    x = layers.Dropout(0.25)(x)

    x = layers.Dense(
        32,
        activation="relu"
    )(x)

    x = layers.Dropout(0.15)(x)

    outputs = layers.Dense(
        3,
        activation="sigmoid",
        name="direction_probabilities"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy"
            )
        ]
    )

    return model


def build_gru_model(
    sequence_length,
    n_features,
    learning_rate=0.001
):
    """
    Multi-horizon GRU classifier.
    """

    inputs = layers.Input(
        shape=(sequence_length, n_features)
    )

    x = layers.GRU(
        64,
        return_sequences=True
    )(inputs)

    x = layers.Dropout(0.25)(x)

    x = layers.GRU(
        32,
        return_sequences=False
    )(x)

    x = layers.Dropout(0.25)(x)

    x = layers.Dense(
        32,
        activation="relu"
    )(x)

    x = layers.Dropout(0.15)(x)

    outputs = layers.Dense(
        3,
        activation="sigmoid",
        name="direction_probabilities"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy"
            )
        ]
    )

    return model


def get_training_callbacks():
    """
    Callbacks for controlling overfitting.
    """

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=0.00001,
        verbose=1
    )

    return [
        early_stopping,
        reduce_lr
    ]