import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    log_loss,
    brier_score_loss
)


def calculate_classification_metrics(
    y_true,
    y_pred,
    y_prob
):

    y_true = np.asarray(y_true)

    y_pred = np.asarray(y_pred)

    y_prob = np.asarray(y_prob)

    positive_prob = y_prob[:, 1]

    try:
        auc = roc_auc_score(
            y_true,
            positive_prob
        )
    except ValueError:
        auc = np.nan

    try:
        loss = log_loss(
            y_true,
            y_prob,
            labels=[0, 1]
        )
    except ValueError:
        loss = np.nan

    try:
        brier = brier_score_loss(
            y_true,
            positive_prob
        )
    except ValueError:
        brier = np.nan

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    return {

        "accuracy": accuracy_score(
            y_true, y_pred
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "roc_auc": auc,

        "log_loss": loss,

        "brier_score": brier,

        "confusion_matrix": cm.tolist()

    }


def create_leaderboard(results):

    leaderboard = pd.DataFrame(results)

    if leaderboard.empty:
        return leaderboard

    leaderboard = leaderboard.sort_values(
        by="balanced_accuracy",
        ascending=False
    )

    return leaderboard.reset_index(drop=True)