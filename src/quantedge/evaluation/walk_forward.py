import numpy as np


class WalkForwardSplitter:

    def __init__(
        self,
        n_splits=5,
        min_train_size=500,
        test_size=None,
        gap=1
    ):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X):

        n_samples = len(X)

        if self.test_size is None:

            available = (
                n_samples
                - self.min_train_size
                - self.gap
            )

            test_size = available // self.n_splits

        else:
            test_size = self.test_size

        if test_size < 1:
            raise ValueError(
                "Insufficient data for walk-forward validation. "
                "Reduce min_train_size or n_splits."
            )

        for fold in range(self.n_splits):

            train_end = (
                self.min_train_size
                + fold * test_size
            )

            validation_start = train_end + self.gap

            validation_end = (
                validation_start + test_size
            )

            if validation_end > n_samples:
                break

            train_indices = np.arange(0, train_end)

            validation_indices = np.arange(
                validation_start,
                validation_end
            )

            yield train_indices, validation_indices