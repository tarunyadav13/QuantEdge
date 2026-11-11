import numpy as np
import pandas as pd

from quantedge.utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataValidator:

    def validate(self, data: pd.DataFrame):

        if data.empty:
            raise ValueError("Dataset is empty.")

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing = [
            col for col in required
            if col not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        if not data.index.is_monotonic_increasing:
            raise ValueError(
                "Dates are not chronologically ordered."
            )

        if data.index.has_duplicates:
            raise ValueError(
                "Duplicate dates detected."
            )

        if data[required].isnull().any().any():
            raise ValueError(
                "Missing OHLCV values detected."
            )

        numeric_data = data[required].to_numpy(
            dtype=float
        )

        if not np.isfinite(numeric_data).all():
            raise ValueError(
                "Infinite or invalid numerical values detected."
            )

        price_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        if (data[price_columns] <= 0).any().any():
            raise ValueError(
                "Non-positive stock prices detected."
            )

        if (data["Volume"] < 0).any():
            raise ValueError(
                "Negative trading volume detected."
            )

        invalid_high = (
            data["High"] <
            data[["Open", "Close", "Low"]].max(axis=1)
        )

        invalid_low = (
            data["Low"] >
            data[["Open", "Close", "High"]].min(axis=1)
        )

        if invalid_high.any() or invalid_low.any():
            raise ValueError(
                "Invalid OHLC relationships detected."
            )

        logger.info(
            "Market data validation successful."
        )

        return True

    def get_quality_report(
        self,
        data: pd.DataFrame
    ):

        report = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "missing_values": int(
                data.isnull().sum().sum()
            ),
            "duplicate_dates": int(
                data.index.duplicated().sum()
            ),
            "start_date": str(data.index.min()),
            "end_date": str(data.index.max()),
            "memory_usage_mb": round(
                data.memory_usage(deep=True).sum()
                / (1024 ** 2),
                3
            )
        }

        return report