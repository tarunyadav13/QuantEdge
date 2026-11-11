import yfinance as yf
import pandas as pd

from quantedge.utils.logger import get_logger

logger = get_logger(__name__)


class YahooFinanceProvider:

    REQUIRED_COLUMNS = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    def fetch_history(
        self,
        ticker: str,
        period: str = "5y",
        interval: str = "1d"
    ) -> pd.DataFrame:

        logger.info(
            f"Fetching market data for {ticker}"
        )

        try:

            stock = yf.Ticker(ticker)

            data = stock.history(
                period=period,
                interval=interval,
                auto_adjust=True,
                actions=True
            )

            if data.empty:
                raise ValueError(
                    f"No market data available for {ticker}"
                )

            # Remove timezone information while preserving
            # the actual market-date representation.
            data.index = pd.DatetimeIndex(data.index)

            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            data.index.name = "Date"

            # Ensure required columns exist.
            missing_columns = [
                col for col in self.REQUIRED_COLUMNS
                if col not in data.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing columns: {missing_columns}"
                )

            data = data[
                self.REQUIRED_COLUMNS + [
                    col for col in ["Dividends", "Stock Splits"]
                    if col in data.columns
                ]
            ]

            data = data.sort_index()

            data = data[
                ~data.index.duplicated(keep="last")
            ]

            logger.info(
                f"Successfully fetched {len(data)} rows for {ticker}"
            )

            return data

        except Exception as error:

            logger.exception(
                f"Data fetching failed for {ticker}"
            )

            raise RuntimeError(
                f"Unable to fetch data: {error}"
            ) from error