from pathlib import Path

import pandas as pd

from quantedge.config import settings
from quantedge.data.providers.yahoo import YahooFinanceProvider
from quantedge.data.validator import MarketDataValidator
from quantedge.utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataService:

    def __init__(self):

        self.provider = YahooFinanceProvider()

        self.validator = MarketDataValidator()

    def fetch_stock_data(
        self,
        ticker: str,
        period: str = "5y",
        interval: str = "1d",
        save: bool = True
    ) -> pd.DataFrame:

        logger.info(
            f"Starting ingestion pipeline: {ticker}"
        )

        data = self.provider.fetch_history(
            ticker=ticker,
            period=period,
            interval=interval
        )

        self.validator.validate(data)

        quality_report = (
            self.validator.get_quality_report(data)
        )

        logger.info(
            f"Data quality report: {quality_report}"
        )

        if save:

            file_name = (
                f"{ticker.replace('.', '_')}_"
                f"{interval}.csv"
            )

            output_path = (
                settings.RAW_DATA_DIR / file_name
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            data.to_csv(output_path)

            logger.info(
                f"Saved dataset: {output_path}"
            )

        return data

    def load_stock_data(
        self,
        ticker: str,
        interval: str = "1d"
    ) -> pd.DataFrame:

        file_name = (
            f"{ticker.replace('.', '_')}_"
            f"{interval}.csv"
        )

        file_path = (
            settings.RAW_DATA_DIR / file_name
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        data = pd.read_csv(
            file_path,
            index_col="Date",
            parse_dates=True
        )

        return data