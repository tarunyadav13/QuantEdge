from pathlib import Path

import yfinance as yf
import pandas as pd


class BenchmarkService:

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[3]

        self.raw_dir = (
            self.project_root / "data" / "raw"
        )

        self.raw_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_benchmark(
        self,
        ticker="NIFTYBEES.NS",
        period="5y"
    ):

        filename = (
            ticker.replace(".", "_")
            + "_1d.csv"
        )

        filepath = self.raw_dir / filename

        if filepath.exists():

            df = pd.read_csv(
                filepath,
                index_col=0,
                parse_dates=True
            )

            df.index = pd.to_datetime(
                df.index
            )

            return df.sort_index()

        print(
            f"Downloading benchmark data: {ticker}"
        )

        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            raise ValueError(
                f"Unable to download benchmark: {ticker}"
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"

        df.to_csv(filepath)

        print(
            f"Benchmark saved: {filepath}"
        )

        return df.sort_index()