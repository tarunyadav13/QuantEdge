import argparse
import logging
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src")
)

import pandas as pd

from quantedge.models.trainer import ModelTrainer

from quantedge.features.feature_pipeline import (
    FeaturePipeline
)

from quantedge.data.market_data import (
    MarketDataService
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def load_stock(ticker):

    service = MarketDataService()

    return service.load_stock_data(
        ticker=ticker
    )


def main():

    parser = argparse.ArgumentParser(
        description="QuantEdge ML Training Engine"
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["RELIANCE.NS"]
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=60
    )

    args = parser.parse_args()

    trainer = ModelTrainer(
        test_size=0.20,
        lookback=args.lookback
    )

    pipeline = FeaturePipeline()

    for ticker in args.tickers:

        print("\n" + "=" * 60)

        print(
            f"QUANTEDGE MODEL TRAINING: {ticker}"
        )

        print("=" * 60)

        try:

            data = load_stock(ticker)

            benchmark = None

            try:

                benchmark = load_stock(
                    "NIFTYBEES.NS"
                )

            except Exception:

                logging.warning(
                    "Benchmark unavailable. "
                    "Continuing without benchmark features."
                )

            features = pipeline.build_features(
                data,
                benchmark_df=benchmark
            )

            leaderboard = trainer.train(
                features,
                ticker
            )

            print("\nMODEL PERFORMANCE LEADERBOARD\n")

            print(
                leaderboard.to_string(
                    index=False
                )
            )

        except Exception as error:

            logging.exception(
                "Training failed for %s: %s",
                ticker,
                error
            )


if __name__ == "__main__":
    main()