import argparse
import json

from quantedge.config import (
    settings,
    create_directories
)

from quantedge.data.market_data import (
    MarketDataService
)


def main():

    parser = argparse.ArgumentParser(
        description="QuantEdge Market Data Downloader"
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=[settings.DEFAULT_TICKER]
    )

    parser.add_argument(
        "--period",
        default=settings.DEFAULT_PERIOD
    )

    parser.add_argument(
        "--interval",
        default=settings.DEFAULT_INTERVAL
    )

    args = parser.parse_args()

    create_directories()

    service = MarketDataService()

    results = {}

    for ticker in args.tickers:

        try:

            data = service.fetch_stock_data(
                ticker=ticker,
                period=args.period,
                interval=args.interval
            )

            results[ticker] = {
                "status": "SUCCESS",
                "rows": len(data),
                "start": str(data.index.min()),
                "end": str(data.index.max()),
                "latest_close": float(
                    data["Close"].iloc[-1]
                )
            }

        except Exception as error:

            results[ticker] = {
                "status": "FAILED",
                "error": str(error)
            }

    print("\nQUANTEDGE DATA INGESTION RESULTS\n")

    print(
        json.dumps(
            results,
            indent=4
        )
    )


if __name__ == "__main__":
    main()