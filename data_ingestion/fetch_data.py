"""CLI entrypoint for fetching OHLCV data from crypto exchanges and stock APIs."""

import argparse
import sys

from .crypto_fetcher import CryptoFetcher, SUPPORTED_EXCHANGES
from .nasdaq_fetcher import NasdaqFetcher
from .data_validator import DataValidator


def main():
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV data for Kronos fine-tuning and prediction."
    )
    parser.add_argument(
        "--source", required=True,
        choices=["coinbase", "kraken", "bybit", "okx", "nasdaq"],
        help="Data source: crypto exchange name or 'nasdaq' for stocks/ETFs",
    )
    parser.add_argument("--symbol", required=True, help="Trading pair (BTC/USD) or ticker (AAPL)")
    parser.add_argument("--timeframe", default="1h", help="Candle interval: 1m, 5m, 15m, 1h, 4h, 1d")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Defaults to now.")
    parser.add_argument("--out", default="./data/", help="Output file or directory path")
    parser.add_argument("--validate", action="store_true", help="Validate data after fetching")

    args = parser.parse_args()

    # Fetch data
    if args.source == "nasdaq":
        fetcher = NasdaqFetcher()
        df = fetcher.fetch_ohlcv(args.symbol, interval=args.timeframe, start_date=args.start, end_date=args.end)
        path = fetcher.save_csv(df, args.out)
    else:
        fetcher = CryptoFetcher(exchange_name=args.source)
        path = fetcher.fetch_and_save(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            output_path=args.out,
        )

    # Validate if requested
    if args.validate:
        print("\nValidating fetched data...")
        validator = DataValidator(expected_timeframe=args.timeframe)
        report = validator.validate(path)
        validator.print_report(report)
        if not report["valid"]:
            print("\nValidation FAILED — see errors above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
