"""CLI entrypoint for validating CSV data files for Kronos compatibility."""

import argparse
import sys

from .data_validator import DataValidator


def main():
    parser = argparse.ArgumentParser(
        description="Validate CSV data for Kronos fine-tuning and prediction."
    )
    parser.add_argument("--input", required=True, help="Path to the CSV file to validate")
    parser.add_argument(
        "--timeframe", default=None,
        help="Expected candle timeframe for gap detection: 1m, 5m, 15m, 1h, 4h, 1d",
    )

    args = parser.parse_args()

    validator = DataValidator(expected_timeframe=args.timeframe)
    report = validator.validate(args.input)
    validator.print_report(report)

    if not report["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
