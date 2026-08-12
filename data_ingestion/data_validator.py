"""Validate OHLCV CSV data for Kronos compatibility — checks format, gaps, NaNs, and outliers."""

import sys
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
OHLC_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


class DataValidator:
    """Validate CSV data for Kronos fine-tuning and prediction pipelines."""

    def __init__(self, expected_timeframe: Optional[str] = None):
        """
        Args:
            expected_timeframe: If provided, checks for gaps in the time series.
                One of: '1m', '5m', '15m', '1h', '4h', '1d'
        """
        self.expected_timeframe = expected_timeframe
        self.timeframe_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }

    def validate(self, csv_path: str) -> dict:
        """
        Validate a CSV file and return a report dict.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            Dict with keys: valid, errors, warnings, stats
        """
        errors = []
        warnings = []
        stats = {}

        # Load
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return {"valid": False, "errors": [f"Failed to read CSV: {e}"], "warnings": [], "stats": {}}

        stats["total_rows"] = len(df)

        # Check required columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return {"valid": False, "errors": errors, "warnings": warnings, "stats": stats}

        # Check timestamps parse
        try:
            ts = pd.to_datetime(df["timestamps"])
        except Exception as e:
            errors.append(f"Failed to parse timestamps: {e}")
            return {"valid": False, "errors": errors, "warnings": warnings, "stats": stats}

        stats["date_range_start"] = str(ts.min())
        stats["date_range_end"] = str(ts.max())
        stats["date_span_days"] = (ts.max() - ts.min()).days

        # Check sorted
        if not ts.is_monotonic_increasing:
            errors.append("Timestamps are not sorted in ascending order")
            df = df.sort_values("timestamps").reset_index(drop=True)
            ts = pd.to_datetime(df["timestamps"])

        # Check duplicates
        dup_count = ts.duplicated().sum()
        stats["duplicate_timestamps"] = int(dup_count)
        if dup_count > 0:
            errors.append(f"Found {dup_count} duplicate timestamps")

        # Check NaNs
        nan_counts = df[NUMERIC_COLUMNS].isnull().sum().to_dict()
        stats["nan_counts"] = {k: int(v) for k, v in nan_counts.items()}
        total_nans = sum(nan_counts.values())
        if total_nans > 0:
            warnings.append(f"Found {total_nans} NaN values across numeric columns: {nan_counts}")

        # Check for zero/negative prices
        for col in OHLC_COLUMNS:
            neg_count = (df[col] <= 0).sum()
            if neg_count > 0:
                warnings.append(f"Column '{col}' has {neg_count} non-positive values")

        # Check OHLC consistency: high >= max(open, close, low), low <= min(open, close, high)
        ohlc_violations = (
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["high"] < df["low"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        ).sum()
        stats["ohlc_violations"] = int(ohlc_violations)
        if ohlc_violations > 0:
            warnings.append(f"Found {ohlc_violations} OHLC consistency violations (high < open/close/low or low > open/close)")

        # Check for gaps if timeframe specified
        if self.expected_timeframe and self.expected_timeframe in self.timeframe_seconds:
            expected_delta = self.timeframe_seconds[self.expected_timeframe]
            deltas = ts.diff().dt.total_seconds().dropna()
            gaps = deltas[deltas > expected_delta * 1.5]
            stats["gaps_found"] = len(gaps)
            if len(gaps) > 0:
                largest_gap = deltas.max()
                warnings.append(
                    f"Found {len(gaps)} gaps larger than 1.5x expected interval ({self.expected_timeframe}). "
                    f"Largest gap: {largest_gap / 3600:.1f} hours"
                )

        # Check minimum data length for Kronos (lookback + predict window)
        min_rows = 512 + 48 + 1  # default lookback + predict + 1
        stats["min_required_rows"] = min_rows
        if len(df) < min_rows:
            warnings.append(
                f"Dataset has {len(df)} rows, but Kronos typically needs at least {min_rows} "
                f"(lookback_window + predict_window + 1)"
            )

        # Summary stats
        stats["price_min"] = float(df["close"].min())
        stats["price_max"] = float(df["close"].max())
        stats["price_mean"] = float(df["close"].mean())
        stats["volume_mean"] = float(df["volume"].mean())

        valid = len(errors) == 0
        return {"valid": valid, "errors": errors, "warnings": warnings, "stats": stats}

    def print_report(self, report: dict):
        """Print a human-readable validation report."""
        print("=" * 60)
        print("Data Validation Report")
        print("=" * 60)

        status = "PASS" if report["valid"] else "FAIL"
        print(f"Status: {status}")
        print()

        if report["errors"]:
            print("ERRORS:")
            for e in report["errors"]:
                print(f"  - {e}")
            print()

        if report["warnings"]:
            print("WARNINGS:")
            for w in report["warnings"]:
                print(f"  - {w}")
            print()

        if report["stats"]:
            print("STATISTICS:")
            for k, v in report["stats"].items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for sub_k, sub_v in v.items():
                        print(f"    {sub_k}: {sub_v}")
                elif isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

        print("=" * 60)
