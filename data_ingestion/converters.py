"""Convert external CSV files with non-standard column names into Kronos-compatible format."""

import os
from typing import Optional

import pandas as pd


KRONOS_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]

# Common column name mappings from various sources
COLUMN_MAPPINGS = {
    # Chinese column names (from existing Kronos examples)
    "日期": "timestamps",
    "时间": "timestamps",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "成交额": "amount",
    # TradingView exports
    "time": "timestamps",
    "Time": "timestamps",
    "Date": "timestamps",
    "date": "timestamps",
    "datetime": "timestamps",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Amount": "amount",
    # Binance/Kraken export formats
    "Date(UTC)": "timestamps",
    "Open time": "timestamps",
    "open_time": "timestamps",
    # Generic variations
    "ts": "timestamps",
    "timestamp": "timestamps",
    "Timestamp": "timestamps",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
    "amt": "amount",
}


class CSVConverter:
    """Convert external CSV files into Kronos-compatible format."""

    def __init__(self, column_mapping: Optional[dict] = None):
        """
        Args:
            column_mapping: Custom column name overrides. Merged with defaults.
        """
        self.column_mapping = {**COLUMN_MAPPINGS}
        if column_mapping:
            self.column_mapping.update(column_mapping)

    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        timestamp_format: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Convert a CSV file to Kronos format.

        Args:
            input_path: Path to the input CSV file.
            output_path: Path to save the converted CSV. If None, auto-generates.
            timestamp_format: Format string for parsing timestamps (e.g., '%Y-%m-%d %H:%M:%S').
                If None, pandas will attempt to infer.

        Returns:
            Converted DataFrame in Kronos format.
        """
        df = pd.read_csv(input_path)

        # Rename columns
        rename_map = {}
        for col in df.columns:
            if col in self.column_mapping:
                rename_map[col] = self.column_mapping[col]
        df = df.rename(columns=rename_map)

        # Check required columns
        missing = [c for c in KRONOS_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Cannot map to Kronos format. Missing columns after rename: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        # Parse timestamps
        if timestamp_format:
            df["timestamps"] = pd.to_datetime(df["timestamps"], format=timestamp_format)
        else:
            df["timestamps"] = pd.to_datetime(df["timestamps"])

        # Sort by timestamp
        df = df.sort_values("timestamps").reset_index(drop=True)

        # Format timestamps as strings
        df["timestamps"] = df["timestamps"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Drop duplicates
        df = df.drop_duplicates(subset=["timestamps"]).reset_index(drop=True)

        # Fill NaN in volume/amount with 0
        df["volume"] = df["volume"].fillna(0)
        df["amount"] = df["amount"].fillna(0)

        # Calculate amount if it's all zeros
        if (df["amount"] == 0).all() and (df["volume"] > 0).any():
            df["amount"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4 * df["volume"]

        result = df[KRONOS_COLUMNS].copy()

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            result.to_csv(output_path, index=False)
            print(f"Converted {len(result)} rows. Saved to {output_path}")
        else:
            print(f"Converted {len(result)} rows (not saved — provide output_path to save).")

        return result
