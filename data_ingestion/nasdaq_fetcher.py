"""Fetch OHLCV data for NASDAQ/NYSE stocks and ETFs via yfinance."""

import os
from typing import Optional

import pandas as pd


KRONOS_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]

# yfinance interval constraints:
#   - 1m, 2m, 5m, 15m, 30m, 60m, 90m: last 60 days
#   - 1h: last 730 days
#   - 1d: full history (~20+ years)
YFINANCE_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "1d"]


class NasdaqFetcher:
    """Fetch OHLCV data for stocks/ETFs via yfinance and output Kronos-compatible CSV."""

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError(
                "yfinance is required. Install with: pip install yfinance"
            )

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a stock/ETF symbol.

        Args:
            symbol: Ticker symbol, e.g. 'AAPL', 'QQQ', 'NVDA'
            interval: '1d' for daily (full history), '1h' for hourly (last ~2 years)
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD). Defaults to today.

        Returns:
            DataFrame with columns: timestamps, open, high, low, close, volume, amount
        """
        if interval not in YFINANCE_INTERVALS:
            raise ValueError(
                f"Interval '{interval}' not supported. Use one of: {YFINANCE_INTERVALS}"
            )

        ticker = self.yf.Ticker(symbol)

        if end_date:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            df = ticker.history(start=start_date, interval=interval)

        if df.empty:
            raise RuntimeError(f"No data returned for {symbol} {interval} from {start_date}")

        # yfinance columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        # Normalize to Kronos format
        result = pd.DataFrame()
        result["timestamps"] = df.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        result["open"] = df["Open"].values
        result["high"] = df["High"].values
        result["low"] = df["Low"].values
        result["close"] = df["Close"].values
        result["volume"] = df["Volume"].values

        # Calculate amount as OHLC average * volume (consistent with Kronos preprocessing)
        result["amount"] = (result["open"] + result["high"] + result["low"] + result["close"]) / 4 * result["volume"]

        # Drop rows with NaN values
        result = result.dropna().reset_index(drop=True)

        result = result[KRONOS_COLUMNS].copy()

        print(f"Fetched {len(result)} candles for {symbol} {interval} from yfinance")
        print(f"  Date range: {result['timestamps'].iloc[0]} to {result['timestamps'].iloc[-1]}")

        return result

    def save_csv(self, df: pd.DataFrame, output_path: str) -> str:
        """Save DataFrame to CSV in Kronos format."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")
        return output_path

    def fetch_and_save(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        output_path: str = "./data/",
    ) -> str:
        """Fetch data and save to CSV in one call."""
        df = self.fetch_ohlcv(symbol, interval, start_date, end_date)

        if os.path.isdir(output_path) or output_path.endswith("/"):
            filename = f"{symbol}_{interval}.csv"
            output_path = os.path.join(output_path, filename)

        return self.save_csv(df, output_path)
