"""Fetch OHLCV candle data from crypto exchanges via CCXT."""

import os
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd
from tqdm import tqdm


SUPPORTED_EXCHANGES = ["coinbase", "kraken", "bybit", "okx"]

# Default exchange — Coinbase handles historical `since` pagination correctly
DEFAULT_EXCHANGE = "coinbase"

TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

KRONOS_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


class CryptoFetcher:
    """Fetch OHLCV data from crypto exchanges and output Kronos-compatible CSV."""

    def __init__(self, exchange_name: str = DEFAULT_EXCHANGE):
        if exchange_name not in SUPPORTED_EXCHANGES:
            raise ValueError(
                f"Exchange '{exchange_name}' not supported. Use one of: {SUPPORTED_EXCHANGES}"
            )
        self.exchange_name = exchange_name
        self.exchange = getattr(ccxt, exchange_name)({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.exchange.load_markets()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a symbol from the exchange.

        Args:
            symbol: Trading pair, e.g. 'BTC/USD' or 'ETH/USD'
            timeframe: Candle interval — '1m', '5m', '15m', '1h', '4h', '1d'
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD). Defaults to now.
            progress: Show progress bar.

        Returns:
            DataFrame with columns: timestamps, open, high, low, close, volume, amount
        """
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(f"Timeframe '{timeframe}' not supported. Use one of: {list(TIMEFRAME_MAP.keys())}")

        if symbol not in self.exchange.markets:
            raise ValueError(f"Symbol '{symbol}' not found on {self.exchange_name}")

        tf = TIMEFRAME_MAP[timeframe]
        start_ms = int(
            pd.Timestamp(start_date, tz="UTC").timestamp() * 1000
        )
        end_ms = (
            int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
            if end_date
            else int(datetime.now(timezone.utc).timestamp() * 1000)
        )

        # Kraken's `since` parameter only works with pagination tokens from
        # previous responses, not arbitrary timestamps. For Kraken, we fetch
        # backwards from the most recent data using the `last` value.
        if self.exchange_name == "kraken":
            return self._fetch_kraken(symbol, tf, timeframe, start_ms, end_ms, progress)

        all_candles = []
        # CCXT fetch_ohlcv returns at most 300-1000 candles per call (exchange-dependent).
        # We paginate by advancing `since` after each call.
        since_ms = start_ms
        if progress:
            pbar = tqdm(desc=f"Fetching {symbol} {timeframe} from {self.exchange_name}", unit=" candles")

        while since_ms < end_ms:
            try:
                candles = self.exchange.fetch_ohlcv(symbol, timeframe=tf, since=since_ms, limit=1000)
            except ccxt.NetworkError as e:
                print(f"Network error, retrying in 5s: {e}")
                time.sleep(5)
                continue
            except ccxt.ExchangeError as e:
                raise RuntimeError(f"Exchange error fetching {symbol}: {e}") from e

            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]

            if last_ts <= since_ms:
                break

            since_ms = last_ts + 1

            if progress:
                pbar.update(len(candles))

            # Respect rate limits
            time.sleep(self.exchange.rateLimit / 1000)

        if progress:
            pbar.close()

        if not all_candles:
            raise RuntimeError(f"No candles returned for {symbol} {timeframe} from {start_date}")

        return self._candles_to_df(all_candles, start_ms, end_ms, symbol, timeframe)

    def _fetch_kraken(
        self, symbol: str, tf: str, timeframe: str, since_ms: int, end_ms: int, progress: bool
    ) -> pd.DataFrame:
        """
        Kraken-specific fetch: paginate backwards from most recent data using
        the `last` value from each response, since Kraken's `since` parameter
        only accepts pagination tokens, not arbitrary timestamps.
        """
        import requests

        # Map Kraken symbol to Kraken's internal pair name
        # CCXT's symbol 'BTC/USD' maps to Kraken's 'XBTUSD' or 'XXBTZUSD'
        market = self.exchange.market(symbol)
        kraken_pair = market.get("id", symbol.replace("/", ""))

        # Kraken interval mapping (in minutes)
        kraken_intervals = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        interval = kraken_intervals.get(timeframe, 60)

        all_candles = []
        # Start from most recent and paginate backwards
        since_param = None

        if progress:
            pbar = tqdm(desc=f"Fetching {symbol} {timeframe} from kraken", unit=" candles")

        while True:
            params = {"pair": kraken_pair, "interval": interval}
            if since_param is not None:
                params["since"] = since_param

            try:
                resp = requests.get("https://api.kraken.com/0/public/OHLC", params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"Error fetching from Kraken: {e}")
                break

            if data.get("error"):
                print(f"Kraken API error: {data['error']}")
                break

            result = data.get("result", {})
            pair_key = [k for k in result.keys() if k != "last"]
            if not pair_key:
                break

            candles_raw = result[pair_key[0]]
            last_val = result.get("last")

            if not candles_raw:
                break

            # Kraken returns [time, o, h, l, c, vwap, volume, count]
            # Convert to CCXT format [ts_ms, o, h, l, c, v]
            batch = []
            for c in candles_raw:
                ts_ms = int(c[0]) * 1000
                if ts_ms < since_ms:
                    continue
                if ts_ms >= end_ms:
                    continue
                batch.append([ts_ms, float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[6])])

            all_candles.extend(batch)

            if progress:
                pbar.update(len(batch))

            # Check if we've gone past our start date
            oldest_ts = int(candles_raw[0][0]) * 1000
            if oldest_ts <= since_ms:
                break

            # Use the `last` value to get older data
            if last_val is None or str(last_val) == str(since_param):
                break
            since_param = last_val

            time.sleep(1.5)  # Respect Kraken rate limits

        if progress:
            pbar.close()

        if not all_candles:
            raise RuntimeError(f"No candles returned for {symbol} {timeframe} from kraken")

        return self._candles_to_df(all_candles, since_ms, end_ms, symbol, timeframe)

    def _candles_to_df(
        self, all_candles: list, start_ms: int, end_ms: int, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """Convert raw candle list to Kronos-format DataFrame."""
        df = pd.DataFrame(all_candles, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["timestamps"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Calculate amount as (O+H+L+C)/4 * volume (consistent with existing Kronos preprocessing)
        df["amount"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4 * df["volume"]

        # Drop duplicates and sort
        df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)

        # Filter to requested date range
        df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] < end_ms)]

        result = df[KRONOS_COLUMNS].copy()

        if len(result) == 0:
            raise RuntimeError(f"No candles in date range for {symbol} {timeframe}")

        print(f"Fetched {len(result)} candles for {symbol} {timeframe} from {self.exchange_name}")
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
        timeframe: str = "1h",
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        output_path: str = "./data/",
    ) -> str:
        """Fetch data and save to CSV in one call."""
        df = self.fetch_ohlcv(symbol, timeframe, start_date, end_date)

        # Auto-generate filename if output_path is a directory
        if os.path.isdir(output_path) or output_path.endswith("/"):
            safe_symbol = symbol.replace("/", "_")
            filename = f"{safe_symbol}_{timeframe}.csv"
            output_path = os.path.join(output_path, filename)

        return self.save_csv(df, output_path)
