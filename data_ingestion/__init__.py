"""Data ingestion package for Kronos — fetches and validates OHLCV data from crypto exchanges and stock APIs."""

# Lazy imports to avoid importing ccxt/yfinance unless needed
__all__ = ["CryptoFetcher", "NasdaqFetcher", "DataValidator", "CSVConverter"]
