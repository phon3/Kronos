"""Backtest package for Kronos — crypto-specific backtesting with prediction-driven strategies."""

# Lazy imports to avoid heavy dependencies (torch, model) unless needed
__all__ = ["CryptoBacktester", "run_backtest", "ReportGenerator"]
