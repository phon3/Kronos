"""Crypto-specific backtester for Kronos predictions — supports long/short, 24/7 markets."""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CryptoBacktester:
    """
    Backtester for crypto markets using Kronos prediction signals.

    Supports long/short positions, configurable thresholds, and produces
    standard performance metrics (Sharpe, max drawdown, win rate, etc.).
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        threshold: float = 0.02,
        allow_short: bool = True,
        fee_pct: float = 0.001,  # 0.1% per trade (typical exchange taker fee)
        slippage_pct: float = 0.0005,  # 0.05% slippage
    ):
        self.initial_capital = initial_capital
        self.threshold = threshold
        self.allow_short = allow_short
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

    def generate_signals(self, predictions: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from Kronos predictions.

        Args:
            predictions: DataFrame with 'close' column (predicted close prices)
            actual: DataFrame with 'close' column (actual close prices, for alignment)

        Returns:
            DataFrame with columns: actual_close, predicted_close, pred_return, signal, position
        """
        combined = pd.DataFrame(index=actual.index)
        combined["actual_close"] = actual["close"]

        # Align predictions to actual data
        pred_close = predictions["close"] if "close" in predictions.columns else predictions.iloc[:, 0]
        combined["predicted_close"] = pred_close.reindex(combined.index)

        # Forward-fill predictions (predictions are made in windows)
        combined["predicted_close"] = combined["predicted_close"].ffill()

        # Calculate predicted return
        combined["pred_return"] = combined["predicted_close"].pct_change()

        # Generate signals
        combined["signal"] = 0
        combined.loc[combined["pred_return"] > self.threshold, "signal"] = 1  # Buy
        if self.allow_short:
            combined.loc[combined["pred_return"] < -self.threshold, "signal"] = -1  # Sell short

        # Position: hold until opposite signal
        combined["position"] = combined["signal"].replace(0, np.nan).ffill().fillna(0)

        return combined

    def run_backtest(self, signals_df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """
        Execute backtest on signal DataFrame.

        Args:
            signals_df: Output from generate_signals()

        Returns:
            Tuple of (results_df, trades_list)
        """
        capital = self.initial_capital  # Cash balance
        position = 0.0  # Number of units (positive=long, negative=short)
        entry_price = 0.0
        trades = []

        results = pd.DataFrame(index=signals_df.index)
        results["capital"] = 0.0
        results["position"] = 0.0
        results["returns"] = 0.0
        results["price"] = signals_df["actual_close"].astype(float)

        prev_signal = 0

        for i, (timestamp, row) in enumerate(signals_df.iterrows()):
            price = row["actual_close"]
            if pd.isna(price):
                continue

            target_signal = int(row["position"])

            # Close existing position when signal changes
            if i > 0 and target_signal != prev_signal and position != 0:
                exec_price = price * (1 + self.slippage_pct) if position > 0 else price * (1 - self.slippage_pct)
                units = abs(position)
                trade_value = units * exec_price
                fee = trade_value * self.fee_pct

                if position > 0:
                    # Close long: sell units
                    capital += trade_value - fee
                else:
                    # Close short: buy back units
                    capital -= trade_value + fee

                pnl = (exec_price - entry_price) * position - fee
                trades.append({
                    "timestamp": timestamp,
                    "action": "CLOSE",
                    "price": exec_price,
                    "units": units,
                    "side": "LONG" if position > 0 else "SHORT",
                    "pnl": pnl,
                    "capital": capital,
                })
                position = 0.0

            # Open new position when signal changes and we're flat
            if i > 0 and target_signal != 0 and position == 0:
                side = 1 if target_signal > 0 else (-1 if self.allow_short else 0)
                if side != 0 and capital > 0:
                    exec_price = price * (1 + self.slippage_pct) if side > 0 else price * (1 - self.slippage_pct)
                    units = capital / exec_price
                    fee = capital * self.fee_pct

                    if side > 0:
                        # Open long: buy units
                        capital -= units * exec_price + fee
                    else:
                        # Open short: sell units
                        capital += units * exec_price - fee

                    position = units * side
                    entry_price = exec_price

                    trades.append({
                        "timestamp": timestamp,
                        "action": "OPEN",
                        "price": exec_price,
                        "units": units,
                        "side": "LONG" if side > 0 else "SHORT",
                        "pnl": 0.0,
                        "capital": capital,
                    })
                    prev_signal = target_signal

            # Mark-to-market
            portfolio_value = capital + position * price
            results.loc[timestamp, "capital"] = portfolio_value
            results.loc[timestamp, "position"] = position

            # Period returns
            if i > 0:
                prev_value = results["capital"].iloc[i - 1]
                if prev_value > 0:
                    results.loc[timestamp, "returns"] = (portfolio_value - prev_value) / prev_value

        # Close any remaining position at the end
        if position != 0:
            last_price = signals_df["actual_close"].dropna().iloc[-1]
            exec_price = last_price * (1 + self.slippage_pct) if position > 0 else last_price * (1 - self.slippage_pct)
            units = abs(position)
            trade_value = units * exec_price
            fee = trade_value * self.fee_pct

            if position > 0:
                capital += trade_value - fee
            else:
                capital -= trade_value + fee

            pnl = (exec_price - entry_price) * position - fee
            trades.append({
                "timestamp": signals_df.index[-1],
                "action": "CLOSE",
                "price": exec_price,
                "units": abs(position),
                "side": "LONG" if position > 0 else "SHORT",
                "pnl": pnl,
                "capital": capital,
            })

        return results, trades

    def calculate_metrics(self, results: pd.DataFrame, trades: list) -> dict:
        """Calculate standard backtest performance metrics."""
        returns = results["returns"].replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) == 0:
            return self._empty_metrics()

        final_capital = results["capital"].dropna().iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # Annualization factor — crypto trades 24/7/365
        # For hourly: 8760 periods/year, for daily: 365
        periods_per_year = self._estimate_periods_per_year(results.index)

        annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1 if total_return > -1 else -1

        volatility = returns.std() * np.sqrt(periods_per_year)
        risk_free_rate = 0.02  # Assume 2% risk-free
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        # Trade statistics
        close_trades = [t for t in trades if t["action"] == "CLOSE"]
        profitable = [t for t in close_trades if t["pnl"] > 0]
        win_rate = len(profitable) / len(close_trades) if close_trades else 0

        total_pnl = sum(t["pnl"] for t in close_trades)
        gross_profit = sum(t["pnl"] for t in close_trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in close_trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Buy-and-hold benchmark
        prices = results["price"].dropna()
        if len(prices) > 1:
            bh_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        else:
            bh_return = 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(close_trades),
            "total_pnl": total_pnl,
            "final_capital": final_capital,
            "buy_hold_return": bh_return,
            "periods_per_year": periods_per_year,
        }

    def _estimate_periods_per_year(self, index: pd.DatetimeIndex) -> int:
        """Estimate number of periods per year from the data frequency."""
        if len(index) < 2:
            return 365
        median_delta = index.to_series().diff().median()
        seconds = median_delta.total_seconds()
        return int(365.25 * 24 * 3600 / seconds)

    def _empty_metrics(self) -> dict:
        return {
            "total_return": 0, "annual_return": 0, "volatility": 0,
            "sharpe_ratio": 0, "max_drawdown": 0, "win_rate": 0,
            "profit_factor": 0, "total_trades": 0, "total_pnl": 0,
            "final_capital": self.initial_capital, "buy_hold_return": 0,
            "periods_per_year": 365,
        }

    def plot_results(self, results: pd.DataFrame, metrics: dict, output_path: str, title: str = "Kronos Backtest"):
        """Generate and save backtest result charts."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # 1. Equity curve
        axes[0].plot(results.index, results["capital"], linewidth=2, label="Strategy", color="#1f77b4")
        axes[0].axhline(y=self.initial_capital, color="red", linestyle="--", label=f"Initial ({self.initial_capital:,.0f})")
        axes[0].set_ylabel("Portfolio Value")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title, fontsize=14, fontweight="bold")

        # 2. Cumulative returns vs buy-and-hold
        cum_returns = (1 + results["returns"].fillna(0)).cumprod()
        axes[1].plot(results.index, cum_returns, linewidth=2, label="Strategy", color="#2ca02c")

        prices = results["price"].ffill()
        bh_cum = prices / prices.iloc[0]
        axes[1].plot(results.index, bh_cum, linewidth=2, label="Buy & Hold", color="#ff7f0e", alpha=0.7)
        axes[1].set_ylabel("Cumulative Return")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. Drawdown
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        axes[2].fill_between(results.index, drawdown, 0, alpha=0.3, color="red", label="Drawdown")
        axes[2].set_ylabel("Drawdown")
        axes[2].set_xlabel("Date")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Metrics text box
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Annual Return: {metrics['annual_return']:.2%}\n"
            f"Sharpe: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Trades: {metrics['total_trades']}\n"
            f"Buy & Hold: {metrics['buy_hold_return']:.2%}"
        )
        axes[0].text(
            0.02, 0.98, metrics_text, transform=axes[0].transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Backtest chart saved to {output_path}")

    def run(
        self,
        predictions: pd.DataFrame,
        actual: pd.DataFrame,
        output_dir: str = "./backtest_results/",
        title: str = "Kronos Crypto Backtest",
    ) -> dict:
        """
        Full backtest pipeline: generate signals, run backtest, calculate metrics, plot results.

        Args:
            predictions: DataFrame with predicted close prices
            actual: DataFrame with actual close prices
            output_dir: Directory to save results
            title: Chart title

        Returns:
            Metrics dict
        """
        signals = self.generate_signals(predictions, actual)
        results, trades = self.run_backtest(signals)
        metrics = self.calculate_metrics(results, trades)

        # Print report
        print("\n" + "=" * 60)
        print(f"Backtest Report: {title}")
        print("=" * 60)
        for key, value in metrics.items():
            if isinstance(value, float):
                if "return" in key or "rate" in key or "drawdown" in key:
                    print(f"  {key}: {value:.2%}")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 60)

        # Save chart
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, "backtest_chart.png")
        self.plot_results(results, metrics, chart_path, title)

        # Save trade log
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_path = os.path.join(output_dir, "trade_log.csv")
            trades_df.to_csv(trades_path, index=False)
            print(f"Trade log saved to {trades_path}")

        # Save results
        results_path = os.path.join(output_dir, "backtest_results.csv")
        results.to_csv(results_path)
        print(f"Results saved to {results_path}")

        return metrics
