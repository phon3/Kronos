"""Tests for the data ingestion and backtest pipeline."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.data_validator import DataValidator
from data_ingestion.converters import CSVConverter
from backtest.crypto_backtest import CryptoBacktester


# --- Fixtures ---

@pytest.fixture
def sample_kronos_csv():
    """Create a small valid Kronos-format CSV for testing."""
    data = {
        "timestamps": pd.date_range("2024-01-01", periods=600, freq="1h").strftime("%Y-%m-%d %H:%M:%S"),
        "open": np.random.uniform(40000, 50000, 600),
        "high": np.random.uniform(50000, 55000, 600),
        "low": np.random.uniform(35000, 40000, 600),
        "close": np.random.uniform(40000, 50000, 600),
        "volume": np.random.uniform(100, 1000, 600),
        "amount": np.random.uniform(1000000, 5000000, 600),
    }
    df = pd.DataFrame(data)
    # Ensure OHLC consistency
    df["high"] = df[["open", "high", "close"]].max(axis=1) + np.random.uniform(0, 100, 600)
    df["low"] = df[["open", "low", "close"]].min(axis=1) - np.random.uniform(0, 100, 600)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_external_csv():
    """Create a CSV with non-standard column names (Chinese format)."""
    data = {
        "日期": pd.date_range("2024-01-01", periods=600, freq="1h").strftime("%Y-%m-%d %H:%M:%S"),
        "开盘价": np.random.uniform(40000, 50000, 600),
        "最高价": np.random.uniform(50000, 55000, 600),
        "最低价": np.random.uniform(35000, 40000, 600),
        "收盘价": np.random.uniform(40000, 50000, 600),
        "成交量": np.random.uniform(100, 1000, 600),
        "成交额": np.random.uniform(1000000, 5000000, 600),
    }
    df = pd.DataFrame(data)
    df["最高价"] = df[["开盘价", "最高价", "收盘价"]].max(axis=1)
    df["最低价"] = df[["开盘价", "最低价", "收盘价"]].min(axis=1)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


# --- Data Validator Tests ---

class TestDataValidator:
    def test_valid_csv_passes(self, sample_kronos_csv):
        validator = DataValidator(expected_timeframe="1h")
        report = validator.validate(sample_kronos_csv)
        assert report["valid"], f"Expected valid, got errors: {report['errors']}"
        assert report["stats"]["total_rows"] == 600

    def test_missing_columns_fails(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame({"timestamps": ["2024-01-01"], "open": [1]}).to_csv(f.name, index=False)
            validator = DataValidator()
            report = validator.validate(f.name)
            assert not report["valid"]
            assert "Missing required columns" in report["errors"][0]
        os.unlink(f.name)

    def test_duplicate_timestamps_detected(self):
        data = {
            "timestamps": ["2024-01-01 00:00:00"] * 10 + pd.date_range("2024-01-01 01:00:00", periods=590, freq="1h").strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "open": [40000] * 600, "high": [41000] * 600, "low": [39000] * 600,
            "close": [40500] * 600, "volume": [100] * 600, "amount": [100000] * 600,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(data).to_csv(f.name, index=False)
            validator = DataValidator()
            report = validator.validate(f.name)
            assert report["stats"]["duplicate_timestamps"] == 9
        os.unlink(f.name)

    def test_ohlc_violations_detected(self):
        data = {
            "timestamps": pd.date_range("2024-01-01", periods=600, freq="1h").strftime("%Y-%m-%d %H:%M:%S"),
            "open": [50000] * 600, "high": [40000] * 600,  # high < open — violation
            "low": [39000] * 600, "close": [45000] * 600,
            "volume": [100] * 600, "amount": [100000] * 600,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(data).to_csv(f.name, index=False)
            validator = DataValidator()
            report = validator.validate(f.name)
            assert report["stats"]["ohlc_violations"] > 0
        os.unlink(f.name)


# --- CSV Converter Tests ---

class TestCSVConverter:
    def test_chinese_column_conversion(self, sample_external_csv):
        converter = CSVConverter()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name
        try:
            df = converter.convert(sample_external_csv, output_path)
            assert list(df.columns) == ["timestamps", "open", "high", "low", "close", "volume", "amount"]
            assert len(df) == 600
            assert os.path.exists(output_path)
        finally:
            os.unlink(output_path)

    def test_custom_mapping(self):
        data = {
            "my_time": pd.date_range("2024-01-01", periods=10, freq="1h").strftime("%Y-%m-%d %H:%M:%S"),
            "my_open": [1] * 10, "my_high": [2] * 10, "my_low": [0.5] * 10,
            "my_close": [1.5] * 10, "my_vol": [100] * 10, "my_amt": [150] * 10,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(data).to_csv(f.name, index=False)
            custom_map = {
                "my_time": "timestamps", "my_open": "open", "my_high": "high",
                "my_low": "low", "my_close": "close", "my_vol": "volume", "my_amt": "amount",
            }
            converter = CSVConverter(column_mapping=custom_map)
            df = converter.convert(f.name)
            assert "timestamps" in df.columns
            assert "open" in df.columns
        os.unlink(f.name)


# --- Backtest Tests ---

class TestCryptoBacktester:
    def test_generate_signals(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        actual = pd.DataFrame({"close": np.linspace(40000, 45000, 100)}, index=dates)
        predictions = pd.DataFrame({"close": np.linspace(40100, 46000, 100)}, index=dates)

        bt = CryptoBacktester(threshold=0.01)
        signals = bt.generate_signals(predictions, actual)

        assert "signal" in signals.columns
        assert "position" in signals.columns
        assert len(signals) == 100

    def test_run_backtest_returns_results(self):
        dates = pd.date_range("2024-01-01", periods=200, freq="1h")
        prices = np.cumsum(np.random.randn(200) * 100) + 40000
        actual = pd.DataFrame({"close": prices}, index=dates)
        # Predictions slightly ahead (simulating model predicting upward trend)
        predictions = pd.DataFrame({"close": prices * 1.001}, index=dates)

        bt = CryptoBacktester(threshold=0.001, initial_capital=10000)
        signals = bt.generate_signals(predictions, actual)
        results, trades = bt.run_backtest(signals)
        metrics = bt.calculate_metrics(results, trades)

        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert isinstance(metrics["total_trades"], int)

    def test_backtest_with_output(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=200, freq="1h")
        prices = np.cumsum(np.random.randn(200) * 100) + 40000
        actual = pd.DataFrame({"close": prices}, index=dates)
        predictions = pd.DataFrame({"close": prices * 1.002}, index=dates)

        bt = CryptoBacktester(threshold=0.001)
        metrics = bt.run(predictions, actual, output_dir=str(tmp_path), title="Test Backtest")

        assert (tmp_path / "backtest_chart.png").exists()
        assert (tmp_path / "backtest_results.csv").exists()

    def test_trend_signal_mode(self, tmp_path):
        """Test backtest using trend-only signals (no model required)."""
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        # Create data with a clear uptrend
        closes = np.linspace(40000, 50000, n) + np.random.randn(n) * 50
        data = pd.DataFrame({
            "timestamps": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "open": closes - 10,
            "high": closes + 50,
            "low": closes - 50,
            "close": closes,
            "volume": np.random.uniform(100, 1000, n),
            "amount": np.random.uniform(1000000, 5000000, n),
        })

        bt = CryptoBacktester(threshold=0.01, initial_capital=10000)
        actual = data.set_index("timestamps")[["close"]]
        actual.index = pd.to_datetime(actual.index)

        metrics = bt.run(
            predictions=pd.DataFrame(),
            actual=actual,
            output_dir=str(tmp_path),
            title="Trend-only Backtest",
            signal_mode="trend",
            full_data=data,
            trend_prd=10,
            trend_ext_break=0.02,
            trend_ext_limit=0.005,
            trend_min_bars=2,
        )

        assert "total_return" in metrics
        assert (tmp_path / "backtest_chart.png").exists()
        assert (tmp_path / "backtest_results.csv").exists()

    def test_combined_signal_mode(self, tmp_path):
        """Test backtest using combined Kronos + trend signals."""
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        closes = np.linspace(40000, 50000, n) + np.random.randn(n) * 50
        data = pd.DataFrame({
            "timestamps": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "open": closes - 10,
            "high": closes + 50,
            "low": closes - 50,
            "close": closes,
            "volume": np.random.uniform(100, 1000, n),
            "amount": np.random.uniform(1000000, 5000000, n),
        })

        actual = data.set_index("timestamps")[["close"]]
        actual.index = pd.to_datetime(actual.index)
        predictions = pd.DataFrame({"close": closes * 1.002}, index=actual.index)

        bt = CryptoBacktester(threshold=0.001, initial_capital=10000)
        metrics = bt.run(
            predictions=predictions,
            actual=actual,
            output_dir=str(tmp_path),
            title="Combined Backtest",
            signal_mode="combined",
            full_data=data,
            trend_prd=10,
            trend_ext_break=0.02,
            trend_ext_limit=0.005,
            trend_min_bars=2,
        )

        assert "total_return" in metrics
        assert (tmp_path / "backtest_chart.png").exists()

    def test_trend_signals_generation(self):
        """Test that generate_trend_signals returns proper DataFrame."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        closes = np.linspace(100, 130, n)
        data = pd.DataFrame({
            "timestamps": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "open": closes - 0.5,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
        })

        bt = CryptoBacktester()
        signals = bt.generate_trend_signals(
            data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2
        )

        assert "actual_close" in signals.columns
        assert "trend_signal" in signals.columns
        assert "signal" in signals.columns
        assert "position" in signals.columns
        assert len(signals) == n
        assert signals["trend_signal"].isin([0, 1, -1]).all()
