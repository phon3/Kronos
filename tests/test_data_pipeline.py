"""Tests for the data ingestion and backtest pipeline."""

import json
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
from backtest import backtest_runner
from backtest.param_sweep import (
    _combine_signals,
    _evaluate_signals,
    _export_candidates,
    _parse_tp_level_sets,
    _risk_grid,
    _score_row,
    _trade_diagnostics,
    run_optimizer,
    run_optimizer_matrix,
)
import webui.app as webui_app
from webui.app import UI_VERSION, app


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

    def test_prediction_window_boundaries_do_not_create_signals(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="1h")
        actual = pd.DataFrame({"close": [100, 101, 200, 202]}, index=dates)
        predictions = pd.DataFrame({
            "close": [100, 101, 200, 202],
            "prediction_window": [0, 0, 1, 1],
        }, index=dates)

        signals = CryptoBacktester(threshold=0.50).generate_signals(predictions, actual)

        assert pd.isna(signals["pred_return"].iloc[2])
        assert signals["signal"].eq(0).all()

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
        assert metrics["average_gross_exposure"] >= 0
        assert metrics["excess_vs_matched_exposure"] == pytest.approx(
            metrics["total_return"] - metrics["exposure_matched_buy_hold_return"]
        )

    @pytest.mark.parametrize("side, expected_entry, expected_exit", [
        (1, 101.0, 99.0),
        (-1, 99.0, 101.0),
    ])
    def test_round_trip_slippage_is_adverse(self, side, expected_entry, expected_exit):
        dates = pd.date_range("2024-01-01", periods=2, freq="1h")
        signals = pd.DataFrame({"actual_close": [100.0, 100.0], "position": [0, side]}, index=dates)
        backtester = CryptoBacktester(
            initial_capital=10_000,
            fee_pct=0,
            slippage_pct=0.01,
            position_size_pct=1.0,
        )

        results, trades = backtester.run_backtest(signals)
        opened = next(trade for trade in trades if trade["action"] == "OPEN")
        closed = next(trade for trade in trades if trade["action"] == "CLOSE")

        assert opened["price"] == pytest.approx(expected_entry)
        assert closed["price"] == pytest.approx(expected_exit)
        assert results["capital"].iloc[-1] < backtester.initial_capital

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


def test_generate_predictions_preserves_disabled_top_k():
    rows = 20
    data = pd.DataFrame({
        "timestamps": pd.date_range("2024-01-01", periods=rows, freq="1h"),
        "open": np.arange(rows),
        "high": np.arange(rows) + 1,
        "low": np.arange(rows) - 1,
        "close": np.arange(rows),
        "volume": np.ones(rows),
        "amount": np.ones(rows),
    })

    class Predictor:
        def __init__(self):
            self.top_k_values = []

        def predict(self, **kwargs):
            self.top_k_values.append(kwargs["top_k"])
            return pd.DataFrame({column: np.ones(kwargs["pred_len"]) for column in [
                "open", "high", "low", "close", "volume", "amount"
            ]})

    predictor = Predictor()
    predictions = backtest_runner.generate_predictions(
        predictor,
        data,
        lookback=10,
        pred_len=5,
        top_k=0,
        sample_count=3,
    )

    assert predictor.top_k_values == [0, 0]
    assert predictions["prediction_window"].tolist() == [0] * 5 + [1] * 5


def test_webui_latest_prediction_uses_final_window_and_seed(monkeypatch, sample_kronos_csv):
    source = pd.read_csv(sample_kronos_csv, parse_dates=["timestamps"])
    captured = {}

    class Predictor:
        def predict(self, **kwargs):
            captured.update(kwargs)
            captured["random_value"] = np.random.randint(0, 1_000_000)
            rows = kwargs["pred_len"]
            return pd.DataFrame({
                "open": np.ones(rows),
                "high": np.ones(rows),
                "low": np.ones(rows),
                "close": np.ones(rows),
                "volume": np.ones(rows),
            })

    monkeypatch.setattr(webui_app, "MODEL_AVAILABLE", True)
    monkeypatch.setattr(webui_app, "predictor", Predictor())
    monkeypatch.setattr(webui_app, "create_prediction_chart", lambda *args, **kwargs: "{}")
    monkeypatch.setattr(webui_app, "save_prediction_results", lambda **kwargs: None)

    response = app.test_client().post("/api/predict", json={
        "file_path": sample_kronos_csv,
        "lookback": 10,
        "pred_len": 5,
        "temperature": 0.6,
        "top_p": 0.9,
        "sample_count": 1,
        "seed": 123,
    })

    assert response.status_code == 200
    assert captured["df"]["close"].iloc[-1] == pytest.approx(source["close"].iloc[-1])
    assert captured["x_timestamp"].iloc[-1] == source["timestamps"].iloc[-1]
    assert captured["y_timestamp"].iloc[0] > source["timestamps"].iloc[-1]
    assert captured["top_k"] == 0
    assert captured["T"] == 0.6
    assert captured["random_value"] == np.random.RandomState(123).randint(0, 1_000_000)
    assert response.get_json()["actual_data"] == []


def test_webui_rejects_stale_backtest_client():
    response = app.test_client().post("/api/backtest", json={"data_path": "unused.csv"})

    assert response.status_code == 409
    assert response.get_json()["code"] == "STALE_UI"
    assert response.get_json()["ui_version"] == UI_VERSION


def test_webui_rejects_model_data_timeframe_mismatch(monkeypatch):
    monkeypatch.setattr(webui_app, "scan_finetuned_models", lambda: {
        "finetuned-btc_usd_1h_test": {
            "name": "Test 1h model",
            "model_path": "unused-model",
            "tokenizer_path": "unused-tokenizer",
            "context_length": 512,
            "params": "test",
            "description": "test",
            "type": "finetuned",
        }
    })

    response = app.test_client().post("/api/backtest", json={
        "ui_version": UI_VERSION,
        "data_path": "BTC_USD_1d.csv",
        "signal_mode": "combined",
        "model_key": "finetuned-btc_usd_1h_test",
        "max_data_rows": 560,
    })

    assert response.status_code == 400
    assert response.get_json()["code"] == "TIMEFRAME_MISMATCH"


def test_trend_backtest_limits_input_rows(monkeypatch, tmp_path):
    rows = 1000
    closes = np.linspace(100, 200, rows)
    data = pd.DataFrame({
        "timestamps": pd.date_range("2024-01-01", periods=rows, freq="15min"),
        "open": closes,
        "high": closes + 1,
        "low": closes - 1,
        "close": closes,
    })
    received = {}

    monkeypatch.setattr(backtest_runner.pd, "read_csv", lambda _: data.copy())
    monkeypatch.setattr(
        backtest_runner.CryptoBacktester,
        "run",
        lambda self, **kwargs: received.update(rows=len(kwargs["full_data"])) or {"total_return": 0},
    )
    monkeypatch.setattr(backtest_runner.ReportGenerator, "generate_report", lambda self, **kwargs: None)

    backtest_runner.run_backtest(
        model_path="",
        tokenizer_path="",
        data_path="unused.csv",
        output_dir=str(tmp_path),
        signal_mode="trend",
        max_data_rows=120,
    )

    assert received["rows"] == 120


def test_combined_optimizer_holds_through_neutral_trend_until_reversal():
    index = pd.date_range("2024-01-01", periods=3, freq="1h")
    kronos = pd.DataFrame({"actual_close": [100, 101, 102], "position": [1, 1, 1]}, index=index)
    trend = pd.DataFrame({"trend_signal": [1, 0, -1], "position": [1, 1, -1]}, index=index)

    combined = _combine_signals(kronos, trend, allow_short=True)

    assert combined["position"].tolist() == [1, 1, 0]


def test_combined_entry_only_ignores_kronos_flips_until_trend_reversal():
    index = pd.date_range("2024-01-01", periods=4, freq="1h")
    kronos = pd.DataFrame({"actual_close": [100, 101, 102, 103], "position": [1, -1, -1, 1]}, index=index)
    trend = pd.DataFrame({
        "trend_signal": [1, 0, 0, -1],
        "position": [1, 1, 1, -1],
    }, index=index)

    continuous = _combine_signals(kronos, trend, allow_short=True, confirmation_policy="continuous")
    entry_only = _combine_signals(kronos, trend, allow_short=True, confirmation_policy="entry_only")

    assert continuous["position"].tolist() == [1, 0, 0, 0]
    assert entry_only["position"].tolist() == [1, 1, 1, 0]


def test_optimizer_trade_diagnostics_separate_partial_and_completed_exits():
    trades = [
        {"timestamp": "2024-01-01 00:00", "action": "OPEN", "reason": "SIGNAL"},
        {"timestamp": "2024-01-01 01:00", "action": "PARTIAL_CLOSE", "reason": "TP1"},
        {"timestamp": "2024-01-01 03:00", "action": "CLOSE", "reason": "SIGNAL"},
    ]

    diagnostics = _trade_diagnostics(trades)

    assert diagnostics["completed_trades"] == 1
    assert diagnostics["partial_exits"] == 1
    assert diagnostics["signal_exits"] == 1
    assert diagnostics["average_holding_hours"] == 3


def test_optimizer_uses_common_timestamp_split():
    index = pd.date_range("2024-01-01", periods=10, freq="1h")
    signals = pd.DataFrame({
        "actual_close": np.arange(100, 110),
        "position": np.zeros(10),
    }, index=index)

    metrics = _evaluate_signals(
        signals,
        risk={},
        initial_capital=10_000,
        train_ratio=0.5,
        split_timestamp=index[7],
    )

    assert metrics["oos_buy_hold_return"] == pytest.approx((109 - 107) / 107)


def test_optimizer_score_filters_low_trade_candidates():
    row = {
        "is_sharpe_ratio": 1.0,
        "oos_sharpe_ratio": 1.2,
        "oos_total_return": 0.10,
        "oos_max_drawdown": -0.05,
        "oos_total_trades": 2,
    }

    assert _score_row(row, "composite", min_trades=3) == float("-inf")
    assert np.isfinite(_score_row(row, "composite", min_trades=2))


def test_optimizer_composite_rewards_excess_return_and_trade_confidence():
    base = {
        "is_sharpe_ratio": 2.0,
        "oos_sharpe_ratio": 2.0,
        "oos_total_return": 0.15,
        "oos_max_drawdown": -0.05,
        "oos_profitable_folds": 3,
        "oos_validation_folds": 3,
        "oos_worst_fold_return": 0.01,
    }
    low_confidence = {**base, "oos_buy_hold_return": 0.10, "oos_total_trades": 10}
    high_confidence = {**base, "oos_buy_hold_return": 0.10, "oos_total_trades": 30}
    underperforming = {**base, "oos_buy_hold_return": 0.20, "oos_total_trades": 30}

    assert _score_row(high_confidence, "composite", 10) > _score_row(low_confidence, "composite", 10)
    assert _score_row(underperforming, "composite", 10) == float("-inf")


def test_optimizer_dry_run_reports_large_grid_without_loading_data():
    results, top = run_optimizer(
        data_path="unused.csv",
        signal_mode="trend",
        grid_profile="exhaustive",
        max_combinations=1,
        dry_run=True,
    )

    assert results.empty
    assert top.empty


def test_optimizer_custom_risk_grid_includes_single_and_multitarget_exits():
    tp_sets = _parse_tp_level_sets("0.03:0.5,0.06:0.5;0.05:0.33,0.10:0.33,0.15:0.34")
    configs = _risk_grid("quick", overrides={
        "position_size_pct": [0.25, 0.5],
        "stop_loss_pct": [0.08],
        "take_profit_pct": [None, 0.10],
        "loss_multiplier": [1.0],
        "daily_loss_limit_pct": [None, 0.03],
        "take_profit_levels": tp_sets,
    })

    assert len(configs) == 16
    assert sum(config["take_profit_levels"] is not None for config in configs) == 8


def test_optimizer_rejects_incomplete_tp_fractions():
    with pytest.raises(ValueError, match="total 1.0"):
        _parse_tp_level_sets("0.03:0.4,0.06:0.4")


def test_optimizer_matrix_dry_run_does_not_load_data():
    results, top = run_optimizer_matrix(
        lookbacks=[256, 512],
        pred_lengths=[24, 48],
        sample_counts=[1, 3],
        output_dir="unused",
        dry_run=True,
        data_path="unused.csv",
    )

    assert results.empty
    assert top.empty


def test_optimizer_exports_empty_candidates_when_all_results_filtered(tmp_path):
    results = pd.DataFrame([{
        "score": float("-inf"),
        "oos_total_return": 0.0,
        "oos_sharpe_ratio": 0.0,
        "oos_max_drawdown": 0.0,
        "oos_win_rate": 0.0,
        "oos_total_trades": 0,
    }])

    top = _export_candidates(results, str(tmp_path), top_n=5)

    assert top.empty
    assert (tmp_path / "all_results.csv").exists()
    assert (tmp_path / "top_candidates.csv").exists()
    assert json.loads((tmp_path / "live_test_candidates.json").read_text()) == []
    manifest = json.loads((tmp_path / "experiment_manifest.json").read_text())
    assert manifest["candidate_count"] == 0


def test_optimizer_exports_top_live_candidates(sample_kronos_csv, tmp_path):
    results, top = run_optimizer(
        data_path=sample_kronos_csv,
        output_dir=str(tmp_path),
        signal_mode="trend",
        grid_profile="quick",
        top_n=3,
        min_trades=0,
        min_excess_return=-10,
        max_data_rows=200,
        grid_overrides={
            "prd": [10],
            "ext_break": [0.03],
            "ext_limit": [0.01],
            "min_bars": [3],
        },
    )

    assert len(results) == 4
    assert {
        "data_file", "data_timeframe", "data_start", "data_end", "evaluation_start",
        "evaluation_end", "oos_start", "oos_end", "oos_excess_return",
        "trade_confidence", "oos_worst_fold_return",
    }.issubset(results.columns)
    assert results["data_timeframe"].eq("1h").all()
    assert 1 <= len(top) <= 3
    assert (tmp_path / "all_results.csv").exists()
    assert (tmp_path / "top_candidates.csv").exists()
    candidates = json.loads((tmp_path / "live_test_candidates.json").read_text())
    assert [candidate["rank"] for candidate in candidates] == list(range(1, len(candidates) + 1))
    assert all("settings" in candidate and "metrics" in candidate for candidate in candidates)
