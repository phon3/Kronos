"""Tests for the trend detection module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.trend_detector import Trends, Movement, TrendParams


# --- Fixtures ---

@pytest.fixture
def trending_up_data():
    """Create OHLC data with a clear uptrend then downtrend."""
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    # Steady uptrend for first 20 bars, then downtrend
    closes = list(np.linspace(100, 130, 20)) + list(np.linspace(130, 105, 10))
    opens = [c - 0.5 for c in closes]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=dates,
    )


@pytest.fixture
def flat_data():
    """Create flat OHLC data with no clear trends."""
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
        },
        index=dates,
    )


@pytest.fixture
def volatile_data():
    """Create OHLC data with realistic volatility and a strong move."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    # Random walk with upward drift
    returns = np.random.normal(0.001, 0.005, n)
    closes = 100 * np.cumprod(1 + returns)
    opens = closes - np.random.uniform(0.1, 0.5, n)
    highs = closes + np.random.uniform(0.2, 1.0, n)
    lows = closes - np.random.uniform(0.2, 1.0, n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=dates,
    )


# --- Tests ---

class TestTrendsInit:
    """Test Trends constructor validation."""

    def test_requires_datetime_index(self, trending_up_data):
        df = trending_up_data.reset_index(drop=True)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            Trends(df, prd=5, ext_break=0.05, ext_limit=0.02)

    def test_missing_columns(self, trending_up_data):
        df = trending_up_data.drop(columns=["high"])
        with pytest.raises(ValueError, match="'high'"):
            Trends(df, prd=5, ext_break=0.05, ext_limit=0.02)

    def test_params_property(self, trending_up_data):
        t = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=3)
        p = t.params
        assert p.prd == 5
        assert p.ext_break == 0.05
        assert p.ext_limit == 0.02
        assert p.min_bars == 3

    def test_gradient_computation(self, trending_up_data):
        t = Trends(trending_up_data, prd=10, ext_break=0.09, ext_limit=0.03)
        assert abs(t.grad_break - 0.09 / 9) < 1e-10
        assert abs(t.grad_limit - 0.03 / 9) < 1e-10


class TestBaseIntact:
    """Test break line evaluation."""

    def test_returns_bool_series(self, trending_up_data):
        t = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02)
        bv = t.base_intact(is_adv=True)
        assert isinstance(bv, pd.Series)
        assert bv.dtype == bool
        assert len(bv) == len(trending_up_data)

    def test_first_prd_bars_are_false(self, trending_up_data):
        prd = 5
        t = Trends(trending_up_data, prd=prd, ext_break=0.05, ext_limit=0.02)
        bv = t.base_intact(is_adv=True)
        # rolling(prd, min_periods=prd) produces NaN for first prd-1 bars
        assert not bv.iloc[:prd - 1].any()


class TestLimitExtended:
    """Test limit line evaluation."""

    def test_returns_bool_series(self, trending_up_data):
        t = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02)
        bv = t.limit_extended(is_adv=True)
        assert isinstance(bv, pd.Series)
        assert bv.dtype == bool
        assert len(bv) == len(trending_up_data)

    def test_first_prd_bars_are_false(self, trending_up_data):
        prd = 5
        t = Trends(trending_up_data, prd=prd, ext_break=0.05, ext_limit=0.02)
        bv = t.limit_extended(is_adv=True)
        # rolling(prd, min_periods=prd) produces NaN for first prd-1 bars
        assert not bv.iloc[:prd - 1].any()


class TestGetMovements:
    """Test full movement detection."""

    def test_returns_list_of_movements(self, trending_up_data):
        t = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        moves = t.get_movements()
        assert isinstance(moves, list)
        for m in moves:
            assert isinstance(m, Movement)

    def test_movements_sorted_by_start(self, trending_up_data):
        t = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        moves = t.get_movements()
        for i in range(1, len(moves)):
            assert moves[i].start >= moves[i - 1].start

    def test_flat_data_no_movements(self, flat_data):
        t = Trends(flat_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        moves = t.get_movements()
        # Flat data should produce few or no movements
        # (limit line won't be extended in flat market)
        assert len(moves) == 0

    def test_uptrend_produces_advance(self, trending_up_data):
        t = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        moves = t.get_movements()
        adv_moves = [m for m in moves if m.is_adv]
        assert len(adv_moves) > 0, "Expected at least one advance movement in uptrend data"
        adv = adv_moves[0]
        assert adv.start_px < adv.end_px, "Advance should have positive price change"
        assert adv.chg_pct > 0

    def test_min_bars_filter(self, trending_up_data):
        t_no_filter = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=0)
        t_filtered = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=100)
        moves_no_filter = t_no_filter.get_movements()
        moves_filtered = t_filtered.get_movements()
        # With min_bars=100 (more than data length), no movements should pass
        assert len(moves_filtered) == 0
        # Without filter, we should get some movements
        assert len(moves_no_filter) > 0

    def test_movement_properties(self, trending_up_data):
        t = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        moves = t.get_movements()
        assert len(moves) > 0
        m = moves[0]
        assert m.duration > 0
        assert isinstance(m.is_adv, bool)
        assert m.trend in (1, -1)
        assert m.is_dec == (not m.is_adv)

    def test_movement_chg_pct(self, trending_up_data):
        t = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        moves = t.get_movements()
        for m in moves:
            expected = (m.end_px - m.start_px) / m.start_px
            assert abs(m.chg_pct - expected) < 1e-10


class TestToDataframe:
    """Test DataFrame export."""

    def test_returns_dataframe(self, trending_up_data):
        t = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        df = t.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "is_adv" in df.columns
            assert "start" in df.columns
            assert "end" in df.columns
            assert "chg_pct" in df.columns
            assert "duration" in df.columns

    def test_empty_dataframe_on_flat_data(self, flat_data):
        t = Trends(flat_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        df = t.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "is_adv" in df.columns


class TestTrendSignals:
    """Test signal generation for backtest integration."""

    def test_signals_aligned_to_index(self, trending_up_data):
        t = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        signals = t.get_trend_signals()
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(trending_up_data)
        assert signals.index.equals(trending_up_data.index)

    def test_confirmed_signals_do_not_start_before_first_confirmation(self, trending_up_data):
        trends = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        movements = trends.get_movements()
        confirmed = trends.get_confirmed_trend_signals()

        assert movements
        first_confirmation = min(movement.start_conf for movement in movements)
        assert confirmed.loc[:first_confirmation].iloc[:-1].eq(0).all()

    def test_signal_values(self, trending_up_data):
        t = Trends(trending_up_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        signals = t.get_trend_signals()
        assert signals.isin([0, 1, -1]).all()

    def test_signals_have_active_periods(self, volatile_data):
        t = Trends(volatile_data, prd=5, ext_break=0.02, ext_limit=0.005, min_bars=2)
        signals = t.get_trend_signals()
        # With volatile upward-drifting data, we should see some active signals
        assert (signals != 0).any(), "Expected some non-zero trend signals"

    def test_signals_with_flat_data(self, flat_data):
        t = Trends(flat_data, prd=5, ext_break=0.05, ext_limit=0.02, min_bars=2)
        signals = t.get_trend_signals()
        assert (signals == 0).all()


class TestMovementComparison:
    """Test Movement comparison operators."""

    def test_movement_sorting(self):
        m1 = Movement(
            is_adv=True,
            start=pd.Timestamp("2024-01-01"),
            start_px=100.0,
            start_conf=pd.Timestamp("2024-01-03"),
            start_conf_px=102.0,
            end=pd.Timestamp("2024-01-10"),
            end_px=110.0,
            end_conf=pd.Timestamp("2024-01-12"),
            end_conf_px=108.0,
            duration=10,
            params=TrendParams(prd=5, ext_break=0.05, ext_limit=0.02),
            by_break=True,
        )
        m2 = Movement(
            is_adv=False,
            start=pd.Timestamp("2024-01-05"),
            start_px=110.0,
            start_conf=pd.Timestamp("2024-01-07"),
            start_conf_px=108.0,
            end=pd.Timestamp("2024-01-15"),
            end_px=100.0,
            end_conf=pd.Timestamp("2024-01-17"),
            end_conf_px=102.0,
            duration=11,
            params=TrendParams(prd=5, ext_break=0.05, ext_limit=0.02),
            by_break=False,
        )
        assert m1 < m2
        assert m2 > m1
        assert m1 <= m2
        assert m2 >= m1
        assert sorted([m2, m1]) == [m1, m2]
