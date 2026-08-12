"""Trend detection algorithm ported from market_analy (maread99/market_analy).

Identifies advance and decline movements in OHLC data using configurable
break lines and limit lines. Each movement records start/end timestamps,
prices, duration, and whether it ended by break or by failing to extend.

Algorithm overview:
    A Movement is confirmed as having started whenever:
        - Break Line over a given period (`prd`) has NOT been broken.
          For an advance/decline, the Break Line is a rising/falling line
          with gradient (`ext_break` / `prd`) passing through the low/high
          of the bar on which the movement started.
        - Limit Line has been extended during `prd`.
          For an advance/decline, the Limit Line is a rising/falling line
          with gradient (`ext_limit` / `prd`) passing through the low/high
          of the bar on which the movement started.

    Movement ends by:
        - break: Break Line is exceeded.
        - limit: Limit Line is not exceeded over the prior `prd` bars.

Original source: https://github.com/maread99/market_analy
License: MIT
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class TrendParams:
    """Parameters for trend detection."""

    prd: int
    ext_break: float
    ext_limit: float
    min_bars: int = 0


@dataclass(frozen=True, eq=False)
class Movement:
    """A single advance or decline movement identified by Trends.

    Attributes
    ----------
    is_adv : bool
        True if advance, False if decline.
    start : pd.Timestamp
        Bar when movement started.
    start_px : float
        Price at movement start (low for advance, high for decline).
    start_conf : pd.Timestamp
        Bar when movement was confirmed as having started.
    start_conf_px : float
        Close price at confirmation bar.
    end : pd.Timestamp
        Bar when movement ended (high/low extreme).
    end_px : float
        Price at movement end.
    end_conf : pd.Timestamp | None
        Bar when movement end was confirmed. None if still open.
    end_conf_px : float | None
        Price when end confirmed. None if still open.
    duration : int
        Number of bars from start to end (inclusive).
    params : TrendParams
        Parameters used to detect this movement.
    by_break : bool | None
        True: ended by break line. False: ended by limit. None: still open.
    """

    is_adv: bool
    start: pd.Timestamp
    start_px: float
    start_conf: pd.Timestamp
    start_conf_px: float
    end: pd.Timestamp
    end_px: float
    end_conf: pd.Timestamp | None
    end_conf_px: float | None
    duration: int
    params: TrendParams
    by_break: bool | None

    @property
    def open(self) -> bool:
        """Movement has not ended."""
        return self.by_break is None

    @property
    def closed(self) -> bool:
        """Movement has ended."""
        return not self.open

    @property
    def is_dec(self) -> bool:
        """Movement is a decline."""
        return not self.is_adv

    @property
    def trend(self) -> int:
        """Trend direction. 1 for advance, -1 for decline."""
        return 1 if self.is_adv else -1

    @property
    def chg(self) -> float:
        """Absolute price change over movement."""
        return self.end_px - self.start_px

    @property
    def chg_pct(self) -> float:
        """Percentage change over movement."""
        return self.chg / self.start_px

    @property
    def conf_chg(self) -> float | None:
        """Absolute change between start and end confirmations."""
        if self.end_conf_px is None:
            return None
        return self.end_conf_px - self.start_conf_px

    @property
    def conf_chg_pct(self) -> float | None:
        """Percentage change between start and end confirmations."""
        if self.conf_chg is None:
            return None
        return self.conf_chg / self.start_conf_px

    def __lt__(self, other: Movement) -> bool:
        return self.start < other.start

    def __le__(self, other: Movement) -> bool:
        return self.start <= other.start

    def __gt__(self, other: Movement) -> bool:
        return self.start > other.start

    def __ge__(self, other: Movement) -> bool:
        return self.start >= other.start


class Trends:
    """Evaluate trends (advance/decline movements) from OHLC data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns "high", "low", and "close".
        Must be indexed with a pd.DatetimeIndex.
    prd : int
        Period (in bars) for break line and limit line evaluation.
    ext_break : float
        Percentage change defining the break line gradient over `prd`.
        e.g. 0.05 for 5%.
    ext_limit : float
        Percentage change defining the limit line gradient over `prd`.
        e.g. 0.02 for 2%.
    min_bars : int, default 0
        Minimum number of bars for a valid movement.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"open": [100, 101, 103, 105, 104, 106, 108, 107, 109, 111]},
    ...     {"high": [101, 102, 104, 106, 105, 107, 109, 108, 110, 112]},
    ...     {"low": [99, 100, 102, 104, 103, 105, 107, 106, 108, 110]},
    ...     {"close": [100, 101, 103, 105, 104, 106, 108, 107, 109, 111]},
    ...     index=pd.date_range("2024-01-01", periods=10, freq="1h"),
    ... )
    >>> trends = Trends(df, prd=3, ext_break=0.05, ext_limit=0.02, min_bars=2)
    >>> movements = trends.get_movements()
    >>> for m in movements:
    ...     print(f"{'ADV' if m.is_adv else 'DEC'}: {m.start} -> {m.end}, {m.chg_pct:.2%}")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        prd: int,
        ext_break: float,
        ext_limit: float,
        min_bars: int = 0,
    ):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("data must be indexed with a pd.DatetimeIndex")
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(f"data must include '{col}' column")

        self.data = data.copy()
        self.prd = prd
        self.ext_break = ext_break
        self.ext_limit = ext_limit
        self.min_bars = min_bars

        self.grad_break = ext_break / (prd - 1)
        self.grad_limit = ext_limit / (prd - 1)

        self.fctrs_pos_break = np.linspace(1, 1 + ext_break, prd)
        self.fctrs_neg_break = np.linspace(1, 1 - ext_break, prd)
        self.fctrs_pos_limit = np.linspace(1, 1 + ext_limit, prd)
        self.fctrs_neg_limit = np.linspace(1, 1 - ext_limit, prd)

    @property
    def params(self) -> TrendParams:
        """Parameters received by constructor."""
        return TrendParams(
            prd=self.prd,
            ext_break=self.ext_break,
            ext_limit=self.ext_limit,
            min_bars=self.min_bars,
        )

    def base_intact(self, is_adv: bool) -> pd.Series:
        """Query if base (break line) was NOT broken over any `prd` bars.

        Returns pd.Series of bool, indexed as self.data.index.
        First `prd` bars are False.
        """

        def func(arr: pd.Series) -> bool:
            fctrs = self.fctrs_pos_break if is_adv else self.fctrs_neg_break
            line = arr.iloc[0] * fctrs[: len(arr)]
            bv = (arr >= line) if is_adv else (arr <= line)
            return bv.all()

        col = self.data["low"] if is_adv else self.data["high"]
        bv = col.rolling(self.prd, min_periods=self.prd).apply(func)
        bv = bv.fillna(False)
        return bv.astype(bool)

    def limit_extended(self, is_adv: bool) -> pd.Series:
        """Query if limit line was extended over any `prd` bars.

        Returns pd.Series of bool, indexed as self.data.index.
        First `prd` bars are False.
        """

        def func(arr: pd.Series) -> bool:
            fctrs = self.fctrs_pos_limit if is_adv else self.fctrs_neg_limit
            line = arr.iloc[0] * fctrs[: len(arr)]
            bv = (arr > line) if is_adv else (arr < line)
            return bv.any()

        col = self.data["high"] if is_adv else self.data["low"]
        bv = col.rolling(self.prd, min_periods=self.prd).apply(func)
        bv = bv.fillna(False)
        return bv.astype(bool)

    def start_confs_all(self, is_adv: bool) -> pd.DatetimeIndex:
        """All bars on which a movement could be confirmed as having started."""
        bv = self.base_intact(is_adv) & self.limit_extended(is_adv)
        return bv[bv].index

    def start_from_start_conf(self, start_conf: pd.Timestamp) -> pd.Timestamp:
        """Get start bar from start confirmed bar."""
        idx_start_conf = self.data.index.get_loc(start_conf)
        idx_start = idx_start_conf - self.prd + 1
        return self.data.index[idx_start]

    def start_conf_from_start(self, start: pd.Timestamp) -> pd.Timestamp | None:
        """Get start confirmed bar corresponding with a start bar.

        Returns None if conf_start would be beyond the end of available data.
        """
        idx_start = self.data.index.get_loc(start)
        idx_start_conf = idx_start + self.prd - 1
        if idx_start_conf > len(self.data) - 1:
            return None
        return self.data.index[idx_start_conf]

    def get_line(
        self, subset: pd.Series, is_adv: bool, limit: bool = True
    ) -> pd.Series:
        """Get a break or limit line for a subset of the data."""
        op = operator.add if is_adv else operator.sub
        grad = self.grad_limit if limit else self.grad_break
        fctrs = np.array([op(1, (i * grad)) for i in range(len(subset))])
        return pd.Series(fctrs * subset.iloc[0], index=subset.index)

    def get_limit_line(
        self, start: pd.Timestamp, end: pd.Timestamp, is_adv: bool
    ) -> pd.Series:
        """Get the limit line from start to end, resetting when extended."""
        col = self.data["high"] if is_adv else self.data["low"]
        subset = col[start:end]

        srss = []
        while True:
            line = self.get_line(subset, is_adv)
            bv = subset > line if is_adv else subset < line
            idx = bv.argmax() if bv.any() else None
            srss.append(line[:idx])
            if idx is None or idx + 1 == len(bv):
                break
            subset = subset[idx:]

        return pd.concat(srss)

    def get_limit(
        self, start_conf: pd.Timestamp, is_adv: bool
    ) -> tuple[pd.Timestamp | None, float | None, pd.Series]:
        """Get bar and price when price has failed to extend limit line."""
        bv = self.limit_extended(is_adv).loc[start_conf:]
        if bv.all():
            bar, px = None, None
            bar_ = self.data.index[-1]
        else:
            bar = bar_ = bv.index[bv.argmin()]
            px = self.data.loc[bar, "close"]

        start = self.start_from_start_conf(start_conf)
        line = self.get_limit_line(start, bar_, is_adv)
        return bar, px, line

    def get_break(
        self, start_conf: pd.Timestamp, is_adv: bool
    ) -> tuple[pd.Timestamp | None, float | None, pd.Series]:
        """Get bar and price when price first breaks the break line."""
        start_confs_all = self.start_confs_all(is_adv)
        idx = start_confs_all.get_loc(start_conf)
        start_confs = start_confs_all[idx:]

        col = self.data["low"] if is_adv else self.data["high"]
        frms = pd.Series(start_confs).apply(self.start_from_start_conf)
        tos = frms.shift(-1)
        srss = []
        for frm, to in zip(frms, tos, strict=True):
            to_ = None if pd.isna(to) else to
            subset = col[frm:to_]
            line = self.get_line(subset, is_adv, limit=False)
            bv = subset < line if is_adv else subset > line
            if not bv.any():
                srss.append(line[1:])
                continue

            idx_brk = bv.argmax()
            srss.append(line[1 : idx_brk + 1])
            line_break = pd.concat(srss)
            break_ = line_break.index[-1]
            f = min if is_adv else max
            break_px = f(line_break.iloc[-1], self.data.loc[break_, "open"])
            return break_, break_px, line_break

        return None, None, pd.concat(srss)

    def get_end_conf(
        self, start_conf: pd.Timestamp, is_adv: bool
    ) -> tuple[
        pd.Timestamp | None,
        float | None,
        bool | None,
    ]:
        """Get bar and price when a movement is confirmed as having ended.

        Returns
        -------
        end_conf : pd.Timestamp | None
        end_conf_px : float | None
        by_break : bool | None
            True if ended by break, False if by limit, None if still open.
        """
        end_break, end_px_break, _ = self.get_break(start_conf, is_adv)
        end_limit, end_px_limit, _ = self.get_limit(start_conf, is_adv)

        if end_break is None and end_limit is None:
            return None, None, None
        if end_break is None:
            return end_limit, end_px_limit, False
        if end_limit is None:
            return end_break, end_px_break, True
        by_break = end_break <= end_limit
        px = end_px_break if by_break else end_px_limit
        return min(end_break, end_limit), px, by_break

    def get_end(
        self, start: pd.Timestamp, end_conf: pd.Timestamp | None, is_adv: bool
    ) -> tuple[pd.Timestamp, float]:
        """Get movement end bar and price (the extreme high/low)."""
        col = self.data["high"] if is_adv else self.data["low"]
        subset = col.loc[start:end_conf]
        _end_idx = subset.argmax() if is_adv else subset.argmin()
        end = subset.index[_end_idx]
        end_px = float(subset.max() if is_adv else subset.min())
        return end, end_px

    def get_start(
        self, start_conf: pd.Timestamp, is_adv: bool
    ) -> tuple[pd.Timestamp, float]:
        """Get movement start bar and price."""
        start = self.start_from_start_conf(start_conf)
        row = self.data.loc[start]
        return start, float(row["low"] if is_adv else row["high"])

    def get_duration(self, start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Get number of bars from start to end (inclusive)."""
        idx_start = self.data.index.get_loc(start)
        idx_end = self.data.index.get_loc(end)
        return idx_end - idx_start + 1

    def get_movement(self, start_conf: pd.Timestamp, is_adv: bool) -> Movement:
        """Get movement corresponding with a given confirmed start bar."""
        end_conf, end_conf_px, by_break = self.get_end_conf(start_conf, is_adv)
        start, start_px = self.get_start(start_conf, is_adv)
        end, end_px = self.get_end(start, end_conf, is_adv)
        start_conf_px = float(self.data.loc[start_conf, "close"])

        return Movement(
            is_adv=is_adv,
            start=start,
            start_px=start_px,
            start_conf=start_conf,
            start_conf_px=start_conf_px,
            end=end,
            end_px=end_px,
            end_conf=end_conf,
            end_conf_px=end_conf_px,
            duration=self.get_duration(start, end),
            params=self.params,
            by_break=by_break,
        )

    def _get_movements(self, is_adv: bool) -> list[Movement]:
        """Return list of movements of a given direction."""
        moves: list[Movement] = []
        start_confs_all = self.start_confs_all(is_adv)
        if len(start_confs_all) == 0:
            return moves

        while True:
            move = self.get_movement(start_confs_all[0], is_adv)
            if move.duration < self.min_bars:
                start_confs_all = start_confs_all[1:]
                if len(start_confs_all) == 0:
                    break
                continue
            moves.append(move)
            if move.end is None:
                break
            minconfstart = self.start_conf_from_start(move.end)
            if minconfstart is None:
                break
            idx = start_confs_all.get_slice_bound(minconfstart, side="right")
            start_confs_all = start_confs_all[idx:]
            if len(start_confs_all) == 0:
                break
        return moves

    def get_movements(self) -> list[Movement]:
        """Return all movements (advances and declines), sorted by start time."""
        advs = self._get_movements(is_adv=True)
        decs = self._get_movements(is_adv=False)
        return sorted(advs + decs)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all movements as a DataFrame.

        Columns: is_adv, start, start_px, start_conf, start_conf_px,
        end, end_px, end_conf, end_conf_px, duration, by_break, chg, chg_pct.
        """
        moves = self.get_movements()
        if not moves:
            return pd.DataFrame(
                columns=[
                    "is_adv",
                    "start",
                    "start_px",
                    "start_conf",
                    "start_conf_px",
                    "end",
                    "end_px",
                    "end_conf",
                    "end_conf_px",
                    "duration",
                    "by_break",
                    "chg",
                    "chg_pct",
                ]
            )
        records = []
        for m in moves:
            records.append(
                {
                    "is_adv": m.is_adv,
                    "start": m.start,
                    "start_px": m.start_px,
                    "start_conf": m.start_conf,
                    "start_conf_px": m.start_conf_px,
                    "end": m.end,
                    "end_px": m.end_px,
                    "end_conf": m.end_conf,
                    "end_conf_px": m.end_conf_px,
                    "duration": m.duration,
                    "by_break": m.by_break,
                    "chg": m.chg,
                    "chg_pct": m.chg_pct,
                }
            )
        return pd.DataFrame(records)

    def get_trend_signals(self) -> pd.Series:
        """Return a trend signal series aligned to self.data.index.

        Values:
            1  -> within an advance movement
            -1 -> within a decline movement
            0  -> no active movement

        This can be used as a feature or signal overlay for backtesting.
        """
        signals = pd.Series(0, index=self.data.index, dtype=int)
        for m in self.get_movements():
            signals.loc[m.start:m.end] = 1 if m.is_adv else -1
        return signals
