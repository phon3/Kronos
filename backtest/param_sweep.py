"""Parameter sweep for trend detection — find robust settings across parameter grid."""

import argparse
import itertools
import json
from datetime import datetime, timezone
import os
import random
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.crypto_backtest import CryptoBacktester
from backtest.backtest_runner import generate_predictions, load_model


def run_sweep(
    data_path: str,
    output_dir: str = "./backtest_results/param_sweep/",
    signal_mode: str = "trend",
    prd_grid: Optional[list] = None,
    ext_break_grid: Optional[list] = None,
    ext_limit_grid: Optional[list] = None,
    min_bars_grid: Optional[list] = None,
    position_size_pct: float = 0.25,
    stop_loss_pct: Optional[float] = 0.08,
    initial_capital: float = 100_000,
    walk_forward: bool = True,
    train_ratio: float = 0.7,
) -> pd.DataFrame:
    """Run parameter sweep across a grid of trend detection parameters.

    Args:
        data_path: Path to CSV with OHLCV data
        output_dir: Directory to save sweep results
        signal_mode: 'trend' or 'combined' (trend params are swept in both)
        prd_grid: List of prd values to test
        ext_break_grid: List of ext_break values to test
        ext_limit_grid: List of ext_limit values to test
        min_bars_grid: List of min_bars values to test
        position_size_pct: Position size for backtests (default 25% — more realistic)
        stop_loss_pct: Stop-loss percentage (default 8%)
        initial_capital: Starting capital
        walk_forward: If True, evaluate on out-of-sample portion only
        train_ratio: In-sample fraction for walk-forward split

    Returns:
        DataFrame with one row per parameter combination, sorted by OOS Sharpe
    """
    if prd_grid is None:
        prd_grid = [30, 60, 90, 120]
    if ext_break_grid is None:
        ext_break_grid = [0.03, 0.05, 0.08, 0.10]
    if ext_limit_grid is None:
        ext_limit_grid = [0.01, 0.02, 0.03]
    if min_bars_grid is None:
        min_bars_grid = [3, 5, 10]

    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    data["timestamps"] = pd.to_datetime(data["timestamps"])
    print(f"  {len(data)} rows, range: {data['timestamps'].min()} to {data['timestamps'].max()}")

    if walk_forward:
        n = len(data)
        split_idx = int(n * train_ratio)
        eval_data = data.iloc[split_idx:]
        print(f"  Walk-forward: evaluating on last {len(eval_data)} bars (out-of-sample)")
    else:
        eval_data = data

    actual = eval_data.set_index("timestamps")[["close"]]

    combos = list(itertools.product(prd_grid, ext_break_grid, ext_limit_grid, min_bars_grid))
    print(f"\nSweeping {len(combos)} parameter combinations...")
    print(f"{'prd':>5} {'ext_brk':>8} {'ext_lim':>8} {'min_brs':>8} {'trades':>7} {'return':>10} {'sharpe':>8} {'maxdd':>8} {'win%':>6}")
    print("-" * 78)

    results = []
    for prd, ext_break, ext_limit, min_bars in combos:
        try:
            bt = CryptoBacktester(
                initial_capital=initial_capital,
                threshold=0.02,
                allow_short=True,
                fee_pct=0.001,
                position_size_pct=position_size_pct,
                stop_loss_pct=stop_loss_pct,
            )
            metrics = bt.run(
                predictions=pd.DataFrame(),
                actual=actual,
                output_dir="/tmp/_sweep_tmp/",
                title=f"prd={prd} brk={ext_break} lim={ext_limit} mb={min_bars}",
                signal_mode="trend",
                full_data=eval_data,
                trend_prd=prd,
                trend_ext_break=ext_break,
                trend_ext_limit=ext_limit,
                trend_min_bars=min_bars,
            )

            row = {
                "prd": prd,
                "ext_break": ext_break,
                "ext_limit": ext_limit,
                "min_bars": min_bars,
                "total_return": metrics["total_return"],
                "annual_return": metrics["annual_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "total_trades": metrics["total_trades"],
                "profit_factor": metrics["profit_factor"],
                "buy_hold_return": metrics["buy_hold_return"],
            }
            results.append(row)

            print(
                f"{prd:>5} {ext_break:>8.3f} {ext_limit:>8.3f} {min_bars:>8} "
                f"{metrics['total_trades']:>7} {metrics['total_return']:>9.1%} "
                f"{metrics['sharpe_ratio']:>8.2f} {metrics['max_drawdown']:>8.1%} "
                f"{metrics['win_rate']:>5.0%}"
            )
        except Exception as e:
            print(f"{prd:>5} {ext_break:>8.3f} {ext_limit:>8.3f} {min_bars:>8}  ERROR: {e}")
            results.append({
                "prd": prd, "ext_break": ext_break, "ext_limit": ext_limit,
                "min_bars": min_bars, "total_return": None, "annual_return": None,
                "sharpe_ratio": None, "max_drawdown": None, "win_rate": None,
                "total_trades": 0, "profit_factor": None, "buy_hold_return": None,
            })

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe_ratio", ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print(f"\n{'=' * 78}")
    print("Top 5 by Sharpe Ratio (out-of-sample):")
    print(f"{'=' * 78}")
    top5 = df.head(5)
    for _, row in top5.iterrows():
        print(
            f"  prd={int(row['prd']):>3} brk={row['ext_break']:.3f} lim={row['ext_limit']:.3f} mb={int(row['min_bars']):>2} "
            f"| ret={row['total_return']:>8.1%} sharpe={row['sharpe_ratio']:>6.2f} "
            f"maxdd={row['max_drawdown']:>7.1%} trades={int(row['total_trades']):>3} win={row['win_rate']:.0%}"
        )

    print(f"\n{'=' * 78}")
    print("Top 5 by Total Return (out-of-sample):")
    print(f"{'=' * 78}")
    top5_ret = df.sort_values("total_return", ascending=False).head(5)
    for _, row in top5_ret.iterrows():
        print(
            f"  prd={int(row['prd']):>3} brk={row['ext_break']:.3f} lim={row['ext_limit']:.3f} mb={int(row['min_bars']):>2} "
            f"| ret={row['total_return']:>8.1%} sharpe={row['sharpe_ratio']:>6.2f} "
            f"maxdd={row['max_drawdown']:>7.1%} trades={int(row['total_trades']):>3} win={row['win_rate']:.0%}"
        )

    # Clean up temp dir
    import shutil
    shutil.rmtree("/tmp/_sweep_tmp/", ignore_errors=True)

    return df


def _grid_profile(name: str) -> dict:
    profiles = {
        "quick": {
            "prd": [30, 60], "ext_break": [0.03, 0.05], "ext_limit": [0.01, 0.02],
            "min_bars": [3, 5], "threshold": [0.003, 0.005], "combined_policy": ["continuous"],
            "macro_prd": [10], "macro_ext_break": [0.03], "macro_ext_limit": [0.03],
            "macro_min_bars": [3],
        },
        "standard": {
            "prd": [30, 60, 90], "ext_break": [0.02, 0.03, 0.05], "ext_limit": [0.01, 0.02],
            "min_bars": [3, 5, 10], "threshold": [0.002, 0.005, 0.01], "combined_policy": ["continuous"],
            "macro_prd": [10, 20], "macro_ext_break": [0.03, 0.05], "macro_ext_limit": [0.02, 0.03],
            "macro_min_bars": [3, 5],
        },
        "exhaustive": {
            "prd": [15, 30, 60, 90, 120], "ext_break": [0.01, 0.02, 0.03, 0.05, 0.08],
            "ext_limit": [0.005, 0.01, 0.02, 0.03], "min_bars": [2, 3, 5, 10],
            "threshold": [0.001, 0.002, 0.003, 0.005, 0.01, 0.02], "combined_policy": ["continuous", "entry_only"],
            "macro_prd": [5, 10, 20, 30], "macro_ext_break": [0.02, 0.03, 0.05],
            "macro_ext_limit": [0.01, 0.02, 0.03], "macro_min_bars": [2, 3, 5],
        },
    }
    if name not in profiles:
        raise ValueError(f"Unknown grid profile: {name}")
    return profiles[name]


def _infer_timeframe(data: pd.DataFrame) -> str:
    if len(data) < 2:
        return "unknown"
    minutes = data["timestamps"].sort_values().diff().dropna().median().total_seconds() / 60
    if minutes <= 15:
        return "15m"
    if minutes <= 60:
        return "1h"
    if minutes >= 20 * 60:
        return "1d"
    return f"{minutes:g}m"


def _model_name(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(model_path))))


def _risk_grid(name: str, overrides: Optional[dict] = None) -> list[dict]:
    presets = {
        "conservative": dict(position_size_pct=0.10, stop_loss_pct=0.04, take_profit_pct=0.08,
                             loss_multiplier=1.0, daily_loss_limit_pct=0.02, take_profit_levels=None),
        "balanced": dict(position_size_pct=0.25, stop_loss_pct=0.08, take_profit_pct=0.12,
                         loss_multiplier=1.0, daily_loss_limit_pct=0.03, take_profit_levels=None),
        "aggressive": dict(position_size_pct=0.50, stop_loss_pct=0.12, take_profit_pct=0.20,
                           loss_multiplier=1.5, daily_loss_limit_pct=0.05, take_profit_levels=None),
        "deeptrade": dict(position_size_pct=0.25, stop_loss_pct=0.08, take_profit_pct=None,
                          loss_multiplier=1.0, daily_loss_limit_pct=0.03,
                          take_profit_levels=[(0.05, 0.33), (0.10, 0.33), (0.15, 0.34)]),
    }
    custom = {key: value for key, value in (overrides or {}).items() if value is not None}
    if custom:
        positions = custom.get("position_size_pct", [0.10, 0.25, 0.50])
        stops = custom.get("stop_loss_pct", [0.04, 0.08, 0.12])
        take_profits = custom.get("take_profit_pct", [None, 0.08, 0.12, 0.20])
        multipliers = custom.get("loss_multiplier", [1.0, 1.5])
        daily_limits = custom.get("daily_loss_limit_pct", [None, 0.03])
        tp_sets = custom.get("take_profit_levels", [])
        configs = [dict(
            risk_profile="custom",
            position_size_pct=position,
            stop_loss_pct=stop,
            take_profit_pct=take_profit,
            loss_multiplier=multiplier,
            daily_loss_limit_pct=daily_limit,
            take_profit_levels=None,
        ) for position, stop, take_profit, multiplier, daily_limit in itertools.product(
            positions, stops, take_profits, multipliers, daily_limits
        )]
        configs.extend(dict(
            risk_profile="custom_multitarget",
            position_size_pct=position,
            stop_loss_pct=stop,
            take_profit_pct=None,
            loss_multiplier=multiplier,
            daily_loss_limit_pct=daily_limit,
            take_profit_levels=levels,
        ) for position, stop, multiplier, daily_limit, levels in itertools.product(
            positions, stops, multipliers, daily_limits, tp_sets
        ))
        return configs
    if name == "quick":
        return [{"risk_profile": key, **value} for key, value in presets.items()]
    positions = [0.10, 0.25, 0.50] if name == "standard" else [0.05, 0.10, 0.25, 0.50, 0.75]
    stops = [0.04, 0.08] if name == "standard" else [0.02, 0.04, 0.08, 0.12]
    take_profits = [None, 0.08, 0.12] if name == "standard" else [None, 0.05, 0.08, 0.12, 0.20]
    multipliers = [1.0, 1.5] if name == "standard" else [1.0, 1.25, 1.5, 2.0]
    daily_limits = [None, 0.03] if name == "standard" else [None, 0.02, 0.03, 0.05]
    configs = []
    for values in itertools.product(positions, stops, take_profits, multipliers, daily_limits):
        position, stop, take_profit, multiplier, daily_limit = values
        configs.append(dict(
            risk_profile="grid", position_size_pct=position, stop_loss_pct=stop,
            take_profit_pct=take_profit, loss_multiplier=multiplier,
            daily_loss_limit_pct=daily_limit, take_profit_levels=None,
        ))
    for position, stop, daily_limit in itertools.product(positions, stops, [0.02, 0.03]):
        configs.append(dict(
            risk_profile="deeptrade", position_size_pct=position, stop_loss_pct=stop,
            take_profit_pct=None, loss_multiplier=1.0, daily_loss_limit_pct=daily_limit,
            take_profit_levels=[(0.05, 0.33), (0.10, 0.33), (0.15, 0.34)],
        ))
    return configs


def _signal_grid(mode: str, grid: dict) -> list[dict]:
    trend = [
        dict(trend_prd=prd, trend_ext_break=ext_break, trend_ext_limit=ext_limit, trend_min_bars=min_bars)
        for prd, ext_break, ext_limit, min_bars in itertools.product(
            grid["prd"], grid["ext_break"], grid["ext_limit"], grid["min_bars"]
        )
    ]
    if mode == "trend":
        return trend
    if mode == "kronos":
        return [dict(threshold=threshold) for threshold in grid["threshold"]]
    if mode == "combined":
        return [
            {**params, "threshold": threshold, "combined_policy": policy}
            for params, threshold, policy in itertools.product(
                trend, grid["threshold"], grid.get("combined_policy", ["continuous"])
            )
        ]
    macro = [
        dict(macro_prd=prd, macro_ext_break=ext_break, macro_ext_limit=ext_limit, macro_min_bars=min_bars)
        for prd, ext_break, ext_limit, min_bars in itertools.product(
            grid["macro_prd"], grid["macro_ext_break"], grid["macro_ext_limit"], grid["macro_min_bars"]
        )
    ]
    if mode == "multi_tf":
        return [{**entry, **macro_params} for entry, macro_params in itertools.product(trend, macro)]
    raise ValueError(f"Unsupported signal mode: {mode}")


def _combine_signals(
    kronos: pd.DataFrame,
    trend: pd.DataFrame,
    allow_short: bool,
    confirmation_policy: str = "continuous",
) -> pd.DataFrame:
    combined = kronos.copy()
    combined["kronos_position"] = combined["position"].astype(int)
    combined["trend_signal"] = trend["trend_signal"].reindex(combined.index).fillna(0)
    combined["trend_position"] = trend["position"].reindex(combined.index).ffill().fillna(0).astype(int)
    combined["signal"] = 0
    combined.loc[(combined["kronos_position"] > 0) & (combined["trend_position"] > 0), "signal"] = 1
    if allow_short:
        combined.loc[(combined["kronos_position"] < 0) & (combined["trend_position"] < 0), "signal"] = -1
    if confirmation_policy == "continuous":
        combined["position"] = combined["signal"].astype(int)
    elif confirmation_policy == "entry_only":
        state = 0
        positions = []
        for kronos_position, trend_position in zip(
            combined["kronos_position"], combined["trend_position"], strict=True
        ):
            if state == 0:
                if (
                    kronos_position == trend_position
                    and trend_position != 0
                    and (trend_position > 0 or allow_short)
                ):
                    state = int(trend_position)
            elif trend_position != state:
                can_enter = kronos_position == trend_position and (trend_position > 0 or allow_short)
                state = int(trend_position) if can_enter else 0
            positions.append(state)
        combined["position"] = positions
    else:
        raise ValueError(f"Unknown confirmation policy: {confirmation_policy}")
    return combined


def _metric_columns(prefix: str, metrics: dict) -> dict:
    keys = [
        "total_return", "annual_return", "sharpe_ratio", "max_drawdown", "win_rate",
        "profit_factor", "total_trades", "total_pnl", "final_capital", "buy_hold_return",
        "completed_trades", "partial_exits", "signal_exits", "stop_loss_exits",
        "take_profit_exits", "end_exits", "average_holding_hours", "median_holding_hours",
        "max_holding_hours", "average_gross_exposure", "average_net_exposure",
        "exposure_matched_buy_hold_return", "excess_vs_matched_exposure",
    ]
    return {f"{prefix}_{key}": metrics.get(key) for key in keys}


def _trade_diagnostics(trades: list) -> dict:
    closes = [trade for trade in trades if trade["action"] == "CLOSE"]
    partials = [trade for trade in trades if trade["action"] == "PARTIAL_CLOSE"]
    durations = []
    opened_at = None
    for trade in trades:
        if trade["action"] == "OPEN":
            opened_at = pd.Timestamp(trade["timestamp"])
        elif trade["action"] == "CLOSE" and opened_at is not None:
            durations.append((pd.Timestamp(trade["timestamp"]) - opened_at).total_seconds() / 3600)
            opened_at = None
    return {
        "completed_trades": len(closes),
        "partial_exits": len(partials),
        "signal_exits": sum(trade.get("reason") == "SIGNAL" for trade in closes),
        "stop_loss_exits": sum(trade.get("reason") == "STOP_LOSS" for trade in closes),
        "take_profit_exits": sum(trade.get("reason") == "TAKE_PROFIT" for trade in closes),
        "end_exits": sum(trade.get("reason") == "END" for trade in closes),
        "average_holding_hours": float(np.mean(durations)) if durations else 0,
        "median_holding_hours": float(np.median(durations)) if durations else 0,
        "max_holding_hours": max(durations, default=0),
    }


def _signal_diagnostics(signals: pd.DataFrame) -> dict:
    position = signals["position"].fillna(0).astype(int)
    diagnostics = {
        "signal_bars": len(signals),
        "long_position_bars": int((position > 0).sum()),
        "short_position_bars": int((position < 0).sum()),
        "flat_position_bars": int((position == 0).sum()),
        "position_entries": int(((position != position.shift()) & (position != 0)).sum()),
    }
    for prefix, column in (("kronos", "kronos_position"), ("trend", "trend_position")):
        if column in signals:
            values = signals[column].fillna(0).astype(int)
            diagnostics[f"{prefix}_long_bars"] = int((values > 0).sum())
            diagnostics[f"{prefix}_short_bars"] = int((values < 0).sum())
            diagnostics[f"{prefix}_neutral_bars"] = int((values == 0).sum())
    if "trend_signal" in signals:
        raw_trend = signals["trend_signal"].fillna(0).astype(int)
        diagnostics["raw_trend_long_bars"] = int((raw_trend > 0).sum())
        diagnostics["raw_trend_short_bars"] = int((raw_trend < 0).sum())
        diagnostics["raw_trend_neutral_bars"] = int((raw_trend == 0).sum())
    return diagnostics


def _score_row(
    row: dict,
    rank_by: str,
    min_trades: int,
    target_trades: int = 30,
    min_excess_return: float = 0,
) -> float:
    trades = row.get("oos_completed_trades", row.get("oos_total_trades", 0)) or 0
    if trades < min_trades:
        return float("-inf")
    sharpe = row.get("oos_sharpe_ratio", 0) or 0
    total_return = row.get("oos_total_return", 0) or 0
    benchmark_return = row.get("oos_buy_hold_return", 0) or 0
    if total_return - benchmark_return < min_excess_return:
        return float("-inf")
    drawdown = abs(row.get("oos_max_drawdown", 0) or 0)
    if not all(np.isfinite(value) for value in (sharpe, total_return, benchmark_return, drawdown)):
        return float("-inf")
    if rank_by == "sharpe":
        return sharpe
    if rank_by == "return":
        return total_return
    if rank_by == "calmar":
        return total_return / drawdown if drawdown > 0 else float("-inf")
    confidence = min(1.0, np.sqrt(trades / max(target_trades, 1)))
    is_sharpe = row.get("is_sharpe_ratio", 0) or 0
    stability_gap = abs(sharpe - is_sharpe) if np.isfinite(is_sharpe) else abs(sharpe)
    excess_return = total_return - benchmark_return
    profitable_folds = row.get("oos_profitable_folds", 0) or 0
    validation_folds = row.get("oos_validation_folds", 1) or 1
    fold_consistency = profitable_folds / validation_folds
    worst_fold_return = row.get("oos_worst_fold_return", 0) or 0
    adjusted_sharpe = np.clip(sharpe, -5, 5) * confidence
    return (
        adjusted_sharpe + 4.0 * excess_return - 2.0 * drawdown
        - 0.25 * min(stability_gap, 5) + fold_consistency
        + 2.0 * min(worst_fold_return, 0)
    )


def _evaluate_signals(
    signals: pd.DataFrame,
    risk: dict,
    initial_capital: float,
    train_ratio: float,
    validation_folds: int = 3,
    split_timestamp: Optional[pd.Timestamp] = None,
    allow_short: bool = True,
) -> dict:
    if split_timestamp is not None and isinstance(signals.index, pd.DatetimeIndex):
        segments = {
            "is": signals[signals.index < split_timestamp],
            "oos": signals[signals.index >= split_timestamp],
        }
    else:
        split_idx = max(1, min(len(signals) - 1, int(len(signals) * train_ratio)))
        segments = {"is": signals.iloc[:split_idx], "oos": signals.iloc[split_idx:]}

    def evaluate(segment: pd.DataFrame) -> dict:
        backtester = CryptoBacktester(initial_capital=initial_capital, allow_short=allow_short, **{
            key: value for key, value in risk.items() if key != "risk_profile"
        })
        results, trades = backtester.run_backtest(segment)
        metrics = backtester.calculate_metrics(results, trades)
        metrics.update(_trade_diagnostics(trades))
        return metrics

    output = {}
    for prefix, segment in segments.items():
        output.update(_metric_columns(prefix, evaluate(segment)))
    fold_indices = np.array_split(np.arange(len(segments["oos"])), validation_folds)
    folds = [segments["oos"].iloc[indices] for indices in fold_indices if len(indices)]
    fold_metrics = [evaluate(fold) for fold in folds]
    fold_returns = [metrics["total_return"] for metrics in fold_metrics]
    output["oos_validation_folds"] = len(fold_metrics)
    output["oos_profitable_folds"] = sum(value > 0 for value in fold_returns)
    output["oos_worst_fold_return"] = min(fold_returns, default=0)
    output["oos_median_fold_return"] = float(np.median(fold_returns)) if fold_returns else 0
    output["oos_worst_fold_sharpe"] = min(
        (metrics["sharpe_ratio"] for metrics in fold_metrics), default=0
    )
    for index, metrics in enumerate(fold_metrics, start=1):
        output[f"oos_fold_{index}_return"] = metrics["total_return"]
        output[f"oos_fold_{index}_trades"] = metrics["total_trades"]
    return output


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _export_candidates(results: pd.DataFrame, output_dir: str, top_n: int) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    results.sort_values("score", ascending=False).to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
    valid = results[np.isfinite(results["score"])].sort_values("score", ascending=False).copy()
    if valid.empty:
        top = valid.copy()
    else:
        performance_columns = [
            "oos_total_return", "oos_sharpe_ratio", "oos_max_drawdown", "oos_win_rate", "oos_total_trades"
        ]
        signatures = valid[performance_columns].round(6).astype(str).apply(lambda row: "|".join(row), axis=1)
        valid["performance_signature"] = signatures
        top = valid.drop_duplicates("performance_signature", keep="first").head(top_n).copy()
        top = top.drop(columns=["performance_signature"])
    top.insert(0, "candidate_rank", range(1, len(top) + 1))
    top.to_csv(os.path.join(output_dir, "top_candidates.csv"), index=False)
    records = []
    metric_prefixes = ("is_", "oos_")
    for row in top.to_dict("records"):
        records.append({
            "rank": row["candidate_rank"],
            "score": _json_value(row["score"]),
            "settings": {key: _json_value(value) for key, value in row.items()
                         if not key.startswith(metric_prefixes) and key not in ("candidate_rank", "score", "signal_signature")},
            "metrics": {key: _json_value(value) for key, value in row.items() if key.startswith(metric_prefixes)},
        })
    with open(os.path.join(output_dir, "live_test_candidates.json"), "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    metadata_columns = [
        "signal_mode", "data_file", "data_timeframe", "data_start", "data_end",
        "model_name", "model_path", "model_timeframe", "macro_data_file", "macro_timeframe",
        "evaluation_start", "evaluation_end", "oos_start", "oos_end", "split_timestamp",
        "lookback", "pred_len", "sample_count", "max_data_rows", "seed",
    ]
    available_metadata = [column for column in metadata_columns if column in results.columns]
    experiments = results[available_metadata].drop_duplicates().to_dict("records") if available_metadata else []
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_count": len(results),
        "qualified_result_count": len(valid),
        "candidate_count": len(top),
        "experiments": _json_value(experiments),
    }
    with open(os.path.join(output_dir, "experiment_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return top


def run_optimizer(
    data_path: str,
    output_dir: str = "./backtest_results/optimizer/",
    signal_mode: str = "trend",
    grid_profile: str = "quick",
    top_n: int = 5,
    rank_by: str = "composite",
    min_trades: int = 10,
    target_trades: int = 30,
    min_excess_return: float = 0,
    validation_folds: int = 3,
    train_ratio: float = 0.7,
    initial_capital: float = 100_000,
    max_data_rows: Optional[int] = None,
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    macro_data_path: Optional[str] = None,
    device: str = "cpu",
    lookback: int = 512,
    pred_len: int = 48,
    sample_count: int = 1,
    seed: int = 123,
    allow_short: bool = True,
    max_combinations: int = 20_000,
    dry_run: bool = False,
    grid_overrides: Optional[dict] = None,
    risk_overrides: Optional[dict] = None,
    fixed_position_size: Optional[float] = None,
    fixed_stop_loss: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimize signal and risk settings and export robust out-of-sample candidates."""
    grid = _grid_profile(grid_profile)
    if grid_overrides:
        grid.update({key: value for key, value in grid_overrides.items() if value is not None})
    signal_configs = _signal_grid(signal_mode, grid)
    risk_configs = _risk_grid(grid_profile, overrides=risk_overrides)
    if fixed_position_size is not None:
        for risk in risk_configs:
            risk["position_size_pct"] = fixed_position_size
    if fixed_stop_loss is not None:
        for risk in risk_configs:
            risk["stop_loss_pct"] = fixed_stop_loss
    risk_configs = list({json.dumps(config, sort_keys=True): config for config in risk_configs}.values())
    combination_count = len(signal_configs) * len(risk_configs)
    print(f"Grid: {len(signal_configs)} signal × {len(risk_configs)} risk = {combination_count} combinations")
    if dry_run:
        return pd.DataFrame(), pd.DataFrame()
    if combination_count > max_combinations:
        raise ValueError(
            f"Grid has {combination_count} combinations, above --max-combinations={max_combinations}. "
            "Use a smaller profile or raise the safety limit explicitly."
        )

    data = pd.read_csv(data_path)
    data["timestamps"] = pd.to_datetime(data["timestamps"])
    data = data.sort_values("timestamps").reset_index(drop=True)
    if max_data_rows is not None:
        data = data.tail(max_data_rows).reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("Backtest data must contain at least two rows")
    split_idx = max(1, min(len(data) - 1, int(len(data) * train_ratio)))
    split_timestamp = data["timestamps"].iloc[split_idx]
    data_file = os.path.basename(data_path)
    data_timeframe = _infer_timeframe(data)
    data_start = data["timestamps"].iloc[0]
    data_end = data["timestamps"].iloc[-1]
    model_name = _model_name(model_path)
    model_timeframe = next((tf for tf in ("15m", "1h", "1d") if model_name and f"_{tf}" in model_name), None)

    predictions = None
    actual = data.set_index("timestamps")[["close"]]
    if signal_mode in ("kronos", "combined"):
        if not model_path or not tokenizer_path:
            raise ValueError("model_path and tokenizer_path are required for Kronos modes")
        model_timeframe = next((tf for tf in ("15m", "1h", "1d") if f"_{tf}" in model_path.lower()), None)
        data_timeframe = next((tf for tf in ("15m", "1h", "1d") if f"_{tf}" in data_path.lower()), None)
        if model_timeframe and data_timeframe and model_timeframe != data_timeframe:
            raise ValueError(f"Model timeframe {model_timeframe} does not match data timeframe {data_timeframe}")
        if len(data) < lookback + pred_len:
            raise ValueError(f"Model optimization needs at least {lookback + pred_len} data rows")
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        predictor = load_model(tokenizer_path, model_path, device=device, max_context=lookback)
        predictions = generate_predictions(
            predictor, data, lookback=lookback, pred_len=pred_len, sample_count=sample_count
        )
        actual = data[data["timestamps"].isin(predictions.index)].set_index("timestamps")[["close"]]

    macro_data = None
    macro_data_file = None
    macro_timeframe = None
    if signal_mode == "multi_tf":
        if not macro_data_path:
            raise ValueError("macro_data_path is required for multi_tf mode")
        macro_data = pd.read_csv(macro_data_path)
        macro_data["timestamps"] = pd.to_datetime(macro_data["timestamps"])
        macro_data = macro_data.sort_values("timestamps").reset_index(drop=True)
        if max_data_rows is not None:
            macro_data = macro_data.tail(max_data_rows).reset_index(drop=True)
        macro_data_file = os.path.basename(macro_data_path)
        macro_timeframe = _infer_timeframe(macro_data)

    trend_cache = {}
    kronos_cache = {}
    results = []
    for signal_index, signal_config in enumerate(signal_configs, start=1):
        signal_bt = CryptoBacktester(threshold=signal_config.get("threshold", 0.005), allow_short=allow_short)
        trend_key = tuple(signal_config.get(key) for key in (
            "trend_prd", "trend_ext_break", "trend_ext_limit", "trend_min_bars"
        ))
        if signal_mode in ("trend", "combined") and trend_key not in trend_cache:
            trend_cache[trend_key] = signal_bt.generate_trend_signals(
                data, prd=signal_config["trend_prd"], ext_break=signal_config["trend_ext_break"],
                ext_limit=signal_config["trend_ext_limit"], min_bars=signal_config["trend_min_bars"]
            )
        if signal_mode == "trend":
            signals = trend_cache[trend_key]
        elif signal_mode == "kronos":
            threshold = signal_config["threshold"]
            if threshold not in kronos_cache:
                kronos_cache[threshold] = signal_bt.generate_signals(predictions, actual)
            signals = kronos_cache[threshold]
        elif signal_mode == "combined":
            threshold = signal_config["threshold"]
            if threshold not in kronos_cache:
                kronos_cache[threshold] = signal_bt.generate_signals(predictions, actual)
            signals = _combine_signals(
                kronos_cache[threshold],
                trend_cache[trend_key],
                allow_short,
                confirmation_policy=signal_config.get("combined_policy", "continuous"),
            )
        else:
            signals = signal_bt.generate_multi_tf_signals(
                data, macro_data,
                entry_prd=signal_config["trend_prd"], entry_ext_break=signal_config["trend_ext_break"],
                entry_ext_limit=signal_config["trend_ext_limit"], entry_min_bars=signal_config["trend_min_bars"],
                macro_prd=signal_config["macro_prd"], macro_ext_break=signal_config["macro_ext_break"],
                macro_ext_limit=signal_config["macro_ext_limit"], macro_min_bars=signal_config["macro_min_bars"],
            )
        signal_diagnostics = _signal_diagnostics(signals)
        evaluation_start = signals.index.min() if len(signals) else None
        evaluation_end = signals.index.max() if len(signals) else None
        oos_signals = signals[signals.index >= split_timestamp] if isinstance(signals.index, pd.DatetimeIndex) else signals
        oos_signal_diagnostics = {f"oos_{key}": value for key, value in _signal_diagnostics(oos_signals).items()}
        oos_start = oos_signals.index.min() if len(oos_signals) else None
        oos_end = oos_signals.index.max() if len(oos_signals) else None
        signature = json.dumps(signal_config, sort_keys=True)
        for risk in risk_configs:
            row = {
                "signal_mode": signal_mode,
                "data_path": data_path,
                "data_file": data_file,
                "data_timeframe": data_timeframe,
                "data_start": str(data_start),
                "data_end": str(data_end),
                "model_name": model_name,
                "model_path": model_path,
                "model_timeframe": model_timeframe,
                "macro_data_file": macro_data_file,
                "macro_timeframe": macro_timeframe,
                "evaluation_start": str(evaluation_start) if evaluation_start is not None else None,
                "evaluation_end": str(evaluation_end) if evaluation_end is not None else None,
                "oos_start": str(oos_start) if oos_start is not None else None,
                "oos_end": str(oos_end) if oos_end is not None else None,
                "lookback": lookback,
                "pred_len": pred_len,
                "sample_count": sample_count,
                "max_data_rows": len(data),
                "seed": seed,
                "allow_short": allow_short,
                "split_timestamp": str(split_timestamp),
                **signal_diagnostics,
                **oos_signal_diagnostics,
                **signal_config,
                **risk,
                "signal_signature": signature,
            }
            try:
                row.update(_evaluate_signals(
                    signals,
                    risk,
                    initial_capital,
                    train_ratio,
                    validation_folds=validation_folds,
                    split_timestamp=split_timestamp,
                    allow_short=allow_short,
                ))
                row["oos_excess_return"] = row["oos_total_return"] - row["oos_buy_hold_return"]
                row["trade_confidence"] = min(
                    1.0, np.sqrt(row["oos_completed_trades"] / max(target_trades, 1))
                )
                row["sharpe_stability_gap"] = abs(row["oos_sharpe_ratio"] - row["is_sharpe_ratio"])
                row["score"] = _score_row(
                    row,
                    rank_by,
                    min_trades,
                    target_trades=target_trades,
                    min_excess_return=min_excess_return,
                )
                row["status"] = "valid" if np.isfinite(row["score"]) else "filtered"
            except Exception as error:
                row.update(score=float("-inf"), status="error", error=str(error))
            results.append(row)
        print(f"Signal configuration {signal_index}/{len(signal_configs)} complete")

    results_df = pd.DataFrame(results)
    top_df = _export_candidates(results_df, output_dir, top_n)
    print(f"Saved {len(results_df)} results and {len(top_df)} live-test candidates to {output_dir}")
    if len(top_df):
        columns = ["candidate_rank", "score", "oos_total_return", "oos_sharpe_ratio", "oos_max_drawdown", "oos_total_trades"]
        print(top_df[columns].to_string(index=False))
    return results_df, top_df


def run_optimizer_matrix(
    lookbacks: list[int],
    pred_lengths: list[int],
    sample_counts: list[int],
    output_dir: str,
    top_n: int = 5,
    dry_run: bool = False,
    **optimizer_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run and rank multiple inference configurations as one experiment."""
    inference_configs = list(itertools.product(lookbacks, pred_lengths, sample_counts))
    print(f"Inference matrix: {len(inference_configs)} configurations")
    if dry_run:
        for lookback, pred_len, sample_count in inference_configs:
            print(f"  lookback={lookback} pred_len={pred_len} samples={sample_count}")
        return pd.DataFrame(), pd.DataFrame()
    result_frames = []
    for index, (lookback, pred_len, sample_count) in enumerate(inference_configs, start=1):
        run_output = os.path.join(output_dir, f"inference_lb{lookback}_pl{pred_len}_s{sample_count}")
        print(
            f"Inference configuration {index}/{len(inference_configs)}: "
            f"lookback={lookback}, pred_len={pred_len}, samples={sample_count}"
        )
        results, _ = run_optimizer(
            output_dir=run_output,
            top_n=top_n,
            lookback=lookback,
            pred_len=pred_len,
            sample_count=sample_count,
            dry_run=False,
            **optimizer_kwargs,
        )
        result_frames.append(results)
    all_results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    top = _export_candidates(all_results, output_dir, top_n)
    print(f"Inference matrix saved {len(all_results)} results and {len(top)} candidates to {output_dir}")
    return all_results, top


def _parse_list(value: Optional[str], value_type) -> Optional[list]:
    return [value_type(item.strip()) for item in value.split(",") if item.strip()] if value else None


def _parse_optional_floats(value: Optional[str]) -> Optional[list]:
    if not value:
        return None
    return [None if item.strip().lower() == "none" else float(item) for item in value.split(",")]


def _parse_tp_level_sets(value: Optional[str]) -> Optional[list]:
    if not value:
        return None
    level_sets = []
    for raw_set in value.split(";"):
        if not raw_set.strip() or raw_set.strip().lower() == "none":
            continue
        levels = [tuple(float(part) for part in level.split(":")) for level in raw_set.split(",")]
        if any(len(level) != 2 or level[0] <= 0 or level[1] <= 0 for level in levels):
            raise ValueError(f"Invalid TP level set: {raw_set}")
        if not np.isclose(sum(level[1] for level in levels), 1.0):
            raise ValueError(f"TP fractions must total 1.0: {raw_set}")
        level_sets.append(levels)
    return level_sets


def main():
    parser = argparse.ArgumentParser(description="Optimize Kronos backtest settings and select live-test candidates")
    parser.add_argument("--data", required=True, help="Path to CSV with OHLCV data")
    parser.add_argument("--output", default="./backtest_results/optimizer/", help="Output directory")
    parser.add_argument("--signal-mode", choices=["trend", "kronos", "combined", "multi_tf"], default="trend")
    parser.add_argument("--grid-profile", choices=["quick", "standard", "exhaustive"], default="quick")
    parser.add_argument("--top-n", type=int, choices=range(1, 21), default=5)
    parser.add_argument("--rank-by", choices=["composite", "sharpe", "return", "calmar"], default="composite")
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--target-trades", type=int, default=30)
    parser.add_argument("--min-excess-return", type=float, default=0)
    parser.add_argument("--validation-folds", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--max-data-rows", type=int, default=5000)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--macro-data", default=None)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--lookback", type=int, default=512)
    parser.add_argument("--pred-len", type=int, default=48)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--lookbacks", default=None)
    parser.add_argument("--pred-lens", default=None)
    parser.add_argument("--sample-counts", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-combinations", type=int, default=20_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prd", default=None)
    parser.add_argument("--ext-break", default=None)
    parser.add_argument("--ext-limit", default=None)
    parser.add_argument("--min-bars", default=None)
    parser.add_argument("--threshold", default=None)
    parser.add_argument("--combined-policies", default=None)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--position-size", type=float, default=None)
    parser.add_argument("--stop-loss", type=float, default=None)
    parser.add_argument("--position-sizes", default=None)
    parser.add_argument("--stop-losses", default=None)
    parser.add_argument("--take-profits", default=None)
    parser.add_argument("--loss-multipliers", default=None)
    parser.add_argument("--daily-loss-limits", default=None)
    parser.add_argument("--tp-level-sets", default=None)
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be between 0 and 1")
    if args.min_trades < 0 or args.target_trades < 1 or args.validation_folds < 1:
        parser.error("trade thresholds and validation folds must be positive")
    if args.signal_mode in ("kronos", "combined") and (not args.model or not args.tokenizer):
        parser.error("--model and --tokenizer are required for Kronos modes")
    if args.signal_mode == "multi_tf" and not args.macro_data:
        parser.error("--macro-data is required for multi_tf mode")

    overrides = {
        "prd": _parse_list(args.prd, int),
        "ext_break": _parse_list(args.ext_break, float),
        "ext_limit": _parse_list(args.ext_limit, float),
        "min_bars": _parse_list(args.min_bars, int),
        "threshold": _parse_list(args.threshold, float),
        "combined_policy": _parse_list(args.combined_policies, str),
    }
    policies = overrides.get("combined_policy") or []
    if any(policy not in ("continuous", "entry_only") for policy in policies):
        parser.error("--combined-policies supports continuous and entry_only")
    try:
        risk_overrides = {
            "position_size_pct": _parse_list(args.position_sizes, float),
            "stop_loss_pct": _parse_list(args.stop_losses, float),
            "take_profit_pct": _parse_optional_floats(args.take_profits),
            "loss_multiplier": _parse_list(args.loss_multipliers, float),
            "daily_loss_limit_pct": _parse_optional_floats(args.daily_loss_limits),
            "take_profit_levels": _parse_tp_level_sets(args.tp_level_sets),
        }
    except ValueError as error:
        parser.error(str(error))
    optimizer_kwargs = dict(
        data_path=args.data,
        signal_mode=args.signal_mode,
        grid_profile=args.grid_profile,
        rank_by=args.rank_by,
        min_trades=args.min_trades,
        target_trades=args.target_trades,
        min_excess_return=args.min_excess_return,
        validation_folds=args.validation_folds,
        train_ratio=args.train_ratio,
        initial_capital=args.capital,
        max_data_rows=args.max_data_rows,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        macro_data_path=args.macro_data,
        device=args.device,
        seed=args.seed,
        allow_short=not args.long_only,
        max_combinations=args.max_combinations,
        grid_overrides=overrides,
        risk_overrides=risk_overrides,
        fixed_position_size=args.position_size,
        fixed_stop_loss=args.stop_loss,
    )
    lookbacks = _parse_list(args.lookbacks, int) or [args.lookback]
    pred_lengths = _parse_list(args.pred_lens, int) or [args.pred_len]
    sample_counts = _parse_list(args.sample_counts, int) or [args.samples]
    if any(value < 1 for value in lookbacks + pred_lengths + sample_counts):
        parser.error("inference grid values must be positive")
    if len(lookbacks) * len(pred_lengths) * len(sample_counts) > 1:
        run_optimizer_matrix(
            lookbacks=lookbacks,
            pred_lengths=pred_lengths,
            sample_counts=sample_counts,
            output_dir=args.output,
            top_n=args.top_n,
            dry_run=args.dry_run,
            **optimizer_kwargs,
        )
    else:
        run_optimizer(
            output_dir=args.output,
            top_n=args.top_n,
            lookback=lookbacks[0],
            pred_len=pred_lengths[0],
            sample_count=sample_counts[0],
            dry_run=args.dry_run,
            **optimizer_kwargs,
        )


if __name__ == "__main__":
    main()
