"""Parameter sweep for trend detection — find robust settings across parameter grid."""

import argparse
import itertools
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.crypto_backtest import CryptoBacktester


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


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for trend detection")
    parser.add_argument("--data", required=True, help="Path to CSV with OHLCV data")
    parser.add_argument("--output", default="./backtest_results/param_sweep/", help="Output directory")
    parser.add_argument("--signal-mode", choices=["trend", "combined"], default="trend", help="Signal mode to sweep")
    parser.add_argument("--prd", type=str, default="30,60,90,120", help="Comma-separated prd values")
    parser.add_argument("--ext-break", type=str, default="0.03,0.05,0.08,0.10", help="Comma-separated ext_break values")
    parser.add_argument("--ext-limit", type=str, default="0.01,0.02,0.03", help="Comma-separated ext_limit values")
    parser.add_argument("--min-bars", type=str, default="3,5,10", help="Comma-separated min_bars values")
    parser.add_argument("--position-size", type=float, default=0.25, help="Position size fraction (default 0.25)")
    parser.add_argument("--stop-loss", type=float, default=0.08, help="Stop-loss percentage (default 0.08)")
    parser.add_argument("--no-walk-forward", action="store_true", help="Use full dataset (no walk-forward split)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="In-sample fraction for walk-forward")

    args = parser.parse_args()

    prd_grid = [int(x) for x in args.prd.split(",")]
    ext_break_grid = [float(x) for x in args.ext_break.split(",")]
    ext_limit_grid = [float(x) for x in args.ext_limit.split(",")]
    min_bars_grid = [int(x) for x in args.min_bars.split(",")]

    run_sweep(
        data_path=args.data,
        output_dir=args.output,
        signal_mode=args.signal_mode,
        prd_grid=prd_grid,
        ext_break_grid=ext_break_grid,
        ext_limit_grid=ext_limit_grid,
        min_bars_grid=min_bars_grid,
        position_size_pct=args.position_size,
        stop_loss_pct=args.stop_loss,
        walk_forward=not args.no_walk_forward,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
