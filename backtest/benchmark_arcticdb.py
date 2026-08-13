"""Benchmark ArcticDB vs CSV for OHLCV data read/write operations.

Tests on real BTC data (15m, 91K rows) to determine if ArcticDB's
columnar storage and date-range slicing provide meaningful speedups
over plain CSV files for our use case.

Usage:
    python -m backtest.benchmark_arcticdb --data ./data/BTC_USD_15m.csv
"""

import argparse
import os
import shutil
import time

import pandas as pd


def benchmark_csv(data_path: str, runs: int = 3) -> dict:
    """Benchmark CSV read/write operations."""
    df = pd.read_csv(data_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    n_rows = len(df)

    # Write benchmark
    write_times = []
    tmp_csv = "/tmp/_bench_data.csv"
    for _ in range(runs):
        t0 = time.perf_counter()
        df.to_csv(tmp_csv, index=False)
        write_times.append(time.perf_counter() - t0)
    csv_size = os.path.getsize(tmp_csv)

    # Full read benchmark
    read_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = pd.read_csv(data_path)
        read_times.append(time.perf_counter() - t0)

    # Date-range slice benchmark (last 30 days)
    end_ts = df["timestamps"].max()
    start_ts = end_ts - pd.Timedelta(days=30)
    slice_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        loaded = pd.read_csv(data_path)
        loaded["timestamps"] = pd.to_datetime(loaded["timestamps"])
        sliced = loaded[(loaded["timestamps"] >= start_ts) & (loaded["timestamps"] <= end_ts)]
        slice_times.append(time.perf_counter() - t0)

    # Column slice benchmark (close only, last 30 days)
    col_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        loaded = pd.read_csv(data_path, usecols=["timestamps", "close"])
        loaded["timestamps"] = pd.to_datetime(loaded["timestamps"])
        sliced = loaded[(loaded["timestamps"] >= start_ts) & (loaded["timestamps"] <= end_ts)]
        col_times.append(time.perf_counter() - t0)

    os.remove(tmp_csv)

    return {
        "n_rows": n_rows,
        "file_size_mb": csv_size / 1024 / 1024,
        "write_avg_ms": sum(write_times) / runs * 1000,
        "full_read_avg_ms": sum(read_times) / runs * 1000,
        "date_slice_avg_ms": sum(slice_times) / runs * 1000,
        "col_slice_avg_ms": sum(col_times) / runs * 1000,
    }


def benchmark_arcticdb(data_path: str, runs: int = 3) -> dict:
    """Benchmark ArcticDB local LMDB store read/write operations."""
    import arcticdb as adb

    df = pd.read_csv(data_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.set_index("timestamps")
    n_rows = len(df)

    # Setup local LMDB store
    db_path = "/tmp/_bench_arcticdb"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    ac = adb.Arctic(f"lmdb://{db_path}")
    lib = ac.create_library("bench")

    # Write benchmark
    write_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        lib.write("btc_15m", df)
        write_times.append(time.perf_counter() - t0)

    # Full read benchmark
    read_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = lib.read("btc_15m").data
        read_times.append(time.perf_counter() - t0)

    # Date-range slice benchmark (last 30 days) — ArcticDB native slicing
    end_ts = df.index.max()
    start_ts = end_ts - pd.Timedelta(days=30)
    slice_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = lib.read("btc_15m", date_range=(start_ts, end_ts)).data
        slice_times.append(time.perf_counter() - t0)

    # Column slice benchmark (close only, last 30 days)
    col_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = lib.read("btc_15m", date_range=(start_ts, end_ts), columns=["close"]).data
        col_times.append(time.perf_counter() - t0)

    # Measure LMDB size on disk
    db_size = 0
    for dirpath, _, filenames in os.walk(db_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            db_size += os.path.getsize(fp)

    shutil.rmtree(db_path)

    return {
        "n_rows": n_rows,
        "file_size_mb": db_size / 1024 / 1024,
        "write_avg_ms": sum(write_times) / runs * 1000,
        "full_read_avg_ms": sum(read_times) / runs * 1000,
        "date_slice_avg_ms": sum(slice_times) / runs * 1000,
        "col_slice_avg_ms": sum(col_times) / runs * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark ArcticDB vs CSV for OHLCV data")
    parser.add_argument("--data", required=True, help="Path to CSV file to benchmark with")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"ArcticDB vs CSV Benchmark")
    print(f"Data: {args.data} ({os.path.basename(args.data)})")
    print(f"Runs: {args.runs} (averaged)")
    print(f"{'=' * 70}")

    print("\nRunning CSV benchmark...")
    csv_results = benchmark_csv(args.data, args.runs)

    print("Running ArcticDB benchmark...")
    arctic_results = benchmark_arcticdb(args.data, args.runs)

    # Comparison table
    print(f"\n{'=' * 70}")
    print(f"{'Operation':<25} {'CSV':>12} {'ArcticDB':>12} {'Speedup':>10}")
    print(f"{'-' * 70}")
    print(f"{'Rows':<25} {csv_results['n_rows']:>12,} {arctic_results['n_rows']:>12,} {'':>10}")
    print(f"{'File size (MB)':<25} {csv_results['file_size_mb']:>12.1f} {arctic_results['file_size_mb']:>12.1f} {'':>10}")
    print(f"{'-' * 70}")
    print(f"{'Write (ms)':<25} {csv_results['write_avg_ms']:>12.1f} {arctic_results['write_avg_ms']:>12.1f} {csv_results['write_avg_ms']/arctic_results['write_avg_ms']:>9.2f}x")
    print(f"{'Full read (ms)':<25} {csv_results['full_read_avg_ms']:>12.1f} {arctic_results['full_read_avg_ms']:>12.1f} {csv_results['full_read_avg_ms']/arctic_results['full_read_avg_ms']:>9.2f}x")
    print(f"{'Date slice 30d (ms)':<25} {csv_results['date_slice_avg_ms']:>12.1f} {arctic_results['date_slice_avg_ms']:>12.1f} {csv_results['date_slice_avg_ms']/arctic_results['date_slice_avg_ms']:>9.2f}x")
    print(f"{'Col slice 30d (ms)':<25} {csv_results['col_slice_avg_ms']:>12.1f} {arctic_results['col_slice_avg_ms']:>12.1f} {csv_results['col_slice_avg_ms']/arctic_results['col_slice_avg_ms']:>9.2f}x")
    print(f"{'=' * 70}")

    # Verdict
    slice_speedup = csv_results["date_slice_avg_ms"] / arctic_results["date_slice_avg_ms"]
    col_speedup = csv_results["col_slice_avg_ms"] / arctic_results["col_slice_avg_ms"]
    full_speedup = csv_results["full_read_avg_ms"] / arctic_results["full_read_avg_ms"]

    print(f"\nVerdict:")
    print(f"  Full read:   {full_speedup:.2f}x {'FASTER' if full_speedup > 1 else 'SLOWER'}")
    print(f"  Date slice:  {slice_speedup:.2f}x {'FASTER' if slice_speedup > 1 else 'SLOWER'}")
    print(f"  Col slice:   {col_speedup:.2f}x {'FASTER' if col_speedup > 1 else 'SLOWER'}")

    if slice_speedup > 2.0:
        print(f"\n  RECOMMENDATION: Integrate ArcticDB — date-range slicing is {slice_speedup:.1f}x faster")
    elif full_speedup > 2.0:
        print(f"\n  RECOMMENDATION: Integrate ArcticDB — full reads are {full_speedup:.1f}x faster")
    else:
        print(f"\n  RECOMMENDATION: Skip ArcticDB — speedup not significant (<2x) for our data sizes")


if __name__ == "__main__":
    main()
