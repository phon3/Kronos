"""Backtest runner — loads fine-tuned Kronos model, generates predictions, runs backtest."""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .crypto_backtest import CryptoBacktester
from .report_generator import ReportGenerator


def load_model(
    tokenizer_path: str,
    model_path: str,
    device: str = "cpu",
    max_context: int = 512,
):
    """Load fine-tuned tokenizer and model into a KronosPredictor."""
    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor

    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer path not found: {tokenizer_path}\n"
            "Ensure you have pulled trained models from the GPU server. "
            "See docs/mac_workflow.md Step 5 or docs/gpu_training_workflow.md Step 5."
        )
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model path not found: {model_path}\n"
            "Ensure you have pulled trained models from the GPU server. "
            "See docs/mac_workflow.md Step 5 or docs/gpu_training_workflow.md Step 5."
        )

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)

    print(f"Loading model from {model_path}...")
    model = Kronos.from_pretrained(model_path)

    device_obj = torch.device(device)
    predictor = KronosPredictor(model, tokenizer, device=device_obj, max_context=max_context)

    print(f"Model loaded on {device}")
    return predictor


def generate_predictions(
    predictor: KronosPredictor,
    data: pd.DataFrame,
    lookback: int = 512,
    pred_len: int = 48,
    feature_list: Optional[list] = None,
    T: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 0,
    sample_count: int = 5,
) -> pd.DataFrame:
    """
    Run sliding-window predictions on the data.

    Args:
        predictor: Loaded KronosPredictor
        data: Full DataFrame with OHLCV data
        lookback: Number of historical bars for context
        pred_len: Number of bars to predict
        feature_list: Features to use (defaults to OHLCV + amount)
        T: Temperature for sampling
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (0 = disabled)
        sample_count: Number of samples to draw

    Returns:
        DataFrame with predicted close prices, indexed by timestamp
    """
    import torch

    if feature_list is None:
        feature_list = ["open", "high", "low", "close", "volume", "amount"]

    df = data.copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    all_preds = []
    step = pred_len  # Non-overlapping windows

    print(f"Generating predictions: {len(df)} bars, lookback={lookback}, pred_len={pred_len}")

    for start in range(0, len(df) - lookback - pred_len + 1, step):
        context = df.iloc[start : start + lookback]
        future = df.iloc[start + lookback : start + lookback + pred_len]

        x_df = context[feature_list].reset_index(drop=True)
        x_timestamp = context["timestamps"].reset_index(drop=True)
        y_timestamp = future["timestamps"].reset_index(drop=True)

        with torch.no_grad():
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=T,
                top_p=top_p,
                top_k=top_k if top_k > 0 else 1,
                sample_count=sample_count,
                verbose=False,
            )

        # Use mean across samples
        pred_df.index = y_timestamp
        all_preds.append(pred_df)

    if not all_preds:
        raise RuntimeError("No predictions generated — data too short for the given lookback/pred_len")

    predictions = pd.concat(all_preds)
    print(f"Generated {len(predictions)} prediction rows")

    return predictions


def run_backtest(
    model_path: str,
    tokenizer_path: str,
    data_path: str,
    output_dir: str = "./backtest_results/",
    device: str = "cpu",
    lookback: int = 512,
    pred_len: int = 48,
    max_context: int = 512,
    initial_capital: float = 100_000,
    threshold: float = 0.02,
    allow_short: bool = True,
    fee_pct: float = 0.001,
    T: float = 0.6,
    top_p: float = 0.9,
    sample_count: int = 5,
    title: str = "Kronos Crypto Backtest",
    signal_mode: str = "kronos",
    trend_prd: int = 60,
    trend_ext_break: float = 0.05,
    trend_ext_limit: float = 0.02,
    trend_min_bars: int = 5,
    position_size_pct: float = 1.0,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    walk_forward: bool = False,
    train_ratio: float = 0.7,
) -> dict:
    """
    Full backtest pipeline: load model, generate predictions, run backtest, generate report.

    Args:
        model_path: Path to fine-tuned Kronos model (required for 'kronos' and 'combined' modes)
        tokenizer_path: Path to fine-tuned Kronos tokenizer (required for 'kronos' and 'combined' modes)
        data_path: Path to CSV with OHLCV data
        output_dir: Directory for output files
        device: 'cpu', 'cuda', or 'mps'
        lookback: Context window size
        pred_len: Prediction length
        max_context: Max context for model
        initial_capital: Starting capital
        threshold: Signal threshold (predicted return %)
        allow_short: Allow short positions
        fee_pct: Trading fee percentage
        T: Sampling temperature
        top_p: Nucleus sampling threshold
        sample_count: Number of samples
        title: Report title
        signal_mode: 'kronos', 'trend', or 'combined'
        trend_prd: Trend detection period (bars)
        trend_ext_break: Trend break line gradient
        trend_ext_limit: Trend limit line gradient
        trend_min_bars: Minimum bars for valid trend movement

    Returns:
        Metrics dict
    """
    # Load data
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    data["timestamps"] = pd.to_datetime(data["timestamps"])
    print(f"  {len(data)} rows, range: {data['timestamps'].min()} to {data['timestamps'].max()}")

    predictions = None
    actual = None

    if signal_mode in ("kronos", "combined"):
        # Load model
        predictor = load_model(tokenizer_path, model_path, device, max_context)

        # Generate predictions
        predictions = generate_predictions(
            predictor, data,
            lookback=lookback, pred_len=pred_len,
            T=T, top_p=top_p, sample_count=sample_count,
        )

        # Prepare actual data for backtest (only the predicted period)
        pred_timestamps = predictions.index
        actual = data[data["timestamps"].isin(pred_timestamps)].set_index("timestamps")[["close"]]

    # Run backtest
    backtester = CryptoBacktester(
        initial_capital=initial_capital,
        threshold=threshold,
        allow_short=allow_short,
        fee_pct=fee_pct,
        position_size_pct=position_size_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    if walk_forward:
        if signal_mode == "trend":
            actual = data.set_index("timestamps")[["close"]]
        # predictions and actual already set above for kronos/combined
        metrics = backtester.run_walk_forward(
            predictions=predictions if predictions is not None else pd.DataFrame(),
            actual=actual if actual is not None else data.set_index("timestamps")[["close"]],
            full_data=data,
            train_ratio=train_ratio,
            output_dir=output_dir,
            title=title,
            signal_mode=signal_mode,
            trend_prd=trend_prd,
            trend_ext_break=trend_ext_break,
            trend_ext_limit=trend_ext_limit,
            trend_min_bars=trend_min_bars,
        )
        report_gen = ReportGenerator()
        report_gen.generate_report(
            metrics=metrics.get("out_of_sample", metrics),
            output_path=os.path.join(output_dir, "backtest_report.html"),
            title=f"{title} (Walk-Forward)",
            data_path=data_path,
            model_path=model_path if signal_mode != "trend" else "trend-only",
        )
        return metrics

    if signal_mode == "trend":
        # Trend-only mode: no model predictions needed
        actual = data.set_index("timestamps")[["close"]]
        metrics = backtester.run(
            predictions=pd.DataFrame(),
            actual=actual,
            output_dir=output_dir,
            title=title,
            signal_mode=signal_mode,
            full_data=data,
            trend_prd=trend_prd,
            trend_ext_break=trend_ext_break,
            trend_ext_limit=trend_ext_limit,
            trend_min_bars=trend_min_bars,
        )
    else:
        metrics = backtester.run(
            predictions=predictions,
            actual=actual,
            output_dir=output_dir,
            title=title,
            signal_mode=signal_mode,
            full_data=data if signal_mode == "combined" else None,
            trend_prd=trend_prd,
            trend_ext_break=trend_ext_break,
            trend_ext_limit=trend_ext_limit,
            trend_min_bars=trend_min_bars,
        )

    # Generate HTML report
    report_gen = ReportGenerator()
    report_gen.generate_report(
        metrics=metrics,
        output_path=os.path.join(output_dir, "backtest_report.html"),
        title=title,
        data_path=data_path,
        model_path=model_path if signal_mode != "trend" else "trend-only",
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Kronos backtest on crypto data")
    parser.add_argument("--model", default=None, help="Path to fine-tuned Kronos model (not required for --signal-mode trend)")
    parser.add_argument("--tokenizer", default=None, help="Path to fine-tuned Kronos tokenizer (not required for --signal-mode trend)")
    parser.add_argument("--data", required=True, help="Path to CSV with OHLCV data")
    parser.add_argument("--output", default="./backtest_results/", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, mps")
    parser.add_argument("--lookback", type=int, default=512, help="Context window size")
    parser.add_argument("--pred-len", type=int, default=48, help="Prediction length")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--threshold", type=float, default=0.02, help="Signal threshold")
    parser.add_argument("--no-short", action="store_true", help="Disable short selling")
    parser.add_argument("--fee", type=float, default=0.001, help="Fee percentage per trade")
    parser.add_argument("--temp", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--samples", type=int, default=5, help="Number of prediction samples")
    parser.add_argument("--signal-mode", choices=["kronos", "trend", "combined"], default="kronos",
                        help="Signal source: kronos (model predictions), trend (trend detection only), combined (both)")
    parser.add_argument("--trend-prd", type=int, default=60, help="Trend detection period (bars)")
    parser.add_argument("--trend-ext-break", type=float, default=0.05, help="Trend break line gradient")
    parser.add_argument("--trend-ext-limit", type=float, default=0.02, help="Trend limit line gradient")
    parser.add_argument("--trend-min-bars", type=int, default=5, help="Minimum bars for valid trend movement")
    parser.add_argument("--position-size", type=float, default=1.0, help="Fraction of capital to allocate per trade (0-1, default 1.0 = all in)")
    parser.add_argument("--stop-loss", type=float, default=None, help="Stop-loss percentage (e.g. 0.05 = 5%% loss closes position)")
    parser.add_argument("--take-profit", type=float, default=None, help="Take-profit percentage (e.g. 0.10 = 10%% gain closes position)")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward backtest (split into in-sample and out-of-sample)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of data for in-sample (walk-forward mode, default 0.7)")

    args = parser.parse_args()

    if args.signal_mode in ("kronos", "combined") and (not args.model or not args.tokenizer):
        parser.error(f"--model and --tokenizer are required when --signal-mode is {args.signal_mode}")

    metrics = run_backtest(
        model_path=args.model or "",
        tokenizer_path=args.tokenizer or "",
        data_path=args.data,
        output_dir=args.output,
        device=args.device,
        lookback=args.lookback,
        pred_len=args.pred_len,
        initial_capital=args.capital,
        threshold=args.threshold,
        allow_short=not args.no_short,
        fee_pct=args.fee,
        T=args.temp,
        sample_count=args.samples,
        signal_mode=args.signal_mode,
        trend_prd=args.trend_prd,
        trend_ext_break=args.trend_ext_break,
        trend_ext_limit=args.trend_ext_limit,
        trend_min_bars=args.trend_min_bars,
        position_size_pct=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        walk_forward=args.walk_forward,
        train_ratio=args.train_ratio,
    )

    if metrics:
        print(f"\nBacktest complete. Results in {args.output}")


if __name__ == "__main__":
    main()
