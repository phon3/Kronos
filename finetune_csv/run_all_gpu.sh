#!/bin/bash
# Run all three timeframe training configs sequentially on a GPU server.
#
# Usage:
#   chmod +x run_all_gpu.sh
#   ./run_all_gpu.sh                    # train all three
#   ./run_all_gpu.sh 1h                 # train only 1h
#   ./run_all_gpu.sh 15m                # train only 15m
#   ./run_all_gpu.sh 1d                 # train only 1d
#   ./run_all_gpu.sh 1h 15m             # train 1h then 15m
#
# Prerequisites:
#   - CUDA available: python -c "import torch; print(torch.cuda.is_available())"
#   - Data fetched: ./data/BTC_USD_{1h,15m,1d}.csv
#   - Dependencies installed: pip install -r requirements.txt
#
# Expected training time on a single modern GPU (e.g., A100, RTX 4090):
#   1h:  ~2-4 hrs  (13K rows, 30+25 epochs, batch=64)
#   15m: ~6-12 hrs (91K rows, 30+25 epochs, batch=64)
#   1d:  ~1-2 hrs  (2.4K rows, 40+35 epochs, batch=32)
#   Total: ~9-18 hrs
#
# On CPU (NOT recommended): 15+ days per config.

set -e

cd "$(dirname "$0")/.."

# Verify CUDA
echo "=== Checking CUDA ==="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available! This script requires a GPU.'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"

# Verify data files
echo "=== Checking data files ==="
for tf in 1h 15m 1d; do
    if [ ! -f "./data/BTC_USD_${tf}.csv" ]; then
        echo "Missing ./data/BTC_USD_${tf}.csv — fetching..."
        python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe ${tf} --start 2020-01-01 --end 2025-08-01 --out ./data/BTC_USD_${tf}.csv --validate
    else
        echo "Found ./data/BTC_USD_${tf}.csv"
    fi
done

# Determine which configs to run
if [ $# -gt 0 ]; then
    TIMEFRAMES="$@"
else
    TIMEFRAMES="1h 15m 1d"
fi

CONFIGS_DIR="finetune_csv/configs"
LOG_DIR="finetune_csv/training_logs"
mkdir -p "$LOG_DIR"

for tf in $TIMEFRAMES; do
    CONFIG="${CONFIGS_DIR}/config_btc_usd_${tf}_prod.yaml"
    LOG_FILE="${LOG_DIR}/train_${tf}_$(date +%Y%m%d_%H%M%S).log"

    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: Config not found: $CONFIG"
        continue
    fi

    echo ""
    echo "=============================================="
    echo "Training: $tf"
    echo "Config:   $CONFIG"
    echo "Log:      $LOG_FILE"
    echo "Started:  $(date)"
    echo "=============================================="

    cd finetune_csv
    python train_sequential.py --config configs/config_btc_usd_${tf}_prod.yaml 2>&1 | tee "../${LOG_FILE}"
    cd ..

    echo ""
    echo "Completed: $tf at $(date)"
    echo "Log saved: $LOG_FILE"
    echo ""
done

echo ""
echo "=============================================="
echo "All training complete: $(date)"
echo "=============================================="
echo ""
echo "Trained models in: finetune_csv/finetuned/"
echo "Training logs in:  finetune_csv/training_logs/"
echo ""
echo "Next steps:"
echo "  1. Run backtests: python -m backtest.backtest_runner --model ./finetune_csv/finetuned/btc_usd_1h_prod/basemodel/best_model --tokenizer ./finetune_csv/finetuned/btc_usd_1h_prod/tokenizer/best_model --data ./data/BTC_USD_1h.csv --device cuda --output ./backtest_results/btc_1h_prod/"
echo "  2. Share models back to Mac: tar czf models.tar.gz finetune_csv/finetuned/btc_usd_*_prod/ && gh release create v0.2-all-tf models.tar.gz"
