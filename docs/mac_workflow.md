# Mac Local Workflow — Dev, Data, Backtest & Light Training

Guide for running the full Kronos pipeline on a Mac (M1/M2/M3/M4 or Intel).
Covers data ingestion, trend-only backtesting (no GPU needed), MPS inference,
and light fine-tuning with the mini model for development.

---

## Architecture

```
Mac (this machine)
├── Write & test code
├── Fetch data (data_ingestion)
├── Validate data
├── Run trend-only backtests (no model, instant)
├── Run Kronos backtests with MPS (Apple Silicon) or CPU
├── Light fine-tuning with kronos-mini (4.1M params, feasible on CPU/MPS)
├── Push code + data to GitHub
└── Pull trained models from GPU server (via GitHub Release or scp)
```

### When to use Mac vs GPU server

| Task | Mac | GPU Server |
|------|-----|------------|
| Data ingestion & validation | ✅ Primary | ✅ Backup |
| Trend-only backtest | ✅ Instant | ✅ |
| Kronos backtest (inference) | ✅ MPS/CPU (slower) | ✅ CUDA (fast) |
| Fine-tuning kronos-mini (4.1M) | ✅ Feasible (~30-60 min) | ✅ Fast |
| Fine-tuning kronos-small (24.7M) | ⚠️ Slow (1-3 hrs on MPS) | ✅ Recommended |
| Fine-tuning kronos-base (102.3M) | ❌ Impractical | ✅ Required |
| Code editing & testing | ✅ Primary | ❌ |
| Reviewing backtest results | ✅ Primary | ✅ |

---

## Step 1: Environment Setup

### 1.1 Clone the repo (if not already done)

```bash
git clone https://github.com/phon3/Kronos.git
cd Kronos
```

### 1.2 Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 1.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 1.4 Verify Apple Silicon MPS (Metal Performance Shaders)

On M-series Macs, PyTorch can use the GPU via MPS:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"
```

Expected output on Apple Silicon:
```
PyTorch: 2.x.x
MPS available: True
MPS built: True
```

On Intel Macs, MPS will not be available — use CPU instead.

### 1.5 Verify model downloads work

```bash
python -c "from model import Kronos, KronosTokenizer; print('Downloading kronos-mini...'); m = Kronos.from_pretrained('NeoQuasar/Kronos-mini'); print(f'Model loaded: {sum(p.numel() for p in m.parameters()):,} params')"
```

---

## Step 2: Fetch & Validate Data

### 2.1 Fetch BTC/USD data from Coinbase (all timeframes)

```bash
# 1h candles (primary timeframe)
python -m data_ingestion.fetch_data \
  --source coinbase --symbol BTC/USD --timeframe 1h \
  --start 2024-01-01 --end 2025-08-01 \
  --out ./data/BTC_USD_1h.csv --validate

# 15m candles (higher frequency, ~91K rows)
python -m data_ingestion.fetch_data \
  --source coinbase --symbol BTC/USD --timeframe 15m \
  --start 2024-01-01 --end 2025-08-01 \
  --out ./data/BTC_USD_15m.csv --validate

# 1d candles (macro trend, ~2.4K rows)
python -m data_ingestion.fetch_data \
  --source coinbase --symbol BTC/USD --timeframe 1d \
  --start 2020-01-01 --end 2025-08-01 \
  --out ./data/BTC_USD_1d.csv --validate
```

> **Note**: CSV files in `data/` are gitignored (too large for git). Fetch them locally on each machine.

### 2.2 Fetch NASDAQ data (Phase 2)

```bash
python -m data_ingestion.fetch_data \
  --source yahoo \
  --symbol QQQ \
  --timeframe 1d \
  --start 2020-01-01 \
  --end 2025-08-01 \
  --out ./data/QQQ_1d.csv \
  --validate
```

### 2.3 Validate an existing CSV

```bash
python -c "
from data_ingestion.data_validator import DataValidator
v = DataValidator(expected_timeframe='1h')
report = v.validate('./data/BTC_USD_1h.csv')
print(f'Valid: {report[\"valid\"]}')
print(f'Rows: {report[\"stats\"][\"total_rows\"]}')
if report['errors']:
    print(f'Errors: {report[\"errors\"]}')
"
```

### 2.4 Data files are gitignored

CSV files in `data/` are too large for git. Fetch them on each machine using the commands in 2.1.
Configs and code are committed normally.

---

## Step 3: Run Backtests

### 3.1 Trend-only backtest (no model, instant)

This uses the ported trend detection algorithm — no model download, no GPU needed.
Best params from param sweep: `prd=60, ext_break=0.03, ext_limit=0.01, min_bars=5` for 1h.

```bash
python -m backtest.backtest_runner \
  --data ./data/BTC_USD_1h.csv \
  --signal-mode trend \
  --trend-prd 60 \
  --trend-ext-break 0.03 \
  --trend-ext-limit 0.01 \
  --trend-min-bars 5 \
  --position-size 0.25 \
  --stop-loss 0.08 \
  --output ./backtest_results/btc_trend/
```

### 3.1b Walk-forward backtest with martingale position sizing

Splits data chronologically (70% in-sample, 30% out-of-sample) and applies
martingale-style loss escalation (1.5x after each loss, capped at 4x):

```bash
python -m backtest.backtest_runner \
  --data ./data/BTC_USD_1h.csv \
  --signal-mode trend \
  --trend-prd 60 --trend-ext-break 0.03 --trend-ext-limit 0.01 --trend-min-bars 5 \
  --position-size 0.25 --stop-loss 0.08 \
  --loss-multiplier 1.5 --max-position-mult 4.0 \
  --walk-forward --train-ratio 0.7 \
  --output ./backtest_results/btc_1h_martingale/
```

### 3.1c Multi-timeframe backtest (1d macro filter + 1h entries)

Uses daily trend as a directional filter — only takes 1h positions when 1d trend agrees:

```bash
python -m backtest.backtest_runner \
  --data ./data/BTC_USD_1h.csv \
  --signal-mode multi_tf \
  --macro-data ./data/BTC_USD_1d.csv \
  --trend-prd 60 --trend-ext-break 0.03 --trend-ext-limit 0.01 --trend-min-bars 5 \
  --macro-prd 10 --macro-ext-break 0.03 --macro-ext-limit 0.03 --macro-min-bars 3 \
  --position-size 0.25 --stop-loss 0.08 \
  --walk-forward \
  --output ./backtest_results/btc_multi_tf_1d_1h/
```

Open the report:
```bash
open backtest_results/btc_trend/backtest_report.html
```

### 3.2 Kronos backtest with MPS (Apple Silicon)

After pulling trained models from the GPU server:

```bash
python -m backtest.backtest_runner \
  --model ./finetune_csv/finetuned/btc_usd_1h_dev/basemodel/best_model \
  --tokenizer ./finetune_csv/finetuned/btc_usd_1h_dev/tokenizer/best_model \
  --data ./data/BTC_USD_1h.csv \
  --device mps \
  --signal-mode kronos \
  --output ./backtest_results/btc_kronos/
```

### 3.3 Kronos backtest with CPU (Intel Mac or fallback)

```bash
python -m backtest.backtest_runner \
  --model ./finetune_csv/finetuned/btc_usd_1h_dev/basemodel/best_model \
  --tokenizer ./finetune_csv/finetuned/btc_usd_1h_dev/tokenizer/best_model \
  --data ./data/BTC_USD_1h.csv \
  --device cpu \
  --signal-mode kronos \
  --output ./backtest_results/btc_kronos/
```

### 3.4 Combined backtest (Kronos + trend confirmation)

```bash
python -m backtest.backtest_runner \
  --model ./finetune_csv/finetuned/btc_usd_1h_dev/basemodel/best_model \
  --tokenizer ./finetune_csv/finetuned/btc_usd_1h_dev/tokenizer/best_model \
  --data ./data/BTC_USD_1h.csv \
  --device mps \
  --signal-mode combined \
  --trend-prd 60 \
  --trend-ext-break 0.05 \
  --trend-ext-limit 0.02 \
  --output ./backtest_results/btc_combined/
```

### 3.5 Backtest output files

```
backtest_results/btc_trend/
├── backtest_chart.png        # Equity curve, cumulative returns, drawdown
├── backtest_results.csv      # Per-bar portfolio values and returns
├── backtest_report.html      # Self-contained HTML report
└── trade_log.csv             # Individual trade entries/exits with PnL
```

### 3.6 Parameter sweep (find best trend params)

Grid-search trend detection parameters with out-of-sample evaluation:

```bash
# 1h sweep
python -m backtest.param_sweep \
  --data ./data/BTC_USD_1h.csv \
  --output ./backtest_results/param_sweep_1h/ \
  --prd 30,60,90,120 --ext-break 0.03,0.05,0.08 --ext-limit 0.01,0.02,0.03 --min-bars 3,5,10 \
  --position-size 0.25 --stop-loss 0.08

# 15m sweep (wider prd range for higher granularity)
python -m backtest.param_sweep \
  --data ./data/BTC_USD_15m.csv \
  --output ./backtest_results/param_sweep_15m/ \
  --prd 60,120,240,480 --ext-break 0.03,0.05,0.08 --ext-limit 0.01,0.02,0.03 --min-bars 3,5,10 \
  --position-size 0.25 --stop-loss 0.08

# 1d sweep (shorter prd for daily bars)
python -m backtest.param_sweep \
  --data ./data/BTC_USD_1d.csv \
  --output ./backtest_results/param_sweep_1d/ \
  --prd 10,20,30,60 --ext-break 0.03,0.05,0.08,0.10 --ext-limit 0.01,0.02,0.03 --min-bars 3,5,10 \
  --position-size 0.25 --stop-loss 0.08
```

### 3.7 Best params found (from sweep results)

| Timeframe | prd | ext_break | ext_limit | min_bars | OOS Return | Sharpe | Win Rate |
|-----------|-----|-----------|-----------|----------|------------|--------|----------|
| 15m       | 240 | 0.03      | 0.01      | 3        | 137.2%     | 17.85  | 100%     |
| 1h        | 60  | 0.03      | 0.01      | 5        | 64.6%      | 16.19  | 100%     |
| 1d        | 10  | 0.03      | 0.03      | 3        | 171.1%     | 5.97   | 90%      |

### 3.8 Compare all modes

```bash
# Trend-only
python -m backtest.backtest_runner --data ./data/BTC_USD_1h.csv --signal-mode trend --trend-prd 60 --trend-ext-break 0.03 --trend-ext-limit 0.01 --trend-min-bars 5 --output ./backtest_results/btc_trend/

# Kronos-only (requires model)
python -m backtest.backtest_runner --model ./finetune_csv/finetuned/btc_usd_1h_dev/basemodel/best_model --tokenizer ./finetune_csv/finetuned/btc_usd_1h_dev/tokenizer/best_model --data ./data/BTC_USD_1h.csv --device mps --signal-mode kronos --output ./backtest_results/btc_kronos/

# Combined
python -m backtest.backtest_runner --model ./finetune_csv/finetuned/btc_usd_1h_dev/basemodel/best_model --tokenizer ./finetune_csv/finetuned/btc_usd_1h_dev/tokenizer/best_model --data ./data/BTC_USD_1h.csv --device mps --signal-mode combined --output ./backtest_results/btc_combined/

# Multi-timeframe (1d filter + 1h entries)
python -m backtest.backtest_runner --data ./data/BTC_USD_1h.csv --signal-mode multi_tf --macro-data ./data/BTC_USD_1d.csv --trend-prd 60 --trend-ext-break 0.03 --trend-ext-limit 0.01 --trend-min-bars 5 --macro-prd 10 --macro-ext-break 0.03 --macro-ext-limit 0.03 --macro-min-bars 3 --output ./backtest_results/btc_multi_tf_1d_1h/

# Open all reports
open backtest_results/btc_trend/backtest_report.html
open backtest_results/btc_kronos/backtest_report.html
open backtest_results/btc_combined/backtest_report.html
open backtest_results/btc_multi_tf_1d_1h/backtest_report.html
```

---

## Step 4: Light Fine-Tuning on Mac (kronos-mini only)

Fine-tuning the mini model (4.1M params) is feasible on Mac for development
and testing. For production training, use the GPU server with kronos-small
or kronos-base.

### 4.1 Create a Mac-specific config

```bash
cat > finetune_csv/configs/config_btc_usd_1h_mac.yaml << 'EOF'
# Mac dev config: Fine-tune Kronos-mini on BTC/USD 1h (lightweight)
data:
  data_path: "./data/BTC_USD_1h.csv"
  lookback_window: 256
  predict_window: 48
  max_context: 2048
  clip: 5.0
  train_ratio: 0.85
  val_ratio: 0.10
  test_ratio: 0.05

training:
  tokenizer_epochs: 10
  basemodel_epochs: 10
  batch_size: 8
  log_interval: 50
  num_workers: 2
  seed: 42
  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000004
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1
  accumulation_steps: 4

model_paths:
  pretrained_tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
  pretrained_predictor: "NeoQuasar/Kronos-mini"
  exp_name: "btc_usd_1h_mac"
  base_path: "./finetune_csv/finetuned/"
  base_save_path: ""
  finetuned_tokenizer: ""
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "btc_usd_finetune_mac"
  description: "BTC/USD 1h fine-tuning on Mac — dev/mini model"
  use_comet: false
  train_tokenizer: true
  train_basemodel: true
  skip_existing: false

device:
  use_cuda: false
  device_id: 0
EOF
```

Key differences from GPU config:
- **kronos-mini** (4.1M params) instead of kronos-small (24.7M)
- **batch_size: 8** (smaller for Mac memory)
- **lookback_window: 256** (shorter context, faster)
- **num_workers: 2** (fewer parallel workers)
- **accumulation_steps: 4** (effective batch size = 32)
- **use_cuda: false** (falls back to CPU)

### 4.2 Run training

```bash
cd finetune_csv
python train_sequential.py --config configs/config_btc_usd_1h_mac.yaml
```

Expected time on M-series Mac (CPU):
- kronos-mini, 13K candles, 10+10 epochs: ~30-60 minutes
- kronos-small would take 1-3 hours — use GPU server instead

### 4.3 Run backtest with the Mac-trained mini model

```bash
python -m backtest.backtest_runner \
  --model ./finetune_csv/finetuned/btc_usd_1h_mac/basemodel/best_model \
  --tokenizer ./finetune_csv/finetuned/btc_usd_1h_mac/tokenizer/best_model \
  --data ./data/BTC_USD_1h.csv \
  --device cpu \
  --signal-mode kronos \
  --output ./backtest_results/btc_mini_mac/
```

---

## Step 5: Pull Trained Models from GPU Server

After the GPU server completes training (see `gpu_training_workflow.md`):

### Option A: GitHub Release

```bash
gh release download v0.1-btc-usd-1h-dev --dir ./finetune_csv/finetuned/
cd finetune_csv/finetuned/
tar xzf btc_usd_1h_dev_models.tar.gz
```

### Option B: scp from GPU server

```bash
scp -r user@gpu-server:/path/to/Kronos/finetune_csv/finetuned/btc_usd_1h_dev ./finetune_csv/finetuned/
```

### Option C: Hugging Face Hub

```bash
python -c "
from model import Kronos, KronosTokenizer
model = Kronos.from_pretrained('your-org/kronos-btc-usd-1h-dev')
tokenizer = KronosTokenizer.from_pretrained('your-org/kronos-btc-usd-1h-dev-tokenizer')
print('Models downloaded')
"
```

---

## Step 6: Run Tests

### 6.1 Run all tests

```bash
python -m pytest tests/ -v
```

### 6.2 Run trend detector tests only

```bash
python -m pytest tests/test_trend_detector.py -v
```

### 6.3 Run data pipeline tests only

```bash
python -m pytest tests/test_data_pipeline.py -v
```

---

## Step 7: Commit & Push Results

```bash
git add backtest_results/
git commit -m "Add BTC/USD backtest results — trend, kronos, combined"
git push origin master
```

---

## Quick Reference: Mac Daily Workflow

```bash
# Activate environment
source venv/bin/activate

# Pull latest
git pull origin master

# Fetch fresh data (all timeframes)
python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe 1h --start 2024-01-01 --end 2025-08-01 --out ./data/BTC_USD_1h.csv --validate
python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe 15m --start 2024-01-01 --end 2025-08-01 --out ./data/BTC_USD_15m.csv --validate
python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe 1d --start 2020-01-01 --end 2025-08-01 --out ./data/BTC_USD_1d.csv --validate

# Run trend-only backtest with walk-forward + martingale (instant, no model)
python -m backtest.backtest_runner --data ./data/BTC_USD_1h.csv --signal-mode trend --trend-prd 60 --trend-ext-break 0.03 --trend-ext-limit 0.01 --trend-min-bars 5 --position-size 0.25 --stop-loss 0.08 --loss-multiplier 1.5 --walk-forward --output ./backtest_results/btc_1h_martingale/

# Run multi-timeframe backtest (1d filter + 1h entries)
python -m backtest.backtest_runner --data ./data/BTC_USD_1h.csv --signal-mode multi_tf --macro-data ./data/BTC_USD_1d.csv --trend-prd 60 --trend-ext-break 0.03 --trend-ext-limit 0.01 --trend-min-bars 5 --macro-prd 10 --macro-ext-break 0.03 --macro-ext-limit 0.03 --macro-min-bars 3 --position-size 0.25 --stop-loss 0.08 --walk-forward --output ./backtest_results/btc_multi_tf_1d_1h/

# Run Kronos backtest (requires model from GPU server)
python -m backtest.backtest_runner --model ./finetune_csv/finetuned/btc_usd_1h_prod/basemodel/best_model --tokenizer ./finetune_csv/finetuned/btc_usd_1h_prod/tokenizer/best_model --data ./data/BTC_USD_1h.csv --device mps --output ./backtest_results/btc_kronos/

# Open reports
open backtest_results/btc_1h_martingale/backtest_report.html
open backtest_results/btc_multi_tf_1d_1h/backtest_report.html
open backtest_results/btc_kronos/backtest_report.html

# Commit results
git add backtest_results/
git commit -m "Update backtest results"
git push origin master
```

---

## Troubleshooting (Mac-specific)

### MPS out of memory
- Reduce `batch_size` in config (try 4 or 2)
- Reduce `lookback_window` (try 128 or 256)
- Use `--device cpu` as fallback

### MPS not available (Intel Mac)
- Use `--device cpu` for all model operations
- Trend-only backtests (`--signal-mode trend`) work on any Mac without a model

### Model download is slow
- Models are cached in `~/.cache/huggingface/` after first download
- Pre-download: `huggingface-cli download NeoQuasar/Kronos-mini`

### Training is slow on Mac
- kronos-mini (4.1M) on CPU: ~30-60 min for 10 epochs on 13K candles
- kronos-small (24.7M) on CPU: 2-4 hours — use GPU server instead
- Use `--skip-tokenizer` if tokenizer is already trained to save time

### ArcticDB for faster data access
ArcticDB provides 28-75x faster reads than CSV for OHLCV data (benchmarked on 91K rows).
Install: `pip install arcticdb`
Benchmark: `python -m backtest.benchmark_arcticdb --data ./data/BTC_USD_15m.csv`
Note: arcticdb requires pandas<3, but pandas 3.x works at runtime despite the constraint.

### Python version issues
- Requires Python 3.10+
- Check: `python3 --version`
- If multiple versions: `python3.12 -m venv venv`

### `huggingface_hub` not found
```bash
pip install huggingface_hub
```

### Port already in use (webui)
```bash
# Kill existing process on port 7070
lsof -ti:7070 | xargs kill -9
python webui/app.py
```
