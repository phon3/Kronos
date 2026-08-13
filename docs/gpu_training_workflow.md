# GPU Server Setup & Training Workflow

Guide for setting up the GPU server, running fine-tuning, and coordinating between machines via GitHub.

---

## Architecture

```
Mac M1 (dev machine)                    GPU Server (training machine)
├── Write code                          ├── Pull from GitHub
├── Fetch data (data_ingestion)         ├── Install dependencies
├── Validate data                       ├── Run fine-tuning (train_sequential.py)
├── Push code + data to GitHub          ├── Push trained models to GitHub (LFS) or share via release
└── Run backtests on predictions        └── Run backtest if GPU available for inference
```

### Coordination via GitHub

- **Code**: Pushed to GitHub normally — both machines pull/push
- **Data**: CSV files (~1-2MB) can be committed directly. For larger datasets, use [Git LFS](https://git-lfs.com) or share via GitHub Releases
- **Trained models**: Model checkpoints (safetensors, ~100MB for kronos-small) are gitignored. Share via:
  - **Option A**: GitHub Releases (attach model files as release assets)
  - **Option B**: Git LFS (if you have it configured)
  - **Option C**: Hugging Face Hub (upload to a private repo, pull from either machine)
  - **Option D**: Direct transfer (scp, rsync, shared drive)

---

## Step 1: GPU Server — Environment Setup

### 1.1 Clone the repo

```bash
git clone https://github.com/<your-org>/Kronos.git
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

### 1.4 Verify GPU is available

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
Device count: 1
Device name: <your GPU name>
```

### 1.5 Verify model downloads work

```bash
python -c "from model import Kronos, KronosTokenizer; print('Downloading kronos-small...'); m = Kronos.from_pretrained('NeoQuasar/Kronos-small'); print(f'Model loaded: {sum(p.numel() for p in m.parameters()):,} params')"
```

---

## Step 2: Get the Data

### Option A: Pull data CSV from GitHub (if committed)

```bash
git pull origin master
ls -la data/BTC_USD_1h.csv data/BTC_USD_15m.csv data/BTC_USD_1d.csv
```

Candle CSV files (`BTC_USD_*.csv`) in `data/` are tracked in Git. They will be available after `git pull`.

### Option B: Fetch data directly on the GPU server

```bash
# 1h (primary)
python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe 1h --start 2024-01-01 --end 2025-08-01 --out ./data/BTC_USD_1h.csv --validate

# 15m (high frequency)
python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe 15m --start 2024-01-01 --end 2025-08-01 --out ./data/BTC_USD_15m.csv --validate

# 1d (macro trend)
python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe 1d --start 2020-01-01 --end 2025-08-01 --out ./data/BTC_USD_1d.csv --validate
```

### Option C: Transfer data from Mac via scp

```bash
# From Mac:
scp ./data/BTC_USD_1h.csv user@gpu-server:/path/to/Kronos/data/BTC_USD_1h.csv
```

---

## Step 3: Run Fine-Tuning

### Available configs

| Config | Timeframe | Model | Lookback | Pred Len | Epochs (tok+base) |
|--------|-----------|-------|----------|----------|-------------------|
| `config_btc_usd_1h_prod.yaml` | 1h | kronos-small | 512 | 48 | 30 + 25 |
| `config_btc_usd_15m_prod.yaml` | 15m | kronos-small | 256 | 24 | 30 + 25 |
| `config_btc_usd_1d_prod.yaml` | 1d | kronos-small | 512 | 14 | 40 + 35 |

### 3.1 Single GPU training (1h — primary)

```bash
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_btc_usd_1h_prod.yaml
```

> **Note**: Run from the repo root, not from `finetune_csv/`. The configs use relative paths like `./data/BTC_USD_1h.csv`.
> If you see "No such file or directory" for data files, either run from repo root or copy data: `cp -r data/ finetune_csv/`

### 3.2 Train all three timeframes

```bash
# From repo root:
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_btc_usd_1h_prod.yaml
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_btc_usd_15m_prod.yaml
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_btc_usd_1d_prod.yaml

# Or use the batch script (runs all three sequentially):
bash finetune_csv/run_all_gpu.sh
```

### 3.3 Multi-GPU training (DDP)

```bash
torchrun --standalone --nproc_per_node=2 finetune_csv/train_sequential.py --config finetune_csv/configs/config_btc_usd_1h_prod.yaml
```

### 3.4 Skip tokenizer training (if already trained)

```bash
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_btc_usd_1h_prod.yaml --skip-tokenizer
```

### 3.5 Skip existing models (resume without retraining)

```bash
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_btc_usd_1h_prod.yaml --skip-existing
```

### 3.6 Use tmux to keep training alive through SSH disconnects

```bash
tmux new -s kronos
bash finetune_csv/run_all_gpu.sh
# Detach: Ctrl+B, then D
# Reconnect: tmux attach -s kronos
```

---

## Step 4: Training Output — What Gets Produced

After training completes, the following files are created:

```
finetune_csv/finetuned/
└── btc_usd_1h_dev/                    # exp_name from config
    ├── tokenizer/
    │   └── best_model/                # Best tokenizer checkpoint (by val loss)
    │       ├── config.json            # Tokenizer architecture config
    │       ├── model.safetensors      # Tokenizer weights
    │       └── ...
    ├── basemodel/
    │   └── best_model/                # Best predictor checkpoint (by val loss)
    │       ├── config.json            # Model architecture config
    │       ├── model.safetensors      # Model weights (~100MB for kronos-small)
    │       └── ...
    └── logs/
        └── btc_usd_finetune_dev_rank0.log  # Training log with per-epoch metrics
```

### Key outputs to check:

- **`basemodel/best_model/`** — This is the fine-tuned Kronos predictor. Use this path as `--model` for backtesting.
- **`tokenizer/best_model/`** — This is the fine-tuned tokenizer. Use this path as `--tokenizer` for backtesting.
- **`logs/*.log`** — Contains training loss, validation loss, and timing per epoch.

### Training log example:

```
--- Epoch 1/15 Summary ---
Training Loss: 2.3456
Validation Loss: 2.1234
Epoch Time: 45.23 seconds

Best model saved to: ./finetune_csv/finetuned/btc_usd_1h_dev/basemodel/best_model (validation loss: 2.1234)
```

---

## Step 5: Share Trained Models Back to Mac

### Option A: GitHub Release (recommended for small models)

On the GPU server:
```bash
# Create a tarball of the trained models
cd finetune_csv/finetuned
tar czf btc_usd_1h_dev_models.tar.gz btc_usd_1h_dev/

# Create a GitHub release and upload
gh release create v0.1-btc-usd-1h-dev ./btc_usd_1h_dev_models.tar.gz --title "BTC/USD 1h Dev Model" --notes "Phase 1 fine-tuned kronos-small on BTC/USD 1h from Coinbase"
```

On Mac:
```bash
gh release download v0.1-btc-usd-1h-dev --dir ./finetune_csv/finetuned/
cd finetune_csv/finetuned/
tar xzf btc_usd_1h_dev_models.tar.gz
```

### Option B: scp directly

```bash
# From GPU server:
scp -r ./finetune_csv/finetuned/btc_usd_1h_dev user@mac:/path/to/Kronos/finetune_csv/finetuned/
```

### Option C: Hugging Face Hub

```bash
# On GPU server — upload
python -c "
from model import Kronos, KronosTokenizer
model = Kronos.from_pretrained('./finetune_csv/finetuned/btc_usd_1h_dev/basemodel/best_model')
model.push_to_hub('your-org/kronos-btc-usd-1h-dev')

tokenizer = KronosTokenizer.from_pretrained('./finetune_csv/finetuned/btc_usd_1h_dev/tokenizer/best_model')
tokenizer.push_to_hub('your-org/kronos-btc-usd-1h-dev-tokenizer')
"

# On Mac — download
python -c "
from model import Kronos, KronosTokenizer
model = Kronos.from_pretrained('your-org/kronos-btc-usd-1h-dev')
tokenizer = KronosTokenizer.from_pretrained('your-org/kronos-btc-usd-1h-dev-tokenizer')
"
```

---

## Step 6: Run Backtest (on either machine)

### On Mac (CPU inference — slower but works):

```bash
python -m backtest.backtest_runner \
  --model ./finetune_csv/finetuned/btc_usd_1h_dev/basemodel/best_model \
  --tokenizer ./finetune_csv/finetuned/btc_usd_1h_dev/tokenizer/best_model \
  --data ./data/BTC_USD_1h.csv \
  --device cpu \
  --output ./backtest_results/btc_usd_1h_dev/
```

### On GPU server (CUDA inference — faster):

```bash
python -m backtest.backtest_runner \
  --model ./finetune_csv/finetuned/btc_usd_1h_dev/basemodel/best_model \
  --tokenizer ./finetune_csv/finetuned/btc_usd_1h_dev/tokenizer/best_model \
  --data ./data/BTC_USD_1h.csv \
  --device cuda \
  --output ./backtest_results/btc_usd_1h_dev/
```

### Backtest output:

```
backtest_results/btc_usd_1h_dev/
├── backtest_chart.png        # Equity curve, cumulative returns, drawdown
├── backtest_results.csv      # Per-bar portfolio values and returns
├── backtest_report.html      # Self-contained HTML report with embedded chart
└── trade_log.csv             # Individual trade entries/exits with PnL
```

---

## Step 7: Share Backtest Results

Backtest results are small (PNG + CSV + HTML). Commit them to GitHub:

```bash
git add backtest_results/
git commit -m "Add BTC/USD 1h dev backtest results"
git push origin main
```

Then pull on the other machine to review.

---

## Backtest Signal Modes

The backtest runner supports four signal modes via `--signal-mode`:

### `kronos` (default) — Model predictions only
Uses Kronos predicted returns to generate long/short signals. Requires `--model` and `--tokenizer`.

```bash
python -m backtest.backtest_runner \
  --model ./finetune_csv/finetuned/btc_usd_1h_prod/basemodel/best_model \
  --tokenizer ./finetune_csv/finetuned/btc_usd_1h_prod/tokenizer/best_model \
  --data ./data/BTC_USD_1h.csv \
  --device cuda \
  --signal-mode kronos \
  --output ./backtest_results/btc_kronos/
```

### `trend` — Trend detection only (no model needed)
Uses the ported trend detection algorithm (break line / limit line) to identify advance/decline movements. No model or tokenizer required — runs on any machine.

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

### `combined` — Kronos predictions + trend confirmation
Only enters a position when both Kronos prediction and trend detection agree on direction. Closes when either signal reverses. Requires model + tokenizer + data.

```bash
python -m backtest.backtest_runner \
  --model ./finetune_csv/finetuned/btc_usd_1h_prod/basemodel/best_model \
  --tokenizer ./finetune_csv/finetuned/btc_usd_1h_prod/tokenizer/best_model \
  --data ./data/BTC_USD_1h.csv \
  --device cuda \
  --signal-mode combined \
  --trend-prd 60 \
  --trend-ext-break 0.03 \
  --trend-ext-limit 0.01 \
  --output ./backtest_results/btc_combined/
```

### `multi_tf` — Multi-timeframe trend confirmation
Uses a higher timeframe (e.g., 1d) trend as a directional filter for lower timeframe (e.g., 1h) entries. Only takes positions when both timeframes agree.

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

### Position sizing & martingale

| Flag | Default | Description |
|------|---------|-------------|
| `--position-size` | 1.0 | Fraction of capital per trade (0.25 = 25%) |
| `--stop-loss` | None | Stop-loss % (0.08 = 8% loss closes position) |
| `--take-profit` | None | Take-profit % (0.10 = 10% gain closes position) |
| `--loss-multiplier` | 1.0 | Martingale: multiply position after each loss (1.5 = 1.5x) |
| `--max-position-mult` | 4.0 | Cap for martingale multiplier |
| `--no-reset-on-win` | false | Don't reset martingale size after a win |
| `--walk-forward` | false | Split into in-sample/out-of-sample |
| `--train-ratio` | 0.7 | In-sample fraction for walk-forward |

### DeepTrade parameters

| Flag | Default | Description |
|-----------|---------|-------------|
| `--tp-levels` | None | Multi-target take-profit levels, format: `pct:fraction,pct:fraction,...` (e.g., `0.05:0.33,0.10:0.33,0.15:0.34` exits 33% at 5% gain, 33% at 10%, 34% at 15%) |
| `--daily-loss-limit` | None | Stop new trades after this daily drawdown % (e.g., `0.03` = 3%) |

These implement the DeepTrade methodology: multi-target take-profit scaling (TP1/TP2/TP3 partial exits) and daily loss limit for risk management.

### Trend detection parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trend-prd` | 60 | Period (bars) for break/limit line evaluation |
| `--trend-ext-break` | 0.05 | Break line gradient (5% over prd bars) |
| `--trend-ext-limit` | 0.02 | Limit line gradient (2% over prd bars) |
| `--trend-min-bars` | 5 | Minimum bars for a valid movement |

### Macro timeframe parameters (multi_tf mode only)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--macro-data` | None | Path to macro timeframe CSV (required for multi_tf) |
| `--macro-prd` | 10 | Macro trend period (bars) |
| `--macro-ext-break` | 0.03 | Macro break line gradient |
| `--macro-ext-limit` | 0.03 | Macro limit line gradient |
| `--macro-min-bars` | 3 | Macro minimum bars |

Lower thresholds = more sensitive (more trades). Higher thresholds = more conservative (fewer, larger movements).

### Best params by timeframe (from param sweep)

| Timeframe | prd | ext_break | ext_limit | min_bars | OOS Return | Sharpe | Win Rate |
|-----------|-----|-----------|-----------|----------|------------|--------|----------|
| 15m       | 240 | 0.03      | 0.01      | 3        | 137.2%     | 17.85  | 100%     |
| 1h        | 60  | 0.03      | 0.01      | 5        | 64.6%      | 16.19  | 100%     |
| 1d        | 10  | 0.03      | 0.03      | 3        | 171.1%     | 5.97   | 90%      |

---

## Quick Reference: Full Workflow

```
# === Mac (dev machine) ===
python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe 1h --start 2024-01-01 --end 2025-08-01 --out ./data/BTC_USD_1h.csv --validate
git add data/BTC_USD_1h.csv finetune_csv/configs/config_btc_usd_1h.yaml
git commit -m "Add BTC/USD 1h data and fine-tune config"
git push origin master

# === GPU Server ===
git pull origin master
pip install -r requirements.txt

# Data is tracked in Git — git pull fetches candle CSVs
# Or fetch fresh data:
python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe 1h --start 2024-01-01 --end 2025-08-01 --out ./data/BTC_USD_1h.csv --validate

# Train all three timeframes (from repo root)
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_btc_usd_1h_prod.yaml
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_btc_usd_15m_prod.yaml
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_btc_usd_1d_prod.yaml
# → produces finetuned/btc_usd_{1h,15m,1d}_prod/{tokenizer,basemodel}/best_model/

# Share models back (choose one):
gh release create v0.2-btc-usd-all-tf ./finetune_csv/finetuned/*_prod_models.tar.gz
# OR: scp -r finetune_csv/finetuned/ mac:~/Kronos/finetune_csv/finetuned/

# Run backtest on GPU (trend-only, no model needed):
python -m backtest.backtest_runner --data ./data/BTC_USD_1h.csv --signal-mode trend --trend-prd 60 --trend-ext-break 0.03 --trend-ext-limit 0.01 --trend-min-bars 5 --position-size 0.25 --stop-loss 0.08 --walk-forward --output ./backtest_results/btc_1h_trend/

# Run multi-timeframe backtest:
python -m backtest.backtest_runner --data ./data/BTC_USD_1h.csv --signal-mode multi_tf --macro-data ./data/BTC_USD_1d.csv --trend-prd 60 --trend-ext-break 0.03 --trend-ext-limit 0.01 --trend-min-bars 5 --macro-prd 10 --macro-ext-break 0.03 --macro-ext-limit 0.03 --macro-min-bars 3 --position-size 0.25 --stop-loss 0.08 --walk-forward --output ./backtest_results/btc_multi_tf_1d_1h/

# === Mac (review results) ===
git pull origin master
# Open backtest_results/btc_usd_1h_dev/backtest_report.html in browser
```

---

## Troubleshooting

### CUDA out of memory
- Reduce `batch_size` in the config (try 16 or 8)
- Reduce `lookback_window` (try 256 instead of 512)

### Model download fails
- Ensure `huggingface_hub` is installed and you have internet access
- Try: `huggingface-cli download NeoQuasar/Kronos-small`

### Training is slow
- kronos-small (24.7M params) on 13K 1h candles: ~15-30 min on a modern GPU
- kronos-small on 91K 15m candles: ~45-90 min (4x more data)
- kronos-small on 2.4K 1d candles: ~10-20 min (small dataset, more epochs)
- If using CPU, expect hours — use GPU
- Check GPU utilization: `nvidia-smi -l 1`

### Data file missing on GPU server
- Candle CSV files (`BTC_USD_*.csv`) in `data/` are tracked in Git — `git pull` should fetch them
- If missing, fetch directly: `python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe 1h --start 2024-01-01 --end 2025-08-01 --out ./data/BTC_USD_1h.csv --validate`
- If running from `finetune_csv/` and getting path errors, copy data: `cp -r data/ finetune_csv/`
- For large datasets, consider ArcticDB (28-75x faster reads than CSV, see `backtest/benchmark_arcticdb.py`)
