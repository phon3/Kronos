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
git pull origin main
ls -la data/BTC_USD_1h.csv
```

### Option B: Fetch data directly on the GPU server

```bash
python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe 1h --start 2024-01-01 --end 2025-08-01 --out ./data/BTC_USD_1h.csv --validate
```

### Option C: Transfer data from Mac via scp

```bash
# From Mac:
scp ./data/BTC_USD_1h.csv user@gpu-server:/path/to/Kronos/data/BTC_USD_1h.csv
```

---

## Step 3: Run Fine-Tuning

### 3.1 Single GPU training

```bash
cd finetune_csv
python train_sequential.py --config configs/config_btc_usd_1h.yaml
```

### 3.2 Multi-GPU training (DDP)

If you have multiple GPUs:

```bash
cd finetune_csv
torchrun --standalone --nproc_per_node=2 train_sequential.py --config configs/config_btc_usd_1h.yaml
```

### 3.3 Skip tokenizer training (if already trained)

```bash
python train_sequential.py --config configs/config_btc_usd_1h.yaml --skip-tokenizer
```

### 3.4 Skip existing models (resume without retraining)

```bash
python train_sequential.py --config configs/config_btc_usd_1h.yaml --skip-existing
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

## Quick Reference: Full Workflow

```
# === Mac (dev machine) ===
python -m data_ingestion.fetch_data --source coinbase --symbol BTC/USD --timeframe 1h --start 2024-01-01 --end 2025-08-01 --out ./data/BTC_USD_1h.csv --validate
git add data/BTC_USD_1h.csv finetune_csv/configs/config_btc_usd_1h.yaml
git commit -m "Add BTC/USD 1h data and fine-tune config"
git push origin main

# === GPU Server ===
git pull origin main
pip install -r requirements.txt
cd finetune_csv
python train_sequential.py --config configs/config_btc_usd_1h.yaml
# → produces finetuned/btc_usd_1h_dev/{tokenizer,basemodel}/best_model/

# Share models back (choose one):
gh release create v0.1-btc-usd-1h-dev ./finetune_csv/finetuned/btc_usd_1h_dev_models.tar.gz
# OR: scp -r finetune_csv/finetuned/btc_usd_1h_dev mac:~/Kronos/finetune_csv/finetuned/

# Run backtest on GPU:
python -m backtest.backtest_runner --model ./finetune_csv/finetuned/btc_usd_1h_dev/basemodel/best_model --tokenizer ./finetune_csv/finetuned/btc_usd_1h_dev/tokenizer/best_model --data ./data/BTC_USD_1h.csv --device cuda

# === Mac (review results) ===
git pull origin main
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
- kronos-small (24.7M params) on 13K candles should take ~15-30 min on a modern GPU
- If using CPU, expect hours — use GPU
- Check GPU utilization: `nvidia-smi -l 1`

### Data file missing on GPU server
- The `data/` directory is not gitignored, but CSV files >100MB may cause issues
- Use Git LFS or fetch directly on the GPU server with `data_ingestion.fetch_data`
