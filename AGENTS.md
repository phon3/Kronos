# Kronos - Agent Quick Reference

## Project Summary

Kronos is the **first open-source foundation model** for financial candlesticks (K-lines), trained on data from over **45 global exchanges**.

### Key Features
- **Novel two-stage framework**: Specialized tokenizer quantizes continuous multi-dimensional K-line data (OHLCV) into hierarchical discrete tokens, then a large autoregressive Transformer is pre-trained on these tokens
- **Multiple model variants**: mini (4.1M params, 2048 context), small (24.7M params, 512 context), base (102.3M params, 512 context)
- **Complete fine-tuning pipeline**: Includes scripts for fine-tuning on custom datasets (e.g., Chinese A-share market with Qlib)
- **Web UI**: Dark-themed Flask interface with Predict and Backtest tabs, fine-tuned model auto-discovery, and DeepTrade risk management parameters
- **DeepTrade integration**: Multi-target take-profit scaling (TP1/TP2/TP3 partial exits) and daily loss limit in backtesting
- **Backtesting framework**: Trend, Kronos, combined, and multi-timeframe signal modes with walk-forward validation and martingale position sizing

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Web UI
```bash
python webui/app.py
```
The web UI will start on **http://localhost:7070**

### 3. Load a Model (via Web UI or API)
- Open the web interface
- Select a model (pretrained: kronos-mini/small/base, or fine-tuned: auto-discovered from `finetune_csv/finetuned/`)
- Choose device (cpu/cuda/mps)
- Load the model

### 4. Predict Tab
- Select data file from `data/` directory (auto-populated)
- Set prediction parameters (lookback, prediction length, temperature, top_p, sample count)
- Optionally set a start date for historical backtesting
- Generate forecasts with interactive candlestick charts

### 5. Backtest Tab
- Choose signal mode: trend, kronos, combined, or multi-timeframe
- Configure risk management (position size, stop loss, take profit, martingale)
- Expand DeepTrade settings for multi-target TP levels and daily loss limit
- Run walk-forward validation
- View results as metric cards (return, Sharpe, drawdown, win rate, profit factor)

## Key Files

| File | Purpose |
|------|---------|
| `webui/app.py` | **Main entrypoint** - Flask web application with Predict & Backtest APIs |
| `webui/templates/index.html` | Dark-themed UI with tab navigation (Predict / Backtest) |
| `model/kronos.py` | Core model implementation (Kronos, KronosTokenizer, KronosPredictor) |
| `model/module.py` | Model components (Transformer blocks, embeddings, etc.) |
| `backtest/backtest_runner.py` | CLI backtest runner with all signal modes + DeepTrade params |
| `backtest/crypto_backtest.py` | Core backtester with multi-target TP and daily loss limit |
| `finetune_csv/train_sequential.py` | Sequential tokenizer + basemodel fine-tuning script |
| `finetune_csv/configs/` | YAML configs for prod training (1h, 15m, 1d) |
| `data/` | Candle CSV data (BTC_USD_1h.csv, BTC_USD_15m.csv, BTC_USD_1d.csv) |
| `examples/prediction_example.py` | Standalone prediction example script |
| `finetune/train_tokenizer.py` | Tokenizer fine-tuning script (original) |
| `finetune/train_predictor.py` | Model fine-tuning script (original) |
| `finetune/qlib_test.py` | Backtesting evaluation script (original) |

## Model Loading (Python API)

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# Load from Hugging Face
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# Create predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)

# Make predictions
pred_df = predictor.predict(df, x_timestamp, y_timestamp, pred_len=120)
```

## Fine-tuning Pipeline (CSV-based)

1. **Configure**: Edit `finetune_csv/configs/config_btc_usd_1h_prod.yaml` with your params
2. **Prepare Data**: Ensure CSV files are in `data/` (fetched via `data_ingestion.fetch_data`)
3. **Train**: `cd finetune_csv && python train_sequential.py --config configs/config_btc_usd_1h_prod.yaml`
4. **Or train all**: `bash finetune_csv/run_all_gpu.sh` (trains 1h, 15m, 1d sequentially)
5. **Backtest via CLI**: `python -m backtest.backtest_runner --model ./finetune_csv/finetuned/<name>/basemodel/best_model --tokenizer ./finetune_csv/finetuned/<name>/tokenizer/best_model --data ./data/BTC_USD_1h.csv --signal-mode kronos --device cpu`
6. **Backtest via Web UI**: Open http://localhost:7070, go to Backtest tab, select model + data, run

## Backtest Signal Modes

| Mode | Description | Model Required |
|------|-------------|----------------|
| `trend` | Trend detection only (break/limit line algorithm) | No |
| `kronos` | Kronos model predictions for long/short signals | Yes |
| `combined` | Kronos + trend confirmation (both must agree) | Yes |
| `multi_tf` | Multi-timeframe (e.g., 1d filter + 1h entries) | No |

## DeepTrade Parameters

| Parameter | CLI Flag | Description |
|-----------|----------|-------------|
| Multi-target TP | `--tp-levels 0.05:0.33,0.10:0.33,0.15:0.34` | Partial exits at multiple TP levels (pct:fraction pairs) |
| Daily loss limit | `--daily-loss-limit 0.03` | Stop new trades after 3% daily drawdown |

## Important Notes

- Models are hosted on Hugging Face Hub (NeoQuasar organization)
- Kronos-small and Kronos-base have max context of **512 tokens**
- Input data must include OHLC columns; volume/amount are optional
- Web UI requires Flask, Flask-CORS, and Plotly: `pip install flask flask-cors plotly`
- Fine-tuned models are auto-discovered from `finetune_csv/finetuned/*/` by the Web UI
- Candle data CSVs in `data/` are tracked in Git (BTC_USD_*.csv)
- For GPU training, use Vast.ai or similar — see `docs/gpu_training_workflow.md`
- Use tmux on GPU instances to keep training alive through SSH disconnects

## Links

- Paper: https://arxiv.org/abs/2508.02739
- Hugging Face: https://huggingface.co/NeoQuasar
- Live Demo: https://shiyu-coder.github.io/Kronos-demo/
