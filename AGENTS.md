# Kronos - Agent Quick Reference

## Project Summary

Kronos is the **first open-source foundation model** for financial candlesticks (K-lines), trained on data from over **45 global exchanges**.

### Key Features
- **Novel two-stage framework**: Specialized tokenizer quantizes continuous multi-dimensional K-line data (OHLCV) into hierarchical discrete tokens, then a large autoregressive Transformer is pre-trained on these tokens
- **Multiple model variants**: mini (4.1M params, 2048 context), small (24.7M params, 512 context), base (102.3M params, 512 context)
- **Complete fine-tuning pipeline**: Includes scripts for fine-tuning on custom datasets (e.g., Chinese A-share market with Qlib)
- **Web UI**: Interactive Flask-based interface for predictions and visualization

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
- Select a model (kronos-mini, kronos-small, or kronos-base)
- Choose device (cpu/cuda)
- Load the model

### 4. Upload Data & Predict
- Upload CSV/feather files with columns: `open`, `high`, `low`, `close` (optional: `volume`, `amount`)
- Set prediction parameters (lookback, prediction length, temperature, etc.)
- Generate forecasts with interactive candlestick charts

## Key Files

| File | Purpose |
|------|---------|
| `webui/app.py` | **Main entrypoint** - Flask web application |
| `model/kronos.py` | Core model implementation (Kronos, KronosTokenizer, KronosPredictor) |
| `model/module.py` | Model components (Transformer blocks, embeddings, etc.) |
| `examples/prediction_example.py` | Standalone prediction example script |
| `finetune/train_tokenizer.py` | Tokenizer fine-tuning script |
| `finetune/train_predictor.py` | Model fine-tuning script |
| `finetune/qlib_test.py` | Backtesting evaluation script |

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

## Fine-tuning Pipeline

1. **Configure**: Edit `finetune/config.py` with your paths
2. **Prepare Data**: `python finetune/qlib_data_preprocess.py`
3. **Fine-tune Tokenizer**: `torchrun --standalone --nproc_per_node=2 finetune/train_tokenizer.py`
4. **Fine-tune Predictor**: `torchrun --standalone --nproc_per_node=2 finetune/train_predictor.py`
5. **Backtest**: `python finetune/qlib_test.py --device cuda:0`

## Important Notes

- Models are hosted on Hugging Face Hub (NeoQuasar organization)
- Kronos-small and Kronos-base have max context of **512 tokens**
- Input data must include OHLC columns; volume/amount are optional
- Web UI requires Flask and Plotly for visualization

## Links

- Paper: https://arxiv.org/abs/2508.02739
- Hugging Face: https://huggingface.co/NeoQuasar
- Live Demo: https://shiyu-coder.github.io/Kronos-demo/
