# Vast.ai GPU Training Guide — Kronos

Quick guide to rent a GPU on Vast.ai and run all three timeframe training configs.

---

## Step 1: Account & Credits

1. Go to [cloud.vast.ai](https://cloud.vast.ai) and sign up
2. Verify your email
3. Go to **Billing → Add Credit** — load **$15-20** (enough for all 3 configs with buffer)
4. Optionally enable **autobilling** to avoid interruptions if balance runs low

---

## Step 2: Set Up SSH Key

On your Mac:

```bash
# Generate SSH key (if you don't already have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Print your public key
cat ~/.ssh/id_ed25519.pub
```

Copy the output, then go to [cloud.vast.ai/manage-keys](https://cloud.vast.ai/manage-keys/) and paste your public key.

---

## Step 3: Find & Rent a GPU

1. Go to [cloud.vast.ai/create](https://cloud.vast.ai/create/) (Search page)
2. Filter for:
   - **GPU**: RTX 4090 or A100 (either works great)
   - **GPU RAM**: ≥ 16GB
   - **Disk space**: 50GB (model files + data + PyTorch images)
   - **Reliability**: ≥ 0.9 (avoids flaky hosts)
3. Sort by **price** — pick the cheapest reliable option (~$0.30-0.80/hr)
4. Under **Template**, select **PyTorch** (pre-built image with CUDA + PyTorch)
5. Under **Image**, use the default PyTorch image (e.g., `pytorch/pytorch:latest-cuda`)
6. Click **Rent**

Expected cost:
- 1h config: ~$1-3 (2-4 hrs)
- 15m config: ~$3-10 (6-12 hrs)
- 1d config: ~$0.50-1.50 (1-2 hrs)
- **Total: ~$5-15**

---

## Step 4: Connect via SSH

Once the instance shows "Running" on the Vast.ai dashboard, click it to see the SSH connection string. It will look like:

```
ssh -p PORT root@IP_ADDRESS
```

Connect from your Mac:

```bash
ssh -p 12345 root@142.214.185.187
```

Type `yes` when prompted about host authenticity.

---

## Step 5: Set Up Environment & Train

Once connected to the GPU instance:

```bash
# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Clone your repo
git clone https://github.com/phon3/Kronos.git
cd Kronos

# Create venv and install deps
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run all three training configs
./finetune_csv/run_all_gpu.sh
```

Or train one at a time:

```bash
./finetune_csv/run_all_gpu.sh 1h     # just 1h
./finetune_csv/run_all_gpu.sh 15m    # just 15m
./finetune_csv/run_all_gpu.sh 1d     # just 1d
```

### Keep training running after disconnect

Use `tmux` so training continues even if your SSH connection drops:

```bash
# Start a tmux session
tmux new -s kronos

# Run training inside tmux
./finetune_csv/run_all_gpu.sh

# Detach: press Ctrl+B, then D
# Reconnect later:
tmux attach -s kronos
```

---

## Step 6: Monitor Training

Check GPU utilization in a separate SSH session:

```bash
nvidia-smi -l 5    # refresh every 5 seconds
```

Or check training logs:

```bash
tail -f finetune_csv/training_logs/train_*.log
```

---

## Step 7: Pull Trained Models Back to Mac

When training is complete, from your Mac:

```bash
# Option A: scp directly
scp -P 12345 -r root@142.214.185.187:/workspace/Kronos/finetune_csv/finetuned/btc_usd_1h_prod ./finetune_csv/finetuned/
scp -P 12345 -r root@142.214.185.187:/workspace/Kronos/finetune_csv/finetuned/btc_usd_15m_prod ./finetune_csv/finetuned/
scp -P 12345 -r root@142.214.185.187:/workspace/Kronos/finetune_csv/finetuned/btc_usd_1d_prod ./finetune_csv/finetuned/

# Option B: tar + scp (faster for multiple files)
ssh -p 12345 root@142.214.185.187 "cd /workspace/Kronos && tar czf /tmp/models.tar.gz finetune_csv/finetuned/btc_usd_*_prod/"
scp -P 12345 root@142.214.185.187:/tmp/models.tar.gz .
tar xzf models.tar.gz
```

---

## Step 8: Delete the Instance

**Important**: Once you've downloaded your models, delete the instance to stop billing:

1. Go to [cloud.vast.ai](https://cloud.vast.ai) → Instances
2. Click **Delete** on your instance
3. Confirm — this stops all charges

---

## Quick Reference: Full GPU Session

```bash
# On Mac: connect
ssh -p PORT root@IP

# On GPU: set up and train
git clone https://github.com/phon3/Kronos.git && cd Kronos
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
tmux new -s kronos
./finetune_csv/run_all_gpu.sh
# Ctrl+B, D to detach

# On Mac: download models when done
scp -P PORT -r root@IP:/workspace/Kronos/finetune_csv/finetuned/btc_usd_*_prod ./finetune_csv/finetuned/

# Delete instance on vast.ai dashboard
```

---

## Troubleshooting

### CUDA out of memory
- Edit the config: reduce `batch_size` to 32 or 16
- Or reduce `lookback_window` to 256

### Instance died mid-training
- If you used `tmux`, your session may be recoverable on a new instance
- Vast.ai instances can be unreliable — that's why we filter by reliability ≥ 0.9
- Training saves checkpoints per epoch, so you can resume with `--skip-existing`

### Data fetch fails on GPU instance
- The `run_all_gpu.sh` script auto-fetches data if missing
- If Coinbase API is rate-limited, scp the data from your Mac instead:
  ```bash
  scp -P PORT ./data/BTC_USD_*.csv root@IP:/workspace/Kronos/data/
  ```

### Disk full
- Model checkpoints + data can use 10-20GB
- If you allocated 50GB disk, you should be fine
- Clean up old logs: `rm finetune_csv/finetuned/*/logs/*.log`
