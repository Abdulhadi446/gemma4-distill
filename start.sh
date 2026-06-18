#!/usr/bin/env bash
set -euo pipefail

echo "================================================"
echo "  Gemma 4 12B Distillation — Setup & Run"
echo "================================================"

# --- Check Python ---
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi

# --- Check CUDA ---
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'" 2>/dev/null \
    || { echo "[ERROR] CUDA GPU not detected. This script requires a CUDA-capable GPU."; exit 1; }

# --- Install deps ---
echo ""
echo "[1/3] Installing Python dependencies..."
pip install -U --quiet \
    torch \
    transformers \
    accelerate \
    bitsandbytes \
    pillow \
    torchaudio \
    librosa \
    opencv-python \
    tqdm \
    scipy

echo "[2/3] Dependencies installed."

# --- Create data dir ---
mkdir -p ./data

# --- Run ---
echo ""
echo "[3/3] Starting distillation data generation..."
echo ""

python3 distill.py

echo ""
echo "Done! Output in ./data/"
