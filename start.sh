#!/usr/bin/env bash
set -e

echo "================================================"
echo "  Gemma 4 12B Distillation — Setup & Run"
echo "================================================"

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found"; exit 1
fi

python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
    || { echo "[ERROR] CUDA GPU not detected."; exit 1; }

echo ""
echo "[1/3] Installing Python dependencies..."
echo ""

# Install only the packages needed by distill.py — skip torch/torchvision/torchaudio
# (Colab already has them) and skip heavy packages unless needed.
pip install transformers accelerate bitsandbytes pillow tqdm scipy

# Fix torchvision if broken (common Colab ABI mismatch).
python3 -c "
import importlib, subprocess, sys
try:
    importlib.import_module('torchvision')
except Exception:
    print('[FIX] Upgrading torchvision...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'torchvision'])
"

# Optional: install audio/video libraries (needed only if processor downloads media)
# pip install librosa opencv-python-headless

echo ""
echo "[2/3] Dependencies ready."

mkdir -p ./data

echo ""
echo "[3/3] Starting distillation data generation..."
echo ""

python3 distill.py

echo ""
echo "Done! Output in ./data/"
