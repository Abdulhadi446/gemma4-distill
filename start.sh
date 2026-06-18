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

# Fix torchvision if torch was upgraded under it (common Colab mismatch).
# This does NOT install torch — only reinstalls torchvision at the matching version.
python3 << 'PYFIX'
import torch, subprocess, sys
base = torch.__version__.split("+")[0]
cu = torch.version.cuda.replace(".", "")
for pkg in ["torchvision", "torchaudio"]:
    try:
        __import__(pkg)
    except Exception:
        print(f"[FIX] Reinstalling {pkg} to match torch {base}+cu{cu}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install",
             f"{pkg}=={base}+cu{cu}",
             "-f", "https://download.pytorch.org/whl/torch_stable.html"]
        )
PYFIX

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
