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

# Fix torchvision/torchaudio if they are broken (common Colab ABI mismatch).
# torchaudio shares torch's version; torchvision uses 0.{torch_minor+15}.{patch}.
python3 << 'PYFIX'
import torch, subprocess, sys

base = torch.__version__.split("+")[0]
cu = torch.version.cuda.replace(".", "")

# torchvision version map: torch 2.12.1 -> tv 0.27.1, torch 2.11.0 -> tv 0.26.0
parts = base.split(".")
tv_ver = f"0.{int(parts[1]) + 15}.{parts[2]}"

for pkg, ver in [("torchvision", tv_ver), ("torchaudio", base)]:
    try:
        __import__(pkg)
    except Exception:
        print(f"[FIX] Reinstalling {pkg}=={ver}+cu{cu}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall",
             f"{pkg}=={ver}+cu{cu}",
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
