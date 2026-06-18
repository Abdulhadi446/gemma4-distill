#!/usr/bin/env bash
set -euo pipefail

echo "================================================"
echo "  Gemma 4 12B Distillation — Setup & Run"
echo "================================================"

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'" 2>/dev/null \
    || { echo "[ERROR] CUDA GPU not detected."; exit 1; }

echo ""
echo "[1/3] Installing Python dependencies..."

# Don't force-upgrade torch/torchvision/torchaudio — Colab pre-installs them at
# the correct CUDA build.  Only add missing packages.
pip install --quiet --upgrade \
    transformers \
    accelerate \
    bitsandbytes \
    pillow \
    librosa \
    "opencv-python-headless" \
    tqdm \
    scipy \
    2>&1 | grep -v "already satisfied\|dependency conflict"

# If torchvision or torchaudio are broken after the pip upgrade above,
# reinstall them at versions matching the installed torch build.
python3 -c "
import torch

def _fix_pkg(pkg, base, cu_short):
    import subprocess, sys, importlib
    try:
        importlib.import_module(pkg)
        # smoke test
        if pkg == 'torchvision':
            import torchvision  # noqa
    except Exception:
        pass
    else:
        return  # already works
    ver = base.split('.')
    ver = '.'.join(ver[:2])  # major.minor
    url = 'https://download.pytorch.org/whl/torch_stable.html'
    pip = [sys.executable, '-m', 'pip', 'install', '--quiet', '--force-reinstall',
           f'{pkg}=={base}+cu{cu_short}', '-f', url]
    print(f'Reinstalling {pkg}=={base}+cu{cu_short}...')
    subprocess.check_call(pip)

base = torch.__version__.split('+')[0]
cu = torch.version.cuda.replace('.', '')
_fix_pkg('torchvision', base, cu)
_fix_pkg('torchaudio', base, cu)
" 2>&1

echo "[2/3] Dependencies ready."

mkdir -p ./data

echo ""
echo "[3/3] Starting distillation data generation..."
echo ""

python3 distill.py

echo ""
echo "Done! Output in ./data/"
