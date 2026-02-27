#!/usr/bin/env bash
# setup_env.sh — PhysicsNeMo Sym 2D Advection-Diffusion Demo
# Creates conda env at /home/featurize/work/env_conda/nemo and installs all dependencies.
# Usage: bash setup_env.sh [--force]
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
ENV_DIR="/home/featurize/work/env_conda/nemo"
CONDA_SH="/environment/miniconda3/etc/profile.d/conda.sh"
PYTHON_VER="3.11"
TORCH_VER="2.4.0+cu121"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
LOG_FILE="$(pwd)/setup_env.log"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# ── Logging ──────────────────────────────────────────────────────────────────
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=========================================="
echo "  PhysicsNeMo Sym Setup — $(date -Iseconds)"
echo "  Log: $LOG_FILE"
echo "=========================================="

# ── 1. GPU Check ─────────────────────────────────────────────────────────────
echo ""
echo "[1/6] Checking GPU environment..."

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA driver is not installed." >&2
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader,nounits | while IFS=',' read -r name driver mem cap; do
    name=$(echo "$name" | xargs)
    driver=$(echo "$driver" | xargs)
    mem=$(echo "$mem" | xargs)
    cap=$(echo "$cap" | xargs)
    echo "  GPU     : $name"
    echo "  Driver  : $driver"
    echo "  VRAM    : ${mem} MiB"
    echo "  Compute : $cap"

    # Driver version check (require >= 450)
    driver_major=$(echo "$driver" | cut -d. -f1)
    if (( driver_major < 450 )); then
        echo "ERROR: Driver $driver is too old (need >= 450)." >&2
        exit 1
    fi

    # VRAM warning if < 8 GB
    if (( mem < 8000 )); then
        echo "WARNING: VRAM ${mem} MiB < 8 GB — demo may OOM on large batch sizes."
    fi
done

# Check CUDA toolkit version (nvcc)
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "  nvcc    : $NVCC_VER"
    NVCC_MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
    if (( NVCC_MAJOR < 11 )); then
        echo "ERROR: CUDA toolkit $NVCC_VER < 11. Please upgrade." >&2
        exit 1
    fi
else
    echo "  nvcc    : not found (proceeding; runtime CUDA from driver)"
fi

echo "[1/6] GPU check passed."

# ── 2. Conda Environment ──────────────────────────────────────────────────────
echo ""
echo "[2/6] Setting up conda environment at $ENV_DIR ..."

if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: conda.sh not found at $CONDA_SH" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"

if [[ -d "$ENV_DIR" ]] && (( FORCE == 0 )); then
    echo "  Env already exists (use --force to recreate). Skipping creation."
else
    if [[ -d "$ENV_DIR" ]] && (( FORCE == 1 )); then
        echo "  --force: removing existing env..."
        conda env remove --prefix "$ENV_DIR" --yes 2>/dev/null || true
    fi
    echo "  Creating env (python=$PYTHON_VER)..."
    conda create --prefix "$ENV_DIR" python="$PYTHON_VER" --yes
fi

echo "[2/6] Conda env ready."

# ── 3. Install Packages ───────────────────────────────────────────────────────
echo ""
echo "[3/6] Installing packages..."

PIP="${ENV_DIR}/bin/pip"
PYTHON="${ENV_DIR}/bin/python"

PYPI="https://pypi.org/simple"

echo "  Upgrading pip/setuptools/wheel..."
"$PIP" install --upgrade pip setuptools wheel --index-url "$PYPI"

echo "  Installing PyTorch $TORCH_VER (cu121)..."
"$PIP" install \
    torch=="${TORCH_VER}" \
    torchvision \
    torchaudio \
    --index-url "$TORCH_INDEX"

echo "  Installing Cython (required by physicsnemo-sym build)..."
"$PIP" install Cython --index-url "$PYPI"

echo "  Installing nvidia-physicsnemo..."
"$PIP" install nvidia-physicsnemo --index-url "$PYPI"

echo "  Installing nvidia-physicsnemo-sym (no-build-isolation)..."
"$PIP" install nvidia-physicsnemo-sym --no-build-isolation --index-url "$PYPI"

echo "  Installing matplotlib and other utilities..."
"$PIP" install matplotlib pandas --index-url "$PYPI"

echo "[3/6] Package installation complete."

# ── 4. Verify Imports ─────────────────────────────────────────────────────────
echo ""
echo "[4/6] Verifying imports..."

"$PYTHON" - <<'PYEOF'
import sys, importlib

checks = [
    ("torch",          lambda m: f"version={m.__version__}, cuda={m.cuda.is_available()}"),
    ("physicsnemo",    lambda m: f"version={getattr(m, '__version__', 'unknown')}"),
    ("physicsnemo.sym",lambda m: f"OK"),
    ("matplotlib",     lambda m: f"version={m.__version__}"),
]

all_ok = True
for name, info_fn in checks:
    try:
        mod = importlib.import_module(name)
        print(f"  [OK] {name}: {info_fn(mod)}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}", file=sys.stderr)
        all_ok = False

if not all_ok:
    sys.exit(1)

# Check CUDA is actually available
import torch
if not torch.cuda.is_available():
    print("  WARNING: torch.cuda.is_available() = False — GPU not accessible to PyTorch.")
PYEOF

echo "[4/6] Import verification passed."

# ── 5. Export Environment ─────────────────────────────────────────────────────
echo ""
echo "[5/6] Exporting environment snapshots..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda env export --prefix "$ENV_DIR" > "${SCRIPT_DIR}/environment.yml"
echo "  Written: environment.yml"

"$PIP" freeze > "${SCRIPT_DIR}/requirements.txt"
echo "  Written: requirements.txt"

echo "[5/6] Snapshots saved."

# ── 6. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Setup complete!"
echo "  ENV_DIR  : $ENV_DIR"
echo "  Python   : $("${ENV_DIR}/bin/python" --version)"
echo "  Log      : $LOG_FILE"
echo ""
echo "Next step: bash run_case.sh"
