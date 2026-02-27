#!/usr/bin/env bash
# run_case.sh — Run the PhysicsNeMo Sym three-fin 2D heat-sink PINN demo
# Usage:
#   bash run_case.sh                      # default 10k steps
#   MAX_STEPS=500000 bash run_case.sh     # full run
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="/home/featurize/work/env_conda/nemo"
PYTHON="${ENV_DIR}/bin/python"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SCRIPT_DIR}/outputs/${TIMESTAMP}"
LATEST_LINK="${SCRIPT_DIR}/outputs/latest"
CASE_SOURCE_REPO="https://github.com/NVIDIA/modulus-sym"
CASE_SUBDIR="examples/three_fin_2d"
CASES_DIR="${SCRIPT_DIR}/cases"
CASE_DIR="${CASES_DIR}/three_fin_2d"
MAX_STEPS="${MAX_STEPS:-10000}"

# ── 1. Pre-flight checks ──────────────────────────────────────────────────────
echo "=========================================="
echo "  PhysicsNeMo Sym Run — $(date -Iseconds)"
echo "=========================================="
echo ""

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    echo "       Please run: bash setup_env.sh" >&2
    exit 1
fi

echo "[1/5] Pre-flight: OK (Python=$("$PYTHON" --version 2>&1))"

# ── 2. Download example via git sparse-checkout ───────────────────────────────
echo ""
echo "[2/5] Acquiring example source..."

if [[ -d "$CASE_DIR" ]] && [[ -f "${CASE_DIR}/heat_sink.py" ]]; then
    echo "  Case dir already exists: $CASE_DIR — skipping download."
else
    mkdir -p "$CASES_DIR"
    SPARSE_TMP="${CASES_DIR}/_sparse_tmp"
    if [[ -d "$SPARSE_TMP" ]]; then
        rm -rf "$SPARSE_TMP"
    fi

    echo "  Cloning $CASE_SUBDIR from $CASE_SOURCE_REPO (sparse, no blobs)..."
    git clone \
        --filter=blob:none \
        --no-checkout \
        --depth=1 \
        --sparse \
        "$CASE_SOURCE_REPO" \
        "$SPARSE_TMP"

    pushd "$SPARSE_TMP" > /dev/null
    git sparse-checkout set "$CASE_SUBDIR"
    git checkout
    popd > /dev/null

    cp -r "${SPARSE_TMP}/${CASE_SUBDIR}" "$CASE_DIR"
    rm -rf "$SPARSE_TMP"
    echo "  Downloaded to: $CASE_DIR"
fi

echo "[2/5] Example source ready."

# ── 3. Prepare output directory ───────────────────────────────────────────────
echo ""
echo "[3/5] Preparing output directory..."

mkdir -p "$OUT_DIR"
# Update latest symlink
ln -sfn "$OUT_DIR" "$LATEST_LINK"
echo "  Output dir : $OUT_DIR"
echo "  Latest link: $LATEST_LINK"

echo "[3/5] Output directory ready."

# ── 4. Start GPU monitor ──────────────────────────────────────────────────────
echo ""
echo "[4/5] Starting GPU monitor..."

GPU_STATS_LOG="${OUT_DIR}/gpu_stats.log"
GPU_MON_PID=""

cleanup_gpu_mon() {
    if [[ -n "$GPU_MON_PID" ]] && kill -0 "$GPU_MON_PID" 2>/dev/null; then
        kill "$GPU_MON_PID" 2>/dev/null || true
        wait "$GPU_MON_PID" 2>/dev/null || true
        echo "  GPU monitor stopped (PID $GPU_MON_PID)."
    fi
}
trap cleanup_gpu_mon EXIT

if command -v nvidia-smi &>/dev/null; then
    # Use --query-gpu loop instead of dmon (dmon fails on some driver configs)
    # Columns: timestamp, util%, mem_used_MiB, mem_total_MiB, temp_C, power_W
    {
        echo "# timestamp                 util_pct  mem_used_MiB  mem_total_MiB  temp_C  power_W"
        while true; do
            nvidia-smi \
                --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
                --format=csv,noheader,nounits 2>/dev/null || true
            sleep 5
        done
    } >> "$GPU_STATS_LOG" 2>&1 &
    GPU_MON_PID=$!
    echo "  GPU monitor PID: $GPU_MON_PID → $GPU_STATS_LOG"
else
    echo "  WARNING: nvidia-smi not found — GPU stats will not be collected."
fi

echo "[4/5] GPU monitor running."

# ── 5. Run the demo ───────────────────────────────────────────────────────────
echo ""
echo "[5/5] Running heat-sink demo (MAX_STEPS=${MAX_STEPS})..."
echo "  Python     : $PYTHON"
echo "  Case dir   : $CASE_DIR"
echo "  Output dir : $OUT_DIR"
echo ""

"$PYTHON" "${SCRIPT_DIR}/run_case.py" \
    --case-dir  "$CASE_DIR" \
    --out-dir   "$OUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --seed      42

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Run complete!"
echo "  Results: $OUT_DIR"
echo "  Report : ${OUT_DIR}/baseline_report.md"
echo "  Charts : ${OUT_DIR}/charts/"
echo "  Latest : $LATEST_LINK"
echo "=========================================="
