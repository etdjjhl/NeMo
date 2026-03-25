#!/usr/bin/env bash
# recover_local.sh — Recover interrupted training data from local disk to archive directory
# Usage:
#   bash recover_local.sh                  # recover all (baseline + param)
#   bash recover_local.sh baseline         # only baseline
#   bash recover_local.sh param            # only param
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BASE="/home/featurize/data/nemo_train"
ARCHIVE_BASE="${SCRIPT_DIR}/outputs"
CASE_DIR="${SCRIPT_DIR}/cases/three_fin_2d"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

FILTER="${1:-all}"  # baseline | param | all

recover_mode() {
    local mode="$1"           # baseline | param
    local run_name="$2"       # run | run_param
    local local_dir="${LOCAL_BASE}/${mode}/${run_name}"

    if [[ ! -d "$local_dir" ]]; then
        echo "[recover] No data found for ${mode}: ${local_dir}"
        return
    fi

    # Count files
    local file_count
    file_count="$(find "$local_dir" -type f | wc -l)"
    if [[ "$file_count" -eq 0 ]]; then
        echo "[recover] Empty directory for ${mode}: ${local_dir}"
        return
    fi

    echo "[recover] Found ${file_count} files in ${local_dir}"

    # Archive destination
    local archive_dir="${ARCHIVE_BASE}/${mode}/recovered_${TIMESTAMP}/hydra_outputs"
    mkdir -p "$archive_dir"

    echo "[recover] Copying: ${local_dir} → ${archive_dir}"
    cp -r "${local_dir}/." "$archive_dir/"
    echo "[recover] Done: ${archive_dir}"

    # Also restore to case_dir/outputs/run if it's a dangling symlink or missing
    local case_run="${CASE_DIR}/outputs/${run_name}"
    if [[ -L "$case_run" ]]; then
        echo "[recover] Removing dangling symlink: ${case_run}"
        rm -f "$case_run"
    fi
    if [[ ! -d "$case_run" ]]; then
        echo "[recover] Restoring case dir: ${local_dir} → ${case_run}"
        mkdir -p "$(dirname "$case_run")"
        cp -r "$local_dir" "$case_run"
    fi

    # Show checkpoint step if available
    local ckpt="${local_dir}/optim_checkpoint.0.pth"
    if [[ -f "$ckpt" ]]; then
        local step
        step="$(python3 -c "
import torch, sys
ckpt = torch.load('${ckpt}', map_location='cpu', weights_only=False)
print(ckpt.get('step', 'unknown'))
" 2>/dev/null || echo "unknown")"
        echo "[recover] Checkpoint step for ${mode}: ${step}"
    fi
}

echo "=========================================="
echo "  Recover local-disk training data"
echo "  Source: ${LOCAL_BASE}"
echo "  Filter: ${FILTER}"
echo "=========================================="
echo ""

if [[ "$FILTER" == "all" || "$FILTER" == "baseline" ]]; then
    recover_mode "baseline" "run"
fi

if [[ "$FILTER" == "all" || "$FILTER" == "param" ]]; then
    recover_mode "param" "run_param"
fi

echo ""
echo "[recover] All done."
