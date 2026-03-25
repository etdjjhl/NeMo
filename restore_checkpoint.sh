#!/usr/bin/env bash
# restore_checkpoint.sh — Restore archived outputs back to case dir for resumed training
#
# Usage:
#   bash restore_checkpoint.sh outputs/baseline/20260318_024109
#   bash restore_checkpoint.sh outputs/param/20260318_XXXXXX
#
# After restoring, resume training with:
#   MAX_STEPS=50000 bash run_case.sh          # baseline
#   MAX_STEPS=50000 MODE=param bash run_case.sh  # param
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="${SCRIPT_DIR}/cases/three_fin_2d"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash restore_checkpoint.sh <archived-output-dir>"
    echo "  e.g. bash restore_checkpoint.sh outputs/baseline/20260318_024109"
    exit 1
fi

ARCHIVE_DIR="$(cd "$1" && pwd)"
HYDRA_SRC="${ARCHIVE_DIR}/hydra_outputs"

if [[ ! -d "$HYDRA_SRC" ]]; then
    echo "ERROR: ${HYDRA_SRC} not found." >&2
    exit 1
fi

# Detect mode from metadata.json or directory path
MODE=""
META="${ARCHIVE_DIR}/metadata.json"
if [[ -f "$META" ]]; then
    MODE=$(python3 -c "import json; print(json.load(open('${META}')).get('params',{}).get('mode',''))" 2>/dev/null || true)
fi
if [[ -z "$MODE" ]]; then
    # Fallback: infer from path (outputs/baseline/... or outputs/param/...)
    case "$ARCHIVE_DIR" in
        *"/baseline/"*) MODE="baseline" ;;
        *"/param/"*)    MODE="param" ;;
        *)              MODE="baseline" ;;
    esac
fi

if [[ "$MODE" == "param" ]]; then
    TARGET="${CASE_DIR}/outputs/run_param"
else
    TARGET="${CASE_DIR}/outputs/run"
fi

echo "Restore checkpoint"
echo "  From : ${HYDRA_SRC}"
echo "  To   : ${TARGET}"
echo "  Mode : ${MODE}"
echo ""

# Back up existing target if present
if [[ -d "$TARGET" ]]; then
    BACKUP="${TARGET}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "  Backing up existing ${TARGET} → ${BACKUP}"
    mv "$TARGET" "$BACKUP"
fi

# Copy archived outputs back
cp -r "$HYDRA_SRC" "$TARGET"
echo "  Restored OK."
echo ""

# Show checkpoint info
OPTIM="${TARGET}/optim_checkpoint.0.pth"
if [[ -f "$OPTIM" ]]; then
    STEP=$(python3 -c "
import torch
ckpt = torch.load('${OPTIM}', map_location='cpu', weights_only=False)
print(ckpt.get('step', '?'))
" 2>/dev/null || echo "?")
    echo "  Checkpoint step: ${STEP}"
fi

echo ""
echo "Ready to resume. Example:"
if [[ "$MODE" == "param" ]]; then
    echo "  MAX_STEPS=50000 MODE=param bash run_case.sh"
else
    echo "  MAX_STEPS=50000 bash run_case.sh"
fi
