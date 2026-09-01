#!/usr/bin/env bash

set -euo pipefail

HWGPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HWGPU_SCRIPT_DIR/env.sh"

SEGY="${SEGY:-$PROJ_DIR/ma2+GathAP_header_edited_small.sgy}"
RANDOM_SEGY="${RANDOM_SEGY:-$PROJ_DIR/random_gaussian_small.sgy}"
RANDOM_SEED="${RANDOM_SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"
PATCH_SIZE="${PATCH_SIZE:-64}"
OVERLAP_SIZE="${OVERLAP_SIZE:-32}"
VALID_RATIO="${VALID_RATIO:-0.3}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_DIR/shot_dataset${PATCH_SIZE}_nano}"
RANDOM_OUTPUT_DIR="${RANDOM_OUTPUT_DIR:-$PROJ_DIR/random_shot_dataset${PATCH_SIZE}_nano}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${SEGY}" ]] || { echo "Small SEG-Y file not found: ${SEGY}" >&2; exit 1; }

echo "Building Nano shot dataset"
echo "  SEG-Y: ${SEGY}"
echo "  patch_size: ${PATCH_SIZE}"
echo "  overlap_size: ${OVERLAP_SIZE}"
echo "  stride: $((PATCH_SIZE - OVERLAP_SIZE))"
echo "  validation ratio: ${VALID_RATIO}"
echo "  output: ${OUTPUT_DIR}"
echo "  random output: ${RANDOM_OUTPUT_DIR}"

"${PYTHON_BIN}" "${CODE_PATH}/BuildRandomSegy.py" \
    --segy "${SEGY}" \
    --output "${RANDOM_SEGY}" \
    --seed "${RANDOM_SEED}"

"${PYTHON_BIN}" "${CODE_PATH}/BuildShotDataset2.py" \
    --segy "${SEGY}" \
    --patch_size "${PATCH_SIZE}" \
    --overlap_size "${OVERLAP_SIZE}" \
    --output_dir "${OUTPUT_DIR}" \
    --valid "${VALID_RATIO}" \
    --valid_mode group_random \
    --seed "${SPLIT_SEED}" \
    --clip -2 2 \
    --slice 0 1501 \
    --normalize

"${PYTHON_BIN}" "${CODE_PATH}/ExtractShot2.py" \
    --segy "${SEGY}" \
    --output_dir "${OUTPUT_DIR}/shot" \
    --clip -2 2 \
    --slice 0 1501

"${PYTHON_BIN}" "${CODE_PATH}/BuildShotDataset2.py" \
    --segy "${RANDOM_SEGY}" \
    --patch_size "${PATCH_SIZE}" \
    --overlap_size "${OVERLAP_SIZE}" \
    --output_dir "${RANDOM_OUTPUT_DIR}" \
    --valid "${VALID_RATIO}" \
    --valid_mode group_random \
    --seed "${SPLIT_SEED}" \
    --slice 0 1501

"${PYTHON_BIN}" "${CODE_PATH}/ExtractShot2.py" \
    --segy "${RANDOM_SEGY}" \
    --output_dir "${RANDOM_OUTPUT_DIR}/shot" \
    --slice 0 1501

echo "Nano shot datasets completed"
