#!/usr/bin/env bash

set -euo pipefail

HWGPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HWGPU_SCRIPT_DIR/env.sh"

SEGY="${SEGY:-$PROJ_DIR/ma2+GathAP_header_edited.sgy}"
RANDOM_SEGY="${RANDOM_SEGY:-$PROJ_DIR/random_gaussian.sgy}"
RANDOM_SEED="${RANDOM_SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${SEGY}" ]] || { echo "SEG-Y file not found: ${SEGY}" >&2; exit 1; }

"${PYTHON_BIN}" "${CODE_PATH}/BuildRandomSegy.py" \
    --segy "${SEGY}" \
    --output "${RANDOM_SEGY}" \
    --seed "${RANDOM_SEED}"

for patch_size in 64 128 256; do
    output_dir="$PROJ_DIR/shot_dataset${patch_size}"
    random_output_dir="$PROJ_DIR/random_shot_dataset${patch_size}"

    "${PYTHON_BIN}" "${CODE_PATH}/BuildShotDataset2.py" \
        --segy "${SEGY}" \
        --patch_size "${patch_size}" \
        --overlap_size 32 \
        --output_dir "${output_dir}" \
        --valid 0.3 \
        --valid_mode group_random \
        --seed "${SPLIT_SEED}" \
        --clip -2 2 \
        --slice 0 1501 \
        --normalize

    "${PYTHON_BIN}" "${CODE_PATH}/ExtractShot2.py" \
        --segy "${SEGY}" \
        --output_dir "${output_dir}/shot" \
        --clip -2 2 \
        --slice 0 1501

    "${PYTHON_BIN}" "${CODE_PATH}/BuildShotDataset2.py" \
        --segy "${RANDOM_SEGY}" \
        --patch_size "${patch_size}" \
        --overlap_size 32 \
        --output_dir "${random_output_dir}" \
        --valid 0.3 \
        --valid_mode group_random \
        --seed "${SPLIT_SEED}" \
        --slice 0 1501

    "${PYTHON_BIN}" "${CODE_PATH}/ExtractShot2.py" \
        --segy "${RANDOM_SEGY}" \
        --output_dir "${random_output_dir}/shot" \
        --slice 0 1501
done
