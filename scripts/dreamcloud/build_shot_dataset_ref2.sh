#!/usr/bin/env bash

set -euo pipefail

DREAMCLOUD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DREAMCLOUD_SCRIPT_DIR/env.sh"

SEGY="${SEGY:-$PROJ_DIR/ma2+GathAP_header_edited.sgy}"
SPLIT_SEED="${SPLIT_SEED:-0}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${SEGY}" ]] || { echo "SEG-Y file not found: ${SEGY}" >&2; exit 1; }

for patch_size in 64 128 256; do
    case "${patch_size}" in
        64) overlap_size="${OVERLAP_SIZE_64:-32}" ;;
        128) overlap_size="${OVERLAP_SIZE_128:-64}" ;;
        256) overlap_size="${OVERLAP_SIZE_256:-128}" ;;
    esac

    output_dir="$PROJ_DIR/shot_dataset${patch_size}_ref2"

    echo "Building patch_size=${patch_size}, overlap_size=${overlap_size}, stride=$((patch_size - overlap_size))"

    "${PYTHON_BIN}" "${CODE_PATH}/BuildShotDataset2.py" \
        --segy "${SEGY}" \
        --patch_size "${patch_size}" \
        --overlap_size "${overlap_size}" \
        --output_dir "${output_dir}" \
        --valid 0.3 \
        --valid_mode group_random \
        --seed "${SPLIT_SEED}" \
        --clip -2 2 \
        --slice 0 1501 \
        --normalize \
        --gen-ref 2

    "${PYTHON_BIN}" "${CODE_PATH}/ExtractShot2.py" \
        --segy "${SEGY}" \
        --output_dir "${output_dir}/shot" \
        --clip -2 2 \
        --slice 0 1501

done
