#!/usr/bin/env bash

set -euo pipefail

HWGPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HWGPU_SCRIPT_DIR/env.sh"

SEGY="${SEGY:-${CODE_PATH}/temp/ma2+GathAP_header_edited.sgy}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${SEGY}" ]] || { echo "SEG-Y file not found: ${SEGY}" >&2; exit 1; }

for patch_size in 64 128 256; do
    output_dir="${CODE_PATH}/temp/shot_dataset${patch_size}"

    "${PYTHON_BIN}" "${CODE_PATH}/build_shot_dataset2.py" \
        --segy "${SEGY}" \
        --patch_size "${patch_size}" \
        --overlap_size 32 \
        --output_dir "${output_dir}" \
        --valid 0.3 \
        --valid_mode group_random \
        --clip -2 2 \
        --slice 0 1501 \
        --normalize

    "${PYTHON_BIN}" "${CODE_PATH}/extract_shot2.py" \
        --segy "${SEGY}" \
        --output_dir "${output_dir}/shot" \
        --clip -2 2 \
        --slice 0 1501
done
