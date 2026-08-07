#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/hwdata/24ydz3d/deeplearning/environments/py313/bin/python3.13}"
SF_PATH="${SF_PATH:-/hwdata/24ydz3d/deeplearning/SeisFlow-main}"
SEGY="${SEGY:-${SF_PATH}/temp/ma2+GathAP_header_edited.sgy}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${SEGY}" ]] || { echo "SEG-Y file not found: ${SEGY}" >&2; exit 1; }

for patch_size in 64 128 256; do
    output_dir="${SF_PATH}/temp/shot_dataset${patch_size}"

    "${PYTHON_BIN}" "${SF_PATH}/build_shot_dataset2.py" \
        --segy "${SEGY}" \
        --patch_size "${patch_size}" \
        --overlap_size 32 \
        --output_dir "${output_dir}" \
        --valid 0.3 \
        --valid_mode group_random \
        --clip -2 2 \
        --slice 0 1501 \
        --normalize

    "${PYTHON_BIN}" "${SF_PATH}/extract_shot2.py" \
        --segy "${SEGY}" \
        --output_dir "${output_dir}/shot" \
        --clip -2 2 \
        --slice 0 1501
done
