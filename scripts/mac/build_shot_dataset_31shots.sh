#!/usr/bin/env bash

set -euo pipefail

SIANDGX_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SIANDGX_SCRIPT_DIR/env.sh"

SEGY="${SEGY:-$PROJ_DIR/shots_196_226_step1_1s_2p5s_4ms.sgy}"
RANDOM_SEGY="${RANDOM_SEGY:-$PROJ_DIR/random_gaussian_31shots.sgy}"
RANDOM_SEED="${RANDOM_SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"

"${PYTHON_BIN}" "${CODE_PATH}/BuildRandomSegy.py" \
    --segy "${SEGY}" \
    --output "${RANDOM_SEGY}" \
    --seed "${RANDOM_SEED}"

for config in "64 32" "64 48" "128 64" "128 96"; do
    read -r patch_size overlap_size <<< "${config}"

    output_dir="$PROJ_DIR/shot_dataset${patch_size}_overlap${overlap_size}_31shots"
    random_output_dir="$PROJ_DIR/random_shot_dataset${patch_size}_overlap${overlap_size}_31shots"

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
        --gen-ref 1 \
        --normalize

    "${PYTHON_BIN}" "${CODE_PATH}/ExtractShot2.py" \
        --segy "${SEGY}" \
        --output_dir "${output_dir}/shot" \
        --clip -2 2 \
        --slice 0 1501

    "${PYTHON_BIN}" "${CODE_PATH}/BuildShotDataset2.py" \
        --segy "${RANDOM_SEGY}" \
        --patch_size "${patch_size}" \
        --overlap_size "${overlap_size}" \
        --output_dir "${random_output_dir}" \
        --valid 0.3 \
        --valid_mode group_random \
        --seed "${SPLIT_SEED}" \
        --gen-ref 1 \
        --slice 0 1501

    "${PYTHON_BIN}" "${CODE_PATH}/ExtractShot2.py" \
        --segy "${RANDOM_SEGY}" \
        --output_dir "${random_output_dir}/shot" \
        --slice 0 1501
done
