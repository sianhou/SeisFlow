#!/usr/bin/env bash

set -euo pipefail

HWGPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HWGPU_SCRIPT_DIR/env.sh"

SCRIPT_NAME="$(basename "$0" .sh)"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/shot_dataset64_nano}"
RUN_DIR="${RUN_DIR:-$PROJ_DIR/$SCRIPT_NAME}"
TRAIN_SCRIPT_NAME="${SCRIPT_NAME/#recon_/train_}"
TRAIN_ROOT="${TRAIN_ROOT:-$PROJ_DIR/$TRAIN_SCRIPT_NAME}"
NPROC_PER_NODE_NANO="${NPROC_PER_NODE_NANO:-$NPROC_PER_NODE}"
RECON_BATCH_SIZE="${RECON_BATCH_SIZE:-32}"
EPOCH_START="${EPOCH_START:-100}"
EPOCH_END="${EPOCH_END:-2000}"
EPOCH_STEP="${EPOCH_STEP:-100}"

[[ -x "${TORCHRUN_BIN}" ]] || { echo "torchrun not found: ${TORCHRUN_BIN}" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_DIR}/valid_dim" ]] || { echo "Nano validation dimension data not found: ${DATA_DIR}/valid_dim" >&2; exit 1; }
[[ -d "${DATA_DIR}/valid_aux" ]] || { echo "Nano validation metadata not found: ${DATA_DIR}/valid_aux" >&2; exit 1; }
[[ -d "${DATA_DIR}/shot" ]] || { echo "Nano original shot data not found: ${DATA_DIR}/shot" >&2; exit 1; }
[[ "${NPROC_PER_NODE_NANO}" =~ ^[1-9][0-9]*$ ]] || { echo "NPROC_PER_NODE_NANO must be a positive integer" >&2; exit 1; }
[[ "${EPOCH_START}" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCH_START must be a positive integer" >&2; exit 1; }
[[ "${EPOCH_END}" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCH_END must be a positive integer" >&2; exit 1; }
[[ "${EPOCH_STEP}" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCH_STEP must be a positive integer" >&2; exit 1; }
(( EPOCH_START <= EPOCH_END )) || { echo "EPOCH_START must not exceed EPOCH_END" >&2; exit 1; }

if [[ -z "${TRAIN_RUN_DIR:-}" ]]; then
    [[ -d "${TRAIN_ROOT}" ]] || { echo "Nano training root not found: ${TRAIN_ROOT}" >&2; exit 1; }
    TRAIN_RUN_DIR="$(find "${TRAIN_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi
[[ -d "${TRAIN_RUN_DIR}" ]] || { echo "Nano training run directory not found: ${TRAIN_RUN_DIR}" >&2; exit 1; }

mkdir -p "${RUN_DIR}"
cd "${CODE_PATH}"

echo "Reconstructing AugmentedDiT Nano checkpoints"
echo "  data: ${DATA_DIR}"
echo "  training run: ${TRAIN_RUN_DIR}"
echo "  output: ${RUN_DIR}"
echo "  processes: ${NPROC_PER_NODE_NANO}"
echo "  epochs: ${EPOCH_START}:${EPOCH_STEP}:${EPOCH_END}"

for epoch in $(seq "${EPOCH_START}" "${EPOCH_STEP}" "${EPOCH_END}"); do
    epoch_name="$(printf '%05d' "${epoch}")"
    checkpoint_dir="${TRAIN_RUN_DIR}/checkpoint_epoch_${epoch_name}"
    patch_output_dir="${RUN_DIR}/valid_epoch_${epoch_name}"
    shot_output_dir="${RUN_DIR}/valid_recon_shot_epoch_${epoch_name}"
    diff_output_dir="${RUN_DIR}/diff_recon_shot_epoch_${epoch_name}"

    [[ -d "${checkpoint_dir}" ]] || { echo "Checkpoint not found: ${checkpoint_dir}" >&2; exit 1; }

    echo "Reconstructing epoch ${epoch} from ${checkpoint_dir}"
    "${TORCHRUN_BIN}" \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE_NANO}" \
        --node_rank=0 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        AugmentedDiTSeisDimReconNeRF.py sample \
        --ckpt "${checkpoint_dir}" \
        --input_dim_dir "${DATA_DIR}/valid_dim" \
        --output_dir "${RUN_DIR}" \
        --log_id "valid_epoch_${epoch_name}" \
        --model_arch Nano \
        --patch_size 4 \
        --max_period 10 \
        --batch_size "${RECON_BATCH_SIZE}" \
        --solver_step_size 0.05 \
        --clip_recon -1 1 \
        --pin_memory \
        --device cuda \
        --nerf_bands 0 \
        --log_console

    echo "Reconstructing shots for epoch ${epoch}"
    "${PYTHON_BIN}" "${CODE_PATH}/ReconShotDataset2.py" \
        --input_dir "${patch_output_dir}" \
        --input_aux_dir "${DATA_DIR}/valid_aux" \
        --output_dir "${shot_output_dir}"

    echo "Generating shot differences for epoch ${epoch}"
    "${PYTHON_BIN}" "${CODE_PATH}/DiffShot.py" \
        --input1_dir "${DATA_DIR}/shot" \
        --input2_dir "${shot_output_dir}" \
        --output_dir "${diff_output_dir}"
done
