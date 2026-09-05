#!/usr/bin/env bash

set -euo pipefail

DREAMCLOUD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DREAMCLOUD_SCRIPT_DIR/env.sh"

SCRIPT_NAME="$(basename "$0" .sh)"
RUN_DIR="$PROJ_DIR/$SCRIPT_NAME"
DATA_DIR="$PROJ_DIR/shot_dataset256"
TRAIN_SCRIPT_NAME="${SCRIPT_NAME/#recon_/train_}"
TRAIN_ROOT="$PROJ_DIR/$TRAIN_SCRIPT_NAME"

if [[ -z "${TRAIN_RUN_DIR:-}" ]]; then
    TRAIN_RUN_DIR="$(find "$TRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

[[ -x "$TORCHRUN_BIN" ]] || { echo "torchrun not found: $TORCHRUN_BIN" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$DATA_DIR/valid_dim" ]] || { echo "Validation dimension data not found: $DATA_DIR/valid_dim" >&2; exit 1; }
[[ -d "$DATA_DIR/valid_aux" ]] || { echo "Validation metadata not found: $DATA_DIR/valid_aux" >&2; exit 1; }
[[ -d "$DATA_DIR/shot" ]] || { echo "Original shot data not found: $DATA_DIR/shot" >&2; exit 1; }
[[ -d "$TRAIN_RUN_DIR" ]] || { echo "Training run directory not found: $TRAIN_RUN_DIR" >&2; exit 1; }

mkdir -p "$RUN_DIR"
cd "$CODE_PATH"

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "TRAIN_RUN_DIR: $TRAIN_RUN_DIR"
echo "OUTPUT_DIR: $RUN_DIR"

for epoch in $(seq 100 100 2000); do
    epoch_name="$(printf '%05d' "$epoch")"
    checkpoint_dir="$TRAIN_RUN_DIR/checkpoint_epoch_${epoch_name}"
    patch_output_dir="$RUN_DIR/valid_ema_epoch_${epoch_name}"
    shot_output_dir="$RUN_DIR/valid_recon_shot_ema_epoch_${epoch_name}"
    diff_output_dir="$RUN_DIR/diff_recon_shot_ema_epoch_${epoch_name}"

    [[ -d "$checkpoint_dir" ]] || { echo "Checkpoint not found: $checkpoint_dir" >&2; exit 1; }

    echo "Reconstructing epoch $epoch with EMA weights from $checkpoint_dir"
    "$TORCHRUN_BIN" \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank=0 \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        PixelDiTSeisDimReconNeRF.py sample \
        --ckpt "$checkpoint_dir" \
        --input_dim_dir "$DATA_DIR/valid_dim" \
        --output_dir "$RUN_DIR" \
        --log_id "valid_ema_epoch_${epoch_name}" \
        --model_arch T \
        --patch_size 8 \
        --batch_size 32 \
        --solver_step_size 0.05 \
        --clip_recon -1 1 \
        --pin_memory \
        --device cuda \
        --nerf_bands 0 \
        --use_ema \
        --log_console

    echo "Reconstructing shots for epoch $epoch with EMA weights"
    "$PYTHON_BIN" "$CODE_PATH/ReconShotDataset2.py" \
        --input_dir "$patch_output_dir" \
        --input_aux_dir "$DATA_DIR/valid_aux" \
        --output_dir "$shot_output_dir"

    echo "Generating shot differences for epoch $epoch with EMA weights"
    "$PYTHON_BIN" "$CODE_PATH/DiffShot.py" \
        --input1_dir "$DATA_DIR/shot" \
        --input2_dir "$shot_output_dir" \
        --output_dir "$diff_output_dir"
done
