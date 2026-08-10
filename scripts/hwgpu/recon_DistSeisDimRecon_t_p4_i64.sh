#!/usr/bin/env bash

set -euo pipefail

MASTER_ADDR="$(hostname -I | awk '{print $1}')"
NPROC_PER_NODE=4
MASTER_PORT=29500

WORKDIR="/hwdata/24ydz3d/deeplearning/SeisFlow"
TORCHRUN_BIN="/hwdata/24ydz3d/deeplearning/environments/py313/bin/torchrun"
PYTHON_BIN="/hwdata/24ydz3d/deeplearning/environments/py313/bin/python3.13"
DATA_DIR="$WORKDIR/temp/shot_dataset64"
SCRIPT_NAME="$(basename "$0" .sh)"
RUN_DIR="$WORKDIR/temp/$SCRIPT_NAME"
TRAIN_ROOT="$WORKDIR/temp/train_DistSeisDimRecon_t_p4_i64"

if [[ -n "${TRAIN_RUN_DIR:-}" ]]; then
    TRAIN_RUN_DIR="${TRAIN_RUN_DIR}"
else
    TRAIN_RUN_DIR="$(find "$TRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

[[ -x "$TORCHRUN_BIN" ]] || {
    echo "torchrun not found: $TORCHRUN_BIN" >&2
    exit 1
}
[[ -x "$PYTHON_BIN" ]] || {
    echo "Python not found: $PYTHON_BIN" >&2
    exit 1
}
[[ -d "$DATA_DIR/valid_dim" ]] || {
    echo "Validation dimension data not found: $DATA_DIR/valid_dim" >&2
    exit 1
}
[[ -d "$DATA_DIR/valid_aux" ]] || {
    echo "Validation metadata not found: $DATA_DIR/valid_aux" >&2
    exit 1
}
[[ -d "$DATA_DIR/shot" ]] || {
    echo "Original shot data not found: $DATA_DIR/shot" >&2
    exit 1
}
[[ -d "$TRAIN_RUN_DIR" ]] || {
    echo "Training run directory not found: $TRAIN_RUN_DIR" >&2
    exit 1
}

mkdir -p "$RUN_DIR"

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "TRAIN_RUN_DIR: $TRAIN_RUN_DIR"
echo "OUTPUT_DIR: $RUN_DIR"

for epoch in $(seq 100 100 1000); do
    epoch_name="$(printf '%05d' "$epoch")"
    checkpoint_dir="$TRAIN_RUN_DIR/checkpoint_epoch_${epoch_name}"
    patch_output_dir="$RUN_DIR/valid_epoch_${epoch_name}"
    shot_output_dir="$RUN_DIR/valid_recon_shot_epoch_${epoch_name}"
    diff_output_dir="$RUN_DIR/diff_recon_shot_epoch_${epoch_name}"

    [[ -d "$checkpoint_dir" ]] || {
        echo "Checkpoint not found: $checkpoint_dir" >&2
        exit 1
    }

    echo "Reconstructing epoch $epoch from $checkpoint_dir"
    "$TORCHRUN_BIN" \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank=0 \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        DistSeisDimRecon.py valid \
        --ckpt "$checkpoint_dir" \
        --train_data_dim_dir "$DATA_DIR/valid_dim" \
        --output_dir "$RUN_DIR" \
        --log_id "valid_epoch_${epoch_name}" \
        --model_arch DiT_T_4 \
        --input_size 64 \
        --batch_size 32 \
        --solver_step_size 0.05 \
        --clip_recon -1 1 \
        --pin_memory \
        --device cuda \
        --log_console

    echo "Reconstructing shots for epoch $epoch"
    "$PYTHON_BIN" "$WORKDIR/recon_shot_dataset2.py" \
        --input_dir "$patch_output_dir" \
        --input_aux_dir "$DATA_DIR/valid_aux" \
        --output_dir "$shot_output_dir"

    echo "Generating shot differences for epoch $epoch"
    "$PYTHON_BIN" "$WORKDIR/diff_shot.py" \
        --input1_dir "$DATA_DIR/shot" \
        --input2_dir "$shot_output_dir" \
        --output_dir "$diff_output_dir"
done
