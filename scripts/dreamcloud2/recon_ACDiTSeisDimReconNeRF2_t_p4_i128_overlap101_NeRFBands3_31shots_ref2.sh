#!/usr/bin/env bash

set -x
set -euo pipefail

DREAMCLOUD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DREAMCLOUD_SCRIPT_DIR/env.sh"

# Workers share CODE_PATH, DATA_DIR, checkpoints and RUN_DIR with the master.
NODES_LIST="${NODES_LIST-node07,node048,node049,node050}"
if [[ -n "$NODES_LIST" ]]; then
    NUM_WORKERS="$(awk -F',' '{print NF}' <<< "$NODES_LIST")"
else
    NUM_WORKERS=0
fi
NUM_NODES=$((NUM_WORKERS + 1))

SCRIPT_NAME="$(basename "$0" .sh)"
RUN_DIR="${RUN_DIR:-$PROJ_DIR/$SCRIPT_NAME}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/shot_dataset128_overlap101}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_SCRIPT_NAME="${SCRIPT_NAME/#recon_/train_}"
TRAIN_ROOT="${TRAIN_ROOT:-$PROJ_DIR/$TRAIN_SCRIPT_NAME}"

if [[ -z "${TRAIN_RUN_DIR:-}" ]]; then
    TRAIN_RUN_DIR="$(find "$TRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

mkdir -p "$RUN_DIR"
cd "$CODE_PATH"

LAUNCH=(
    "$TORCHRUN_BIN" --nnodes="$NUM_NODES" --nproc_per_node="$NPROC_PER_NODE"
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
)
FIRST_EPOCH="${FIRST_EPOCH:-100}"
LAST_EPOCH="${LAST_EPOCH:-1000}"
EPOCH_STEP="${EPOCH_STEP:-100}"
for epoch in $(seq "$FIRST_EPOCH" "$EPOCH_STEP" "$LAST_EPOCH"); do
    epoch_name="$(printf '%05d' "$epoch")"
    checkpoint_dir="$TRAIN_RUN_DIR/checkpoint_epoch_${epoch_name}"
    patch_output_dir="$RUN_DIR/valid_ema_epoch_${epoch_name}"
    shot_output_dir="$RUN_DIR/valid_recon_shot_ema_epoch_${epoch_name}"
    diff_output_dir="$RUN_DIR/diff_recon_shot_ema_epoch_${epoch_name}"

    SAMPLE_JOB=(
        "$CODE_PATH/ACDiTSeisDimReconNeRF2.py" sample
        --ckpt "$checkpoint_dir" --input_dim_dir "$DATA_DIR/valid_dim"
        --ref_dir1 "$DATA_DIR/valid_ref" --ref_dim_dir1 "$DATA_DIR/valid_ref_dim"
        --ref_dir2 "$DATA_DIR/valid_ref2" --ref_dim_dir2 "$DATA_DIR/valid_ref2_dim"
        --output_dir "$RUN_DIR" --log_id "valid_ema_epoch_${epoch_name}"
        --model_arch T --patch_size 4 --batch_size "$BATCH_SIZE"
        --solver_step_size 0.05 --clip_recon -1 1 --pin_memory --device cuda
        --nerf_bands 3 --use_ema --log_console
    )
    rank=1
    worker_pids=""
    for node in $(tr ',' ' ' <<< "$NODES_LIST"); do
        printf -v remote_command '%q ' "${LAUNCH[@]}" --node_rank="$rank" "${SAMPLE_JOB[@]}"
        printf -v remote_directory '%q' "$CODE_PATH"
        ssh "$node" "cd $remote_directory && $remote_command" \
            > "$RUN_DIR/sample_${node}_epoch_${epoch_name}.log" 2>&1 &
        worker_pids="$worker_pids $!"
        rank=$((rank + 1))
    done
    "${LAUNCH[@]}" --node_rank=0 "${SAMPLE_JOB[@]}"
    for pid in $worker_pids; do
        wait "$pid"
    done

    # Sampling partitions files by global rank and ends with a distributed barrier.
    # Only this master shell merges/evaluates after every node has exited successfully.
    "$PYTHON_BIN" "$CODE_PATH/ReconShotDataset2.py" \
        --input_dir "$patch_output_dir" \
        --input_aux_dir "$DATA_DIR/valid_aux" \
        --output_dir "$shot_output_dir"

    "$PYTHON_BIN" "$CODE_PATH/DiffShot.py" \
        --input1_dir "$DATA_DIR/shot" \
        --input2_dir "$shot_output_dir" \
        --output_dir "$diff_output_dir"
done
