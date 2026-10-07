#!/usr/bin/env bash
set -x
set -euo pipefail

DREAMCLOUD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DREAMCLOUD_SCRIPT_DIR/env.sh"

# NODES_LIST contains workers only; the launching host is node rank zero.
# An explicitly empty NODES_LIST enables single-node, multi-GPU training.
NODES_LIST="${NODES_LIST-node07,node048,node049,node050}"
if [[ -n "$NODES_LIST" ]]; then
    NUM_WORKERS="$(awk -F',' '{print NF}' <<< "$NODES_LIST")"
else
    NUM_WORKERS=0
fi
NUM_NODES=$((NUM_WORKERS + 1))

DATA_DIR="${DATA_DIR:-$PROJ_DIR/shot_dataset64_overlap32}"
SCRIPT_NAME="$(basename "$0" .sh)"
RUN_DIR="${RUN_DIR:-$PROJ_DIR/$SCRIPT_NAME}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-100}"

mkdir -p "$RUN_DIR"

TRAIN_JOB=(
    "$CODE_PATH/ACDiTSeisDimReconNeRF2.py" train
    --input_dir "$DATA_DIR/train"
    --input_dim_dir "$DATA_DIR/train_dim"
    --ref_dir1 "$DATA_DIR/train_ref" --ref_dim_dir1 "$DATA_DIR/train_ref_dim"
    --ref_dir2 "$DATA_DIR/train_ref2" --ref_dim_dir2 "$DATA_DIR/train_ref2_dim"
    --output_dir "$RUN_DIR"
    --model_arch T --patch_size 4 --batch_size "$BATCH_SIZE"
    --num_epochs "$NUM_EPOCHS" --save_every_epochs "$SAVE_EVERY_EPOCHS"
    --pin_memory --device cuda --nerf_bands 0 --upcast_attention --log_console
)
LAUNCH=(
    "$TORCHRUN_BIN" --nnodes="$NUM_NODES" --nproc_per_node="$NPROC_PER_NODE"
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
)

cd "$CODE_PATH"
rank=1
worker_pids=""
for node in $(tr ',' ' ' <<< "$NODES_LIST"); do
    printf -v remote_command '%q ' "${LAUNCH[@]}" --node_rank="$rank" "${TRAIN_JOB[@]}"
    printf -v remote_directory '%q' "$CODE_PATH"
    ssh "$node" "cd $remote_directory && $remote_command" > "$RUN_DIR/train_${node}.log" 2>&1 &
    worker_pids="$worker_pids $!"
    rank=$((rank + 1))
done

"${LAUNCH[@]}" --node_rank=0 "${TRAIN_JOB[@]}"
for pid in $worker_pids; do
    wait "$pid"
done
