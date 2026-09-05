#!/usr/bin/env bash
set -x
set -euo pipefail

DREAMCLOUD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DREAMCLOUD_SCRIPT_DIR/env.sh"

DATA_DIR="$PROJ_DIR/shot_dataset256"
NODES_LIST="${NODES_LIST:-node045,node046}"
if [[ -n "$NODES_LIST" ]]; then
    NUM_WORKERS="$(awk -F',' '{print NF}' <<< "$NODES_LIST")"
else
    NUM_WORKERS=0
fi
NUM_NODES=$((NUM_WORKERS + 1))
SCRIPT_NAME="$(basename "$0" .sh)"
RUN_DIR="$PROJ_DIR/$SCRIPT_NAME"
LOG_DIR="$RUN_DIR"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_JOB="DiTSeisDimReconNeRF.py train \
--input_dir $DATA_DIR/train/ \
--input_dim_dir $DATA_DIR/train_dim/ \
--output_dir $RUN_DIR/ \
--model_arch DiT_T_8 \
--input_size 256 \
--batch_size $BATCH_SIZE \
--num_epochs 2000 \
--save_every_epochs 100 \
--pin_memory \
--device cuda \
--nerf_bands 0 \
--upcast_attention \
--log_console"

[[ -x "$TORCHRUN_BIN" ]] || { echo "torchrun not found: $TORCHRUN_BIN" >&2; exit 1; }
[[ -d "$DATA_DIR/train" ]] || { echo "Training data not found: $DATA_DIR/train" >&2; exit 1; }
[[ -d "$DATA_DIR/train_dim" ]] || { echo "Training dimension data not found: $DATA_DIR/train_dim" >&2; exit 1; }

echo "MASTER: $MASTER"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODES_LIST: $NODES_LIST"
echo "NUM_NODES: $NUM_NODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "TRAIN_JOB: $TRAIN_JOB"

mkdir -p "$LOG_DIR"

rank=1
for node in $(tr ',' ' ' <<< "$NODES_LIST"); do
    echo "Starting training on $node (rank=$rank)..."
    ssh "$node" "cd $CODE_PATH && \
        $TORCHRUN_BIN \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=$rank \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        ${TRAIN_JOB}" > "$LOG_DIR/train_${node}.log" 2>&1 &
    rank=$((rank + 1))
    sleep 2s
done

echo "Starting training on master (rank=0)..."
cd "$CODE_PATH"
"$TORCHRUN_BIN" \
    --nnodes="$NUM_NODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    ${TRAIN_JOB}
