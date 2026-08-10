#!/bin/bash

set -euo pipefail

# CONFIG NODES
MASTER=$(hostname)
MASTER_ADDR=$(hostname -I | awk '{print $1}')

# Nodes configuration - ensure master is NOT in this list
NODES_LIST="clsgpu11,clsgpu12"
NUM_WORKERS=$(echo "$NODES_LIST" | awk -F',' '{print NF}')
NUM_NODES=$((NUM_WORKERS + 1))
NPROC_PER_NODE=4
MASTER_PORT=29500

WORKDIR="/hwdata/24ydz3d/deeplearning/SeisFlow"
TORCHRUN_BIN="/hwdata/24ydz3d/deeplearning/environments/py313/bin/torchrun"
DATA_DIR="$WORKDIR/temp/shot_dataset128"
SCRIPT_NAME="$(basename "$0" .sh)"
RUN_DIR="$WORKDIR/temp/$SCRIPT_NAME"
LOG_DIR="$RUN_DIR"
TRAIN_JOB="PixelDiTSeisDimNeRFRecon.py train \
--input_dir $DATA_DIR/train/ \
--input_dim_dir $DATA_DIR/train_dim/ \
--output_dir $RUN_DIR/ \
--model_arch T \
--batch_size 32 \
--num_epochs 1001 \
--save_every_epochs 100 \
--pin_memory \
--device cuda \
--nerf_bands 6 \
--log_console"

echo "MASTER: $MASTER"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODES_LIST: $NODES_LIST"
echo "NUM_NODES: $NUM_NODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "TRAIN_JOB: $TRAIN_JOB"

mkdir -p "$LOG_DIR"

rank=1
for node in $(echo "$NODES_LIST" | tr ',' ' '); do
	echo "Starting training on $node(rank=$rank)..."
	ssh "$node" "cd $WORKDIR && \
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

echo "Starting training on master(rank=0)..."
cd "$WORKDIR"
"$TORCHRUN_BIN" \
	--nnodes="$NUM_NODES" \
	--nproc_per_node="$NPROC_PER_NODE" \
	--node_rank=0 \
	--master_addr="$MASTER_ADDR" \
	--master_port="$MASTER_PORT" \
	${TRAIN_JOB}
