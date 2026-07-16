#!/bin/bash

set -euo pipefail

# CONFIG NODES
MASTER=$(hostname)
MASTER_ADDR=$(hostname -I | awk '{print $1}')

# Nodes configuration - ensure master is NOT in this list
NODES_LIST="clsgpu04"
NUM_WORKERS=$(echo "$NODES_LIST" | awk -F',' '{print NF}')
NUM_NODES=$((NUM_WORKERS + 1))
NPROC_PER_NODE=4
MASTER_PORT=29500

WORKDIR="/hwdata/24ydz3d/deeplearning/SeisFlow-main"
TORCHRUN_BIN="/hwdata/24ydz3d/deeplearning/environments/py313/bin/torchrun"
LOG_DIR="$WORKDIR/logs/vae_recon_dit_t4_i256o64"
TRAIN_JOB="SeisVaeDimRecon.py \
--train_data_dir /hwdata/24ydz3d/deeplearning/SeisFlow-main/shot_dataset256_random/train/ \
--train_data_dim_dir /hwdata/24ydz3d/deeplearning/SeisFlow-main/shot_dataset256_random/train_dim/ \
--output_dir /hwdata/24ydz3d/deeplearning/SeisFlow-main/output_vae_recon_dit_t4_i256o64/ \
--input_size 256 \
--ckpt_vae /hwdata/24ydz3d/deeplearning/SeisFlow-main/output_vae_i256o64/20260716_114834_580185/checkpoint_epoch_00300/
--batch_size 32 \
--num_epochs 1000 \
--save_every_epochs 50 \
--pin_memory
--device cuda \
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
