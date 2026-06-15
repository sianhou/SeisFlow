#!/bin/bash

set -euo pipefail

# CONFIG NODES
MASTER=$(hostname)
MASTER_ADDR=$(hostname -I | awk '{print $1}')

# Nodes configuration - ensure master is NOT in this list
NODES_LIST="node050"
NUM_WORKERS=$(echo "$NODES_LIST" | awk -F',' '{print NF}')
NUM_NODES=$((NUM_WORKERS + 1))
NPROC_PER_NODE=3
MASTER_PORT=29500

WORKDIR="/dreamdata/24yds3d/deeplearning/SeisFlow4"
TORCHRUN_BIN="/dreamdata/24yds3d/deeplearning/python313/bin/torchrun"
LOG_DIR="$WORKDIR/logs/train_vae_i256_o64_kl_1e-6"
TRAIN_JOB="train_seismic_vae.py \
--train_data_dir $WORKDIR/train_dataset256/ \
--output_dir $WORKDIR/output_train_vae_i256_o64_kl_1e-6/ \
--input_size 256 \
--latent_size 64 \
--input_channels 1 \
--latent_channels 4 \
--hidden_channels 32 \
--batch_size 32 \
--num_workers 4 \
--pin_memory \
--learning_rate 1e-4 \
--num_epochs 500 \
--save_every_epochs 10 \
--kl_weight 1e-6 \
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
