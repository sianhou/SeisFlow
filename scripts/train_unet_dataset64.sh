#!/bin/bash

set -u

# CONFIG NODES
MASTER=$(hostname)
MASTER_ADDR=$(hostname -I | awk '{print $1}') # default the current node

# Nodes configuration - ensure master is NOT in this list
NODES_LIST="clsgpu11,clsgpu12,clsgpu13"  # worker nodes only
NUM_WORKERS=$(echo $NODES_LIST | awk -F',' '{print NF}')
NUM_NODES=$((NUM_WORKERS + 1))
NPROC_PER_NODE=4
TRAIN_JOB="train4.py --train_data_dir /hwdata/24ydz3d/SeisFlow/train_dataset64/ --output_dir /hwdata/24ydz3d/SeisFlow/output_train_unet_size64/ --model_arch unet --log_console --num_workers 0 --pin_memory --batch_size 32"
WORKDIR="/hwdata/24ydz3d/SeisFlow/"
TORCHRUN_BIN="/hwdata/24ydz3d/deeplearning/environments/py313/bin/torchrun"
LOG_DIR="$WORKDIR/logs"

echo "MASTER: $MASTER"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODES_LIST: $NODES_LIST"
echo "NUM_NODES: $NUM_NODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "TRAIN_JOB: $TRAIN_JOB"

mkdir -p "$LOG_DIR"

rank=1
for node in $(echo $NODES_LIST | tr ',' ' '); do
	echo "Starting training on $node(rank=$rank)..."
	ssh "$node" "cd $WORKDIR && \
		$TORCHRUN_BIN \
		--nnodes=$NUM_NODES \
		--nproc_per_node=$NPROC_PER_NODE \
		--node_rank=$rank \
		--master_addr=$MASTER_ADDR \
		--master_port=29500 \
		${TRAIN_JOB}" > "$LOG_DIR/train_${node}.log" 2>&1 &
	rank=$(( $rank + 1))
	sleep 2s
done

echo "Starting training on master(rank=0)..."
cd "$WORKDIR"
"$TORCHRUN_BIN" \
	--nnodes=$NUM_NODES \
	--nproc_per_node=$NPROC_PER_NODE \
	--node_rank=0 \
	--master_addr=$MASTER_ADDR \
	--master_port=29500 \
  	${TRAIN_JOB}
