#!/bin/bash

set -euo pipefail

# CONFIG NODES
MASTER=$(hostname)
MASTER_ADDR=$(hostname -I | awk '{print $1}')

# Nodes configuration - ensure master is NOT in this list
NODES_LIST="clsgpu05,clsgpu06"
NUM_WORKERS=$(echo "$NODES_LIST" | awk -F',' '{print NF}')
NUM_NODES=$((NUM_WORKERS + 1))
NPROC_PER_NODE=4
MASTER_PORT=29500
DEVICE=cuda

for ((i = 100; i <=2000; i += 100)); do
	j=$(printf '%05d' "$i")
	echo "processing epoch = $j"

	WORKDIR="/hwdata/24ydz3d/deeplearning/SeisFlow-main"
	TORCHRUN_BIN="/hwdata/24ydz3d/deeplearning/environments/py313/bin/torchrun"
	LOG_DIR="$WORKDIR/logs/train_recon_nerf_dit_t2_i64"
	TRAIN_JOB="DistSeisDimReconNerf.py valid \
	--ckpt /hwdata/24ydz3d/deeplearning/SeisFlow-main/output_recon_nerf_dit_t2_i64/20260801_195609_124424/checkpoint_epoch_$j/ \
	--train_data_dim_dir /hwdata/24ydz3d/deeplearning/SeisFlow-main/shot_dataset64/valid_dim/ \
	--output_dir /hwdata/24ydz3d/deeplearning/SeisFlow-main/shot_dataset64/valid_nerf_dit_t2_epoch__$j/ \
	--model_arch DiT_T_2 \
	--input_size 64 \
	--batch_size 32 \
	--solver_step_size 0.05 \
	--clip_recon -1 1 \
	--pin_memory \
	--device $DEVICE \
	--log_id 0 \
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
done


