#!/usr/bin/env bash

set -euo pipefail

DREAMCLOUD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DREAMCLOUD_SCRIPT_DIR/env.sh"

SEGY="${SEGY:-$PROJ_DIR/shots_196_226_step1_1s_2p5s_4ms.sgy}"
RANDOM_SEGY="${RANDOM_SEGY:-$PROJ_DIR/random_gaussian_31shots.sgy}"
RANDOM_SEED="${RANDOM_SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"

# BuildShotDataset2 is not a distributed program. Assign the eight independent
# (patch/overlap, real/random) datasets across nodes; each output has one owner.
if [[ "${1:-}" != "--worker" ]]; then
    NODES_LIST="${NODES_LIST-node046,node047,node048,node049}"
    if [[ -n "$NODES_LIST" ]]; then
        NUM_WORKERS="$(awk -F',' '{print NF}' <<< "$NODES_LIST")"
    else
        NUM_WORKERS=0
    fi
    NUM_NODES=$((NUM_WORKERS + 1))
    SCRIPT_NAME="$(basename "$0" .sh)"
    RUN_DIR="${RUN_DIR:-$PROJ_DIR/$SCRIPT_NAME}"
    mkdir -p "$RUN_DIR"

    # Generate once on shared storage before any node reads the random SEG-Y.
    "$PYTHON_BIN" "$CODE_PATH/BuildRandomSegy.py" \
        --segy "$SEGY" --output "$RANDOM_SEGY" --seed "$RANDOM_SEED"

    echo "NODES_LIST: $NODES_LIST; NUM_NODES: $NUM_NODES; dataset jobs: 8"
    rank=1
    worker_pids=""
    for node in $(tr ',' ' ' <<< "$NODES_LIST"); do
        printf -v remote_command '%q ' env \
            "CODE_PATH=$CODE_PATH" "PROJ_DIR=$PROJ_DIR" "PYTHON_BIN=$PYTHON_BIN" \
            "SEGY=$SEGY" "RANDOM_SEGY=$RANDOM_SEGY" "SPLIT_SEED=$SPLIT_SEED" \
            bash "$DREAMCLOUD_SCRIPT_DIR/$SCRIPT_NAME.sh" --worker "$rank" "$NUM_NODES"
        ssh "$node" "$remote_command" > "$RUN_DIR/build_${node}.log" 2>&1 &
        worker_pids="$worker_pids $!"
        rank=$((rank + 1))
    done
    SEGY="$SEGY" RANDOM_SEGY="$RANDOM_SEGY" SPLIT_SEED="$SPLIT_SEED" \
        bash "$DREAMCLOUD_SCRIPT_DIR/$SCRIPT_NAME.sh" --worker 0 "$NUM_NODES"
    for pid in $worker_pids; do
        wait "$pid"
    done
    exit 0
fi

node_rank="$2"
num_nodes="$3"
task_index=0

for config in "64 32" "64 48" "128 64" "128 96"; do
    read -r patch_size overlap_size <<< "${config}"

    for dataset_kind in real random; do
        owner=$((task_index % num_nodes))
        task_index=$((task_index + 1))
        if [[ "$owner" -ne "$node_rank" ]]; then
            continue
        fi
        output_dir="$PROJ_DIR/shot_dataset${patch_size}_overlap${overlap_size}_31shots_ref2"
        input_segy="$SEGY"
        build_options=(--slice 0 1501 --clip -2 2 --normalize)
        extract_options=(--slice 0 1501 --clip -2 2)
        if [[ "$dataset_kind" == random ]]; then
            output_dir="$PROJ_DIR/random_shot_dataset${patch_size}_overlap${overlap_size}_31shots_ref2"
            input_segy="$RANDOM_SEGY"
            build_options=(--slice 0 1501)
            extract_options=(--slice 0 1501)
        fi

        echo "Node rank $node_rank: $dataset_kind patch=$patch_size overlap=$overlap_size refs=2"
        "$PYTHON_BIN" "$CODE_PATH/BuildShotDataset2.py" \
            --segy "$input_segy" --patch_size "$patch_size" --overlap_size "$overlap_size" \
            --output_dir "$output_dir" --valid 0.3 --valid_mode group_random \
            --seed "$SPLIT_SEED" --gen-ref 2 "${build_options[@]}"

        "$PYTHON_BIN" "$CODE_PATH/ExtractShot2.py" \
            --segy "$input_segy" --output_dir "$output_dir/shot" "${extract_options[@]}"
    done
done
