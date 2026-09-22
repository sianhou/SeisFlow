#!/usr/bin/env bash
set -euo pipefail

SIANDGX_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Separate defaults allow the two configurations to run concurrently.
export MASTER_PORT="${MASTER_PORT:-29511}"
source "$SIANDGX_SCRIPT_DIR/env.sh"

DATA_DIR="${DATA_DIR:-$PROJ_DIR/shot_dataset128_50shots}"
NODES_LIST="${NODES_LIST:-0}"
[[ "$NODES_LIST" == "0" ]] || { echo "This siandgx script requires NODES_LIST=0." >&2; exit 1; }
SCRIPT_NAME="$(basename "$0" .sh)"
RUN_DIR="${RUN_DIR:-$PROJ_DIR/$SCRIPT_NAME}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-100}"
NERF_BANDS="${NERF_BANDS:-6}"

[[ -x "$TORCHRUN_BIN" ]] || { echo "torchrun not found: $TORCHRUN_BIN" >&2; exit 1; }
[[ -d "$DATA_DIR/train" ]] || { echo "Training data not found: $DATA_DIR/train" >&2; exit 1; }
[[ -d "$DATA_DIR/train_dim" ]] || { echo "Training coordinates not found: $DATA_DIR/train_dim" >&2; exit 1; }
mkdir -p "$RUN_DIR"
cd "$CODE_PATH"

# Coordinates -> amplitudes, fixed t=0; no FM noise or ODE solver.
"$TORCHRUN_BIN" \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    AugmentedDiTSeisDimReconNeRFDirect.py train \
    --input_dir "$DATA_DIR/train" \
    --input_dim_dir "$DATA_DIR/train_dim" \
    --output_dir "$RUN_DIR" \
    --model_arch T \
    --patch_size 4 \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --save_every_epochs "$SAVE_EVERY_EPOCHS" \
    --nerf_bands "$NERF_BANDS" \
    --upcast_attention \
    --pin_memory \
    --device cuda \
    --use_ema \
    --log_console
