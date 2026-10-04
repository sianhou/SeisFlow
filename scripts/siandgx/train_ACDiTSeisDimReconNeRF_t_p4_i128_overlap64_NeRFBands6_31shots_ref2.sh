#!/usr/bin/env bash
set -euo pipefail

SIANDGX_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SIANDGX_SCRIPT_DIR/env.sh"

# Ignore inherited distributed-launch settings; run one local Python process.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK SLURM_PROCID

DATA_DIR="${DATA_DIR:-$PROJ_DIR/shot_dataset128_overlap64_31shots_ref2}"
SCRIPT_NAME="$(basename "$0" .sh)"
RUN_DIR="${RUN_DIR:-$PROJ_DIR/$SCRIPT_NAME}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-100}"

mkdir -p "$RUN_DIR"

cd "$CODE_PATH"
nohup "$PYTHON_BIN" "$CODE_PATH/ACDiTSeisDimReconNeRF.py" train \
    --input_dir "$DATA_DIR/train" \
    --input_dim_dir "$DATA_DIR/train_dim" \
    --ref_dir1 "$DATA_DIR/train_ref" \
    --ref_dim_dir1 "$DATA_DIR/train_ref_dim" \
    --ref_dir2 "$DATA_DIR/train_ref2" \
    --ref_dim_dir2 "$DATA_DIR/train_ref2_dim" \
    --output_dir "$RUN_DIR" \
    --model_arch T \
    --patch_size 4 \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --save_every_epochs "$SAVE_EVERY_EPOCHS" \
    --pin_memory \
    --device cuda \
    --nerf_bands 6 \
    --upcast_attention \
    --log_console > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
echo "$!" > "$RUN_DIR/launcher.pid"
