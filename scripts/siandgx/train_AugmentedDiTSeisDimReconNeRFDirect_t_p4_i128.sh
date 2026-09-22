#!/usr/bin/env bash
set -euo pipefail

SIANDGX_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SIANDGX_SCRIPT_DIR/env.sh"

# Run one local Python process, even if distributed settings were inherited.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK SLURM_PROCID

DATA_DIR="${DATA_DIR:-$PROJ_DIR/shot_dataset128_50shots}"
SCRIPT_NAME="$(basename "$0" .sh)"
RUN_DIR="${RUN_DIR:-$PROJ_DIR/$SCRIPT_NAME}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-100}"
NERF_BANDS="${NERF_BANDS:-6}"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$DATA_DIR/train" ]] || { echo "Training data not found: $DATA_DIR/train" >&2; exit 1; }
[[ -d "$DATA_DIR/train_dim" ]] || { echo "Training coordinates not found: $DATA_DIR/train_dim" >&2; exit 1; }
mkdir -p "$RUN_DIR"
cd "$CODE_PATH"

# Coordinates -> amplitudes, fixed t=0; no FM noise or ODE solver.
nohup "$PYTHON_BIN" "$CODE_PATH/AugmentedDiTSeisDimReconNeRFDirect.py" train \
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
    --log_console > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
echo "$!" > "$RUN_DIR/launcher.pid"
