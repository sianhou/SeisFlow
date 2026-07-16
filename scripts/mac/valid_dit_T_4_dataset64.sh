#!/bin/bash

set -euo pipefail

WORKDIR="/Users/housian/Workplaces/SeisFlow"
PYTHON_BIN="/Users/housian/Workplaces/ucas/.venv/SeisFlow/bin/python3.13"

CHECKPOINT="/Users/housian/Workplaces/SeisFlow/output_train_dit_DiT_T_4/20260616_075636_360343/checkpoint_epoch_00400/"
SHOT_INTERVAL="${2:-10}"

"$PYTHON_BIN" valid5.py \
  --segy "$WORKDIR/ma2+GathAP.sgy" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "/Users/housian/Workplaces/SeisFlow/output_train_dit_DiT_T_4/20260616_075636_360343/valid_epoch_00400/" \
  --model_arch DiT_T_4 \
  --patch_size 64 \
  --overlap_size 32 \
  --batch_size 32 \
  --mask_ratio 0.5 \
  --shot_interval "$SHOT_INTERVAL" \
  --device cpu \
  --clip_vmin -2 \
  --clip_vmax 2 \
  --log_console
