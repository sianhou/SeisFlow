#!/bin/bash

set -euo pipefail

WORKDIR="/Users/housian/Workplaces/SeisFlow"
PYTHON_BIN="/Users/housian/Workplaces/ucas/.venv/SeisFlow/bin/python3.13"

CHECKPOINT="${1:?Usage: $0 <checkpoint_dir> [shot_interval]}"
SHOT_INTERVAL="${2:-10}"

"$PYTHON_BIN" valid5.py \
  --segy "$WORKDIR/ma2+GathAP.sgy" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$WORKDIR/output_valid_dit_T_2_dataset64/" \
  --model_arch DiT_T_2 \
  --patch_size 64 \
  --overlap_size 32 \
  --batch_size 32 \
  --mask_ratio 0.5 \
  --shot_interval "$SHOT_INTERVAL" \
  --device cpu \
  --clip_vmin -2 \
  --clip_vmax 2 \
  --log_console
