#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH=""
SF_PATH="/Users/housian/Workplaces/SeisFlow"
RUN_DIR="output_train_vae_dataset256/20260603_175851_759388/"
SEGY_FILE="ma2+GathAP.sgy"
BATCH_SIZE=16
DEVICE="cpu"
MISSING_RATIO=0.5

for CHECKPOINT in "${RUN_DIR}"/checkpoint_epoch_*.pth; do
  echo "Validating ${CHECKPOINT}"
  #${PYTHON_PATH}/bin/python3.13 valid_seismic_vae.py \
  python valid_seismic_vae.py \
    --ckpt "${CHECKPOINT}" \
    --segy "${SEGY_FILE}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --missing_ratio "${MISSING_RATIO}" \
    --clip_vmin -2 --clip_vmax 2
done
