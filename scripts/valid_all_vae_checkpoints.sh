#!/usr/bin/env bash
set -euo pipefail

SF_PATH="/Users/housian/Workplaces/SeisFlow"
RUN_DIR="/Users/housian/Workplaces/SeisFlow/output_vae_i256_o32/"
SEGY_FILE="/Users/housian/Workplaces/SeisFlow/ma2+GathAP.sgy"
BATCH_SIZE=16
DEVICE="cpu"
MISSING_RATIO=0.5

cd "${SF_PATH}"
shopt -s nullglob

CHECKPOINTS=("${RUN_DIR}"/checkpoint_epoch_*)
if (( ${#CHECKPOINTS[@]} == 0 )); then
  echo "No checkpoint directories found in ${RUN_DIR}" >&2
  exit 1
fi

VALIDATED=0
SKIPPED=0
FAILED=0

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
  if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "Skipping non-directory checkpoint: ${CHECKPOINT}" >&2
    ((SKIPPED += 1))
    continue
  fi

  if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
    echo "Skipping checkpoint without config.json: ${CHECKPOINT}" >&2
    ((SKIPPED += 1))
    continue
  fi

  if [[ ! -f "${CHECKPOINT}/diffusion_pytorch_model.safetensors" \
        && ! -f "${CHECKPOINT}/diffusion_pytorch_model.bin" ]]; then
    echo "Skipping checkpoint without Diffusers model weights: ${CHECKPOINT}" >&2
    ((SKIPPED += 1))
    continue
  fi

  echo "Validating ${CHECKPOINT}"
  if python valid_seismic_vae.py \
      --ckpt "${CHECKPOINT}" \
      --segy "${SEGY_FILE}" \
      --batch_size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --missing_ratio "${MISSING_RATIO}" \
      --data_range 4 \
      --clip_vmin -2 --clip_vmax 2; then
    ((VALIDATED += 1))
  else
    echo "Validation failed: ${CHECKPOINT}" >&2
    ((FAILED += 1))
  fi
done

echo "Validation complete: ${VALIDATED} succeeded, ${SKIPPED} skipped, ${FAILED} failed."
(( FAILED == 0 ))
