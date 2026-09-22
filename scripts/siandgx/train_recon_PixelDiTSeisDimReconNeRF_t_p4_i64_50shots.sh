#!/usr/bin/env bash
# Sequential single-process training and reconstruction: bands 0 -> 6 -> 3.
set -euo pipefail

SIANDGX_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SIANDGX_SCRIPT_DIR/env.sh"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/shot_dataset64_50shots}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
RUN_ROOT="${RUN_ROOT:-$PROJ_DIR/train_recon_PixelDiT_p4_i64_50shots_$(date +%Y%m%d_%H%M%S)_$$}"

# Avoid inherited launcher settings enabling distributed mode.
unset RANK WORLD_SIZE LOCAL_RANK SLURM_PROCID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
for directory in train train_dim valid_dim valid_aux shot; do
    [[ -d "$DATA_DIR/$directory" ]] || { echo "Missing data directory: $DATA_DIR/$directory" >&2; exit 1; }
done
# Each invocation owns its output root; never select another run's checkpoints.
mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT"
cd "$CODE_PATH"

for bands in 0 6 3; do
    band_root="$RUN_ROOT/NeRFBands${bands}"
    train_run="$band_root/train"
    recon_root="$band_root/recon"
    echo "Training NeRFBands${bands}: 2000 epochs, output=$train_run"
    "$PYTHON_BIN" "$CODE_PATH/PixelDiTSeisDimReconNeRF.py" train \
        --input_dir "$DATA_DIR/train" \
        --input_dim_dir "$DATA_DIR/train_dim" \
        --output_dir "$band_root" --log_id train \
        --model_arch T --patch_size 4 --nerf_bands "$bands" \
        --batch_size "$BATCH_SIZE" --num_epochs 2000 --save_every_epochs 100 \
        --num_workers 0 --pin_memory --device "$DEVICE" \
        --upcast_attention --log_console

    for epoch in $(seq 100 100 2000); do
        epoch_name="$(printf '%05d' "$epoch")"
        checkpoint_dir="$train_run/checkpoint_epoch_${epoch_name}"
        log_id="valid_ema_epoch_${epoch_name}"
        patch_output_dir="$recon_root/$log_id"
        shot_output_dir="$recon_root/valid_recon_shot_ema_epoch_${epoch_name}"
        diff_output_dir="$recon_root/diff_recon_shot_ema_epoch_${epoch_name}"
        [[ -d "$checkpoint_dir" ]] || { echo "Checkpoint not found: $checkpoint_dir" >&2; exit 1; }

        echo "Reconstructing NeRFBands${bands}, epoch $epoch"
        "$PYTHON_BIN" "$CODE_PATH/PixelDiTSeisDimReconNeRF.py" sample \
            --ckpt "$checkpoint_dir" \
            --input_dim_dir "$DATA_DIR/valid_dim" \
            --output_dir "$recon_root" --log_id "$log_id" \
            --model_arch T --patch_size 4 --nerf_bands "$bands" \
            --batch_size "$BATCH_SIZE" --solver_step_size 0.05 --clip_recon -1 1 \
            --num_workers 0 --pin_memory --device "$DEVICE" --use_ema --log_console

        "$PYTHON_BIN" "$CODE_PATH/ReconShotDataset2.py" \
            --input_dir "$patch_output_dir" --input_aux_dir "$DATA_DIR/valid_aux" \
            --output_dir "$shot_output_dir" --workers 1
        "$PYTHON_BIN" "$CODE_PATH/DiffShot.py" \
            --input1_dir "$DATA_DIR/shot" --input2_dir "$shot_output_dir" \
            --output_dir "$diff_output_dir" --workers 1
    done
done
echo "All training and reconstruction completed: $RUN_ROOT"
