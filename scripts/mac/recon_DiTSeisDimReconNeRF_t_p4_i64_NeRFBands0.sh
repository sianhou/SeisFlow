#!/usr/bin/env bash

set -euo pipefail

MAC_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MAC_SCRIPT_DIR/env.sh"

EPOCH="${1:?Usage: $0 <epoch>}"
SCRIPT_NAME="$(basename "$0" .sh)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJ_DIR/new2}"
DATA_DIR="${DATA_DIR:-$PROJ_DIR/shot_dataset64}"
RUN_DIR="${RUN_DIR:-$EXPERIMENT_DIR/$SCRIPT_NAME}"
TRAIN_SCRIPT_NAME="${SCRIPT_NAME/#recon_/train_}"
TRAIN_ROOT="${TRAIN_ROOT:-$EXPERIMENT_DIR/$TRAIN_SCRIPT_NAME}"
DEVICE="${DEVICE:-cpu}"
SHOT_ID="0107"

if [[ -z "${TRAIN_RUN_DIR:-}" ]]; then
    TRAIN_RUN_DIR="$(find "$TRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$DATA_DIR/valid_dim" ]] || { echo "Validation dimension data not found: $DATA_DIR/valid_dim" >&2; exit 1; }
[[ -d "$DATA_DIR/valid_aux" ]] || { echo "Validation metadata not found: $DATA_DIR/valid_aux" >&2; exit 1; }
[[ -d "$DATA_DIR/shot" ]] || { echo "Original shot data not found: $DATA_DIR/shot" >&2; exit 1; }
[[ -d "$TRAIN_RUN_DIR" ]] || { echo "Training run directory not found: $TRAIN_RUN_DIR" >&2; exit 1; }
[[ -f "$DATA_DIR/valid_dim/patches_${SHOT_ID}.npy" ]] || { echo "Shot dimension patches not found: patches_${SHOT_ID}.npy" >&2; exit 1; }
[[ -f "$DATA_DIR/valid_aux/patches_${SHOT_ID}.npz" ]] || { echo "Shot metadata not found: patches_${SHOT_ID}.npz" >&2; exit 1; }

SUBSET_DIR="$(mktemp -d "${TMPDIR:-/tmp}/seisflow-shot-${SHOT_ID}.XXXXXX")"
trap 'rm -rf -- "$SUBSET_DIR"' EXIT
SUBSET_DIM_DIR="$SUBSET_DIR/valid_dim"
SUBSET_AUX_DIR="$SUBSET_DIR/valid_aux"
mkdir -p "$SUBSET_DIM_DIR" "$SUBSET_AUX_DIR"
ln -s "$DATA_DIR/valid_dim/patches_${SHOT_ID}.npy" "$SUBSET_DIM_DIR/"
ln -s "$DATA_DIR/valid_aux/patches_${SHOT_ID}.npz" "$SUBSET_AUX_DIR/"

mkdir -p "$RUN_DIR"
cd "$CODE_PATH"

echo "TRAIN_RUN_DIR: $TRAIN_RUN_DIR"
echo "SHOT_ID: $SHOT_ID"
echo "DEVICE: $DEVICE"
echo "OUTPUT_DIR: $RUN_DIR"

epoch_name="$(printf '%05d' "$EPOCH")"
checkpoint_dir="$TRAIN_RUN_DIR/checkpoint_epoch_${epoch_name}"

[[ -d "$checkpoint_dir" ]] || { echo "Checkpoint not found: $checkpoint_dir" >&2; exit 1; }

for weight_set in ema no_ema; do
    if [[ "$weight_set" == "ema" ]]; then
        ema_flag="--use_ema"
    else
        ema_flag="--no-use_ema"
    fi

    sample_files=()
    for seed in 0 1 2 3; do
        seed_name="$(printf '%04d' "$seed")"
        log_id="sample_${weight_set}_seed_${seed_name}_shot_${SHOT_ID}_epoch_${epoch_name}"
        sample_output_dir="$RUN_DIR/$log_id"

        echo "Sampling epoch $EPOCH with $weight_set weights, seed $seed"
        "$PYTHON_BIN" "$CODE_PATH/DiTSeisDimReconNeRF.py" sample \
            --ckpt "$checkpoint_dir" \
            --input_dim_dir "$SUBSET_DIM_DIR" \
            --output_dir "$RUN_DIR" \
            --log_id "$log_id" \
            --model_arch DiT_T_4 \
            --input_size 64 \
            --batch_size 32 \
            --solver_step_size 0.05 \
            --clip_recon -1 1 \
            --device "$DEVICE" \
            --nerf_bands 0 \
            --seed "$seed" \
            "$ema_flag" \
            --log_console

        sample_files+=("$sample_output_dir/patches_${SHOT_ID}.npy")
    done

    for sample_count in 1 2 4; do
        averaged_patch_dir="$RUN_DIR/valid_${weight_set}_samples_${sample_count}_shot_${SHOT_ID}_epoch_${epoch_name}"
        averaged_patch_file="$averaged_patch_dir/patches_${SHOT_ID}.npy"
        shot_output_dir="$RUN_DIR/valid_recon_shot_${weight_set}_samples_${sample_count}_shot_${SHOT_ID}_epoch_${epoch_name}"
        diff_output_dir="$RUN_DIR/diff_recon_shot_${weight_set}_samples_${sample_count}_shot_${SHOT_ID}_epoch_${epoch_name}"

        echo "Averaging $sample_count independent samples for $weight_set weights"
        "$PYTHON_BIN" "$CODE_PATH/AverageNpy.py" \
            --inputs "${sample_files[@]:0:$sample_count}" \
            --output "$averaged_patch_file"

        echo "Reconstructing shot $SHOT_ID from $sample_count averaged samples"
        "$PYTHON_BIN" "$CODE_PATH/ReconShotDataset2.py" \
            --input_dir "$averaged_patch_dir" \
            --input_aux_dir "$SUBSET_AUX_DIR" \
            --output_dir "$shot_output_dir"

        echo "Generating differences for $sample_count averaged samples"
        "$PYTHON_BIN" "$CODE_PATH/DiffShot.py" \
            --input1_dir "$DATA_DIR/shot" \
            --input2_dir "$shot_output_dir" \
            --output_dir "$diff_output_dir"
    done
done
