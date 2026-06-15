#!/bin/bash

set -u

PYTHON_PATH=""
SF_PATH="/Users/housian/Workplaces/SeisFlow"

python ${SF_PATH}/scripts/build_patch_dataset.py \
	--segy /Users/housian/Workplaces/SeisFlow/ma2+GathAP.sgy \
	--patch_size 64 \
	--overlap_size 32 \
	--output_dir ${SF_PATH}/train_dataset64 \
	--clip_vmin -2 \
	--clip_vmax 2 \
	--normalize \
	--plot_interval 1000
