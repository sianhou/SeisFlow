#!/bin/bash

set -u

PYTHON_PATH="/hwdata/24ydz3d/deeplearning/environments/py313/"
SF_PATH="/hwdata/24ydz3d/SeisFlow"

${PYTHON_PATH}/bin/python3.13 ${SF_PATH}/scripts/build_patch_dataset.py \
	--segy /hwdata/24ydz3d/SeisFlow/ma2+GathAP.sgy \
	--patch_size 256 \
	--overlap_size 240 \
	--output_dir ${SF_PATH}/train_dataset256 \
	--clip_vmin -2 \
	--clip_vmax 2 \
	--normalize \
	--plot_interval 10000 
