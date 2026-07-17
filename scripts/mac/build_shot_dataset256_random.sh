#!/bin/bash

set -u

PYTHON_PATH=""
SF_PATH="/Users/housian/Workplaces/SeisFlow"

python ${SF_PATH}/build_shot_dataset.py \
	--segy ${SF_PATH}/ma2+GathAP.sgy \
	--patch_size 256 \
	--overlap_size 64 \
	--output_dir ${SF_PATH}/shot_dataset256_random \
	--valid 0.3 \
	--valid_mode random \
	--clip -2 2 \
	--normalize