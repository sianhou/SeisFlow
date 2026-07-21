#!/bin/bash

set -u

PYTHON_PATH=""
SF_PATH="/Users/housian/Workplaces/SeisFlow"

python ${SF_PATH}/build_shot_dataset2.py \
	--segy /Users/housian/Workplaces/SeisFlow/ma2+GathAP.sgy \
	--patch_size 64 \
	--overlap_size 32 \
	--output_dir ${SF_PATH}/shot_dataset64_nonorm \
	--valid 0.0 \
	--clip -2 2 \
