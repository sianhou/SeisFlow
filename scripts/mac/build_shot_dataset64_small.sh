#!/bin/bash

set -u

PYTHON_PATH=""
SF_PATH="/Users/housian/Workplaces/SeisFlow"

python ${SF_PATH}/build_shot_dataset.py \
	--segy /Users/housian/Workplaces/SeisFlow/ma2+GathAP.sgy \
	--patch_size 64 \
	--overlap_size 32 \
	--output_dir ${SF_PATH}/shot_dataset64_small \
	--valid 0.0 \
	--clip -2 2 \
	--normalize

rm ${SF_PATH}/shot_dataset64_small/train/patches_00*
rm ${SF_PATH}/shot_dataset64_small/train/patches_01*
rm ${SF_PATH}/shot_dataset64_small/train/patches_020*
rm ${SF_PATH}/shot_dataset64_small/train/patches_021*
rm ${SF_PATH}/shot_dataset64_small/train/patches_022*
rm ${SF_PATH}/shot_dataset64_small/train/patches_023*
rm ${SF_PATH}/shot_dataset64_small/train/patches_024*

rm ${SF_PATH}/shot_dataset64_small/train_dim/patches_00*
rm ${SF_PATH}/shot_dataset64_small/train_dim/patches_01*
rm ${SF_PATH}/shot_dataset64_small/train_dim/patches_020*
rm ${SF_PATH}/shot_dataset64_small/train_dim/patches_021*
rm ${SF_PATH}/shot_dataset64_small/train_dim/patches_022*
rm ${SF_PATH}/shot_dataset64_small/train_dim/patches_023*
rm ${SF_PATH}/shot_dataset64_small/train_dim/patches_024*

rm ${SF_PATH}/shot_dataset64_small/train_aux/patches_00*
rm ${SF_PATH}/shot_dataset64_small/train_aux/patches_01*
rm ${SF_PATH}/shot_dataset64_small/train_aux/patches_020*
rm ${SF_PATH}/shot_dataset64_small/train_aux/patches_021*
rm ${SF_PATH}/shot_dataset64_small/train_aux/patches_022*
rm ${SF_PATH}/shot_dataset64_small/train_aux/patches_023*
rm ${SF_PATH}/shot_dataset64_small/train_aux/patches_024*
