#!/bin/bash

python train6.py \
  --train_data_dir /Users/housian/Workplaces/SeisFlow/train_dataset64b/ \
  --train_data_aux_dir /Users/housian/Workplaces/SeisFlow/train_dataset64b_aux/ \
  --output_dir /Users/housian/Workplaces/SeisFlow/output_train_dit_size64b/ \
  --model_arch DiT_T_4 \
  --input_size 64 \
  --batch_size 32 \
  --num_workers 0 \
  --pin_memory \
  --learning_rate 1e-4 \
  --num_epochs 1000 \
  --save_every_epochs 50 \
  --device cpu \
  --log_console
