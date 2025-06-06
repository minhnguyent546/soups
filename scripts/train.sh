#!/usr/bin/env bash

python train.py \
  --seed 42 \
  --model timm/coatnet_0_rw_224.sw_in1k \
  --dataset_dir data/ICH-17-processed-2 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --num_epochs 50 \
  --lr 1e-4 \
  --min_lr 0.0 \
  --weight_decay 1.0e-4 \
  --label_smoothing 0.1 \
  --scheduler_T_0 10 \
  --scheduler_T_mult 3 \
  --wandb_logging \
  --wandb_project soups \
  --wandb_name example_expr
