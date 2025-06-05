#!/usr/bin/env bash

python train.py \
  --dataset_dir data/ICH-17 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --lr 1.0e-4 \
  --num_epochs 10 \
  --weight_decay 1.0e-4 \
  --label_smoothing 0.1 \
  --wandb_logging \
  --wandb_project soups \
  --wandb_name expr_xx
