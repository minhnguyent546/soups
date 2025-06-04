#!/usr/bin/env bash

python train.py \
  --dataset_dir datasets/vietnamese_cultural_dataset \
  --train_batch_size=32 \
  --eval_batch_size=32 \
