#!/usr/bin/env bash

python train.py \
  --seed 111 \
  --model timm/coatnet_0_rw_224.sw_in1k \
  --dataset_dir data/seed-111/ICH-17-processed-3-seed-111 \
  --train_batch_size 64 \
  --eval_batch_size 64 \
  --num_epochs 50 \
  --label_smoothing 0.0 \
  --num_workers 4 \
  --mixed_precision fp16 \
  --gradient_accum_steps 2 \
  --lr 1e-4 \
  --min_lr 0.0 \
  --weight_decay 1.0e-3 \
  --scheduler_T_0 3 \
  --scheduler_T_mult 1 \
  --best_checkpoint_metrics loss accuracy f1 \
  --save_best_k 8 \
  --use_mixup_cutmix \
  --max_grad_norm 1.0 \
  --wandb_logging \
  --wandb_project soups \
  --wandb_name example_expr
