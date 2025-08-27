#!/usr/bin/env bash

# This script is used for testing the model with multiple checkpoints

CHECKPOINTS_DIR='/path/to/saved/checkpoints'

python -m soups.run_test_multiple_checkpoints \
  --seed 111 \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --model timm/coatnet_0_rw_224.sw_in1k \
  --dataset_dir data/ich-split-renamed \
  --output_file "${CHECKPOINTS_DIR}/test_results.json"
