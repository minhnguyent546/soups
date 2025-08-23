#!/usr/bin/env bash

# This script is used for testing the model with model soups

python_command="python"
if command -v uv &> /dev/null; then
  python_command="uv run python"
fi

CHECKPOINTS_DIR='/path/to/saved/checkpoints'

$python_command -m soups.run_test_with_model_soups \
  --seed 111 \
  --checkpoint_path "$CHECKPOINTS_DIR" \
  --model timm/coatnet_0_rw_224.sw_in1k \
  --uniform_soup \
  --greedy_soup \
  --dataset_dir data/ich-split-renamed \
  --output_dir "${CHECKPOINTS_DIR}/soups_results"
