# Soups

> *We are cooking*

## Quick Start

### Prerequisites

- Python 3.10+
- uv 

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/minhnguyent546/soups.git
   cd soups
   ```

2. **Set up Python environment using uv:**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync

   # Active virtual environment
   source .venv/bin/activate
   ```

3. **Verify installation:**
   ```bash
   python -m soups.train --help
   ```

### Preparing datasets

> Will be updated.

### Dataset structure

```
dataset_name
├── train
│   ├── class_1
│   │   ├── img1.jpg
│   │   ├── ...
│   ├── class_2
│   │   ├── img1.jpg
│   │   ├── ...
├── val
│   ├── class_1
│   │   ├── img1.jpg
│   │   ├── ...
├── test
│   ├── class_1
│   │   ├── img1.jpg
│   │   ├── ...
```

## Usage

### Training

To train the model, you can run the following command:
```bash
python -m soups.train \
  --seed 42 \
  --model timm/coatnet_0_rw_224.sw_in1k \
  --dataset_dir data/ICH-17 \
  --use_mixup_cutmix \
  --train_batch_size 32 \
  --eval_batch_size 64 \
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
```

### Inference

To run inference on a trained model, you can use the following command:
```bash
python -m soups.train \
  --run_test_only \
  --from_checkpoint /PATH/TO/CHECKPOINT.pth \
  --seed 42 \
  --model timm/coatnet_0_rw_224.sw_in1k \
  --dataset_dir data/ICH-17 \
  --eval_batch_size 64 \
  --wandb_logging \
  --wandb_project soups \
  --wandb_name example_expr
```
