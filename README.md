# Leveraging Model Soups to Classify Intangible Cultural Heritage Images from the Mekong Delta

**Abstract:** The classification of Intangible Cultural Heritage (ICH) images in the Mekong Delta poses unique challenges due to limited annotated data, high visual similarity among classes, and domain heterogeneity. In such low-resource settings, conventional deep learning models often suffer from high variance or overfit to spurious correlations, leading to poor generalization. To address these limitations, we propose a robust framework that integrates the hybrid CoAtNet architecture with *model soups*, a lightweight weight-space ensembling technique that averages checkpoints from a single training trajectory - *without increasing inference cost*. CoAtNet captures both local and global patterns through stage-wise fusion of convolution and self-attention. We apply two ensembling strategies - *greedy* and *uniform* soup - to selectively combine diverse checkpoints into a final model. Beyond performance improvements, we analyze the ensembling effect through the lens of bias–variance decomposition. Our findings show that *model soups* reduces variance by stabilizing predictions across diverse model snapshots, while introducing minimal additional bias. Furthermore, using cross-entropy-based distance metrics and Multidimensional Scaling (MDS), we show that *model soups* selects geometrically diverse checkpoints, unlike Soft Voting, which blends redundant models centered in output space. Evaluated on the ICH-17 dataset (7,406 images across 17 classes), our approach achieves **state-of-the-art** results with **72.36**\% top-1 accuracy and **69.28**\% macro F1-score, outperforming strong baselines including ResNet-50, DenseNet-121, and ViT and previous studies on the ICH-17 dataset. These results underscore that diversity-aware checkpoint averaging provides a principled and efficient way to reduce variance and enhance generalization in culturally rich, data-scarce classification tasks.

---

[TOC]

## 1. Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) - A Package and Project manager for Python

### Setup

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

   # Activate virtual environment
   source .venv/bin/activate
   ```

3. **Verify installation:**
   ```bash
   python -m soups.train --help
   ```

## 2. Datasets

> The dataset used in this study is not publicly available due to institutional or licensing restrictions. However, it can be made available for academic use upon reasonable request. Interested researchers may contact the authors for further information.

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

## 3. Training

To train the model, you can run the following command:
```bash
python -m soups.train \
  --seed 42 \
  --model timm/coatnet_0_rw_224.sw_in1k \
  --dataset_dir data/ich-17 \
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
```

## 4. Inference

To run inference on a trained model, you can use the following command:
```bash
python -m soups.train \
  --run_test_only \
  --from_checkpoint /path/to/checkpoint.pth \
  --seed 42 \
  --model timm/coatnet_0_rw_224.sw_in1k \
  --dataset_dir data/ich-17 \
  --eval_batch_size 64
```

To run inference on multiple saved checkpoints:
```bash
python -m soups.run_test_multiple_checkpoints \
  --seed 42 \
  --checkpoints_dir /path/to/checkpoints/directory \
  --model timm/coatnet_0_rw_224.sw_in1k \
  --dataset_dir data/ich-17 \
  --output_file ./test_results.json
```

To run inference with *model soups*:
```bash
python -m soups.run_test_with_model_soups \
  --seed 42 \
  --checkpoint_path "/path/to/directory" "/path/to/checkpoint_1.pth" "/path/to/checkpoint_2.pth" \
  --model timm/coatnet_0_rw_224.sw_in1k \
  --uniform_soup \
  --greedy_soup \
  --dataset_dir data/ich-17 \
  --output_dir ./soups_results

```

## 5. Acknowledgment

We would like to express our sincere gratitude to the AniAge project for providing the ICH dataset used in our experiments. This valuable resource enabled a fair and rigorous evaluation of the proposed methods in a culturally grounded context.

## 6. Citing

If you find this repository useful for your research, please consider citing:

> Will be updated soon.
