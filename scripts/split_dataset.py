#!/usr/bin/env python3

"""
This script is used for splitting an image dataset into training, validation, and test sets.

Expected input format:
```
dataset_name
├── class_1
│     ├── image.jpg
│     ├── image.png
│     ├── ...
├── class_2
│     ├── image.jpg
│     ├── ...
├── ...
```

Output format:
```
dataset_name
├── train
│     ├── class_1
│     │     ├── image.jpg
│     │     ├── ...
│     ├── class_2
│     │     ├── image.jpg
│     │     ├── ...
├── val
│     ├── class_1
│     │     ├── image.jpg
│     │     ├── ...
├── test
│     ├── class_1
│     │     ├── image.jpg
│     │     ├── ...
```
"""

import argparse
import os
import random
import shutil

import torch
import torchvision
from sklearn.model_selection import train_test_split


def make_dataset_splits(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = torchvision.datasets.ImageFolder(
        root=args.dataset_dir,
    )
    print(f'Total image found: {len(dataset)}')

    # using train_test_split to split this dataset into train, test, and val splits
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.1,
        random_state=args.seed,
        stratify=[target for _, target in dataset.samples],
    )
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.1,
        random_state=args.seed,
        stratify=[dataset.samples[i][1] for i in train_indices],
    )
    print(
        f'Train size: {len(train_indices)}, '
        f'Test size: {len(test_indices)}, '
        f'Val size: {len(val_indices)}'
    )

    # create directories for splits
    os.makedirs(args.output_dir, exist_ok=True)
    split_names = ['train', 'test', 'val']

    # save the splits
    for split, indices in zip(
        split_names, [train_indices, test_indices, val_indices], strict=True
    ):
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for class_name in dataset.classes:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        for idx in indices:
            src_path, label = dataset.samples[idx]
            class_name = dataset.classes[label]
            dst_path = os.path.join(split_dir, class_name, os.path.basename(src_path))
            shutil.copyfile(src_path, dst_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Make dataset splits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed',
        default=42,
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Path to the dataset directory',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the output directory',
    )

    args = parser.parse_args()
    make_dataset_splits(args)


if __name__ == '__main__':
    main()
