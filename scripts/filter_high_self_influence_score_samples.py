#!/usr/bin/env python3

"""Filter samples with high self-influence scores and saved the filtered dataset."""

import argparse
import os
import shutil
import sys

import torchvision
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from soups.utils.logger import init_logger, logger


class SelfInfluenceScoreItem(BaseModel):
    file_name: str
    class_name: str
    self_influence_score: float


class SelfInfluenceResults(BaseModel):
    model_name: str
    checkpoint_path_list: list[str]
    self_influence_scores: list[SelfInfluenceScoreItem]
    train_dataset_dir: str
    num_classes: int
    num_samples: int


def filter_high_self_influence_score_samples(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.self_influence_results) or not args.self_influence_results.endswith(
        '.json'
    ):
        logger.error(f'Invalid self-influence results file: {args.self_influence_results}')
        exit(1)
    if os.path.isdir(args.output_dataset_dir) and len(os.listdir(args.output_dataset_dir)) > 0:
        logger.error(
            f'Output dataset directory already exists and is not empty: {args.output_dataset_dir}'
        )
        exit(1)

    init_logger(compact=True)

    with open(args.self_influence_results, 'r', encoding='utf-8') as f:
        self_influence_results_str = f.read()

    try:
        self_influence_results = SelfInfluenceResults.model_validate_json(
            self_influence_results_str
        )
    except ValueError as err:
        logger.error(f'Invalid self-influence results file format: {args.self_influence_results}')
        logger.error(err)
        exit(1)

    # loading dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.dataset_dir, 'train'),
    )
    label_to_class = train_dataset.classes
    class_to_label = train_dataset.class_to_idx
    num_classes = len(label_to_class)

    if self_influence_results.num_classes != num_classes:
        logger.error(
            f'Number of classes in self-influence results ({self_influence_results.num_classes}) does not match the dataset ({num_classes})'
        )
        exit(1)
    if self_influence_results.num_samples != len(train_dataset):
        logger.error(
            f'Number of samples in self-influence results ({self_influence_results.num_samples}) does not match the dataset ({len(train_dataset)})'
        )
        exit(1)

    # copy test and val splits
    for split in ['val', 'test', 'train']:
        src_split_dir = os.path.join(args.dataset_dir, split)
        dst_split_dir = os.path.join(args.output_dataset_dir, split)
        if not os.path.isdir(src_split_dir):
            logger.error(f'Split directory does not exist: {src_split_dir}')
            exit(1)

        logger.info(f'Copying {split} split from {src_split_dir} to {dst_split_dir}')
        shutil.copytree(src_split_dir, dst_split_dir)

    self_influence_scores = self_influence_results.self_influence_scores
    if args.num_top_samples_to_remove is not None:
        self_influence_scores = self_influence_scores[: args.num_top_samples_to_remove]

    num_removed_samples_per_class: dict[int, int] = dict.fromkeys(range(num_classes), 0)

    # determine max number of removed samples per class
    max_num_removed_samples_per_class: list[int]
    if args.max_num_removed_samples_per_class < 1.0:
        max_num_removed_samples_per_class = [
            int(args.max_num_removed_samples_per_class * train_dataset.targets.count(c))
            for c in range(num_classes)
        ]
    else:
        max_num_removed_samples_per_class = [
            int(args.max_num_removed_samples_per_class) for _ in range(num_classes)
        ]

    logger.debug('  Max number of removed samples per class:')
    for class_label, max_num_removed in enumerate(max_num_removed_samples_per_class):
        logger.debug(f'   - {label_to_class[class_label]}: {max_num_removed}')

    logger.info('Filtering top self-influence score samples...')
    for item in self_influence_scores:
        item_label = class_to_label[item.class_name]
        if (
            num_removed_samples_per_class[item_label]
            >= max_num_removed_samples_per_class[item_label]
        ):
            # if we have reached the max number of removed samples for this class, skip it
            continue
        num_removed_samples_per_class[item_label] += 1
        file_to_remove = os.path.join(
            args.output_dataset_dir, 'train', item.class_name, item.file_name
        )
        try:
            os.remove(file_to_remove)
        except FileNotFoundError:
            logger.warning(f'File not found, skipping: {file_to_remove}')
        except PermissionError as e:
            logger.warning(f'Permission denied when removing {file_to_remove}: {e}')

    logger.info(' ** Summary ** ')
    logger.info(f'  Total samples in the original training set: {len(train_dataset)}')
    num_removed_samples = sum(num_removed_samples_per_class.values())
    logger.info(f'  Total samples removed: {num_removed_samples}')
    logger.info('  Removed samples per class:')
    for class_label, num_removed in num_removed_samples_per_class.items():
        logger.info(f'   - {label_to_class[class_label]}: {num_removed}')


def _add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--self_influence_results',
        type=str,
        help='Path to the self-influence results JSON file.',
        required=True,
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='Path to the dataset',
        default='./data/ICH-17',
    )
    parser.add_argument(
        '--output_dataset_dir',
        type=str,
        help='Path to the directory to save the filtered dataset',
        default='./data/ICH-17-filtered',
    )
    parser.add_argument(
        '--num_top_samples_to_remove',
        type=int,
        help='Number of samples (top self-influence scores samples) to consider removing. Leave `None` to consider all samples.',
        default=None,
    )
    parser.add_argument(
        '--max_num_removed_samples_per_class',
        type=float,
        help='Max number of samples to remove per class (use float for percentage). Leave `None` to disable this option',
        default=None,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Filter samples with high self-influence scores and saved the filtered dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_opts(parser)
    args = parser.parse_args()

    filter_high_self_influence_score_samples(args)


if __name__ == '__main__':
    main()
