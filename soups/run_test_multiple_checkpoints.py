"""Run test for multiple checkpoints and write results to a json file"""

import argparse
import json
import os
import time
from datetime import timedelta

import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

import soups.utils as utils
from soups.opts import add_test_multiple_checkpoints_opts
from soups.utils.logger import init_logger, logger
from soups.utils.training import EvalResults, convert_eval_results_to_dict, eval_model, make_model


def test_multiple_checkpoints(args: argparse.Namespace) -> None:
    if not args.output_file.endswith('.json'):
        logger.error('Output file must be a .json file')
        exit(1)
    if os.path.isfile(args.output_file):
        logger.error(f'Output file already exists: {args.output_file}')
        exit(1)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    log_file_path = os.path.join(
        os.path.dirname(args.output_file), 'test_with_multiple_checkpoints.log'
    )
    init_logger(log_file=log_file_path, compact=True)

    utils.set_seed(args.seed)
    logger.info(f'Using seed: {args.seed}')

    device = utils.get_device(args.device)
    logger.info(f'Using device: {device}')

    # find all model checkpoint files
    checkpoint_paths = utils.find_checkpoint_files(checkpoint_files_or_dirs=args.checkpoint_paths)
    if not checkpoint_paths:
        logger.error('No model checkpoints found.')
        exit(1)

    logger.info(f'Found total {len(checkpoint_paths)} model checkpoints')

    # test dataset and test data loader
    eval_transforms = v2.Compose([
        v2.Resize(size=args.eval_resize_size),
        v2.CenterCrop(size=args.eval_crop_size),
        v2.ToTensor(),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.dataset_dir, 'val'),
        transform=eval_transforms,
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.dataset_dir, 'test'),
        transform=eval_transforms,
    )
    class_names = test_dataset.classes
    num_classes = len(class_names)

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(1, min(16, (os.cpu_count() or 1) // 2)),
        pin_memory=True,
        persistent_workers=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(1, min(16, (os.cpu_count() or 1) // 2)),
        pin_memory=True,
        persistent_workers=True,
    )

    model = make_model(
        model_name=args.model,
        num_classes=num_classes,
    ).to(device)

    test_data = {}
    best_test_results: EvalResults | None = (
        None  # test results for checkpoint with best (f1, accuracy)
    )
    best_test_results_checkpoint_path = None
    best_val_f1: float = float('-inf')  # best val f1 score
    best_val_f1_checkpoint_path = None  # checkpoint with best val f1

    test_start_time = time.perf_counter()
    for i, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_dict['model_state_dict'])

        logger.info(f'Testing checkpoint [{i} / {len(checkpoint_paths)}]: {checkpoint_path}')
        val_results = eval_model(
            model=model,
            eval_data_loader=val_data_loader,
            device=device,
            num_classes=num_classes,
        )
        test_results = eval_model(
            model=model,
            eval_data_loader=test_data_loader,
            device=device,
            num_classes=num_classes,
        )
        test_data[checkpoint_path] = {}
        test_data[checkpoint_path]['val'] = convert_eval_results_to_dict(
            eval_results=val_results,
            class_names=class_names,
        )
        test_data[checkpoint_path]['test'] = convert_eval_results_to_dict(
            eval_results=test_results,
            class_names=class_names,
        )

        # choose the best checkpoint based on val f1
        if val_results['f1'] > best_val_f1:
            best_val_f1 = val_results['f1']
            best_val_f1_checkpoint_path = checkpoint_path

        # choose the best checkpoint based on (f1, accuracy) score
        if best_test_results is None:
            best_test_results = test_results
            best_test_results_checkpoint_path = checkpoint_path
        elif round(best_test_results['f1'], 6) < round(test_results['f1'], 6) or (
            round(best_test_results['f1'], 6) == round(test_results['f1'], 6)
            and round(best_test_results['accuracy'], 6) < round(test_results['accuracy'], 6)
        ):
            best_test_results = test_results
            best_test_results_checkpoint_path = checkpoint_path

    assert best_test_results is not None
    assert best_test_results_checkpoint_path is not None
    assert best_val_f1_checkpoint_path is not None

    test_data['best_val_f1_checkpoint_test_results'] = test_data[best_val_f1_checkpoint_path][
        'test'
    ]
    test_data['best_val_f1_checkpoint_test_results']['checkpoint_path'] = (
        best_val_f1_checkpoint_path
    )

    test_data[f'best_test_results@{len(checkpoint_paths)}'] = convert_eval_results_to_dict(
        eval_results=best_test_results, class_names=class_names
    )
    test_data[f'best_test_results@{len(checkpoint_paths)}']['checkpoint_path'] = (
        best_test_results_checkpoint_path
    )

    with open(args.output_file, 'w') as f:
        json.dump(test_data, f, indent=4)

    logger.info(f'Test results saved to {args.output_file}')

    test_end_time = time.perf_counter()
    total_test_time = test_end_time - test_start_time
    total_test_time_str = str(timedelta(seconds=int(total_test_time)))
    logger.info(f'Cooking time: {total_test_time_str}')


def main():
    parser = argparse.ArgumentParser(
        description='Run test for multiple checkpoints and output results to a json file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_test_multiple_checkpoints_opts(parser)
    args = parser.parse_args()

    test_multiple_checkpoints(args)


if __name__ == '__main__':
    main()
