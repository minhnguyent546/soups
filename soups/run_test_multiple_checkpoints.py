"""Run test for multiple checkpoints and write results to a json file"""

import argparse
import json
import os

import torch
import torchvision
from torch.utils.data import DataLoader

import soups.utils as utils
from soups.opts import add_test_multiple_checkpoints_opts
from soups.utils.logger import init_logger, logger
from soups.utils.training import eval_model, make_model


def test_with_model_soups(args: argparse.Namespace) -> None:
    if not args.output_file.endswith('.json'):
        logger.error('Output file must be a .json file')
        exit(1)
    if os.path.isfile(args.output_file):
        logger.error(f'Output file already exists: {args.output_file}')
        exit(1)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    init_logger(compact=True)

    utils.set_seed(args.seed)
    logger.info(f'Using seed: {args.seed}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # find all model checkpoint files
    checkpoint_paths: list[str] = []
    for checkpoint_path in os.listdir(args.checkpoints_dir):
        checkpoint_path = os.path.join(args.checkpoints_dir, checkpoint_path)
        if os.path.isfile(checkpoint_path) and checkpoint_path.endswith('.pth'):
            checkpoint_paths.append(checkpoint_path)

    if not checkpoint_paths:
        logger.error('No model checkpoints found.')
        exit(1)

    logger.info(f'Found total {len(checkpoint_paths)} model checkpoints')

    # test dataset and test data loader
    eval_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.dataset_dir, 'test'),
        transform=eval_transforms,
    )
    class_names = test_dataset.classes
    num_classes = len(class_names)

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
    best_results = None
    for checkpoint_path in checkpoint_paths:
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_dict['model_state_dict'])

        logger.info(f'Testing checkpoint: {checkpoint_path}')
        test_results = eval_model(
            model=model,
            eval_data_loader=test_data_loader,
            device=device,
            num_classes=num_classes,
        )
        test_data[checkpoint_path] = {
            'loss': f'{test_results["loss"]:0.4f}',
            'accuracy': f'{test_results["accuracy"]:0.4f}',
            'precision': f'{test_results["precision"]:0.4f}',
            'recall': f'{test_results["recall"]:0.4f}',
            'f1': f'{test_results["f1"]:0.4f}',
        }
        for per_class_metric in (
            'per_class_accuracy', 'per_class_precision', 'per_class_recall', 'per_class_f1',
        ):
            test_data[checkpoint_path][per_class_metric] = {}
            for i, class_name in enumerate(class_names):
                test_data[checkpoint_path][per_class_metric][class_name] = (
                    f'{test_results[per_class_metric][i]:0.4f}'
                )

        # choose the best checkpoint based on (f1, accuracy) score
        if best_results is None:
            best_results = test_results
        elif round(best_results['f1'], 4) < round(test_results['f1'], 4) or (
            round(best_results['f1'], 4) == round(test_results['f1'], 4) and
            round(best_results['accuracy'], 4) < round(test_results['accuracy'], 4)
        ):
            best_results = test_results

    if best_results is not None:
        test_data['best_results'] = {
            'loss': f'{best_results["loss"]:0.4f}',
            'accuracy': f'{best_results["accuracy"]:0.4f}',
            'precision': f'{best_results["precision"]:0.4f}',
            'recall': f'{best_results["recall"]:0.4f}',
            'f1': f'{best_results["f1"]:0.4f}',
        }
    with open(args.output_file, 'w') as f:
        json.dump(test_data, f, indent=4)

    logger.info(f'Test results saved to {args.output_file}')

def main():
    parser = argparse.ArgumentParser(
        description='Run test for multiple checkpoints and output results to a json file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_test_multiple_checkpoints_opts(parser)
    args = parser.parse_args()

    test_with_model_soups(args)


if __name__ == '__main__':
    main()
