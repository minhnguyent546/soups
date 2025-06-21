"""Run test with soft voting"""

import argparse
import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as Fun
import torchvision
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

import soups.utils as utils
from soups.opts import add_test_with_soft_voting_opts
from soups.utils.logger import init_logger, logger
from soups.utils.training import EvalResults, make_model


@dataclass
class Candidate:
    model_path: str
    eval_results: EvalResults

def test_with_soft_voting(args: argparse.Namespace) -> None:
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

    # find all model checkpoint files
    checkpoint_paths: list[str] = []
    for checkpoint_path in args.checkpoint_path:
        if os.path.isfile(checkpoint_path) and checkpoint_path.endswith('.pth'):
            checkpoint_paths.append(checkpoint_path)
        elif os.path.isdir(checkpoint_path):
            checkpoint_paths.extend(
                os.path.join(checkpoint_path, f)
                for f in os.listdir(checkpoint_path) if f.endswith('.pth')
            )

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

    num_checkpoints = len(checkpoint_paths)
    all_logits: list[list[float]] = [[] for _ in range(num_checkpoints)]
    all_labels: list[int] = []
    for i, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        model.eval()

        logger.info(f'Testing checkpoint: {checkpoint_path}')

        with torch.no_grad():
            for images, labels in test_data_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                all_logits[i].extend(logits.detach().cpu().numpy())
                if i == 0:
                    all_labels.extend(labels.detach().cpu().numpy())

    # soft voting with mean logits
    mean_logits = torch.tensor(all_logits).mean(dim=0)
    labels = torch.tensor(all_labels)
    preds = mean_logits.argmax(dim=1)

    # compute loss, accuracy, precision, recall, f1
    loss = Fun.cross_entropy(mean_logits, labels)
    accuracy = (preds == labels).sum().item() / len(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=labels,
        y_pred=preds,
        average='macro',
        zero_division=0,  # pyright: ignore[reportArgumentType]
        labels=range(num_classes),
    )

    # per class accuracy
    conf_matrix = confusion_matrix(
        y_true=labels,
        y_pred=preds,
        labels=range(num_classes),
    )
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # per class precision, recall, f1
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true=labels,
        y_pred=preds,
        average=None,
        zero_division=0,  # pyright: ignore[reportArgumentType]
        labels=range(num_classes),
    )

    # log data to json file
    test_data = {
        'checkpoints': checkpoint_paths,
        'loss': f'{loss:0.4f}',
        'accuracy': f'{accuracy:0.4f}',
        'precision': f'{precision:0.4f}',
        'recall': f'{recall:0.4f}',
        'f1': f'{f1:0.4f}',
        'per_class_accuracy': {
            class_names[i]: f'{per_class_accuracy[i]:0.4f}'
            for i in range(num_classes)
        },
        'per_class_precision': {
            class_names[i]: f'{per_class_precision[i]:0.4f}'  # pyright: ignore[reportIndexIssue]
            for i in range(num_classes)
        },
        'per_class_recall': {
            class_names[i]: f'{per_class_recall[i]:0.4f}'  # pyright: ignore[reportIndexIssue]
            for i in range(num_classes)
        },
        'per_class_f1': {
            class_names[i]: f'{per_class_f1[i]:0.4f}'  # pyright: ignore[reportIndexIssue]
            for i in range(num_classes)
        }
    }

    with open(args.output_file, 'w') as f:
        json.dump(test_data, f, indent=4)

    logger.info(f'Test results saved to {args.output_file}')

def main():
    parser = argparse.ArgumentParser(
        description='Run test with soft voting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_test_with_soft_voting_opts(parser)
    args = parser.parse_args()

    test_with_soft_voting(args)


if __name__ == '__main__':
    main()
