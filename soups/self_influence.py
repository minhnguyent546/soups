"""Compute self-influence scores for the training set"""

import argparse
import json
import os
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
from captum.influence import TracInCPFast
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import soups.constants as C
import soups.utils as utils
from soups.opts import add_self_influence_opts
from soups.utils.logger import init_logger, logger
from soups.utils.training import infer_final_fc, make_model


def self_influence(args: argparse.Namespace) -> None:
    init_logger(compact=True)

    utils.set_seed(args.seed)
    logger.info(f'Seed: {args.seed}')

    if os.path.isfile(args.output_file):
        raise FileExistsError(f'Output file already exists: {args.output_file}')
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    device = utils.get_device(args.device)
    logger.info(f'Using device: {device}')

    # find all model checkpoint files
    checkpoint_paths: list[str] = []
    for checkpoint_path in args.checkpoint_paths:
        if os.path.isfile(checkpoint_path) and checkpoint_path.endswith('.pth'):
            checkpoint_paths.append(checkpoint_path)
        elif os.path.isdir(checkpoint_path):
            checkpoint_paths.extend(
                os.path.join(checkpoint_path, f)
                for f in os.listdir(checkpoint_path)
                if f.endswith('.pth')
            )

    if not checkpoint_paths:
        logger.error('No model checkpoints found.')
        exit(1)

    logger.info(
        f'Found total {len(checkpoint_paths)} model checkpoints for calculating self-influence'
    )

    # load dataset
    train_transforms = v2.Compose([
        v2.RandomResizedCrop(size=args.train_crop_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor(),
        v2.Normalize(
            mean=C.IMAGENET_DEFAULT_MEAN,
            std=C.IMAGENET_DEFAULT_STD,
        ),
    ])
    eval_transforms = v2.Compose([
        v2.Resize(size=args.eval_resize_size),
        v2.CenterCrop(size=args.eval_crop_size),
        v2.ToTensor(),
        v2.Normalize(
            mean=C.IMAGENET_DEFAULT_MEAN,
            std=C.IMAGENET_DEFAULT_STD,
        ),
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.dataset_dir, 'train'),
        transform=train_transforms,
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.dataset_dir, 'test'),
        transform=eval_transforms,
    )
    class_names = test_dataset.classes
    num_classes = len(class_names)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    _test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = make_model(
        model_name=args.model,
        num_classes=num_classes,
    ).to(device)
    model.eval()

    final_fc_layer = infer_final_fc(model=model)
    assert isinstance(final_fc_layer, nn.Linear)
    logger.debug(f'final_fc_layer: {final_fc_layer = }')

    tracin_cp_fast = TracInCPFast(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=train_data_loader,
        checkpoints=checkpoint_paths,
        checkpoints_load_func=partial(checkpoints_load_func, device=device),
        loss_fn=nn.CrossEntropyLoss(),
    )

    start_time = datetime.now()
    logger.info('Start computing self-influence scores...')
    self_influence_scores = tracin_cp_fast.self_influence(show_progress=True)
    total_minutes = (datetime.now() - start_time).total_seconds() / 60.0
    print(
        'computed self influence scores for %d examples in %.2f minutes'
        % (len(self_influence_scores), total_minutes)
    )

    self_influence_scores_list: list[float] = self_influence_scores.detach().cpu().tolist()
    sorted_score_indices = sorted(
        range(len(self_influence_scores_list)),
        key=lambda i: self_influence_scores_list[i],
    )
    sorted_samples = [
        {
            'file_name': train_dataset.samples[idx][0].split('/')[-1],
            'class_name': class_names[train_dataset.samples[idx][1]],
            'self_influence_score': self_influence_scores_list[idx],
        }
        for idx in sorted_score_indices
    ]

    predicted: list[bool] = []
    with torch.no_grad():
        model.to(device)
        checkpoints_load_func(model=model, checkpoint_path=checkpoint_paths[0], device=device)
        model.eval()

        # for idx in tqdm(sorted_score_indices, desc='Calculating accuracies', unit=' samples'):
        train_iter = tqdm(
            range(0, len(sorted_score_indices), args.eval_batch_size),
            desc='Calculating accuracies',
        )
        for i in train_iter:
            start_index = i
            end_index = min(len(sorted_score_indices), i + args.eval_batch_size)
            batch_indices = sorted_score_indices[start_index:end_index]
            images = torch.stack([train_dataset[idx][0] for idx in batch_indices], dim=0).to(
                device
            )
            labels = torch.tensor([train_dataset[idx][1] for idx in batch_indices], device=device)

            logits = model(images)
            predictions = logits.argmax(dim=1)
            predicted.extend((predictions == labels).tolist())

    plot_accuracies(predicted)

    to_save_data = {
        'self_influence_scores': sorted_samples,
        'train_dataset_dir': os.path.join(args.dataset_dir, 'train'),
        'num_classes': num_classes,
        'num_samples': len(train_dataset),
    }
    # save this to a file
    with open(args.output_file, 'w') as f:
        json.dump(to_save_data, f, ensure_ascii=False, indent=2)


def plot_accuracies(predicted: list[bool], show: bool = False) -> None:
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(8, 6))
    increasing_scores_accuracies = []
    decreasing_scores_accuracies = []

    num_corrects = 0
    for i in range(len(predicted)):
        num_corrects += int(predicted[i])
        increasing_scores_accuracies.append(num_corrects / (i + 1))

    num_corrects = 0
    for i in range(len(predicted)):
        num_corrects += int(predicted[-i - 1])
        decreasing_scores_accuracies.append(num_corrects / (i + 1))

    plt.plot(increasing_scores_accuracies, label='Increasing self-influence scores', color='blue')
    plt.plot(decreasing_scores_accuracies, label='Decreasing self-influence scores', color='red')
    plt.xlabel('Number of examples considered')
    plt.ylabel('Cumulative Accuracy')

    plt.title('Model Accuracy vs Self-influence Score Order')
    plt.legend()
    plt.grid(True)

    logger.info('Saved accuracies plot to ./accuracies.png')
    plt.savefig('./accuracies.png', dpi=300, bbox_inches='tight')

    if show:
        plt.show()


def checkpoints_load_func(model: nn.Module, checkpoint_path: str, device: torch.device) -> float:
    """
    When this function is used in TracInCP implementations, this function should
    return the learning rate at the checkpoint. However, if that learning rate
    is not available, it is safe to simply return 1, as we do, because it turns
    out TracInCP implementations are not sensitive to that learning rate.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return 1.0


def main():
    parser = argparse.ArgumentParser(
        description='Compute self-influence scores for the training set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_self_influence_opts(parser)
    args = parser.parse_args()

    self_influence(args)


if __name__ == '__main__':
    main()
