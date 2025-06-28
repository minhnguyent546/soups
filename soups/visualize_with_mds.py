"""Visualizing predictions from various checkpoints"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import torch
import torch.nn.functional as Fun
import torchvision
from sklearn.manifold import MDS
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import soups.utils as utils
from soups.opts import add_visualize_predictions_opts
from soups.utils.logger import init_logger, logger
from soups.utils.metric import AverageMeter
from soups.utils.training import make_model
import soups.utils.dist as dist_fun


def visualize_predictions(args: argparse.Namespace) -> None:
    init_logger(compact=True)
    utils.set_seed(args.seed)
    logger.info(f'Using seed: {args.seed}')

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu',
    )
    logger.info(f'Using device: {device}')

    if os.path.isfile(args.output_file):
        logger.error(f'Output file already exists: {args.output_file}')
        exit(1)

    # find all model checkpoint files
    model_paths: list[str] = []
    greedy_checkpoint_idx = None
    uniform_checkpoint_idx = None
    if (
        args.greedy_soup_checkpoint is not None and
        args.greedy_soup_checkpoint.endswith('.pth')
    ):
        model_paths.append(args.greedy_soup_checkpoint)
        greedy_checkpoint_idx = len(model_paths) - 1
    if (
        args.uniform_soup_checkpoint is not None and
        args.uniform_soup_checkpoint.endswith('.pth')
    ):
        model_paths.append(args.uniform_soup_checkpoint)
        uniform_checkpoint_idx = len(model_paths) - 1

    for model_path in args.checkpoint_path:
        if os.path.isfile(model_path) and model_path.endswith('.pth'):
            model_paths.append(model_path)
        elif os.path.isdir(model_path):
            model_paths.extend(
                os.path.join(model_path, f)
                for f in os.listdir(model_path) if f.endswith('.pth')
            )

    model_paths = list(set(model_paths))  # remove duplicates
    if not model_paths:
        logger.error('No model checkpoints found.')
        exit(1)

    logger.info(f'Found total {len(model_paths)} model checkpoints')

    # test dataset and test data loader
    logger.info(f'Run visualization on {args.eval_split} split')
    eval_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    eval_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.dataset_dir, args.eval_split),
        transform=eval_transforms,
    )
    classes = eval_dataset.classes
    num_classes = len(classes)

    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(1, min(16, (os.cpu_count() or 1) // 2)),
        pin_memory=True,
        persistent_workers=True,
    )

    all_probs_list: list[list[list[float]]] = []
    model = make_model(
        model_name=args.model,
        num_classes=num_classes,
    ).to(device)

    for model_path in model_paths:
        logger.info(f'Evaluating checkpoint: {model_path}')
        model.load_state_dict(
            torch.load(model_path, map_location=device)['model_state_dict'],
        )

        eval_iter = tqdm(eval_data_loader, desc='Evaluating model')
        eval_loss = AverageMeter('eval_loss', fmt=':0.4f')

        # for each image in the dataset, we store the probability for each class
        probs_list: list[list[float]] = []

        with torch.no_grad():
            for images, labels in eval_iter:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                probs = Fun.softmax(logits, dim=1)
                loss = Fun.cross_entropy(input=logits, target=labels)
                eval_loss.update(loss.item(), labels.shape[0])

                probs_list.extend(probs.detach().cpu().numpy())

                eval_iter.set_postfix({
                    'loss': f'{loss:0.4f}',
                })

        all_probs_list.append(probs_list)

    # calculate pairwise distance with cross_entropy_dist_fn
    all_probs_list = np.array(all_probs_list)  # pyright: ignore[reportAssignmentType]
    dist = dist_fun.pairwise_cross_entropy_dist(torch.tensor(all_probs_list)).tolist()
    for i in range(len(dist)):
        dist[i][i] = 0

    embedding = MDS(
        n_components=2,
        max_iter=500,
        verbose=2,
        n_init=4,  # pyright: ignore[reportArgumentType]
        random_state=args.seed,
        dissimilarity='precomputed',
    )

    embeddings = embedding.fit_transform(dist)
    plot_embeddings(
        embeddings,  # pyright: ignore[reportArgumentType]
        greedy_checkpoint_idx=greedy_checkpoint_idx,
        uniform_checkpoint_idx=uniform_checkpoint_idx,
        save_path=args.output_file,
        show=False,
    )

def plot_embeddings(
    embeddings: np.ndarray,  # pyright: ignore[reportMissingTypeArgument]
    greedy_checkpoint_idx: int | None = None,
    uniform_checkpoint_idx: int | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    plt.style.use('science')
    plt.figure(figsize=(8, 8))

    spec_indices = []
    if greedy_checkpoint_idx is not None and greedy_checkpoint_idx < embeddings.shape[0]:
        spec_indices.append(greedy_checkpoint_idx)
        plt.scatter(
            embeddings[greedy_checkpoint_idx, 0], embeddings[greedy_checkpoint_idx, 1],
            c='red', s=50, alpha=0.5, label='Greedy Model',
        )
    if uniform_checkpoint_idx is not None and uniform_checkpoint_idx < embeddings.shape[0]:
        spec_indices.append(uniform_checkpoint_idx)
        plt.scatter(
            embeddings[uniform_checkpoint_idx, 0], embeddings[uniform_checkpoint_idx, 1],
            c='green', s=50, alpha=0.5, label='Uniform Model',
        )

    # plot remaining embeddings
    if spec_indices:
        embeddings = np.delete(embeddings, spec_indices, axis=0)
    if embeddings.shape[0] > 0:
        plt.scatter(
            embeddings[:, 0], embeddings[:, 1], s=50, alpha=0.5, label='Ingredient Models',
        )

    plt.legend()
    plt.title('MDS Visualization of Model Predictions')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    if save_path is not None:
        plt.savefig(save_path)
        logger.info(f'Saved plot to {save_path}')

    if show:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Visualize predictions from various checkpoints',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_visualize_predictions_opts(parser)
    args = parser.parse_args()

    visualize_predictions(args)


if __name__ == '__main__':
    main()
