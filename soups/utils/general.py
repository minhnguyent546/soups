import argparse
import os
import random
from typing import Any

import numpy as np
import torch
import yaml
from wandb.sdk.wandb_run import Run as WandbRun


def save_metadata_to_checkpoint(
    checkpoint_dir: str,
    args: argparse.Namespace,
    wandb_run: WandbRun | None = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    metadata = {
        'args': vars(args),
    }

    if wandb_run is not None:
        metadata['wandb'] = {
            'id': wandb_run.id,
            'name': wandb_run.name,
            'project': wandb_run.project,
            'tags': wandb_run.tags,
            'notes': wandb_run.notes,
        }

    metadata_path = os.path.join(checkpoint_dir, 'metadata.yml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_tensor_or_np(obj: Any) -> bool:
    return isinstance(obj, (torch.Tensor, np.ndarray))

def get_device(device_type: str = 'auto') -> torch.device:
    if device_type == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    try:
        return torch.device(device_type)
    except Exception:
        return torch.device('cpu')

def get_batch_samples(
    data_iter,
    num_batches: int,
    labels_key: str | None = None,
    labels_index: int | None = None,
    ignore_index: int = -100,
) -> tuple[list[Any], int | None]:
    """
    Get a batch of samples from the data iterator.

    Note: this function is applied for data iterators that produce items
    that have labels with labels key `labels_key` if items are
    of type dict, or labels index `labels_index` if items are
    of type list or tuple.
    """
    if labels_key is not None and labels_index is not None:
        raise ValueError("Specify either 'labels_key' or 'labels_index', not both.")

    batch_samples = []
    num_items_in_batch = None
    for _ in range(num_batches):
        try:
            batch_samples.append(next(data_iter))
        except StopIteration:
            break

    # there are no samples left
    if not batch_samples:
        return batch_samples, None

    # TODO: for simplicity, we handle for Tensor only here and
    # do not support list or numpy array
    labels_id = labels_key or labels_index
    if not isinstance(batch_samples[0][labels_id], torch.Tensor):
        raise ValueError(f'Expected labels in batch samples to be of type torch.Tensor, found type {type(batch_samples[0][labels_id])}')

    if batch_samples[0][labels_id].ndim == 1:
        num_items_in_batch = sum(
            torch.count_nonzero(batch_sample[labels_id] != ignore_index).item()
            for batch_sample in batch_samples
        )
    else:
        num_items_in_batch = sum(
            batch_sample[labels_id].shape[0]
            for batch_sample in batch_samples
        )
    return batch_samples, num_items_in_batch
