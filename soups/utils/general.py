import argparse
import os
import random
import yaml

import numpy as np
import torch
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
