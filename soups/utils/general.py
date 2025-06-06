import os
import yaml
import argparse

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
