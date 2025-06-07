import argparse
import os


def add_general_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('General')
    group.add_argument(
        '--seed',
        type=int,
        help='Seed',
        default=42,
    )
    group.add_argument(
        '--model',
        type=str,
        help='Name of the model to use (e.g., resnet50, densenet121, timm/coatnet_0_rw_224.sw_in1k, timm/maxvit_base_tf_224.in1k)',
        default='coatnet_0_rw_224.sw_in1k',
    )
    group.add_argument(
        '--dataset_dir',
        type=str,
        help='Path to the dataset',
        default='./data/vietnamese_cultural_dataset',
    )
    group.add_argument(
        '--image_size',
        type=int,
        help='Image size for the model',
        default=224,
    )

def add_training_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Training')

    # test only mode
    group.add_argument(
        '--run_test_only',
        action='store_true',
        help='Run testing only',
    )

    # basic stuff
    group.add_argument(
        '--checkpoints_dir',
        type=str,
        help='Checkpoints directory for saving stuff',
        default='./checkpoints',
    )
    group.add_argument(
        '--from_checkpoint',
        type=str,
        help='Path to the checkpoint storing the model state',
    )
    group.add_argument(
        '--train_batch_size',
        type=int,
        help='Training batch size',
        default=32,
    )
    group.add_argument(
        '--eval_batch_size',
        type=int,
        help='Evaluation batch size',
        default=32,
    )
    group.add_argument(
        '--num_epochs',
        type=int,
        help='Number of training epochs',
        default=10,
    )
    group.add_argument(
        '--label_smoothing',
        type=float,
        help='Label smoothing value',
        default=0.0,
    )
    group.add_argument(
        '--num_workers',
        type=int,
        default=min(os.cpu_count() or 1, 16),  # too large can cause insufficient shared memory
        help='Number of workers for data loading',
    )

    # mixed precision training
    group.add_argument(
        '--mixed_precision',
        type=str,
        choices=['fp16', 'bf16'],
        help='Whether to enable mixed precision training (fp16 or bf16)',
    )

    # optimizer
    group.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=1.0e-4,
    )
    group.add_argument(
        '--weight_decay',
        type=float,
        help='Weight decay',
        default=1e-4,
    )

    # scheduler
    group.add_argument(
        '--min_lr',
        type=float,
        help='Learning rate',
        default=0.0,
    )
    group.add_argument(
        '--scheduler_T_0',
        type=int,
        help='Number of iterations for the first restart in CosineAnnealingWarmRestarts',
        default=10,
    )
    group.add_argument(
        '--scheduler_T_mult',
        type=int,
        help='Multiplier for the period of the cosine annealing scheduler',
        default=3,
    )

    # EMA
    group.add_argument(
        '--use_ema',
        action='store_true',
        help='Whether to use Exponential Moving Average (EMA) for the model',
    )
    group.add_argument(
        '--model_ema_decay',
        type=float,
        help='Decay factor for Model EMA',
        default=0.9999,
    )
    group.add_argument(
        '--model_ema_warmup',
        action='store_true',
        help='Whether to use warmup for Model EMA',
    )

    # other
    group.add_argument(
        '--use_mixup_cutmix',
        action='store_true',
        help='Whether to use MixUp & CutMiX',
    )
    group.add_argument(
        '--max_grad_norm',
        type=float,
        help='Maximum gradient norm for gradient clipping',
        default=0.0,
    )

def add_wandb_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Wandb')
    group.add_argument(
        '--wandb_logging',
        action='store_true',
        help='Enable logging to wandb',
    )
    group.add_argument(
        '--wandb_project',
        type=str,
        help='Project name',
        default='medical-llama2',
    )
    group.add_argument(
        '--wandb_name',
        type=str,
        help='Experiment name',
        default='base',
    )
    group.add_argument(
        '--wandb_resume_id',
        type=str,
        help='Id to resume a run from',
    )
    group.add_argument(
        '--wandb_notes',
        type=str,
        help='Wandb notes',
    )
    group.add_argument(
        '--wandb_tags',
        type=str,
        help='Wandb tags',
    )
