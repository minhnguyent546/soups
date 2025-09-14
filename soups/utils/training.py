import argparse
import heapq
import os
from contextlib import nullcontext
from typing import Any, TypedDict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from wandb.sdk.wandb_run import Run as WandbRun

from soups.utils.logger import logger
from soups.utils.metric import AverageMeter


class EvalResults(TypedDict):
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    per_class_accuracy: list[float]
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]


class EarlyStopping:
    def __init__(
        self, patience: int = 5, min_delta: float = 0.0, enabled: bool = True, verbose: bool = True
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.enabled = enabled
        self.counter = 0
        self.min_val_loss = float('inf')
        self.verbose = verbose

    def early_stop(self, val_loss: float) -> bool:
        if not self.enabled:
            # do nothing
            return False

        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.verbose:
                logger.info(
                    f'No improvement in validation loss for {self.counter} consecutive epochs'
                )
            if self.counter >= self.patience:
                return True
        return False

    def is_enabled(self) -> bool:
        return self.enabled


def make_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model_classifier = model.fc
    elif model_name == 'densenet121':
        model = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None,
        )
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        model_classifier = model.classifier
    elif model_name.startswith('timm/'):
        model = timm.create_model(
            model_name[len('timm/') :],
            pretrained=pretrained,
            num_classes=num_classes,
        )
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            model_classifier = model.head
        elif (
            hasattr(model, 'head')
            and hasattr(model.head, 'fc')
            and isinstance(model.head.fc, nn.Linear)
        ):
            model_classifier = model.head.fc
        elif (
            hasattr(model, 'head')
            and hasattr(model.head, 'fc')
            and isinstance(model.head.fc, timm.models.metaformer.MlpHead)
        ):
            model_classifier = model.head.fc.fc2
        else:
            raise ValueError(f'Unable to determine classification head for model {model_name}')
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    # initializing classification head
    nn.init.xavier_uniform_(model_classifier.weight)
    if model_classifier.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
        nn.init.zeros_(model_classifier.bias)

    return model


def infer_final_fc(model: nn.Module) -> nn.Module:
    if isinstance(model, torchvision.models.ResNet):
        final_fc = model.fc
    elif isinstance(model, torchvision.models.DenseNet):
        final_fc = model.classifier
    else:
        # timm models
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            final_fc = model.head
        elif (
            hasattr(model, 'head')
            and hasattr(model.head, 'fc')
            and isinstance(model.head, nn.Module)
            and isinstance(model.head.fc, nn.Linear)
        ):
            final_fc = model.head.fc
        elif (
            hasattr(model, 'head')
            and hasattr(model.head, 'fc')
            and isinstance(model.head, nn.Module)
            and isinstance(model.head.fc, timm.models.metaformer.MlpHead)
        ):
            final_fc = model.head.fc.fc2
        else:
            raise ValueError('Unsupported model type for inferring final fc layer')

    return final_fc


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    args: argparse.Namespace,
    **kwargs,
):
    if scheduler_name == 'cosine_annealing':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=args.cosine_annealing_T_0,
            T_mult=args.cosine_annealing_T_mult,
            eta_min=args.min_lr,
            **kwargs,
        )
    elif scheduler_name == 'one_cycle_lr':
        required_args = ['epochs', 'steps_per_epoch']
        for required_arg in required_args:
            if required_arg not in kwargs:
                raise ValueError(
                    f'Argument `{required_arg}` is required for OneCycleLR scheduler but missing in `kwargs`'
                )

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=args.lr,
            pct_start=args.one_cycle_lr_pct_start,
            **kwargs,
        )
    else:
        raise ValueError(f'Unsupported scheduler: {scheduler_name}')

    return scheduler


def eval_model(
    model: nn.Module,
    eval_data_loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
    device: torch.device,
    num_classes: int,
    autocast_context=None,
) -> EvalResults:
    if autocast_context is None:
        autocast_context = nullcontext()

    model_mode_before = model.training
    model.eval()

    eval_iter = tqdm(eval_data_loader, desc='Evaluating model')
    eval_loss = AverageMeter('eval_loss', fmt=':0.4f')
    eval_accuracy = AverageMeter('eval_accuracy', fmt=':0.4f')
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in eval_iter:
            images = images.to(device)
            labels = labels.to(device)

            with autocast_context:
                logits = model(images)
                loss = Fun.cross_entropy(input=logits, target=labels)

            preds = logits.argmax(dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            eval_loss.update(loss.item(), labels.shape[0])

            num_corrects = (preds == labels).sum().item()
            cur_accuracy = num_corrects / labels.shape[0]
            eval_accuracy.update(cur_accuracy, labels.shape[0])

            eval_iter.set_postfix({
                'loss': f'{loss:0.4f}',
            })

    # set model back to the original mode
    model.train(model_mode_before)

    # precision, recall, f1
    label_names = list(range(num_classes))
    eval_precision, eval_recall, eval_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0,  # pyright: ignore[reportArgumentType]
        labels=label_names,
    )

    # per class accuracy
    conf_matrix = confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        labels=label_names,
    )
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # per class precision, recall, f1
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true=all_labels,
        y_pred=all_preds,
        average=None,
        zero_division=0,  # pyright: ignore[reportArgumentType]
        labels=label_names,
    )

    return {
        'loss': eval_loss.avg,
        'accuracy': eval_accuracy.avg,
        'precision': float(eval_precision),
        'recall': float(eval_recall),
        'f1': float(eval_f1),
        'per_class_accuracy': per_class_accuracy.tolist(),
        'per_class_precision': per_class_precision.tolist(),  # pyright: ignore[reportAttributeAccessIssue]
        'per_class_recall': per_class_recall.tolist(),  # pyright: ignore[reportAttributeAccessIssue]
        'per_class_f1': per_class_f1.tolist(),  # pyright: ignore[reportAttributeAccessIssue]
    }


def save_top_k_checkpoints(
    criterion_metrics: list[str],
    top_k: int,
    val_results: EvalResults,
    best_val_results: dict[str, list[tuple[float, str]]],
    state_dict_to_save: dict[str, Any],
    checkpoint_path_template: str,
) -> None:
    """
    Save the top-k model checkpoints based on validation metrics.

    This function implements a checkpoint management strategy that
    maintains only the top-k best checkpoints for each specified metric.
    It uses a min-heap data structure to efficiently track and manage
    the best performing models, automatically removing worse checkpoints
    when the limit is exceeded.

    Behavior:
        - For each metric in `criterion_metrics`, evaluates if the current model performance
          warrants saving a checkpoint
        - For loss metrics, uses negative values to maintain consistent "higher is better" semantics
        - Maintains up to `top_k` checkpoints per metric using a min-heap
        - When the checkpoint limit is reached, replaces the worst checkpoint if current performance is better
        - Automatically deletes old checkpoint files from disk when they are replaced
        - Saves complete checkpoint state including model weights, validation results, epoch, and global step

    File Naming:
        Checkpoint files are named as: "model_1_epoch_{epoch}_{metric}_{metric_value:.4f}.pth"

    Note:
        - The function modifies best_val_results in-place to maintain state across training epochs
        - Loss values are negated internally for heap operations but displayed as positive values in logs
        - Only saves checkpoints that are among the top-k performing for their respective metrics
    """
    for metric in criterion_metrics:
        current_metric_value = val_results[metric]
        if metric == 'loss':
            # For loss, we want to save the lowest value, so we negate it
            current_metric_value = -current_metric_value

        current_checkpoint_path = checkpoint_path_template.format(
            metric=metric, metric_value=abs(current_metric_value)
        )

        if len(best_val_results[metric]) < top_k:
            # If we haven't saved `top_k`` checkpoints yet, just add this one.
            # Store the actual positive metric value.
            heapq.heappush(
                best_val_results[metric],
                (current_metric_value, current_checkpoint_path),
            )

            # Save the model state and other relevant information
            torch.save(
                state_dict_to_save,
                current_checkpoint_path,
            )
            logger.info(
                f'Saved checkpoint for {metric}: {abs(current_metric_value):.4f} to {current_checkpoint_path}'
            )
        else:
            # If we already have args.save_best_k checkpoints, check if the current one is better than the worst of them.
            # The worst of the k is at the top of the min-heap (best_val_results[metric][0]).
            worst_of_k_value = best_val_results[metric][0][
                0
            ]  # This correctly gets the smallest (worst) value in the heap

            if current_metric_value > worst_of_k_value:
                # Current checkpoint is better, so replace the worst one in the heap
                # heapq.heapreplace pops the smallest item and then pushes the new item
                old_worst_checkpoint_tuple = heapq.heapreplace(
                    best_val_results[metric],
                    (current_metric_value, current_checkpoint_path),
                )
                old_worst_path = old_worst_checkpoint_tuple[1]

                # Delete the old worst checkpoint file from disk
                if os.path.exists(old_worst_path):
                    os.remove(old_worst_path)
                    print(f'Deleted old worst checkpoint: {old_worst_path}')

                # Save the new better checkpoint
                torch.save(
                    state_dict_to_save,
                    current_checkpoint_path,
                )
                logger.info(
                    f'Replaced checkpoint for {metric}: {abs(current_metric_value):.4f} '
                    f'(old worst: {abs(worst_of_k_value):.4f}) to {current_checkpoint_path}',
                )


def maybe_log_eval_results(
    eval_results: EvalResults,
    epoch: int,
    prefix: str = 'val',
    class_names: list[str] | None = None,
    wandb_run: WandbRun | None = None,
    wandb_log_step: int | None = None,
) -> None:
    if wandb_run is None:
        return

    assert wandb_log_step is not None
    num_classes = len(eval_results['per_class_accuracy'])
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]

    log_data = {
        f'{prefix}/loss': eval_results['loss'],
        f'{prefix}/accuracy': eval_results['accuracy'],
        f'{prefix}/precision': eval_results['precision'],
        f'{prefix}/recall': eval_results['recall'],
        f'{prefix}/f1': eval_results['f1'],
        'epoch': epoch + 1,
    }
    for i, class_name in enumerate(class_names):
        log_data[f'{prefix}/{class_name}_accuracy'] = eval_results['per_class_accuracy'][i]
        # TODO: we might want to log per class precision, recall, and f1 here also,
        # but currently I think it is a bit messy.

    wandb_run.log(log_data, step=wandb_log_step)


def print_eval_results(
    eval_results: EvalResults,
    prefix: str,  # either 'val' or 'test'
    epoch: int | None = None,
) -> None:
    if epoch is None:
        print(f'{prefix} results: ', end='')
    else:
        print(f'{prefix} results on epoch {epoch}: ', end='')

    print(
        f'{prefix}_loss {eval_results["loss"]:0.4f} | '
        f'{prefix}_acc {eval_results["accuracy"]:0.4f} | '
        f'{prefix}_precision {eval_results["precision"]:0.4f} | '
        f'{prefix}_recall {eval_results["recall"]:0.4f} | '
        f'{prefix}_f1 {eval_results["f1"]:0.4f}'
    )


def select_samples_for_co_teaching(
    logits_1: Tensor, logits_2: Tensor, labels: Tensor, forget_rate: float
) -> tuple[Tensor, Tensor]:
    """
    Select samples from the two models based on the cross-entropy loss for Co-Teaching.
    """
    loss_1 = Fun.cross_entropy(input=logits_1, target=labels, reduction='none')
    indices_1 = torch.argsort(loss_1)
    loss_2 = Fun.cross_entropy(input=logits_2, target=labels, reduction='none')
    indices_2 = torch.argsort(loss_2)

    num_remembers = int((1.0 - forget_rate) * len(indices_1))

    selected_indices_1 = indices_1[:num_remembers]
    selected_indices_2 = indices_2[:num_remembers]

    return selected_indices_1, selected_indices_2
