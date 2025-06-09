from contextlib import nullcontext
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as Fun
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from wandb.sdk.wandb_run import Run as WandbRun

from soups.utils.metric import AverageMeter


class EvalResults(TypedDict):
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    per_class_accuracy: list[float]

def eval_model(
    model: nn.Module,
    eval_data_loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
    device: torch.device,
    num_classes: int | None = None,
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

    eval_precision, eval_recall, eval_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    conf_matrix = confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        labels=list(range(num_classes)) if num_classes is not None else None,
    )
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    return {
        'loss': eval_loss.avg,
        'accuracy': eval_accuracy.avg,
        'precision': float(eval_precision),
        'recall': float(eval_recall),
        'f1': float(eval_f1),
        'per_class_accuracy': per_class_accuracy.tolist(),
    }

def maybe_log_eval_results(
    eval_results: EvalResults,
    epoch: int,
    prefix: str = 'val',
    class_names: list[str] | None = None,
    wandb_run: WandbRun | None = None,
) -> None:
    if wandb_run is None:
        return

    num_classes = len(eval_results['per_class_accuracy'])
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]

    log_data = {
        f'{prefix}/loss': eval_results["loss"],
        f'{prefix}/accuracy': eval_results['accuracy'],
        f'{prefix}/precision': eval_results['precision'],
        f'{prefix}/recall': eval_results['recall'],
        f'{prefix}/f1': eval_results['f1'],
        'epoch': epoch + 1,
    }
    for i, class_name in enumerate(class_names):
        log_data[f'{prefix}/{class_name}_accuracy'] = eval_results['per_class_accuracy'][i]

    wandb_run.log(log_data)
