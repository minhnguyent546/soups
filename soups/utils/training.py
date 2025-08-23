from contextlib import nullcontext
from typing import TypedDict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import torchvision
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
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
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]

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
            model_name[len('timm/'):],
            pretrained=pretrained,
            num_classes=num_classes,
        )
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            model_classifier = model.head
        elif hasattr(model, 'head') and hasattr(model.head, 'fc') and isinstance(model.head.fc, nn.Linear):
            model_classifier = model.head.fc
        else:
            raise ValueError(f'Unable to determine classification head for model {model_name}')
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    # initializing classification head
    nn.init.xavier_uniform_(model_classifier.weight)
    if model_classifier.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
        nn.init.zeros_(model_classifier.bias)

    return model

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
        f'{prefix}/loss': eval_results["loss"],
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
