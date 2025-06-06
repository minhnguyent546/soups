import argparse
import os
from datetime import datetime
from typing import TypedDict

import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
import wandb
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support
from timm.utils.model_ema import ModelEmaV3
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from tqdm.autonotebook import tqdm

import soups.opts as opts
from soups.utils import save_metadata_to_checkpoint, set_seed
from soups.utils.metric import AverageMeter


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    logger.info(f'Seed: {args.seed}')

    checkpoint_dir = os.path.join(
        args.checkpoints_dir,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # loading dataset
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    eval_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
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
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.dataset_dir, 'val'),
        transform=eval_transforms,
    )
    num_classes = len(train_dataset.classes)

    logger.info(
        f'num_classes = {num_classes}, '
        f'train_size = {len(train_dataset)}, '
        f'test_size = {len(test_dataset)}, '
        f'val_size = {len(val_dataset)}'
    )

    # CutMiX & MixUp
    if args.use_mixup_cutmix:
        logger.info('MixUp & CutMix enabled')
        cutmix = v2.CutMix(alpha=1.0, num_classes=num_classes)
        mixup = v2.MixUp(alpha=1.0, num_classes=num_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = v2.Identity()

    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate((batch)))

    # creating data loaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # creating model
    if args.model == 'resnet50' or args.model == 'densenet121':
        if args.model == 'resnet50':
            model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
            )
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            nn.init.xavier_uniform_(model.fc.weight)
            if model.fc.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(model.fc.bias)
        else:
            model = torchvision.models.densenet121(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
            )
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            nn.init.xavier_uniform_(model.classifier.weight)
            if model.classifier.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(model.classifier.bias)
    elif args.model.startswith('coatnet'):
        model = timm.create_model(
            args.model,
            pretrained=True,
            num_classes=num_classes,
        )
        nn.init.xavier_uniform_(model.head.fc.weight)
        if model.head.fc.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            nn.init.zeros_(model.head.fc.bias)
    else:
        raise ValueError(f'Unsupported model: {args.model}')

    model.to(device)
    if args.from_checkpoint is not None:
        logger.info(f'Loading model from checkpoint: {args.from_checkpoint}')
        checkpoint = torch.load(args.from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # setting up logging with wandb
    wandb_run = None
    if args.wandb_logging:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            id=args.wandb_resume_id,
            resume='must' if args.wandb_resume_id is not None else None,
        )
    save_metadata_to_checkpoint(
        checkpoint_dir=checkpoint_dir,
        args=args,
        wandb_run=wandb_run,
    )

    num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Using model: {args.model}')
    logger.info(f'Num_params: {num_model_params / 1e6:.2f}M')

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=args.scheduler_T_0,
        T_mult=args.scheduler_T_mult,
        eta_min=args.min_lr,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    if args.run_test_only:
        test_results = eval_model(
            model=model,
            eval_data_loader=test_data_loader,
            device=device,
            criterion=criterion,
        )
        print(
            '** Test results **\n'
            f'  Loss: {test_results["loss"]:0.4f}\n'
            f'  Accuracy: {test_results["accuracy"]:0.4f}\n'
            f'  Precision: {test_results["precision"]:0.4f}\n'
            f'  Recall: {test_results["recall"]:0.4f}\n'
            f'  F1: {test_results["f1"]:0.4f}\n'
        )
        return

    model_ema = None
    if args.use_ema:
        model_ema = ModelEmaV3(
            model=model,
            device=device,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
        )
        logger.info('EMA enabled')

    global_step = 0
    training_loss = AverageMeter(name='training_loss', fmt=':0.4f')
    num_train_iters = len(train_data_loader)
    for epoch in range(args.num_epochs):
        model.train()
        train_iter = tqdm(train_data_loader, desc=f'Training epoch {epoch + 1}/{args.num_epochs}')
        for i, (images, labels) in enumerate(train_iter):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if model_ema is not None:
                model_ema.update(model)

            if wandb_run is not None:
                log_data = {
                    f'learning_rate/group_{group_id}': group_lr
                    for group_id, group_lr in enumerate(scheduler.get_last_lr())
                }
                log_data['train/loss'] = loss.item()
                wandb_run.log(log_data, step=global_step)

            scheduler.step(epoch + i / num_train_iters)  # pyright: ignore[reportArgumentType]

            training_loss.update(loss.item(), labels.shape[0])
            train_iter.set_postfix({
                'loss': f'{loss:0.4f}'
            })
            global_step += 1

        # validation
        val_results = eval_model(
            model=model_ema.module if model_ema is not None else model,
            eval_data_loader=val_data_loader,
            device=device,
            criterion=eval_criterion,
        )
        print(
            f'Epoch {epoch + 1}: val_loss {val_results["loss"]:0.4f} | '
            f'val_acc {val_results["accuracy"]:0.4f} | '
            f'val_precision {val_results["precision"]:0.4f} | '
            f'val_recall {val_results["recall"]:0.4f} | '
            f'val_f1 {val_results["f1"]:0.4f}'
        )
        if wandb_run is not None:
            wandb_run.log({
                'val/loss': val_results["loss"],
                'val/accuracy': val_results['accuracy'],
                'val/precision': val_results['precision'],
                'val/recall': val_results['recall'],
                'val/f1': val_results['f1'],
                'train/epoch_loss': training_loss.avg,
            }, step=global_step)

        # saving checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'model_epoch_{epoch + 1}.pth',
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
        }, checkpoint_path)

    # TODO: acc with mixup&cutmix
    # TODO: acc@k

class EvalResults(TypedDict):
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float

def eval_model(
    model: nn.Module,
    eval_data_loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument]
    device: torch.device,
    criterion,
) -> EvalResults:
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

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            eval_loss.update(loss.item(), labels.shape[0])

            num_corrects = (preds == labels).sum().item()
            cur_accuracy = num_corrects / labels.shape[0]
            eval_accuracy.update(cur_accuracy, labels.shape[0])

            eval_iter.set_postfix({
                'loss': f'{loss:0.4f}',
            })

    model.train(model_mode_before)

    eval_precision, eval_recall, eval_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )

    return {
        'loss': eval_loss.avg,
        'accuracy': eval_accuracy.avg,
        'precision': float(eval_precision),
        'recall': float(eval_recall),
        'f1': float(eval_f1),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Training model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opts.add_general_opts(parser)
    opts.add_training_opts(parser)
    opts.add_wandb_opts(parser)
    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()
