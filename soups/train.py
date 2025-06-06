import argparse
import os
import random

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


def train_model(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logger.info(f'Seed: {args.seed}')

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
    model.to(device)
    model_ema = ModelEmaV3(
        model=model,
        device=device,
        decay=args.model_ema_decay,
        use_warmup=args.model_ema_warmup,
    )

    global_step = 0
    num_train_iters = len(train_data_loader)
    for epoch in range(args.num_epochs):
        model.train()
        train_iter = tqdm(train_data_loader, desc=f'Training epoch {epoch + 1}/{args.num_epochs}')
        for i, (images, labels) in enumerate(train_iter):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            model_ema.update(model)

            if wandb_run is not None:
                log_data = {
                    f'learning_rate/group_{group_id}': group_lr
                    for group_id, group_lr in enumerate(scheduler.get_last_lr())
                }
                log_data['train/loss'] = loss.item()
                wandb_run.log(log_data, step=global_step)

            scheduler.step(epoch + i / num_train_iters)  # pyright: ignore[reportArgumentType]

            train_iter.set_postfix({
                'loss': f'{loss:0.4f}'
            })
            global_step += 1

        # validation
        ema_model = model_ema.module
        ema_model.eval()

        val_iter = tqdm(val_data_loader, desc=f'Validating epoch {epoch + 1}')
        val_loss = 0.0
        num_corrects = 0
        num_totals = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_iter:
                images = images.to(device)
                labels = labels.to(device)

                outputs = ema_model(images)
                preds = outputs.argmax(dim=1)
                loss = criterion(outputs, labels)

                num_totals += labels.shape[0]
                num_corrects += (preds == labels).sum().item()
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                val_loss += loss.item()
                val_iter.set_postfix({
                    'loss': f'{loss:0.4f}',
                })

            val_loss /= len(val_iter)
            val_acc = num_corrects / num_totals
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                all_labels,
                all_preds,
                average='macro',
                zero_division=0,
            )
            print(
                f'Epoch {epoch + 1}: val_loss {val_loss:0.4f} | '
                f'val_acc {val_acc:0.4f} | '
                f'val_precision {val_precision:0.4f} | '
                f'val_recall {val_recall:0.4f} | '
                f'val_f1 {val_f1:0.4f}'
            )
            if wandb_run is not None:
                wandb_run.log({
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'val/precision': val_precision,
                    'val/recall': val_recall,
                    'val/f1': val_f1,
                }, step=global_step)

    # TODO: save model checkpoints
    # TODO: test model

def main():
    parser = argparse.ArgumentParser(
        description='Train a model using timm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opts.add_general_opts(parser)
    opts.add_training_opts(parser)
    opts.add_wandb_opts(parser)
    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()
