import argparse

import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
import wandb
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support
from timm.data.config import resolve_data_config
from timm.data.dataset_factory import create_dataset
from timm.data.transforms_factory import create_transform
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from tqdm.autonotebook import tqdm

import soups.opts as opts


def train_model(args: argparse.Namespace) -> None:
    # TODO: set random seed

    NUM_CLASSES = 17  # TODO: infer number of classes from train_dataset

    if args.model == 'resnet50' or args.model == 'densenet121':
        if args.model == 'resnet50':
            model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
            )
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        else:
            model = torchvision.models.densenet121(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
            )
            model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        eval_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.model.startswith('coatnet'):
        model = timm.create_model(args.model, pretrained=True, num_classes=NUM_CLASSES)
        # use default transforms from coatnet model
        # TODO: consider changing it for fair comparison with other models
        data_config = resolve_data_config(model=model)
        train_transforms = create_transform(**data_config, is_training=True)
        eval_transforms = create_transform(**data_config, is_training=False)
        # train_transforms = Compose(
        #     RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=bicubic)
        #     RandomHorizontalFlip(p=0.5)
        #     ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=None)
        #     MaybeToTensor()
        #     Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
        # )
        # eval_transforms = Compose(
        #     Resize(size=235, interpolation=bicubic, max_size=None, antialias=True)
        #     CenterCrop(size=(224, 224))
        #     MaybeToTensor()
        #     Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
        # )
    else:
        raise ValueError(f'Unsupported model: {args.model}')

    # load datasets from the specified directory
    train_dataset = create_dataset(
        name='42',
        root=args.dataset_dir,
        split='train',
        transform=train_transforms,
        is_training=True,
    )
    test_dataset = create_dataset(
        name='42',
        root=args.dataset_dir,
        split='test',
        transform=eval_transforms,
        is_training=False,
    )
    val_dataset = create_dataset(
        name='42',
        root=args.dataset_dir,
        split='val',
        transform=eval_transforms,
        is_training=False,
    )
    logger.info(
        f'Train_size = {len(train_dataset)}, '
        f'test_size = {len(test_dataset)}, '
        f'val_size = {len(val_dataset)}'
    )

    # CutMiX & MixUp
    if args.use_mixup_cutmix:
        logger.info('MixUp & CutMix enabled')
        cutmix = v2.CutMix(alpha=1.0, num_classes=NUM_CLASSES)
        mixup = v2.MixUp(alpha=1.0, num_classes=NUM_CLASSES)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = v2.Identity()
    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate((batch)))

    # create data loaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # logging to wandb
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

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    model.to(device)

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        train_iter = tqdm(train_data_loader, desc=f'Training epoch {epoch + 1}/{args.num_epochs}')
        for images, labels in train_iter:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if wandb_run is not None:
                wandb_run.log({
                    'train/loss': loss.item(),
                }, step=global_step)

            train_iter.set_postfix({
                'loss': f'{loss:0.4f}'
            })
            global_step += 1

        # validation
        model.eval()
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

                outputs = model(images)
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
    # TODO: integrate EMA (Exponential Moving Average) for model weights

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
