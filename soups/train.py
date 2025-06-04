import argparse

import timm
import torch
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support
from timm.data.config import resolve_data_config
from timm.data.dataset import ImageDataset
from timm.data.dataset_factory import create_dataset
from timm.data.transforms_factory import create_transform
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm


def train_model(args: argparse.Namespace) -> None:
    model = timm.create_model(args.model, pretrained=True, num_classes=12)  # TODO: stop hardcoding
    data_config = resolve_data_config(model=model)
    transforms = create_transform(**data_config, is_training=True)

    # load datasets from the specified directory
    train_dataset = create_dataset(
        name='42',
        root=args.dataset_dir,
        split='train',
        transform=transforms,
        is_training=True,
    )
    test_dataset = create_dataset(
        name='42',
        root=args.dataset_dir,
        split='test',
        transform=transforms,
        is_training=False,
    )
    val_dataset = create_dataset(
        name='42',
        root=args.dataset_dir,
        split='val',
        transform=transforms,
        is_training=False,
    )
    logger.info(f'Train size = {len(train_dataset)}, test size = {len(test_dataset)}, val size = {len(val_dataset)}')

    # create a data loader
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = LabelSmoothingCrossEntropy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    model.to(device)

    for epoch in range(args.num_epochs):
        model.train()
        train_iter = tqdm(train_data_loader, desc=f'Training epoch {epoch + 1}/{args.num_epochs + 1}')
        num_corrects = 0
        num_totals = 0
        all_preds = []
        all_labels = []
        for images, labels in train_iter:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            num_totals += labels.shape[0]
            num_corrects += (preds == labels).sum().item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            train_iter.set_postfix({
                'loss': f'{loss:0.4f}'
            })
        train_acc = num_corrects / num_totals
        print(f'Epoch {epoch + 1}: train acc {train_acc}')

        model.eval()
        val_iter = tqdm(val_data_loader, desc=f'Validating epoch {epoch + 1}')
        val_loss = 0.0
        num_corrects = 0
        num_totals = 0
        with torch.no_grad():
            for images, labels in val_iter:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)
                loss = criterion(outputs, labels)

                num_totals += labels.shape[0]
                num_corrects += (preds == labels).sum().item()
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
            print(f'Epoch {epoch + 1}: val_loss {val_loss:0.4f} | val_acc {val_acc:0.4f} | val_precision {val_precision:0.4f} | val_recall {val_recall:0.4f} | val_f1 {val_f1:0.4f}')

def main():
    parser = argparse.ArgumentParser(description='Train a model using timm')
    parser.add_argument(
        '--model',
        type=str,
        help='Name of the model to use',
        default='coatnet_0_rw_224.sw_in1k',
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='Path to the dataset',
        default='./data/vietnamese_cultural_dataset',
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        help='Training batch size',
        default=32,
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        help='Evaluation batch size',
        default=32,
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=1.0e-4,
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of training epochs',
        default=10,
    )
    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()
