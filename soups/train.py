import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as Fun
import torchvision
import torchvision.transforms.v2 as v2
import wandb
from timm.utils.model_ema import ModelEmaV3
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler, default_collate
from tqdm.autonotebook import tqdm

import soups.utils as utils
from soups.opts import add_training_opts
from soups.utils.logger import init_logger, logger
from soups.utils.metric import AverageMeter
from soups.utils.training import (
    EarlyStopping,
    eval_model,
    make_model,
    maybe_log_eval_results,
    print_eval_results,
    save_top_k_checkpoints,
)


def train_model(args: argparse.Namespace) -> None:
    init_logger()
    utils.set_seed(args.seed)
    logger.info(f'Seed: {args.seed}')

    checkpoint_dir = None
    if not args.run_test_only:
        checkpoint_dir = os.path.join(
            args.checkpoints_dir,
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

    # training device
    device = utils.get_device(args.device)
    logger.info(f'Using device: {device}')

    # loading dataset
    train_transforms = v2.Compose([
        v2.RandomResizedCrop(size=args.train_crop_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor(),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    eval_transforms = v2.Compose([
        v2.Resize(size=args.eval_resize_size),
        v2.CenterCrop(size=args.eval_crop_size),
        v2.ToTensor(),
        v2.Normalize(
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
    class_names = train_dataset.classes
    num_classes = len(class_names)

    logger.info(
        f'num_classes = {num_classes}, '
        f'train_size = {len(train_dataset)}, '
        f'test_size = {len(test_dataset)}, '
        f'val_size = {len(val_dataset)}'
    )

    # class weighting for train_dataset
    train_sampler = None
    if args.class_weighting:
        logger.info('Computing class weights for train_dataset')
        train_unique_labels, train_label_counts = np.unique(
            train_dataset.targets, return_counts=True
        )
        assert np.all(train_label_counts > 0), 'All classes must have at least one sample'
        train_weights = 1.0 / train_label_counts
        label_to_weight = dict(zip(train_unique_labels, train_weights, strict=True))
        train_samples_weights = [label_to_weight[label] for label in train_dataset.targets]

        train_sampler = WeightedRandomSampler(
            weights=train_samples_weights, num_samples=len(train_samples_weights), replacement=True
        )

    # CutMiX & MixUp
    if args.use_mixup_cutmix:
        logger.info('MixUp & CutMix enabled')
        cutmix = v2.CutMix(alpha=args.cutmix_alpha, num_classes=num_classes)
        mixup = v2.MixUp(alpha=args.mixup_alpha, num_classes=num_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = v2.Identity()

    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate((batch)))

    # creating data loaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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

    # mixed precision training
    mp_dtype = torch.float32
    if device.type == 'cuda' and args.mixed_precision == 'fp16':
        mp_dtype = torch.float16
    elif device.type == 'cuda' and args.mixed_precision == 'bf16':
        if torch.cuda.is_bf16_supported():
            mp_dtype = torch.bfloat16
        else:
            mp_dtype = torch.float16
    if mp_dtype != torch.float32:
        logger.info(f'Mixed precision training enabled with dtype {mp_dtype}')

    autocast_context = torch.autocast(
        device_type=device.type,
        dtype=mp_dtype,
        enabled=(mp_dtype in (torch.float16, torch.bfloat16)),
    )
    scaler = torch.amp.grad_scaler.GradScaler(
        device=device.type,
        enabled=(mp_dtype == torch.float16),
    )

    # creating model
    model = make_model(args.model, num_classes=num_classes, pretrained=not args.random_weights)
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
        wandb_run.define_metric(name='val/*', step_metric='epoch')
        wandb_run.define_metric(name='test/*', step_metric='epoch')
        wandb_run.define_metric(name='train/epoch_loss', step_metric='epoch')
    if not args.run_test_only:
        assert checkpoint_dir is not None
        utils.save_metadata_to_checkpoint(
            checkpoint_dir=checkpoint_dir,
            args=args,
            wandb_run=wandb_run,
        )

    num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Using model: {args.model}')
    logger.info(f'Num_params: {num_model_params / 1e6:.2f}M')

    if args.run_test_only:
        test_results = eval_model(
            model=model,
            eval_data_loader=test_data_loader,
            device=device,
            num_classes=num_classes,
        )
        print(
            '** Test results **\n'
            f'  Loss: {test_results["loss"]:0.4f}\n'
            f'  Accuracy: {test_results["accuracy"]:0.4f}\n'
            f'  Precision: {test_results["precision"]:0.4f}\n'
            f'  Recall: {test_results["recall"]:0.4f}\n'
            f'  F1: {test_results["f1"]:0.4f}\n'
        )
        print('  Per class results (acc | pre | recall | f1):')
        for i, class_name in enumerate(class_names):
            print(
                f'    {class_name}: {test_results["per_class_accuracy"][i]:0.4f} |'
                f'{test_results["per_class_precision"][i]:0.4f} |'
                f'{test_results["per_class_recall"][i]:0.4f} |'
                f'{test_results["per_class_f1"][i]:0.4f}'
            )

        return

    assert checkpoint_dir is not None

    model_ema = None
    if args.use_ema:
        model_ema = ModelEmaV3(
            model=model,
            device=device,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
        )
        logger.info('EMA enabled')

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_updates_per_epoch = (
        len(train_data_loader) + args.gradient_accum_steps - 1
    ) // args.gradient_accum_steps
    if args.scheduler == 'cosine_annealing':
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=args.cosine_annealing_T_0,
            T_mult=args.cosine_annealing_T_mult,
            eta_min=args.min_lr,
        )
    elif args.scheduler == 'one_cycle_lr':
        main_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=args.lr,
            pct_start=0.0,  # warmup is now can be used via a separate scheduler
            epochs=args.num_epochs - args.lr_warmup_epochs,
            steps_per_epoch=num_updates_per_epoch,
        )
    else:
        raise ValueError(f'Unsupported scheduler: {args.scheduler}')

    # learning rate warmup
    warmup_lr_scheduler = None
    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == 'linear':
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs * num_updates_per_epoch,
            )
        elif args.lr_warmup_method == 'constant':
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs * num_updates_per_epoch,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only `linear` and `constant` are supported."
            )

    global_step = 0
    training_loss = AverageMeter(name='training_loss', fmt=':0.4f')
    optimizer.zero_grad()
    if args.max_grad_norm > 0:
        logger.info(f'Using gradient clipping with max norm {args.max_grad_norm}')

    # results for each metric will be sorted in decreasing order
    if args.best_checkpoint_metrics is None:
        args.best_checkpoint_metrics = []

    # best_val_results[metric] = list of tuples (value, checkpoint_path)
    best_val_results: dict[str, list[tuple[float, str]]] = {
        metric: [] for metric in args.best_checkpoint_metrics
    }
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience, min_delta=0.0, enabled=args.early_stopping
    )
    if early_stopping.is_enabled():
        logger.info(f'Early stopping enabled with patience {early_stopping.patience}')

    for epoch in range(args.num_epochs):
        model.train()

        train_data_iter = iter(train_data_loader)
        total_num_samples = len(train_data_loader)
        last_iter_num_batches = total_num_samples % args.gradient_accum_steps
        if last_iter_num_batches == 0:
            last_iter_num_batches = args.gradient_accum_steps

        # determine the number of updates for the current epoch
        # based on gradient accumulation steps
        num_updates_per_epoch = (
            total_num_samples + args.gradient_accum_steps - 1
        ) // args.gradient_accum_steps  # ceil_div

        train_progressbar = tqdm(
            range(num_updates_per_epoch),
            desc=f'Training epoch {epoch + 1}/{args.num_epochs}',
        )

        for update_step in train_progressbar:
            num_batches = (
                args.gradient_accum_steps
                if update_step + 1 < num_updates_per_epoch
                else last_iter_num_batches
            )
            batches, num_items_in_batch = utils.get_batch_samples(
                data_iter=train_data_iter,
                num_batches=num_batches,
                labels_index=1,
            )
            assert num_items_in_batch is not None
            num_batches = len(batches)  # actual number batches retrieved

            batch_loss: float = 0.0
            for images, labels in batches:
                images = images.to(device)
                labels = labels.to(device)

                with autocast_context:
                    logits = model(images)
                    loss = Fun.cross_entropy(
                        input=logits,
                        target=labels,
                        reduction='sum',
                        label_smoothing=args.label_smoothing,
                    )
                    if num_items_in_batch > 0:
                        loss = loss / num_items_in_batch

                scaler.scale(loss).backward()
                batch_loss += loss.detach().item()

            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.max_grad_norm,
                    norm_type=2,
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if wandb_run is not None:
                log_data = {
                    f'learning_rate/group_{group_id}': param_group['lr']
                    for group_id, param_group in enumerate(optimizer.param_groups)
                }
                log_data['train/loss'] = batch_loss
                wandb_run.log(log_data, step=global_step)

            if (epoch <= args.lr_warmup_epochs - 1) and (warmup_lr_scheduler is not None):
                # still in the warmup phase
                warmup_lr_scheduler.step()
            else:
                if args.scheduler == 'cosine_annealing':
                    main_lr_scheduler.step(
                        epoch - args.lr_warmup_epochs + update_step / num_updates_per_epoch
                    )  # pyright: ignore[reportArgumentType]
                else:
                    main_lr_scheduler.step()

            training_loss.update(batch_loss, num_items_in_batch)
            train_progressbar.set_postfix({'loss': f'{batch_loss:0.4f}'})
            global_step += 1

        # validation
        val_results = eval_model(
            model=model_ema.module if model_ema is not None else model,
            eval_data_loader=val_data_loader,
            device=device,
            num_classes=num_classes,
        )
        print_eval_results(eval_results=val_results, prefix='val', epoch=epoch + 1)

        assert len(val_results['per_class_accuracy']) == num_classes
        maybe_log_eval_results(
            eval_results=val_results,
            epoch=epoch,
            prefix='val',
            class_names=class_names,
            wandb_run=wandb_run,
            wandb_log_step=global_step,
        )

        # testing
        test_results = eval_model(
            model=model_ema.module if model_ema is not None else model,
            eval_data_loader=test_data_loader,
            device=device,
            num_classes=num_classes,
        )
        print_eval_results(eval_results=test_results, prefix='test', epoch=epoch + 1)

        assert len(test_results['per_class_accuracy']) == num_classes
        maybe_log_eval_results(
            eval_results=test_results,
            epoch=epoch,
            prefix='test',
            class_names=class_names,
            wandb_run=wandb_run,
            wandb_log_step=global_step,
        )

        # saving checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'model_epoch_{epoch + 1}.pth',
        )
        state_dict_to_save = {
            'model_state_dict': model.state_dict(),
            'val_results': val_results,
            'epoch': epoch,
            'global_step': global_step,
        }
        torch.save(state_dict_to_save, checkpoint_path)
        save_top_k_checkpoints(
            criterion_metrics=args.best_checkpoint_metrics,
            top_k=args.save_best_k,
            val_results=val_results,
            best_val_results=best_val_results,
            state_dict_to_save=state_dict_to_save,
            checkpoint_path_template=os.path.join(
                checkpoint_dir, f'model_epoch_{epoch}_{{metric}}_{{metric_value:.4f}}.pth'
            ),
        )

        if early_stopping.early_stop(val_loss=val_results['loss']):
            logger.info(
                f'Early stopped. No improvement in validation loss for '
                f'{early_stopping.patience} consecutive epochs.'
            )
            break


def main():
    parser = argparse.ArgumentParser(
        description='Training model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_training_opts(parser)
    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()
