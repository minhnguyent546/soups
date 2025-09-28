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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, default_collate
from tqdm.autonotebook import tqdm

import soups.utils as utils
from soups.opts import add_training_with_co_teaching_opts
from soups.utils.logger import init_logger, logger
from soups.utils.metric import AverageMeter
from soups.utils.training import (
    eval_model,
    make_model,
    maybe_log_eval_results,
    print_eval_results,
    save_top_k_checkpoints,
    select_samples_for_co_teaching,
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
    scaler_1 = torch.amp.grad_scaler.GradScaler(
        device=device.type,
        enabled=(mp_dtype == torch.float16),
    )
    scaler_2 = torch.amp.grad_scaler.GradScaler(
        device=device.type,
        enabled=(mp_dtype == torch.float16),
    )

    # creating model
    model_1 = make_model(args.model, num_classes=num_classes, pretrained=not args.random_weights)
    model_2 = make_model(args.model, num_classes=num_classes, pretrained=not args.random_weights)
    model_1.to(device)
    model_2.to(device)

    if args.from_checkpoint is not None:
        logger.info(f'Loading model from checkpoint: {args.from_checkpoint}')
        checkpoint = torch.load(args.from_checkpoint, map_location=device)
        model_1.load_state_dict(checkpoint['model_state_dict'])

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

    num_model_params = sum(p.numel() for p in model_1.parameters() if p.requires_grad)
    logger.info(f'Using model: {args.model}')
    logger.info(f'Num_params: {num_model_params / 1e6:.2f}M')

    optimizer_1 = AdamW(
        model_1.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer_2 = AdamW(
        model_2.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler_1 = CosineAnnealingWarmRestarts(
        optimizer=optimizer_1,
        T_0=args.scheduler_T_0,
        T_mult=args.scheduler_T_mult,
        eta_min=args.min_lr,
    )
    scheduler_2 = CosineAnnealingWarmRestarts(
        optimizer=optimizer_2,
        T_0=args.scheduler_T_0,
        T_mult=args.scheduler_T_mult,
        eta_min=args.min_lr,
    )

    # define drop rate schedule
    rate_schedule = np.ones(args.num_epochs) * args.forget_rate
    rate_schedule[: args.num_gradual_epochs] = np.linspace(
        0, args.forget_rate**args.forget_rate_exponent, args.num_gradual_epochs
    )

    if args.run_test_only:
        test_results_1 = eval_model(
            model=model_1,
            eval_data_loader=test_data_loader,
            device=device,
            num_classes=num_classes,
        )
        print(
            '** Test results **\n'
            f'  Loss: {test_results_1["loss"]:0.4f}\n'
            f'  Accuracy: {test_results_1["accuracy"]:0.4f}\n'
            f'  Accuracy@5: {test_results_1["accuracy5"]:0.4f}\n'
            f'  Precision: {test_results_1["precision"]:0.4f}\n'
            f'  Recall: {test_results_1["recall"]:0.4f}\n'
            f'  F1: {test_results_1["f1"]:0.4f}\n'
        )
        print('  Per class results (acc | pre | recall | f1):')
        for i, class_name in enumerate(class_names):
            print(
                f'    {class_name}: {test_results_1["per_class_accuracy"][i]:0.4f} |'
                f'{test_results_1["per_class_precision"][i]:0.4f} |'
                f'{test_results_1["per_class_recall"][i]:0.4f} |'
                f'{test_results_1["per_class_f1"][i]:0.4f}'
            )

        return

    assert checkpoint_dir is not None

    model_ema_1 = None
    model_ema_2 = None
    if args.use_ema:
        model_ema_1 = ModelEmaV3(
            model=model_1,
            device=device,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
        )
        model_ema_2 = ModelEmaV3(
            model=model_2,
            device=device,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
        )
        logger.info('EMA enabled')

    global_step = 0
    training_loss_1 = AverageMeter(name='training_loss_1', fmt=':0.4f')
    training_loss_2 = AverageMeter(name='training_loss_2', fmt=':0.4f')
    optimizer_1.zero_grad()
    optimizer_2.zero_grad()
    if args.max_grad_norm > 0:
        logger.info(f'Using gradient clipping with max norm {args.max_grad_norm}')

    # results for each metric will be sorted in decreasing order
    if args.best_checkpoint_metrics is None:
        args.best_checkpoint_metrics = []

    # best_val_results[metric] = list of tuples (value, checkpoint_path)
    best_val_results_1: dict[str, list[tuple[float, str]]] = {
        metric: [] for metric in args.best_checkpoint_metrics
    }
    best_val_results_2: dict[str, list[tuple[float, str]]] = {
        metric: [] for metric in args.best_checkpoint_metrics
    }

    for epoch in range(args.num_epochs):
        model_1.train()
        model_2.train()

        train_data_iter = iter(train_data_loader)
        total_num_samples = len(train_data_loader)
        last_iter_num_batches = total_num_samples % args.gradient_accum_steps
        if last_iter_num_batches == 0:
            last_iter_num_batches = args.gradient_accum_steps

        # determine the number of updates for the current epoch
        # based on gradient accumulation steps
        total_updates = (
            total_num_samples + args.gradient_accum_steps - 1
        ) // args.gradient_accum_steps  # ceil_div

        train_progressbar = tqdm(
            range(total_updates),
            desc=f'Training epoch {epoch + 1}/{args.num_epochs}',
        )
        for update_step in train_progressbar:
            num_batches = (
                args.gradient_accum_steps
                if update_step + 1 < total_updates
                else last_iter_num_batches
            )
            batches, num_items_in_batch = utils.get_batch_samples(
                data_iter=train_data_iter,
                num_batches=num_batches,
                labels_index=1,
            )
            assert num_items_in_batch is not None
            num_batches = len(batches)  # actual number batches retrieved

            batch_loss_1: float = 0.0
            batch_loss_2: float = 0.0
            for images, labels in batches:
                images = images.to(device)
                labels = labels.to(device)

                with autocast_context:
                    logits_1 = model_1(images)
                    logits_2 = model_2(images)
                    selected_indices_1, selected_indices_2 = select_samples_for_co_teaching(
                        logits_1=logits_1,
                        logits_2=logits_2,
                        labels=labels,
                        forget_rate=rate_schedule[epoch],
                    )
                    loss_1 = Fun.cross_entropy(
                        input=logits_1[selected_indices_2],
                        target=labels[selected_indices_2],
                        reduction='sum',
                        label_smoothing=args.label_smoothing,
                    )
                    loss_2 = Fun.cross_entropy(
                        input=logits_2[selected_indices_1],
                        target=labels[selected_indices_1],
                        reduction='sum',
                        label_smoothing=args.label_smoothing,
                    )
                    if num_items_in_batch > 0:
                        loss_1 = loss_1 / num_items_in_batch
                        loss_2 = loss_2 / num_items_in_batch

                scaler_1.scale(loss_1).backward()
                scaler_2.scale(loss_2).backward()
                batch_loss_1 += loss_1.detach().item()
                batch_loss_2 += loss_2.detach().item()

            if args.max_grad_norm > 0:
                scaler_1.unscale_(optimizer_1)
                scaler_2.unscale_(optimizer_2)
                torch.nn.utils.clip_grad_norm_(
                    model_1.parameters(),
                    max_norm=args.max_grad_norm,
                    norm_type=2,
                )
                torch.nn.utils.clip_grad_norm_(
                    model_2.parameters(),
                    max_norm=args.max_grad_norm,
                    norm_type=2,
                )

            scaler_1.step(optimizer_1)
            scaler_2.step(optimizer_2)

            scaler_1.update()
            scaler_2.update()
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            if model_ema_1 is not None:
                model_ema_1.update(model_1)
            if model_ema_2 is not None:
                model_ema_2.update(model_2)

            if wandb_run is not None:
                log_data = {
                    f'learning_rate/group_{group_id}': group_lr
                    for group_id, group_lr in enumerate(scheduler_1.get_last_lr())
                }
                log_data['train/loss'] = batch_loss_1
                wandb_run.log(log_data, step=global_step)

            scheduler_1.step(epoch + update_step / total_updates)  # pyright: ignore[reportArgumentType]
            scheduler_2.step(epoch + update_step / total_updates)  # pyright: ignore[reportArgumentType]

            training_loss_1.update(batch_loss_1, num_items_in_batch)
            training_loss_2.update(batch_loss_2, num_items_in_batch)
            train_progressbar.set_postfix({
                'loss_1': f'{batch_loss_1:0.4f}',
                'loss_2': f'{batch_loss_2:0.4f}',
            })
            global_step += 1

        # validation
        val_results_1 = eval_model(
            model=model_ema_1.module if model_ema_1 is not None else model_1,
            eval_data_loader=val_data_loader,
            device=device,
            num_classes=num_classes,
        )
        val_results_2 = eval_model(
            model=model_ema_2.module if model_ema_2 is not None else model_2,
            eval_data_loader=val_data_loader,
            device=device,
            num_classes=num_classes,
        )

        print_eval_results(eval_results=val_results_1, prefix='val_1', epoch=epoch + 1)
        print_eval_results(eval_results=val_results_2, prefix='val_2', epoch=epoch + 1)

        assert len(val_results_1['per_class_accuracy']) == num_classes
        assert len(val_results_2['per_class_accuracy']) == num_classes
        maybe_log_eval_results(
            eval_results=val_results_1,
            epoch=epoch,
            prefix='val_1',
            class_names=class_names,
            wandb_run=wandb_run,
            wandb_log_step=global_step,
        )
        maybe_log_eval_results(
            eval_results=val_results_2,
            epoch=epoch,
            prefix='val_2',
            class_names=class_names,
            wandb_run=wandb_run,
            wandb_log_step=global_step,
        )

        # testing
        test_results_1 = eval_model(
            model=model_ema_1.module if model_ema_1 is not None else model_1,
            eval_data_loader=test_data_loader,
            device=device,
            num_classes=num_classes,
        )
        test_results_2 = eval_model(
            model=model_ema_2.module if model_ema_2 is not None else model_2,
            eval_data_loader=test_data_loader,
            device=device,
            num_classes=num_classes,
        )
        print_eval_results(eval_results=test_results_1, prefix='test_1', epoch=epoch + 1)
        print_eval_results(eval_results=test_results_2, prefix='test_2', epoch=epoch + 1)

        assert len(test_results_1['per_class_accuracy']) == num_classes
        assert len(test_results_2['per_class_accuracy']) == num_classes
        maybe_log_eval_results(
            eval_results=test_results_1,
            epoch=epoch,
            prefix='test_1',
            class_names=class_names,
            wandb_run=wandb_run,
            wandb_log_step=global_step,
        )
        maybe_log_eval_results(
            eval_results=test_results_2,
            epoch=epoch,
            prefix='test_2',
            class_names=class_names,
            wandb_run=wandb_run,
            wandb_log_step=global_step,
        )

        # saving checkpoint
        checkpoint_path_1 = os.path.join(
            checkpoint_dir,
            f'model_1_epoch_{epoch + 1}.pth',
        )
        checkpoint_path_2 = os.path.join(
            checkpoint_dir,
            f'model_2_epoch_{epoch + 1}.pth',
        )
        state_dict_to_save_1 = {
            'model_state_dict': model_1.state_dict(),
            'val_results': val_results_1,
            'epoch': epoch,
            'global_step': global_step,
        }
        state_dict_to_save_2 = {
            'model_state_dict': model_2.state_dict(),
            'val_results': val_results_2,
            'epoch': epoch,
            'global_step': global_step,
        }

        torch.save(state_dict_to_save_1, checkpoint_path_1)
        torch.save(state_dict_to_save_2, checkpoint_path_2)

        save_top_k_checkpoints(
            criterion_metrics=args.best_checkpoint_metrics,
            top_k=args.save_best_k,
            val_results=val_results_1,
            best_val_results=best_val_results_1,
            state_dict_to_save=state_dict_to_save_1,
            checkpoint_path_template=os.path.join(
                checkpoint_dir, f'model_1_epoch_{epoch}_{{metric}}_{{metric_value:.4f}}.pth'
            ),
        )

        save_top_k_checkpoints(
            criterion_metrics=args.best_checkpoint_metrics,
            top_k=args.save_best_k,
            val_results=val_results_2,
            best_val_results=best_val_results_2,
            state_dict_to_save=state_dict_to_save_2,
            checkpoint_path_template=os.path.join(
                checkpoint_dir, f'model_2_epoch_{epoch}_{{metric}}_{{metric_value:.4f}}.pth'
            ),
        )


def main():
    parser = argparse.ArgumentParser(
        description='Training model with Co-Teaching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_training_with_co_teaching_opts(parser)
    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()
