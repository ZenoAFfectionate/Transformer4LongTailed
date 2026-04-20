"""Stage1 training entry for Transformer4LongTailed framework.

Usage::

    python train_stage1.py --cfg config/ViT/Base/CIFAR10_balance.yaml \
        model_name ViT num_classes 10 data_path /home/kemove/data

Features:
* YAML + CLI ``opts`` config merge via ``utils.update_config``.
* AdamW + linear warmup + cosine LR schedule.
* AMP (``torch.cuda.amp``) mixed precision when available.
* Mixup / ReMix data augmentation (toggle via cfg).
* MoE auxiliary-loss accumulation for ``model_name in {'MoE4ViT','MoE4SwT'}``.
* Top-1/Top-5, head/medium/tail group accuracy, ECE.
* Best-only checkpoint (``ckps/ckp_best.pth.tar``) — non-best epochs skip IO.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from dataset.cifar10 import CIFAR10_LT
from dataset.cifar100 import CIFAR100_LT
from dataset.fashion_mnist import FashionMNIST_LT
from utils import (
    AverageMeter,
    ProgressMeter,
    accuracy,
    build_loss,
    build_model,
    calibration,
    config,
    create_logger,
    forward_logits,
    load_checkpoint,
    log_experiment_details,
    mixup_criterion,
    mixup_data,
    remix_data,
    save_checkpoint,
    update_config,
    _to_builtin,
)


# ------------------------------------------------------------------
# Arg parsing / boilerplate
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='MoE4ViT stage1 training')
    parser.add_argument('--cfg', required=True, type=str, help='experiment YAML')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='override cfg via KEY VAL pairs')
    args = parser.parse_args()
    update_config(config, args)
    return args


def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_optimizer(model, cfg):
    betas_raw = cfg.betas
    if isinstance(betas_raw, (tuple, list)):
        betas = tuple(float(b) for b in betas_raw)
    else:
        betas = tuple(float(b.strip()) for b in str(betas_raw).split(','))
    if len(betas) != 2:
        raise ValueError(f'cfg.betas must resolve to exactly 2 values, got: {betas_raw!r}')
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        betas=betas,
        weight_decay=float(cfg.weight_decay),
    )


def build_scheduler(optimizer, cfg):
    """Linear warmup for ``warmup_epochs``, then cosine to ``n_epochs``."""
    warmup = int(cfg.warmup_epochs)
    total = int(cfg.n_epochs)

    def lr_lambda(epoch):
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        if total <= warmup:
            return 1.0
        progress = (epoch - warmup) / float(total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_dataset(cfg):
    name = str(cfg.dataset).lower()
    if cfg.longtail:
        imb_type = str(cfg.imb_type)
        imb_factor = float(cfg.imb_factor)
    else:
        imb_type = 'none'
        imb_factor = 1.0

    kwargs = dict(
        distributed=False,
        root=str(cfg.data_path),
        imb_type=imb_type,
        imb_factor=imb_factor,
        batch_size=int(cfg.batch_size),
        num_works=int(cfg.workers),
        test_imb_factor=cfg.test_imb_factor,
    )

    if name == 'cifar10':
        return CIFAR10_LT(**kwargs)
    if name == 'cifar100':
        return CIFAR100_LT(**kwargs)
    if name in ('fashionmnist', 'fashion_mnist'):
        return FashionMNIST_LT(**kwargs)
    raise ValueError(f'Unsupported cfg.dataset: {cfg.dataset!r} '
                     '(stage1 supports CIFAR10 / CIFAR100 / FashionMNIST)')


def resolve_group_indices(cfg):
    """Return ``(head_slice, med_slice, tail_slice)`` with safe clipping."""
    num_classes = int(cfg.num_classes)

    def _clip(pair):
        lo, hi = int(pair[0]), int(pair[1])
        lo = max(0, min(lo, num_classes))
        hi = max(lo, min(hi, num_classes))
        return (lo, hi)

    return _clip(cfg.head_class_idx), _clip(cfg.med_class_idx), _clip(cfg.tail_class_idx)


def safe_slice_mean(tensor, start, end):
    if end <= start:
        return torch.tensor(0.0, device=tensor.device)
    return tensor[start:end].mean()


# ------------------------------------------------------------------
# Train / Validate
# ------------------------------------------------------------------

def train_one_epoch(train_loader, model, criterion, optimizer, scaler,
                    epoch, cfg, logger, cls_num_list, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    aux_meter = AverageMeter('Aux', ':.4f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, aux_meter],
        prefix=f'Epoch: [{epoch + 1}]',
    )

    model.train()
    amp_enabled = scaler is not None and scaler.is_enabled()
    aux_weight = float(getattr(cfg, 'moe_aux_weight', 1.0))
    model_is_moe = str(cfg.model_name) in {'MoE4ViT', 'MoE4SwT'}

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        aux_loss_value = torch.tensor(0.0, device=device)

        with autocast(device_type='cuda', enabled=amp_enabled):
            if bool(cfg.mixup):
                mixed, targets_a, targets_b, lam = mixup_data(
                    images, target, alpha=float(cfg.alpha), use_cuda=(device.type == 'cuda'))
                output = forward_logits(model, mixed)
                loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
            elif bool(cfg.remix):
                mixed, targets_a, targets_b, _, lam_y = remix_data(
                    images, target, class_counts=cls_num_list, alpha=float(cfg.alpha))
                output = forward_logits(model, mixed)
                ce_none_a = F.cross_entropy(output, targets_a, reduction='none')
                ce_none_b = F.cross_entropy(output, targets_b, reduction='none')
                lam_y = lam_y.to(ce_none_a.dtype).to(ce_none_a.device)
                loss = (lam_y * ce_none_a + (1.0 - lam_y) * ce_none_b).mean()
            else:
                output = forward_logits(model, images)
                loss = criterion(output, target)

            if model_is_moe:
                target_module = model.module if hasattr(model, 'module') else model
                if hasattr(target_module, 'get_auxiliary_loss'):
                    aux = target_module.get_auxiliary_loss()
                    if torch.is_tensor(aux):
                        aux_loss_value = aux
                        loss = loss + aux_weight * aux_loss_value

        acc1, acc5 = accuracy(output.detach().float(), target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
        aux_meter.update(
            float(aux_loss_value.detach().item()) if torch.is_tensor(aux_loss_value) else 0.0,
            images.size(0),
        )

        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # DeepSeek-V3 aux-loss-free balancing: nudge expert biases toward
        # uniform load after every optimizer step. Cheap (O(num_experts)),
        # no grad, and reads ``expert_load`` buffers populated by the last
        # forward, so no extra compute beyond one sign() per gate.
        if model_is_moe:
            target_module = model.module if hasattr(model, 'module') else model
            if hasattr(target_module, 'update_expert_bias'):
                target_module.update_expert_bias()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % int(cfg.print_freq) == 0:
            progress.display(i, logger)

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, cfg, logger, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Eval: ',
    )

    model.eval()
    num_classes = int(cfg.num_classes)
    class_num = torch.zeros(num_classes, device=device)
    correct = torch.zeros(num_classes, device=device)

    conf_chunks, pred_chunks, true_chunks = [], [], []

    amp_eval = bool(cfg.use_amp) and device.type == 'cuda'

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=amp_eval):
                output = forward_logits(model, images)
                loss = criterion(output, target)

            output = output.float()
            if torch.is_tensor(loss) and loss.ndim > 0:
                loss_val = float(loss.mean().item())
            else:
                loss_val = float(loss.item())

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss_val, images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            _, predicted = output.max(1)
            class_num += torch.bincount(target, minlength=num_classes).to(class_num.dtype)
            correct_mask = predicted == target
            if correct_mask.any():
                correct += torch.bincount(target[correct_mask], minlength=num_classes).to(correct.dtype)

            prob = torch.softmax(output, dim=1)
            conf_part, pred_part = torch.max(prob, dim=1)
            conf_chunks.append(conf_part.cpu().numpy())
            pred_chunks.append(pred_part.cpu().numpy())
            true_chunks.append(target.cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            if i % int(cfg.print_freq) == 0:
                progress.display(i, logger)

    confidence = np.concatenate(conf_chunks) if conf_chunks else np.array([])
    pred_class = np.concatenate(pred_chunks) if pred_chunks else np.array([])
    true_class = np.concatenate(true_chunks) if true_chunks else np.array([])

    acc_classes = correct / class_num.clamp(min=1)
    head_slice, med_slice, tail_slice = resolve_group_indices(cfg)
    head_acc = float(safe_slice_mean(acc_classes, *head_slice).item() * 100)
    med_acc = float(safe_slice_mean(acc_classes, *med_slice).item() * 100)
    tail_acc = float(safe_slice_mean(acc_classes, *tail_slice).item() * 100)

    cal = calibration(true_class, pred_class, confidence, num_bins=15)
    raw_ece = cal['expected_calibration_error']
    ece = float(raw_ece) * 100 if raw_ece == raw_ece else float('nan')

    logger.info(
        '* Acc@1 %.3f%% Acc@5 %.3f%% HAcc %.3f%% MAcc %.3f%% TAcc %.3f%% ECE %.3f%%',
        top1.avg, top5.avg, head_acc, med_acc, tail_acc, ece,
    )

    return top1.avg, top5.avg, losses.avg, head_acc, med_acc, tail_acc, ece


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg)
    log_experiment_details(logger, args, config, args.cfg, model_dir)

    set_seed(int(config.seed), bool(config.deterministic))
    cuda_available = torch.cuda.is_available()
    if cuda_available and config.gpu is not None:
        torch.cuda.set_device(int(config.gpu))
        device = torch.device(f'cuda:{int(config.gpu)}')
    elif cuda_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info('Using device: %s', device)

    # --- data ---
    dataset = build_dataset(config)
    train_loader = dataset.train_instance
    val_loader = dataset.eval
    cls_num_list = dataset.cls_num_list
    logger.info(
        'Dataset %s | train samples: %d | val samples: %d | cls_num_list: %s',
        config.dataset, len(train_loader.dataset), len(val_loader.dataset),
        cls_num_list,
    )

    # --- model / loss / optim ---
    model = build_model(config).to(device)
    criterion = build_loss(config, cls_num_list, device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler('cuda', enabled=bool(config.use_amp) and cuda_available)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        'Model %s | params: %.2fM | loss: %s | optim: AdamW(lr=%.3e, wd=%.3e)',
        config.model_name, n_params / 1e6, config.loss_type,
        float(config.lr), float(config.weight_decay),
    )

    # --- resume ---
    start_epoch = 0
    best_acc1 = 0.0
    its_ece = float('inf')
    if config.resume:
        if os.path.isfile(str(config.resume)):
            logger.info('=> loading checkpoint %s', config.resume)
            ckpt = load_checkpoint(
                str(config.resume),
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                map_location=str(device),
            )
            start_epoch = int(ckpt.get('epoch', 0))
            best_acc1 = float(ckpt.get('best_acc1', 0.0))
            its_ece = float(ckpt.get('its_ece', float('inf')))
            logger.info('=> resumed from epoch %d (best_acc1=%.3f)', start_epoch, best_acc1)
        else:
            logger.info('=> no checkpoint found at %s', config.resume)

    # --- epoch loop ---
    for epoch in range(start_epoch, int(config.n_epochs)):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info('Epoch [%d/%d] lr=%.6f', epoch + 1, int(config.n_epochs), current_lr)

        train_loss, train_acc1, train_acc5 = train_one_epoch(
            train_loader, model, criterion, optimizer, scaler,
            epoch, config, logger, cls_num_list, device,
        )

        val_acc1, val_acc5, val_loss, head_acc, med_acc, tail_acc, ece = validate(
            val_loader, model, criterion, config, logger, device,
        )

        scheduler.step()

        is_best = val_acc1 > best_acc1
        if is_best:
            best_acc1 = val_acc1
            its_ece = ece

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_model': model.state_dict(),
            'state_dict_classifier': None,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'best_acc1': best_acc1,
            'its_ece': its_ece,
            'cls_num_list': list(cls_num_list),
            'cfg': _to_builtin(config),
        }, is_best, model_dir)

        logger.info(
            'Epoch [%d/%d] | train_loss %.3f train_acc1 %.3f | val_acc1 %.3f val_acc5 %.3f ECE %.3f | '
            'best %.3f (ECE %.3f)\n',
            epoch + 1, int(config.n_epochs),
            train_loss, train_acc1, val_acc1, val_acc5, ece, best_acc1, its_ece,
        )

    logger.info('Final best Acc@1: %.3f%% (ECE %.3f%%)', best_acc1, its_ece)


if __name__ == '__main__':
    main()
