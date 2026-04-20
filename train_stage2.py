"""Stage2 training entry — classifier re-training on frozen backbone features.

Backbone dispatch via ``cfg.model_name``: ``ViT`` / ``MoE4ViT`` (CLS-token
features, dim ``2 * emb_size``), ``SwT`` (mean-pooled multi-scale features,
dim ``10 * emb_size``), or ``MoE4SwT`` (CLS-token, dim ``2 * emb_size``).

Supported classifiers (``cfg.classifier``):

* ``linear`` / ``lws`` / ``lws_plus`` — PyTorch SGD feature retraining.
* ``adaptive_bls`` / ``bls`` — Adaptive Re-weighted BLS closed-form solve,
  with optional enhancement-node epochs.
* ``elm`` — Extreme Learning Machine closed-form solve.

Usage::

    python train_stage2.py --cfg config/ViT/Base/CIFAR10_balance.yaml \
        model_name ViT num_classes 10 data_path /home/kemove/data \
        classifier linear resume results/.../ckps/ckp_best.pth.tar
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.amp import autocast

from classifier.arbn import ARBN
from classifier.elm import ELM
from dataset.cifar10 import CIFAR10_LT
from dataset.cifar100 import CIFAR100_LT
from dataset.fashion_mnist import FashionMNIST_LT
from utils import (
    AverageMeter,
    ProgressMeter,
    accuracy,
    build_stage1_components_vit,
    config,
    create_logger,
    extract_features,
    log_experiment_details,
    save_checkpoint,
    update_config,
    wrap_lws_head,
    _to_builtin,
)
from utils.loss import LabelAwareSmoothing


GROUP_NAMES = ('head', 'medium', 'tail')
GROUP_RATIOS = (0.33, 0.33, 0.34)
ADAPTIVE_BLS_CLASSIFIERS = {'dcbls', 'adaptive_bls', 'bls'}
ELM_CLASSIFIERS = {'elm'}
FEATURE_RETRAIN_CLASSIFIERS = {'linear', 'fc', 'stage1_fc', 'ffn', 'lws', 'lws_plus'}
LWS_FAMILY_CLASSIFIERS = {'lws', 'lws_plus'}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='MoE4ViT stage2 classifier retraining')
    parser.add_argument('--cfg', required=True, type=str, help='experiment YAML')
    parser.add_argument('--seed', default=996, type=int, help='random seed override')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='override cfg via KEY VAL pairs')
    args = parser.parse_args()
    update_config(config, args)
    return args


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def uses_adaptive_bls(name): return name in ADAPTIVE_BLS_CLASSIFIERS
def uses_elm(name):          return name in ELM_CLASSIFIERS
def uses_sklearn(name):      return uses_adaptive_bls(name) or uses_elm(name)
def uses_feature_retrain(name): return name in FEATURE_RETRAIN_CLASSIFIERS
def uses_lws_family(name):   return name in LWS_FAMILY_CLASSIFIERS


def cfg_value(cfg, key, default):
    value = getattr(cfg, key, default)
    return default if value is None else value


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def extract_features_from_dataset(model, data_loader, cfg, device):
    """Run the backbone over ``data_loader`` and collect pre-classifier feats."""
    model.eval()
    use_amp = bool(cfg_value(cfg, 'use_amp', True)) and device.type == 'cuda'

    total = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else None
    feats_buf = None
    labels_buf = None
    feats_list, labels_list = [], []
    offset = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=use_amp):
                features = extract_features(model, images)

            features = features.float().cpu().numpy()
            targets_np = targets.cpu().numpy()

            if total is not None:
                if feats_buf is None:
                    feats_buf = np.empty((total, features.shape[1]), dtype=np.float32)
                    labels_buf = np.empty((total,), dtype=targets_np.dtype)
                end = offset + features.shape[0]
                feats_buf[offset:end] = features
                labels_buf[offset:end] = targets_np
                offset = end
            else:
                feats_list.append(features)
                labels_list.append(targets_np)

    if total is not None and feats_buf is not None:
        return feats_buf[:offset], labels_buf[:offset]
    if feats_list:
        return np.vstack(feats_list), np.concatenate(labels_list)
    return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)


# ------------------------------------------------------------------
# HMT partitioning
# ------------------------------------------------------------------

def allocate_hmt_group_sizes(num_classes):
    exact = [num_classes * r for r in GROUP_RATIOS]
    sizes = [int(np.floor(s)) for s in exact]
    remaining = num_classes - sum(sizes)
    if remaining > 0:
        order = sorted(range(len(GROUP_NAMES)),
                       key=lambda i: (-(exact[i] - sizes[i]), i))
        for i in order[:remaining]:
            sizes[i] += 1
    return dict(zip(GROUP_NAMES, sizes))


def build_hmt_partitions(cls_num_list):
    cls_num_array = np.asarray(cls_num_list, dtype=np.float64)
    num_classes = len(cls_num_array)
    sorted_ids = np.argsort(-cls_num_array, kind='mergesort')
    sizes = allocate_hmt_group_sizes(num_classes)

    partitions = {
        'num_classes': num_classes,
        'sorted_class_ids': sorted_ids,
        'sorted_class_counts': cls_num_array[sorted_ids],
    }
    offset = 0
    for group in GROUP_NAMES:
        nxt = offset + sizes[group]
        ids = sorted_ids[offset:nxt]
        partitions[f'{group}_classes'] = ids
        partitions[f'{group}_counts'] = cls_num_array[ids]
        offset = nxt
    return partitions


def log_hmt_partitions(logger, partitions):
    logger.info(
        'Stage2 HMT split: 33/33/34 by training frequency. total=%d head=%d med=%d tail=%d',
        partitions['num_classes'],
        len(partitions['head_classes']),
        len(partitions['medium_classes']),
        len(partitions['tail_classes']),
    )


# ------------------------------------------------------------------
# ECE / metrics for sklearn-style predictions
# ------------------------------------------------------------------

def calculate_ece(y_true, y_pred, confidence, num_bins=15):
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        prop = in_bin.mean()
        if prop > 0:
            acc_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
            conf_in_bin = confidence[in_bin].mean()
            ece += np.abs(conf_in_bin - acc_in_bin) * prop
    return float(ece * 100)


def summarize_predictions(y_true, y_pred, y_pred_proba, cls_num_list, partitions):
    num_classes = len(cls_num_list)
    val_accuracy = float(np.mean(y_pred == y_true) * 100)

    class_total = np.bincount(y_true.astype(np.int64), minlength=num_classes).astype(np.float64)
    class_correct = np.bincount(
        y_true[y_pred == y_true].astype(np.int64), minlength=num_classes
    ).astype(np.float64)
    class_acc = np.divide(
        class_correct, class_total,
        out=np.zeros(num_classes, dtype=np.float64),
        where=class_total > 0,
    ) * 100

    group_acc = {}
    for group in GROUP_NAMES:
        ids = partitions[f'{group}_classes']
        group_acc[group] = float(class_acc[ids].mean()) if len(ids) else 0.0

    confidence = np.max(y_pred_proba, axis=1)
    ece = calculate_ece(y_true, y_pred, confidence)

    return {
        'val_accuracy': val_accuracy,
        'ece': ece,
        'class_acc': class_acc,
        'group_acc': group_acc,
    }


def log_metrics(logger, metrics):
    g = metrics['group_acc']
    logger.info('  *Head Acc: %.2f%%', g['head'])
    logger.info('  *Med  Acc: %.2f%%', g['medium'])
    logger.info('  *Tail Acc: %.2f%%', g['tail'])
    logger.info('  *Val  Acc: %.2f%%', metrics['val_accuracy'])
    logger.info('  *ECE     : %.3f%%', metrics['ece'])


# ------------------------------------------------------------------
# Feature retrain (PyTorch)
# ------------------------------------------------------------------

def build_feature_loader(X, y, batch_size, shuffle):
    features = torch.from_numpy(np.asarray(X)).float()
    labels = torch.from_numpy(np.asarray(y).astype(np.int64))
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def evaluate_feature_classifier(classifier, loader, cls_num_list, partitions, device):
    classifier.eval()
    preds, probs_all, labels = [], [], []
    with torch.no_grad():
        for feats, target in loader:
            feats = feats.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = classifier(feats)
            probs = torch.softmax(logits, dim=1)
            preds.append(probs.argmax(dim=1).cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            labels.append(target.cpu().numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    y_proba = np.concatenate(probs_all, axis=0)
    return summarize_predictions(y_true, y_pred, y_proba, cls_num_list, partitions)


def train_feature_classifier_epoch(classifier, loader, criterion, optimizer, device,
                                   epoch, num_epochs, logger, print_freq):
    classifier.train()
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    progress = ProgressMeter(len(loader), [losses, top1],
                             prefix=f'Stage2 Epoch [{epoch + 1}/{num_epochs}] ')
    for step, (feats, target) in enumerate(loader):
        feats = feats.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = classifier(feats)
        loss = criterion(logits, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc1 = accuracy(logits, target, topk=(1,))[0]
        losses.update(loss.item(), feats.size(0))
        top1.update(float(acc1.item()), feats.size(0))
        if step % max(1, print_freq) == 0:
            progress.display(step, logger)
    return {'loss': float(losses.avg), 'acc1': float(top1.avg)}


def save_feature_classifier_checkpoint(state, is_best, model_dir):
    return save_checkpoint(state, is_best, model_dir, best_name='stage2_best.pth.tar')


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg)
    log_experiment_details(logger, args, config, args.cfg, model_dir)

    seed = int(args.seed)
    config.seed = seed
    set_seed(seed, bool(getattr(config, 'deterministic', True)))

    cuda_available = torch.cuda.is_available()
    if cuda_available and config.gpu is not None:
        torch.cuda.set_device(int(config.gpu))
        device = torch.device(f'cuda:{int(config.gpu)}')
    elif cuda_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info('Using device: %s', device)

    # --- backbone + linear classifier stub ---
    model, classifier, feat_dim = build_stage1_components_vit(config)
    model = model.to(device)
    classifier = classifier.to(device)
    num_classes = int(config.num_classes)

    classifier_name = str(config.classifier).lower()
    if not (uses_adaptive_bls(classifier_name)
            or uses_feature_retrain(classifier_name)
            or uses_elm(classifier_name)):
        raise ValueError(f'Unsupported cfg.classifier: {classifier_name!r}')
    logger.info('Backbone %s | feature dim %d | stage2 classifier %s',
                config.model_name, feat_dim, classifier_name)

    # --- load stage1 checkpoint ---
    if not config.resume or not os.path.isfile(str(config.resume)):
        raise FileNotFoundError(
            f'Stage2 requires cfg.resume pointing to a stage1 checkpoint. Got: {config.resume!r}'
        )
    logger.info('=> loading stage1 checkpoint %s', config.resume)
    ckpt = torch.load(str(config.resume), map_location=str(device), weights_only=False)
    state_dict_model = ckpt.get('state_dict_model', ckpt.get('state_dict'))
    if state_dict_model is None:
        raise KeyError('Stage1 checkpoint is missing state_dict_model/state_dict')
    model.load_state_dict(state_dict_model, strict=True)
    logger.info('=> stage1 backbone loaded')

    # --- dataset ---
    name = str(config.dataset).lower()
    imb_type = str(config.imb_type) if bool(config.longtail) else 'none'
    imb_factor = float(config.imb_factor) if bool(config.longtail) else 1.0
    ds_kwargs = dict(
        distributed=False, root=str(config.data_path),
        imb_type=imb_type, imb_factor=imb_factor,
        batch_size=int(config.batch_size), num_works=int(config.workers),
        test_imb_factor=config.test_imb_factor,
    )
    if name == 'cifar10':
        dataset = CIFAR10_LT(**ds_kwargs)
    elif name == 'cifar100':
        dataset = CIFAR100_LT(**ds_kwargs)
    elif name in ('fashionmnist', 'fashion_mnist'):
        dataset = FashionMNIST_LT(**ds_kwargs)
    else:
        raise ValueError(f'Unsupported cfg.dataset: {config.dataset}')

    if uses_sklearn(classifier_name):
        train_loader = dataset.train_instance
        loader_desc = 'instance distribution'
    else:
        train_loader = dataset.train_balance
        loader_desc = 'class-balanced distribution'
    val_loader = dataset.eval
    cls_num_list = dataset.cls_num_list

    # --- feature extraction ---
    logger.info('> Extracting features from train (%s) ...', loader_desc)
    X_train, y_train = extract_features_from_dataset(model, train_loader, config, device)
    logger.info('  train features: %s, labels: %s', X_train.shape, y_train.shape)

    logger.info('> Extracting features from val ...')
    X_val, y_val = extract_features_from_dataset(model, val_loader, config, device)
    logger.info('  val features: %s, labels: %s', X_val.shape, y_val.shape)

    _, counts = np.unique(y_train, return_counts=True)
    logger.info('Training feature distribution - min=%d max=%d total=%d',
                int(counts.min()), int(counts.max()), int(len(y_train)))
    logger.info('Original imbalance ratio: %.2f', max(cls_num_list) / min(cls_num_list))

    partitions = build_hmt_partitions(cls_num_list)
    log_hmt_partitions(logger, partitions)

    # ---------------- Feature-retrain classifier ----------------
    if uses_feature_retrain(classifier_name):
        classifier_head = classifier
        state_dict_cls = ckpt.get('state_dict_classifier')
        if state_dict_cls is not None:
            try:
                classifier_head.load_state_dict(state_dict_cls, strict=True)
                logger.info('=> loaded stage1 classifier state_dict into LinearClassifier')
            except RuntimeError as exc:
                logger.warning(
                    'state_dict_classifier shape mismatch (%s); keeping random init.', exc,
                )

        if uses_lws_family(classifier_name):
            classifier_head = wrap_lws_head(
                classifier_name, classifier_head, feat_dim, num_classes
            ).to(device)
            logger.info('Wrapped head into %s (base frozen, scale trained)', classifier_name)

        train_bs = int(cfg_value(config, 'batch_size', 128))
        val_bs = max(train_bs, 256)
        tr_fl = build_feature_loader(X_train, y_train, train_bs, shuffle=True)
        va_fl = build_feature_loader(X_val, y_val, val_bs, shuffle=False)

        smooth_head = getattr(config, 'smooth_head', None)
        smooth_tail = getattr(config, 'smooth_tail', None)
        if smooth_head is not None and smooth_tail is not None and bool(config.longtail):
            criterion = LabelAwareSmoothing(cls_num_list, float(smooth_head), float(smooth_tail))
            logger.info('Stage2 loss: LabelAwareSmoothing(head=%.3f, tail=%.3f)',
                        float(smooth_head), float(smooth_tail))
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info('Stage2 loss: plain CrossEntropy')
        criterion = criterion.to(device)

        stage2_lr = float(cfg_value(config, 'lr', 0.1)) * float(cfg_value(config, 'lr_factor', 1.0))
        optimizer = torch.optim.SGD(
            [p for p in classifier_head.parameters() if p.requires_grad],
            lr=stage2_lr,
            momentum=float(cfg_value(config, 'momentum', 0.9)),
            weight_decay=float(cfg_value(config, 'weight_decay', 0.0)),
            nesterov=True,
        )
        num_epochs = int(cfg_value(config, 'num_epochs', 30))
        milestones = sorted({
            e for e in (int(num_epochs * 0.6), int(num_epochs * 0.8))
            if 0 < e < num_epochs
        })
        scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
                     if milestones else None)
        logger.info('Stage2 optimizer: SGD(lr=%.4f, momentum=%.3f, wd=%.4e), milestones=%s',
                    stage2_lr, float(cfg_value(config, 'momentum', 0.9)),
                    float(cfg_value(config, 'weight_decay', 0.0)), milestones)

        logger.info('> Pre-retrain val evaluation ...')
        metrics = evaluate_feature_classifier(classifier_head, va_fl, cls_num_list, partitions, device)
        log_metrics(logger, metrics)

        best_acc1 = metrics['val_accuracy']
        best_ece = metrics['ece']
        best_group = dict(metrics['group_acc'])
        save_feature_classifier_checkpoint({
            'epoch': 0,
            'state_dict_model': model.state_dict(),
            'state_dict_classifier': classifier_head.state_dict(),
            'best_acc1': best_acc1, 'its_ece': best_ece,
            'best_group_acc': best_group,
            'stage2_classifier': classifier_name,
            'cfg': _to_builtin(config),
        }, True, model_dir)

        for epoch in range(num_epochs):
            logger.info('%s', '=' * 60)
            logger.info('Stage2 epoch %d/%d', epoch + 1, num_epochs)
            logger.info('%s', '=' * 60)

            stats = train_feature_classifier_epoch(
                classifier_head, tr_fl, criterion, optimizer, device,
                epoch, num_epochs, logger, int(cfg_value(config, 'print_freq', 20)),
            )
            logger.info('  train loss=%.4f acc@1=%.2f%%', stats['loss'], stats['acc1'])

            metrics = evaluate_feature_classifier(classifier_head, va_fl, cls_num_list, partitions, device)
            log_metrics(logger, metrics)

            is_best = metrics['val_accuracy'] > best_acc1
            if is_best:
                best_acc1 = metrics['val_accuracy']
                best_ece = metrics['ece']
                best_group = dict(metrics['group_acc'])
                save_feature_classifier_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                    'state_dict_classifier': classifier_head.state_dict(),
                    'best_acc1': best_acc1, 'its_ece': best_ece,
                    'best_group_acc': best_group,
                    'stage2_classifier': classifier_name,
                    'cfg': _to_builtin(config),
                }, True, model_dir)

            if scheduler is not None:
                scheduler.step()

            logger.info('Best Prec@1 %.3f%% ECE %.3f%% | Head %.2f Med %.2f Tail %.2f',
                        best_acc1, best_ece,
                        best_group.get('head', 0.0), best_group.get('medium', 0.0), best_group.get('tail', 0.0))

        logger.info('Final best Acc@1: %.3f%% (ECE %.3f%%)', best_acc1, best_ece)
        return

    # ---------------- ELM ----------------
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    best_acc1 = -1.0
    best_ece = float('inf')
    best_group = {g: 0.0 for g in GROUP_NAMES}

    if uses_elm(classifier_name):
        elm_adaptive = bool(cfg_value(config, 'elm_adaptive', True))
        elm = ELM(
            n_hidden=int(cfg_value(config, 'elm_n_hidden', 1024)),
            n_classes=num_classes,
            activation=str(cfg_value(config, 'elm_activation', 'relu')),
            reg=float(cfg_value(config, 'elm_reg', 1e-2)),
            cls_num_list=cls_num_list if elm_adaptive else None,
            class_weight_beta=float(cfg_value(config, 'elm_weight_beta', 0.5)),
            random_state=seed,
            orthogonalize=bool(cfg_value(config, 'elm_orthogonalize', False)),
        )
        logger.info('> Fitting ELM ...')
        t0 = time.time()
        elm.fit(X_train, y_train)
        logger.info('  ELM fit done in %.1fs', time.time() - t0)

        y_pred = elm.predict(X_val)
        y_proba = elm.predict_proba(X_val)
        metrics = summarize_predictions(y_val, y_pred, y_proba, cls_num_list, partitions)
        log_metrics(logger, metrics)
        best_acc1 = metrics['val_accuracy']
        best_ece = metrics['ece']
        best_group = dict(metrics['group_acc'])

        if bool(cfg_value(config, 'elm_storing', False)):
            with open(os.path.join(model_dir, 'elm_best.pkl'), 'wb') as f:
                pickle.dump(elm, f)
            logger.info('  Saved ELM pickle to %s/elm_best.pkl', model_dir)

        logger.info('Final best Acc@1: %.3f%% (ECE %.3f%%) | Head %.2f Med %.2f Tail %.2f',
                    best_acc1, best_ece, best_group['head'], best_group['medium'], best_group['tail'])
        return

    # ---------------- Adaptive BLS (ARBN) ----------------
    bls = ARBN(
        feature_times=int(cfg_value(config, 'bls_feature_times', 10)),
        enhance_times=int(cfg_value(config, 'bls_enhance_times', 10)),
        feature_size=int(cfg_value(config, 'bls_feature_size', 256)),
        n_classes=num_classes,
        mapping_function=str(cfg_value(config, 'bls_mapping_function', 'linear')),
        enhance_function=str(cfg_value(config, 'bls_enhance_function', 'relu')),
        reg=float(cfg_value(config, 'bls_reg', 0.005)),
        use_sparse=bool(cfg_value(config, 'bls_use_sparse', False)),
        cls_num_list=cls_num_list if bool(cfg_value(config, 'bls_adaptive_reg', True)) else None,
        adaptive_reg=bool(cfg_value(config, 'bls_adaptive_reg', True)),
        class_weight_beta=float(cfg_value(config, 'bls_weight_beta', 0.5)),
    )
    persist = bool(cfg_value(config, 'bls_storing', False))
    logger.info('> Fitting Adaptive BLS (ARBN) ...')
    t0 = time.time()
    bls.fit(X_train, y_train)
    logger.info('  ARBN fit done in %.1fs', time.time() - t0)

    y_pred = bls.predict(X_val)
    y_proba = bls.predict_proba(X_val)
    metrics = summarize_predictions(y_val, y_pred, y_proba, cls_num_list, partitions)
    log_metrics(logger, metrics)
    best_acc1 = metrics['val_accuracy']
    best_ece = metrics['ece']
    best_group = dict(metrics['group_acc'])

    if persist:
        with open(os.path.join(model_dir, 'adaptive_bls_best.pkl'), 'wb') as f:
            pickle.dump(bls, f)
        logger.info('  Saved ARBN pickle to %s/adaptive_bls_best.pkl', model_dir)

    enhance_epochs = int(cfg_value(config, 'bls_enhance_epochs', 0))
    enhance_nodes = int(cfg_value(config, 'bls_enhance_nodes', 10))
    for epoch in range(enhance_epochs):
        logger.info('%s', '=' * 60)
        logger.info('Enhancement epoch %d/%d', epoch + 1, enhance_epochs)
        bls.add_enhancement_nodes(X_train, y_train, enhance_nodes)
        y_pred = bls.predict(X_val)
        y_proba = bls.predict_proba(X_val)
        metrics = summarize_predictions(y_val, y_pred, y_proba, cls_num_list, partitions)
        log_metrics(logger, metrics)
        if metrics['val_accuracy'] > best_acc1:
            best_acc1 = metrics['val_accuracy']
            best_ece = metrics['ece']
            best_group = dict(metrics['group_acc'])
            if persist:
                with open(os.path.join(model_dir, 'adaptive_bls_best.pkl'), 'wb') as f:
                    pickle.dump(bls, f)
        logger.info('Best Prec@1 %.3f%% ECE %.3f%% | Head %.2f Med %.2f Tail %.2f',
                    best_acc1, best_ece, best_group['head'], best_group['medium'], best_group['tail'])

    logger.info('Final best Acc@1: %.3f%% (ECE %.3f%%) | Head %.2f Med %.2f Tail %.2f',
                best_acc1, best_ece, best_group['head'], best_group['medium'], best_group['tail'])


if __name__ == '__main__':
    main()
