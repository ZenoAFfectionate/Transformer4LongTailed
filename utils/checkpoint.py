"""Best-only checkpoint save/load helpers.

``save_checkpoint`` short-circuits when ``is_best=False`` so non-best epochs
are cheap no-ops. Only ``<best_name>`` is ever written.
"""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(state, is_best, model_dir, best_name='ckp_best.pth.tar'):
    """Persist ``state`` to ``<model_dir>/<best_name>`` only when ``is_best``."""
    if not is_best:
        return None
    path = Path(model_dir) / best_name
    torch.save(state, path)
    return str(path)


def load_checkpoint(path, model=None, optimizer=None, scheduler=None,
                    scaler=None, map_location='cpu', strict=True):
    """Load a checkpoint and optionally restore optimizer/scheduler/scaler state.

    Returns the full checkpoint dictionary so callers can pull extra keys
    (``best_acc1``, ``epoch``, ``cfg``, etc.) without re-loading.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        state_dict = checkpoint.get('state_dict_model', checkpoint.get('state_dict'))
        if state_dict is None:
            raise KeyError(
                'Checkpoint does not contain `state_dict_model` or `state_dict`; '
                f'available keys: {list(checkpoint.keys())}'
            )
        model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if scaler is not None and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
        scaler.load_state_dict(checkpoint['scaler'])

    return checkpoint


__all__ = ['save_checkpoint', 'load_checkpoint']
