"""Config + logging infrastructure for Transformer4LongTailed training.

Adapted from DS-AWBN/utils/logger.py with a project-specific ``_C`` default
graph. Supports:

* ``yacs.config.CfgNode`` when available; otherwise a minimal in-house
  fallback that parses YAML via ``pyyaml`` and overrides via dotted paths.
* ``create_logger(cfg, cfg_name)`` produces ``results/<cfg>_<ts>/{logs,ckps}``
  with a shared file + console handler.
* ``log_experiment_details`` dumps CLI args + raw YAML + resolved config.
"""

from __future__ import annotations

import ast
import logging
import os
import pprint
import time
from pathlib import Path

import yaml

try:
    from yacs.config import CfgNode as CN
except ImportError:  # minimal fallback — same shape as DS-AWBN's
    class CN(dict):  # type: ignore[no-redef]
        def __getattr__(self, name):
            if name in self:
                return self[name]
            raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

        def defrost(self):
            return None

        def freeze(self):
            return None

        def merge_from_file(self, cfg_file):
            with open(cfg_file, 'r', encoding='utf-8') as handle:
                data = yaml.safe_load(handle) or {}
            self._merge_dict(data)

        def merge_from_list(self, cfg_list):
            if not cfg_list:
                return
            if len(cfg_list) % 2 != 0:
                raise ValueError('Config override list must contain key/value pairs.')
            for key, raw_value in zip(cfg_list[0::2], cfg_list[1::2]):
                self._set_by_path(key.split('.'), self._decode_value(raw_value))

        def _merge_dict(self, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    node = self.get(key, CN())
                    if not isinstance(node, CN):
                        node = CN(node)
                    node._merge_dict(value)
                    self[key] = node
                else:
                    self[key] = self._normalize_value(value)

        def _set_by_path(self, keys, value):
            node = self
            for key in keys[:-1]:
                child = node.get(key, CN())
                if not isinstance(child, CN):
                    child = CN(child)
                node[key] = child
                node = child
            node[keys[-1]] = value

        @staticmethod
        def _decode_value(raw_value):
            try:
                value = yaml.safe_load(raw_value)
            except yaml.YAMLError:
                value = raw_value
            return CN._normalize_value(value)

        @staticmethod
        def _normalize_value(value):
            if isinstance(value, dict):
                node = CN()
                node._merge_dict(value)
                return node
            if isinstance(value, list):
                return [CN._normalize_value(item) for item in value]
            if isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return value
            return value


# ==================================================================
# Default configuration graph
# ==================================================================
_C = CN()

# Meta
_C.name        = ''
_C.print_freq  = 40
_C.workers     = 8
_C.log_dir     = 'logs'
_C.model_dir   = 'ckps'
_C.seed        = 0
_C.deterministic = True
_C.gpu         = 0
_C.resume      = ''

# Data
_C.dataset     = 'CIFAR10'
_C.data_path   = '/home/kemove/data'
_C.num_classes = 10
_C.longtail    = False
_C.imb_type    = 'exp'
_C.imb_factor  = 0.01
_C.test_imb_factor = None
_C.head_class_idx = [0, 3]
_C.med_class_idx  = [3, 7]
_C.tail_class_idx = [7, 10]

# Model
_C.model_name    = 'ViT'      # ViT | MoE4ViT
_C.channels      = 3
_C.image_size    = 32
_C.patch_size    = 4
_C.depth         = 12
_C.emb_size      = 768
_C.head_num      = 12
_C.drop_p        = 0.1              # dropout inside FFN + patch embed + residual proj
_C.attn_drop     = 0.0              # dropout applied on the attention softmax
_C.max_drop_path = 0.1              # linearly scheduled stochastic-depth peak rate
_C.num_experts   = 8
_C.share_experts = 2
_C.top_k         = 2
_C.window_size   = 7                # SwT / MoE4SwT shifted-window attention
_C.num_registers = 4                # ViT/MoE4ViT/MoE4SwT register tokens (0 = off)
_C.qk_norm       = True             # RMSNorm on Q, K before attention (all backbones)

# Optim
_C.lr           = 1e-4
_C.betas        = '0.9,0.99'
_C.weight_decay = 0.05
_C.momentum     = 0.9              # only used by stage2 SGD
_C.batch_size   = 256
_C.n_epochs     = 200
_C.warmup_epochs = 10
_C.mixup        = True
_C.alpha        = 1.0
_C.remix        = False
_C.loss_type    = 'CE'             # CE|LDAM|BalancedSoftmax|LogitAdjustment|Focal|LAS|CB
_C.label_smoothing = 0.0            # applied inside CE (composes with mixup)
_C.use_amp      = True
_C.smooth_head  = 0.1               # LAS head smoothing
_C.smooth_tail  = 0.0               # LAS tail smoothing
_C.tau          = 1.0               # LogitAdjustment tau
_C.max_m        = 0.5               # LDAM max margin
_C.moe_aux_weight = 1.0             # scales MoE auxiliary loss

# Stage2
_C.classifier  = 'linear'          # linear|lws|lws_plus|adaptive_bls|bls|elm
_C.lr_factor   = 1.0
_C.num_epochs  = 30                 # stage2 epoch count alias
_C.bls_feature_times    = 10
_C.bls_enhance_times    = 10
_C.bls_feature_size     = 256
_C.bls_mapping_function = 'linear'
_C.bls_enhance_function = 'relu'
_C.bls_reg              = 0.005
_C.bls_use_sparse       = False
_C.bls_adaptive_reg     = True
_C.bls_weight_beta      = 0.5
_C.bls_enhance_epochs   = 0
_C.bls_enhance_nodes    = 10
_C.bls_storing          = False
_C.bls_loading          = False
_C.bls_max_train_samples = None
_C.elm_n_hidden      = 1024
_C.elm_activation    = 'relu'
_C.elm_reg           = 1e-2
_C.elm_adaptive      = True
_C.elm_weight_beta   = 0.5
_C.elm_orthogonalize = False
_C.elm_storing       = False


def update_config(cfg, args):
    """Merge a YAML file and CLI ``opts`` (list of KEY VAL pairs) into ``cfg``."""
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if getattr(args, 'opts', None):
        cfg.merge_from_list(args.opts)


def create_logger(cfg, cfg_name):
    """Create per-run ``results/<cfg_basename>_<timestamp>/{logs,ckps}``.

    Returns ``(logger, model_dir)``.
    """
    time_str = os.environ.get('T4LT_RUN_TAG', time.strftime('%Y%m%d%H%M%S'))
    cfg_basename = os.path.basename(cfg_name).split('.')[0]

    repo_root = Path(__file__).resolve().parents[1]
    result_root = Path(os.environ.get('T4LT_RESULT_DIR', str(repo_root / 'results')))
    run_root = result_root / f'{cfg_basename}_{time_str}'

    log_dir = run_root / Path(cfg.log_dir)
    print(f'=> creating {log_dir}')
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'{cfg_basename}.txt'
    head = '%(asctime)-15s %(message)s'

    logger = logging.getLogger(f'{cfg_basename}_{time_str}')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(logging.Formatter(head))
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(head))
    logger.addHandler(console)
    logger.propagate = False

    model_dir = run_root / Path(cfg.model_dir)
    print(f'=> creating {model_dir}')
    model_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(model_dir)


def _to_builtin(value):
    """Recursively convert a ``CfgNode`` to plain dict/list/scalar."""
    if isinstance(value, CN):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    return value


def log_experiment_details(logger, args, cfg, cfg_path, model_dir):
    """Dump CLI args, raw YAML, and the fully resolved config."""
    cfg_path = Path(cfg_path).resolve()
    logger.info('=' * 100)
    logger.info('Experiment Configuration')
    logger.info('Config path: %s', cfg_path)
    logger.info('Model/checkpoint dir: %s', model_dir)
    logger.info('CLI args:\n%s', pprint.pformat(vars(args)))

    try:
        raw_yaml = cfg_path.read_text(encoding='utf-8')
    except OSError as exc:
        raw_yaml = f'<failed to read yaml: {exc}>'

    logger.info('Loaded YAML content:\n%s', raw_yaml)
    logger.info('Resolved config:\n%s', pprint.pformat(_to_builtin(cfg)))
    logger.info('=' * 100)


__all__ = [
    '_C',
    'CN',
    'update_config',
    'create_logger',
    'log_experiment_details',
    '_to_builtin',
]
