"""Two-stage pipeline entry for Transformer4LongTailed.

Runs stage1 (backbone pretraining) followed by stage2 (classifier retraining
on frozen backbone features) in a single invocation. Supports all four
backbones (``ViT`` / ``MoE4ViT`` / ``SwT`` / ``MoE4SwT``) and all the
classifiers supported by stage2 (linear / lws / lws_plus / adaptive_bls / bls
/ elm).

Usage
=====

Run both stages back-to-back using the Tiny-SwT preset on CIFAR10-balance::

    python main.py --cfg config/SwT/Tiny/CIFAR10_balance.yaml \
        --classifier linear --run-tag swt_tiny_cifar10

Only run stage1 (e.g. long pretraining run on a shared server)::

    python main.py --cfg config/SwT/Base/CIFAR100_longtail.yaml \
        --stage 1 --run-tag swt_base_c100_lt

Only run stage2 given an existing stage1 checkpoint::

    python main.py --cfg config/SwT/Base/CIFAR100_longtail.yaml \
        --stage 2 --classifier adaptive_bls \
        --resume results/CIFAR100_longtail_swt_base_c100_lt/ckps/ckp_best.pth.tar

Override any config field via key/value pairs after the flags::

    python main.py --cfg config/MoE4SwT/Tiny/CIFAR10_balance.yaml \
        --classifier linear --stage both \
        n_epochs 50 batch_size 128 lr 1e-4

Arguments
---------
``--cfg``        Path to the backbone/dataset YAML.
``--stage``      ``1`` | ``2`` | ``both`` (default).
``--classifier`` Stage2 classifier (default ``linear``). Passed via
                 ``cfg.classifier`` override.
``--resume``     Stage1 checkpoint to use as stage2 input (only for
                 ``--stage 2``). If omitted in ``both`` mode, we auto-pick
                 the ``ckp_best.pth.tar`` produced by stage1.
``--run-tag``    Shared ``T4LT_RUN_TAG`` used as timestamp suffix so that
                 stage1 and stage2 end up under the same
                 ``results/<cfg>_<tag>/`` directory.
``opts``         Arbitrary ``KEY VAL`` pairs appended to the stage1 / stage2
                 command lines.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description='Transformer4LongTailed one-stop entry — runs stage1 + stage2 back-to-back.',
    )
    parser.add_argument('--cfg', required=True, type=str, help='experiment YAML')
    parser.add_argument('--stage', default='both', choices=('1', '2', 'both'),
                        help='which stage(s) to run (default: both)')
    parser.add_argument('--classifier', default='linear',
                        choices=('linear', 'lws', 'lws_plus',
                                 'adaptive_bls', 'bls', 'elm'),
                        help='stage2 classifier head (default: linear)')
    parser.add_argument('--resume', default=None, type=str,
                        help='stage1 checkpoint to feed into stage2 '
                             '(required for --stage 2; auto-detected otherwise)')
    parser.add_argument('--run-tag', default=None, type=str,
                        help='shared results-directory suffix; defaults to a '
                             'timestamp so repeat runs do not collide')
    parser.add_argument('--python', default=sys.executable, type=str,
                        help='python interpreter to use for the subprocesses')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='override cfg via KEY VAL pairs (forwarded verbatim)')
    args = parser.parse_args()

    if args.stage == '2' and args.resume is None:
        parser.error('--stage 2 requires --resume <ckp_path>')

    if args.run_tag is None:
        args.run_tag = time.strftime('%Y%m%d%H%M%S')

    return args


def resolve_run_dir(cfg_path: str, run_tag: str) -> Path:
    """Mirror utils.logger.create_logger directory layout."""
    cfg_basename = Path(cfg_path).name.split('.')[0]
    result_root = Path(os.environ.get('T4LT_RESULT_DIR',
                                      str(REPO_ROOT / 'results')))
    return result_root / f'{cfg_basename}_{run_tag}'


def run_subprocess(cmd, env_extra=None):
    """Launch a subprocess, stream output, and raise on non-zero exit."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print(f'\n>>> {" ".join(cmd)}\n', flush=True)
    completed = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        raise RuntimeError(
            f'subprocess exited with code {completed.returncode}: {" ".join(cmd)}'
        )


def run_stage1(args):
    cmd = [
        args.python, str(REPO_ROOT / 'train_stage1.py'),
        '--cfg', args.cfg,
    ]
    if args.opts:
        cmd.extend(args.opts)
    run_subprocess(cmd, env_extra={'T4LT_RUN_TAG': args.run_tag})


def run_stage2(args, resume_path: str):
    cmd = [
        args.python, str(REPO_ROOT / 'train_stage2.py'),
        '--cfg', args.cfg,
        'classifier', args.classifier,
        'resume', resume_path,
    ]
    if args.opts:
        cmd.extend(args.opts)
    # stage2's run-tag suffix includes _s2 so the stage1 and stage2 results
    # sit side-by-side under results/.
    run_subprocess(cmd, env_extra={'T4LT_RUN_TAG': f'{args.run_tag}_s2'})


def main():
    args = parse_args()

    run_dir = resolve_run_dir(args.cfg, args.run_tag)
    ckp_path = run_dir / 'ckps' / 'ckp_best.pth.tar'

    if args.stage in ('1', 'both'):
        print(f'=== Stage1: training {args.cfg} (tag={args.run_tag}) ===')
        run_stage1(args)

    if args.stage in ('2', 'both'):
        resume = args.resume or str(ckp_path)
        if not Path(resume).is_file():
            raise FileNotFoundError(
                f'Stage2 input checkpoint not found: {resume}. '
                f'Did stage1 fail, or is --resume wrong?'
            )
        print(f'=== Stage2: classifier={args.classifier} resume={resume} ===')
        run_stage2(args, resume)

    print('\n=== all requested stages complete ===')
    print(f'results root: {run_dir}')


if __name__ == '__main__':
    main()
