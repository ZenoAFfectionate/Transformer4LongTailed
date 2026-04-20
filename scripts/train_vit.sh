#!/usr/bin/env bash
# Stage1 — train ViT-Base on CIFAR-10 or CIFAR-100.
#
# Usage:
#   ./train_vit.sh                         # defaults to cifar10
#   DATASET=cifar100 ./train_vit.sh
#   ./train_vit.sh cifar100                # positional form
#   ./train_vit.sh cifar100 longtail true  # extra opts forwarded to train_stage1.py
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASET="${1:-${DATASET:-cifar10}}"
shift || true
DATASET_LOWER="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"

case "${DATASET_LOWER}" in
  cifar10)
    CFG="${REPO_ROOT}/config/ViT/Base/CIFAR10_balance.yaml"
    NUM_CLASSES=10
    HEAD_IDX="[0,3]"
    MED_IDX="[3,7]"
    TAIL_IDX="[7,10]"
    ;;
  cifar100)
    CFG="${REPO_ROOT}/config/ViT/Base/CIFAR100_balance.yaml"
    NUM_CLASSES=100
    HEAD_IDX="[0,33]"
    MED_IDX="[33,67]"
    TAIL_IDX="[67,100]"
    ;;
  fashion_mnist|fashionmnist)
    CFG="${REPO_ROOT}/config/ViT/Base/FashionMNIST_balance.yaml"
    NUM_CLASSES=10
    HEAD_IDX="[0,3]"
    MED_IDX="[3,7]"
    TAIL_IDX="[7,10]"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET} (expected cifar10|cifar100|fashion_mnist)" >&2
    exit 1
    ;;
esac

CUDA_VISIBLE_DEVICES="${GPU}" python "${REPO_ROOT}/train_stage1.py" \
  --cfg "${CFG}" \
  model_name ViT \
  num_classes "${NUM_CLASSES}" \
  head_class_idx "${HEAD_IDX}" \
  med_class_idx  "${MED_IDX}" \
  tail_class_idx "${TAIL_IDX}" \
  data_path "${DATA_ROOT}" \
  workers "${WORKERS}" \
  "$@"
