#!/usr/bin/env bash
# Stage2 — retrain classifier on frozen ViT-Base features.
#
# Usage:
#   STAGE1_CKPT=results/.../ckp_best.pth.tar ./stage2_vit.sh            # cifar10 + linear
#   STAGE1_CKPT=...  CLASSIFIER=adaptive_bls ./stage2_vit.sh cifar100
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STAGE1_CKPT="${STAGE1_CKPT:?must point to a stage1 ckp_best.pth.tar}"
CLASSIFIER="${CLASSIFIER:-linear}"   # linear|lws|lws_plus|adaptive_bls|bls|elm

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

CUDA_VISIBLE_DEVICES="${GPU}" python "${REPO_ROOT}/train_stage2.py" \
  --cfg "${CFG}" \
  model_name ViT \
  num_classes "${NUM_CLASSES}" \
  head_class_idx "${HEAD_IDX}" \
  med_class_idx  "${MED_IDX}" \
  tail_class_idx "${TAIL_IDX}" \
  data_path "${DATA_ROOT}" \
  workers "${WORKERS}" \
  classifier "${CLASSIFIER}" \
  resume "${STAGE1_CKPT}" \
  "$@"
