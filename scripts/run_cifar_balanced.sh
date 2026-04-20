#!/usr/bin/env bash
# run_cifar_balanced.sh — run 8 experiments sequentially on one GPU.
#
# Usage:
#   bash script/run_cifar_balanced.sh                     # default: GPU 0, 200 epochs, DeiT-style ViT
#   GPU=1 EPOCHS=200 bash script/run_cifar_balanced.sh    # pick GPU
#   EPOCHS=5 bash script/run_cifar_balanced.sh            # quick smoke test
#   VIT_SCALE=base bash script/run_cifar_balanced.sh      # use original ViT-Base-768 instead of DeiT-Tiny
#   ONLY='SwT MoE4SwT' bash script/run_cifar_balanced.sh  # run only some backbones
#
# Env overrides:
#   GPU         single GPU id                       (default: 0)
#   DATA        dataset root                        (default: /home/kemove/data)
#   EPOCHS      n_epochs for stage1                 (default: 200)
#   S2EP        num_epochs for stage2 classifier    (default: 30)
#   BATCH       batch_size                          (default: 256)
#   WORKERS     dataloader workers                  (default: 8)
#   VIT_SCALE   "deit_tiny" (emb=192,head=3) or     (default: deit_tiny)
#               "base"      (emb=768,head=12)       — affects ViT & MoE4ViT only;
#               SwT/MoE4SwT always use their Base cfg
#   ONLY        space-separated backbone filter     (default: all 4)
#   DATASETS    space-separated datasets filter     (default: CIFAR10 CIFAR100)
#
# Each experiment runs stage1 (backbone pretrain) + stage2 (linear classifier).
# Results in:
#   results/<dataset>_balance_<tag>/        (stage1)
#   results/<dataset>_balance_<tag>_s2/     (stage2)
# Full stdout/stderr log:  results/_batch_logs/<tag>.log

set -uo pipefail

# --- paths ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- config ---
GPU="${GPU:-0}"
DATA="${DATA:-/home/kemove/data}"
EPOCHS="${EPOCHS:-200}"
S2EP="${S2EP:-30}"
BATCH="${BATCH:-256}"
WORKERS="${WORKERS:-8}"
VIT_SCALE="${VIT_SCALE:-deit_tiny}"
ONLY="${ONLY:-ViT MoE4ViT SwT MoE4SwT}"
DATASETS="${DATASETS:-CIFAR10 CIFAR100}"

LOG_DIR="$REPO_ROOT/results/_batch_logs"
mkdir -p "$LOG_DIR"

# --- ViT scale presets ---
# DeiT-Tiny shape is standard for CIFAR-from-scratch experiments.
# Base shape (768) is the paper-original ViT-Base — works but heavily over-parametrized on CIFAR.
case "$VIT_SCALE" in
    deit_tiny) VIT_OVERRIDE="emb_size 192 head_num 3" ;;
    base)      VIT_OVERRIDE="emb_size 768 head_num 12" ;;
    *)         echo "ERROR: unknown VIT_SCALE=$VIT_SCALE (expected deit_tiny|base)"; exit 1 ;;
esac

# --- CIFAR class-group boundaries for Head/Med/Tail evaluation ---
HMT_C10="head_class_idx [0,3] med_class_idx [3,7] tail_class_idx [7,10]"
HMT_C100="head_class_idx [0,36] med_class_idx [36,71] tail_class_idx [71,100]"

# --- experiment builder ---
# Returns: "backbone:dataset:cfg:extra-overrides"
build_row() {
    local backbone="$1"
    local dataset="$2"
    local cfg="" extra=""
    local hmt
    if [ "$dataset" = "CIFAR10" ]; then hmt="$HMT_C10"; else hmt="$HMT_C100"; fi
    local num_cls
    if [ "$dataset" = "CIFAR10" ]; then num_cls="10"; else num_cls="100"; fi

    case "$backbone" in
        ViT)
            cfg="config/ViT/Base/${dataset}_balance.yaml"
            extra="$VIT_OVERRIDE num_classes $num_cls $hmt"
            ;;
        MoE4ViT)
            cfg="config/ViT/Base/${dataset}_balance.yaml"
            extra="model_name MoE4ViT $VIT_OVERRIDE num_experts 8 share_experts 2 top_k 2 num_classes $num_cls $hmt"
            ;;
        SwT)
            cfg="config/SwT/Base/${dataset}_balance.yaml"
            extra=""
            ;;
        MoE4SwT)
            cfg="config/MoE4SwT/Base/${dataset}_balance.yaml"
            extra=""
            ;;
        *)
            echo "ERROR: unknown backbone $backbone"; return 1 ;;
    esac
    echo "${backbone}:${dataset}:${cfg}:${extra}"
}

# --- build experiment list ---
EXPERIMENTS=()
for ds in $DATASETS; do
    for bb in $ONLY; do
        row=$(build_row "$bb" "$ds") || exit 1
        EXPERIMENTS+=("$row")
    done
done

# --- pre-flight ---
for entry in "${EXPERIMENTS[@]}"; do
    cfg=$(echo "$entry" | cut -d: -f3)
    if [ ! -f "$cfg" ]; then
        echo "ERROR: config not found: $cfg"
        exit 1
    fi
done
if [ ! -d "$DATA" ]; then
    echo "WARNING: data path $DATA does not exist. torchvision will try to download."
fi

echo "================================================================"
echo "CIFAR balanced — sequential run on GPU $GPU"
echo "================================================================"
echo "  repo:       $REPO_ROOT"
echo "  gpu:        $GPU"
echo "  data:       $DATA"
echo "  epochs:     $EPOCHS (stage1) / $S2EP (stage2)"
echo "  batch:      $BATCH"
echo "  workers:    $WORKERS"
echo "  ViT scale:  $VIT_SCALE"
echo "  backbones:  $ONLY"
echo "  datasets:   $DATASETS"
echo "  #experiments: ${#EXPERIMENTS[@]}"
echo "  logs:       $LOG_DIR"
echo "================================================================"
echo ""

# --- Ctrl-C handler ---
current_pid=""
cleanup() {
    echo ""
    echo "=== interrupted ==="
    [ -n "$current_pid" ] && kill "$current_pid" 2>/dev/null
    exit 130
}
trap cleanup INT TERM

# --- sequential launcher ---
batch_start=$(date +%s)
for i in "${!EXPERIMENTS[@]}"; do
    entry="${EXPERIMENTS[$i]}"
    backbone=$(echo "$entry" | cut -d: -f1)
    dataset=$(echo "$entry" | cut -d: -f2)
    cfg=$(echo "$entry" | cut -d: -f3)
    extra=$(echo "$entry" | cut -d: -f4-)

    tag="${backbone,,}_$(echo ${dataset,,})_b"
    log_file="$LOG_DIR/${tag}.log"

    echo "----------------------------------------------------------------"
    echo "[$((i+1))/${#EXPERIMENTS[@]}] $backbone × $dataset   ($(date +%H:%M:%S))"
    echo "  cfg:  $cfg"
    echo "  tag:  $tag"
    echo "  log:  $log_file"
    echo "  extra: ${extra:-<none>}"
    echo "----------------------------------------------------------------"

    run_start=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --cfg "$cfg" \
        --stage both \
        --classifier linear \
        --run-tag "$tag" \
        data_path "$DATA" \
        n_epochs "$EPOCHS" \
        num_epochs "$S2EP" \
        batch_size "$BATCH" \
        workers "$WORKERS" \
        gpu 0 \
        $extra \
        2>&1 | tee "$log_file" &
    current_pid=$!
    wait "$current_pid"
    rc=$?
    current_pid=""
    run_min=$(( ($(date +%s) - run_start) / 60 ))
    if [ $rc -ne 0 ]; then
        echo "  ! experiment exited non-zero (rc=$rc) — continuing with next"
    else
        echo "  ✓ done in ${run_min} min"
    fi
    echo ""
done

elapsed_min=$(( ($(date +%s) - batch_start) / 60 ))

# --- summary ---
echo "================================================================"
echo "ALL EXPERIMENTS COMPLETE — total ${elapsed_min} min"
echo "================================================================"
echo ""
printf "  %-25s %-14s %-14s\n" "experiment" "stage1 Acc1" "stage2 Acc1"
printf "  %-25s %-14s %-14s\n" "-------------------------" "-----------" "-----------"
for entry in "${EXPERIMENTS[@]}"; do
    backbone=$(echo "$entry" | cut -d: -f1)
    dataset=$(echo "$entry" | cut -d: -f2)
    tag="${backbone,,}_$(echo ${dataset,,})_b"
    s1_dir="results/${dataset}_balance_${tag}"
    s2_dir="results/${dataset}_balance_${tag}_s2"
    s1_acc=$(grep -h "Final best" "$s1_dir/logs/"*.txt 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+%" | head -1)
    s2_acc=$(grep -h "Final best" "$s2_dir/logs/"*.txt 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+%" | head -1)
    printf "  %-25s %-14s %-14s\n" "${backbone}_${dataset}" "${s1_acc:-N/A}" "${s2_acc:-N/A}"
done
echo ""
echo "Per-experiment logs: $LOG_DIR"
