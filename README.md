# Transformer4LongTailed

**A unified Vision Transformer framework for long-tailed image classification, implemented in PyTorch.**

---

## Overview

`Transformer4LongTailed` (read as *Transformer **for** Long-Tailed*) is a research-oriented codebase that studies how modern Vision Transformers behave under severe class imbalance. It brings together four Transformer backbones, six classifier heads, five standard long-tailed benchmarks, and a decoupled two-stage training protocol — all behind a single command-line entry point and a consistent YAML configuration system.

The central motivation is reproducibility and controlled comparison. Every backbone shares the same data loaders, optimizer, learning-rate schedule, mixed-precision training loop, evaluation protocol, and checkpoint format; the only variable is the encoder block design. This makes it straightforward to isolate the effect of any single architectural choice — Mixture-of-Experts routing, shifted-window attention, local enhancement convolutions, or depth-path regularization — on head / medium / tail accuracy.

The codebase targets researchers and practitioners working on long-tailed recognition, representation-vs-classifier decoupling, MoE architectures for vision, or head-bias mitigation in deep models. It is not an out-of-the-box production library; it is an experiment platform with production-grade engineering underneath.

---

## Key Features

- **Four unified Transformer backbones.** `ViT`, `MoE4ViT` (ViT with DeepSeek-V3-style shared + routed MoE), `SwT` (four-stage Swin Transformer with W-MSA / SW-MSA and patch merging), and `MoE4SwT` (ViT + MoE + Local Enhancement Head in the first block). All expose the same `build_model` / `extract_features` interface through `utils/model_factory.py`.
- **Decoupled two-stage training.** *Stage 1* performs representation learning on the full training distribution; *Stage 2* freezes the backbone and retrains a classifier head on balanced features, a protocol shown to substantially reduce the head-bias of long-tailed classifiers.
- **Six classifier heads.** Linear, LWS, LWS+, Adaptive Re-weighted Broad Learning System (Adaptive-BLS), vanilla BLS, and Extreme Learning Machine (ELM). The first three are trained with SGD + MultiStepLR; the latter three use closed-form solvers.
- **Long-tailed data augmentation.** Mixup for generalization and ReMix with tail-frequency-weighted mixing coefficients λᵧ to amplify minority-class signal during representation learning.
- **Shallow-plus-semantic feature head.** Concatenation of a shallow CLS token (or Swin stage-1 pooled feature) with the final semantic token, which empirically preserves fine-grained cues that help tail classes.
- **Pre-attention local enhancement (LEH).** A `1×1 → depthwise 3×3 → 1×1` convolution stack inserted before attention to inject local inductive bias and smooth window / patch boundary artifacts.
- **Aux-loss-free MoE load balancing.** Routed experts are selected by `sigmoid(logits) + expert_bias` where `expert_bias` is a non-trainable buffer nudged after every optimizer step (DeepSeek-V3 recipe). Selection is biased; the token→expert mixing weight uses the *unbiased* sigmoid score, so balancing pressure never distorts the features. A weak sequence-wise auxiliary term (α ≈ 1e-3) is retained as a safety net.
- **Sort-based fixed-capacity dispatch.** MoE tokens are sorted by expert id and routed into a static `[num_experts, capacity, d]` buffer — no variable-length splits, friendly to `torch.compile`, and significantly faster than per-expert Python loops.
- **Modern training stack.** PyTorch 2.0 scaled-dot-product attention (FlashAttention-2 kernel when available), linear-scaled DropPath schedule, AMP with `GradScaler`, and AdamW + warmup + cosine decay.
- **Deterministic evaluation.** All stochastic components (DropPath, MoE routing) are gated by `self.training`; evaluation and checkpoint re-loading are fully reproducible. Shifted-window masks are cached lazily per `(H, W, device)` and invalidated on `.to()`.
- **Ready-to-run presets.** Twenty-four YAML configurations span `{SwT, MoE4SwT} × {Tiny, Base} × {CIFAR-10, CIFAR-100, ImageNet} × {balanced, long-tailed}`, with additional ViT configs covering Tiny / Base / Large / Huge.

---

## Installation

All dependencies are declared in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate MoE4Vision   # environment name is kept for backward compatibility
```

**Minimum requirements**

- Python ≥ 3.10
- PyTorch ≥ 2.0 (requires `torch.nn.functional.scaled_dot_product_attention`)
- CUDA ≥ 11.8 (Ampere or Hopper GPU recommended for the FlashAttention-2 kernel)

---

## Quick Start

### End-to-end pipeline in a single command

```bash
python main.py \
    --cfg config/SwT/Base/CIFAR10_longtail.yaml \
    --classifier linear \
    --run-tag swt_base_c10lt_exp1
```

Stage 1 and Stage 2 run sequentially and write their outputs to:

```
results/CIFAR10_longtail_swt_base_c10lt_exp1/      # Stage 1
results/CIFAR10_longtail_swt_base_c10lt_exp1_s2/   # Stage 2
```

### Stage 1 only (backbone pretraining)

```bash
python main.py --cfg config/MoE4SwT/Base/CIFAR100_longtail.yaml \
    --stage 1 --run-tag moeswt_c100lt_exp1
```

### Stage 2 only (classifier ablation on an existing checkpoint)

```bash
# Linear head
python main.py --cfg config/SwT/Base/CIFAR10_longtail.yaml \
    --stage 2 --classifier linear \
    --resume results/CIFAR10_longtail_swt_base_c10lt_exp1/ckps/ckp_best.pth.tar

# Adaptive-BLS head
python main.py --cfg config/SwT/Base/CIFAR10_longtail.yaml \
    --stage 2 --classifier adaptive_bls \
    --resume results/CIFAR10_longtail_swt_base_c10lt_exp1/ckps/ckp_best.pth.tar

# ELM head
python main.py --cfg config/SwT/Base/CIFAR10_longtail.yaml \
    --stage 2 --classifier elm \
    --resume results/CIFAR10_longtail_swt_base_c10lt_exp1/ckps/ckp_best.pth.tar
```

### Overriding YAML fields from the command line

Any `KEY VAL` pair after the flags is forwarded to both stages' `opts`:

```bash
python main.py --cfg config/MoE4SwT/Tiny/CIFAR10_balance.yaml \
    --classifier linear --stage both --run-tag tune_lr \
    n_epochs 100 batch_size 256 lr 2e-4 moe_aux_weight 0.5 max_drop_path 0.2
```

---

## Supported Backbones

| Backbone   | Architecture                                     | CLS token | Stage-2 feature dim | MoE | LEH                     | Reference                     |
|------------|--------------------------------------------------|-----------|---------------------|-----|-------------------------|-------------------------------|
| `ViT`      | Single-resolution, 12 blocks                     | Yes       | `2 × emb_size`      | No  | No                      | Dosovitskiy et al., ICLR 2021 |
| `MoE4ViT`  | ViT with sparse Shared-Expert MoE                | Yes       | `2 × emb_size`      | Yes | No                      | This project                  |
| `SwT`      | Four-stage hierarchical, W-MSA / SW-MSA          | No        | `10 × emb_size`     | No  | Yes (first two stages)  | Liu et al., ICCV 2021         |
| `MoE4SwT`  | ViT + MoE + LEH on the first block               | Yes       | `2 × emb_size`      | Yes | Yes (first block only)  | This project                  |

**Default MoE settings.** `num_experts=8`, `share_experts=1`, `top_k=2`, bias-update speed `1e-3`, sequence-wise aux-loss weight `seq_aux_alpha=1e-3`, capacity factor `1.25`.

---

## Training Pipeline

### Stage 1 — Backbone Pretraining

- AdamW optimizer with linear warmup and cosine learning-rate decay.
- Mixed-precision training via `torch.amp.autocast('cuda')` and `GradScaler`.
- Optional Mixup or ReMix augmentation (mutually exclusive).
- MoE backbones balance expert load via a **bias-update controller** (DeepSeek-V3): after every `optimizer.step()`, each gate's `expert_bias` buffer is nudged by `±bias_update_speed` toward a uniform load distribution. A weak sequence-wise balance loss (α ≈ 1e-3) is added as a safety net.
- Metrics logged per epoch: `Acc@1`, `Acc@5`, Head / Medium / Tail accuracy, and Expected Calibration Error (ECE). The best-ECE checkpoint is saved as `ckp_best.pth.tar`.

### Stage 2 — Classifier Retraining

- Backbone weights are frozen; pre-classifier features are extracted once (CLS concatenation or mean-pool fusion, dimension per the table above).
- **SGD-trained heads** (`linear`, `lws`, `lws_plus`) use SGD + MultiStepLR on a class-balanced sampler.
- **Closed-form heads** (`adaptive_bls`, `bls`, `elm`) solve a ridge-regularized least-squares system; BLS variants additionally support enhancement-node epochs.
- Evaluation partitions classes into Head / Medium / Tail thirds by training-set frequency so that tail recall is tracked independently of overall accuracy.

---

## Configuration Reference

Every experiment is fully described by a YAML file. The key fields are:

| Field                                                            | Purpose                                                                              | Default     |
|------------------------------------------------------------------|--------------------------------------------------------------------------------------|-------------|
| `model_name`                                                     | `ViT` / `MoE4ViT` / `SwT` / `MoE4SwT`                                                | —           |
| `dataset`                                                        | `CIFAR10` / `CIFAR100` / `ImageNet` / `Places` / `iNaturalist2018`                   | —           |
| `num_classes`, `channels`, `image_size`, `patch_size`            | Data geometry                                                                        | —           |
| `window_size`                                                    | Swin-only attention window                                                           | `7`         |
| `emb_size`, `depth`, `head_num`                                  | Backbone capacity                                                                    | —           |
| `num_experts`, `share_experts`, `top_k`                          | MoE expert pool                                                                      | `8, 1, 2`   |
| `capacity_factor`, `bias_update_speed`, `seq_aux_alpha`, `moe_aux_weight` | MoE dispatch + aux-loss-free balancing                                       | `1.25, 1e-3, 1e-3, 1.0` |
| `batch_size`, `n_epochs`, `warmup_epochs`, `lr`, `weight_decay`, `betas` | Optimizer / schedule                                                         | —           |
| `mixup`, `remix`, `alpha`                                        | Data augmentation                                                                    | —           |
| `longtail`, `imb_type`, `imb_factor`                             | Long-tailed sampling                                                                 | —           |
| `loss_type`                                                      | `CE` / `LDAM` / `BalancedSoftmax` / `LogitAdjustment` / `Focal` / `LAS` / `CB`       | `CE`        |
| `drop_p`, `attn_drop`, `max_drop_path`, `label_smoothing`        | Regularization                                                                       | `0.1, 0.0, 0.1, 0.0` |
| `classifier`                                                     | Stage-2 head: `linear` / `lws` / `lws_plus` / `adaptive_bls` / `bls` / `elm`         | `linear`    |
| `head_class_idx`, `med_class_idx`, `tail_class_idx`              | Head / Medium / Tail boundaries (class-index ranges)                                 | —           |

**Predefined configurations.** Twenty-four YAMLs are shipped under `config/`:

```
config/SwT/{Tiny,Base}/{CIFAR10,CIFAR100,ImageNet}_{balance,longtail}.yaml
config/MoE4SwT/{Tiny,Base}/{CIFAR10,CIFAR100,ImageNet}_{balance,longtail}.yaml
config/ViT/{Tiny,Base,Large,Huge}/{CIFAR10,CIFAR100,ImageNet}_{balance,longtail}.yaml
```

---

## Repository Layout

```
Transformer4LongTailed/
├── main.py                       # One-stop entry: stage1 → stage2
├── train_stage1.py               # Stage 1 training loop
├── train_stage2.py               # Stage 2 training loop
├── model/
│   ├── attention.py              # MultiHeadAttention (SDPA) + ShiftedWindowAttention
│   ├── moe.py                    # MoEGate / SparseMoE / ShareExpertMoE (aux-loss-free)
│   ├── utils.py                  # Patch embedding, patch merging, LEH, FFN, DropPath
│   ├── ViT.py                    # Vision Transformer backbone
│   ├── MoE4ViT.py                # ViT + MoE backbone
│   ├── SwT.py                    # Swin Transformer backbone
│   └── MoE4SwT.py                # ViT + MoE + LEH backbone
├── classifier/
│   ├── linear.py                 # Linear / LWS / LWS+ heads
│   ├── arbn.py                   # Adaptive Re-weighted BLS
│   ├── bls.py                    # Vanilla BLS
│   └── elm.py                    # Extreme Learning Machine
├── dataset/                      # CIFAR-10/100, ImageNet, Places, iNat2018 loaders
├── utils/
│   ├── model_factory.py          # Routes build_model / extract_features by cfg.model_name
│   ├── logger.py                 # YACS config + logger
│   ├── loss.py                   # CE / LDAM / BalancedSoftmax / LogitAdjustment / Focal / LAS / CB
│   ├── mixup.py                  # Mixup / ReMix
│   ├── checkpoint.py             # Save / load
│   └── meter.py, metric.py       # AverageMeter, Top-k accuracy, ECE
├── config/                       # Predefined YAMLs (ViT / MoE4ViT / SwT / MoE4SwT)
├── docs/
│   ├── specs/                    # Design specifications
│   └── plans/                    # Implementation plans
├── scripts/                      # Shell launchers
└── results/                      # Per-run output directories (auto-created)
```

---

## Results

Results tables will be populated as experiments complete. Each configuration is run at least three times; cells report mean ± standard deviation. Empty cells are shown as `—`.

### Balanced setting — Acc@1 (%)

| Model     | Scale | CIFAR-10 | CIFAR-100 | ImageNet-1K | Params (M) |
|-----------|-------|----------|-----------|-------------|------------|
| ViT       | Tiny  | —        | —         | —           | —          |
| ViT       | Base  | —        | —         | —           | —          |
| MoE4ViT   | Tiny  | —        | —         | —           | —          |
| MoE4ViT   | Base  | —        | —         | —           | —          |
| SwT       | Tiny  | —        | —         | —           | —          |
| SwT       | Base  | —        | —         | —           | —          |
| MoE4SwT   | Tiny  | —        | —         | —           | —          |
| MoE4SwT   | Base  | —        | —         | —           | —          |

### Long-tailed setting — Acc@1 / Head / Medium / Tail / ECE (%)

**CIFAR-10-LT (`imb_factor=0.01`)**

| Model      | Stage-2 head   | Acc@1 | Head | Medium | Tail | ECE |
|------------|----------------|-------|------|--------|------|-----|
| ViT-B      | linear         | —     | —    | —      | —    | —   |
| ViT-B      | LWS            | —     | —    | —      | —    | —   |
| ViT-B      | adaptive_bls   | —     | —    | —      | —    | —   |
| MoE4ViT-B  | linear         | —     | —    | —      | —    | —   |
| MoE4ViT-B  | adaptive_bls   | —     | —    | —      | —    | —   |
| SwT-B      | linear         | —     | —    | —      | —    | —   |
| SwT-B      | LWS            | —     | —    | —      | —    | —   |
| SwT-B      | adaptive_bls   | —     | —    | —      | —    | —   |
| MoE4SwT-B  | linear         | —     | —    | —      | —    | —   |
| MoE4SwT-B  | LWS            | —     | —    | —      | —    | —   |
| MoE4SwT-B  | adaptive_bls   | —     | —    | —      | —    | —   |

**CIFAR-100-LT (`imb_factor=0.01`)**

| Model      | Stage-2 head   | Acc@1 | Head | Medium | Tail | ECE |
|------------|----------------|-------|------|--------|------|-----|
| ViT-B      | linear         | —     | —    | —      | —    | —   |
| MoE4ViT-B  | adaptive_bls   | —     | —    | —      | —    | —   |
| SwT-B      | linear         | —     | —    | —      | —    | —   |
| SwT-B      | adaptive_bls   | —     | —    | —      | —    | —   |
| MoE4SwT-B  | adaptive_bls   | —     | —    | —      | —    | —   |

**ImageNet-LT**

| Model      | Stage-2 head | Acc@1 | Head | Medium | Tail | ECE |
|------------|--------------|-------|------|--------|------|-----|
| SwT-B      | linear       | —     | —    | —      | —    | —   |
| SwT-B      | LWS+         | —     | —    | —      | —    | —   |
| MoE4SwT-B  | linear       | —     | —    | —      | —    | —   |
| MoE4SwT-B  | LWS+         | —     | —    | —      | —    | —   |

### Ablations

**Effect of LEH placement on MoE4SwT (CIFAR-100-LT, `emb=384`, `depth=12`)**

| Configuration                       | Acc@1 | Head | Medium | Tail |
|-------------------------------------|-------|------|--------|------|
| No LEH (equivalent to MoE4ViT)      | —     | —    | —      | —    |
| LEH on the first block only         | —     | —    | —      | —    |
| LEH on every block                  | —     | —    | —      | —    |

**Effect of DropPath schedule on SwT (CIFAR-100-LT)**

| `max_drop_path` | Acc@1 | ECE |
|-----------------|-------|-----|
| 0.0             | —     | —   |
| 0.1             | —     | —   |
| 0.2             | —     | —   |
| 0.3             | —     | —   |

**SDPA vs native einsum throughput (224×224, bs=128, SwT-Base, A100)**

| Implementation         | Train ms/iter | Peak memory (GB) |
|------------------------|---------------|------------------|
| einsum                 | —             | —                |
| SDPA (FlashAttention-2)| —             | —                |

---

## Environment Variables

| Variable                | Purpose                                                              | Default                      |
|-------------------------|----------------------------------------------------------------------|------------------------------|
| `T4LT_RUN_TAG`          | Suffix appended to each run's result directory to prevent collisions | Timestamp `YYYYMMDDHHMMSS`   |
| `T4LT_RESULT_DIR`       | Override the root directory for `results/`                           | `<repo>/results`             |
| `CUDA_VISIBLE_DEVICES`  | Standard CUDA device mask                                            | —                            |

---

## Testing

A minimal end-to-end sanity check:

```bash
python -c "
import torch
from model.MoE4ViT import MoE4ViT
from model.MoE4SwT import MoE4SwT
from model.ViT import VisionTransformer
from model.SwT import SwinTransformer

for name, M in [('ViT', VisionTransformer), ('SwT', SwinTransformer),
                ('MoE4ViT', MoE4ViT), ('MoE4SwT', MoE4SwT)]:
    # Construct each backbone with small dims and run a forward pass.
    ...
"
```

The official smoke / system-verification tests previously under `tests/` have been retired; per-backbone forward/backward coverage is now expected to be asserted in the contributing PR (see below).

---

## Contributing

Internal design notes and implementation plans live under `docs/specs/` and `docs/plans/`; please consult them before proposing substantive changes. Pull requests that add new backbones, classifier heads, or datasets should include:

1. A minimal forward + backward test demonstrating the new component runs end-to-end without grad errors.
2. At least one YAML configuration in `config/`.
3. Results on CIFAR-10-LT and CIFAR-100-LT for any claim of improvement on the long-tailed setting.

---

## License

Released under the Apache License 2.0. Please update the `LICENSE` file if a different license is required for your fork.

---

## Citation

If this repository is useful in your research, please consider citing it:

```bibtex
@misc{transformer4longtailed2026,
  title  = {Transformer4LongTailed: A Unified Vision Transformer Framework for Long-Tailed Classification},
  author = {TODO},
  year   = {2026},
  note   = {\url{https://github.com/TODO/Transformer4LongTailed}}
}
```

### References

- Dosovitskiy, A. et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
- Dai, D. et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* arXiv 2024.
- Liu, A. et al. *DeepSeek-V3 Technical Report* (aux-loss-free load balancing via expert bias). arXiv 2024.
- Cao, K. et al. *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss.* NeurIPS 2019.
- Kang, B. et al. *Decoupling Representation and Classifier for Long-Tailed Recognition.* ICLR 2020.
- Zhong, Z. et al. *Improving Calibration for Long-Tailed Recognition (LAS / MiSLAS).* CVPR 2021.
- Dao, T. et al. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023.
