"""Multi-head self-attention for visual tokens.

Post-cleanup:

* Dropped the never-applied ``RotaryEmbedding`` (its call site was commented
  out).
* Dropped ``manifold_type`` / ``lambda_manifold`` / ``spatial_window`` — the
  forward only referenced them via ``lambda_manifold * 0.0`` (a no-op).
* Dropped the ``H, W`` positional args — they were stored but never used.
* Restored attention dropout: ``attn = self.attn_drop(attn)`` is live again.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """Fused-QKV multi-head self-attention backed by PyTorch SDPA.

    ``F.scaled_dot_product_attention`` dispatches to FlashAttention-2 on
    modern GPUs (Ampere+ with CUDA 11+, PyTorch 2.0+) — faster and lower
    memory than the hand-written ``einsum + softmax`` path. A fallback to
    the explicit einsum path is available via ``use_sdpa=False`` for
    debugging / reproducibility.
    """

    def __init__(self, emb_size, head_num, attn_dropout=0.0, proj_dropout=0.0,
                 use_sdpa=True, qk_norm=True):
        super().__init__()
        assert emb_size % head_num == 0, 'embedding size is not divisible by head number'
        self.emb_size = emb_size
        self.head_num = head_num
        self.head_dim = emb_size // head_num
        self.scale = self.head_dim ** -0.5
        self.attn_dropout_p = float(attn_dropout)
        self.use_sdpa = use_sdpa

        self.qkv = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.proj_drop = nn.Dropout(proj_dropout)

        # QK-Norm (ViT-22B / SD3 / Chameleon) stabilizes attention logits
        # in fp16/bf16 by normalizing Q and K per-head before the dot product.
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x):
        qkv = rearrange(self.qkv(x), 'b n (h d qkv) -> (qkv) b h n d',
                        h=self.head_num, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_sdpa:
            # Fused kernel: FlashAttention-2 where available.
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn = torch.einsum('bhqd, bhkd -> bhqk', q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.einsum('bhal, bhlv -> bhav', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return self.proj_drop(out)


class RelativeEmbedding(nn.Module):
    """Relative Position Embedding for Shifted Window Attention.

    Exposes two APIs: ``bias()`` returns the per-head bias tensor of shape
    ``[num_heads, M, M]`` (used by SDPA-based callers); ``forward(att_scores)``
    adds the bias in-place to ``att_scores`` of shape ``[..., num_heads, M, M]``
    (used by the explicit einsum path).
    """

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        coords = torch.arange(window_size)
        relative_coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        relative_coords = torch.flatten(relative_coords, 1)
        relative_coords = relative_coords[:, :, None] - relative_coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= window_size * 2 - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def bias(self):
        """Return ``[num_heads, M, M]`` per-head bias tensor."""
        bias = self.bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            self.num_heads,
        ).permute(2, 0, 1).contiguous()
        return bias

    def forward(self, att_scores):
        return att_scores + self.bias().unsqueeze(0)


class ShiftedWindowAttention(nn.Module):
    """Shifted Window Multi-Head Self-Attention (Swin V1).

    The attention mask for the cyclic-shifted pass depends on the input
    spatial size, so we build it lazily on first forward and cache it keyed
    by ``(H, W, device)``. The original MoE4SwT implementation attempted to
    precompute it at ``__init__`` time with a broken view; this version uses
    the standard Swin masking recipe.
    """

    def __init__(self, emb_size, head_num, window_size, shift=False,
                 attn_dropout=0.0, use_sdpa=True, qk_norm=True):
        super().__init__()
        assert emb_size % head_num == 0, 'emb_size must be divisible by head_num'
        self.emb_size = emb_size
        self.head_num = head_num
        self.head_dim = emb_size // head_num
        self.window_size = window_size
        self.shift = shift
        self.attn_dropout_p = float(attn_dropout)
        self.use_sdpa = use_sdpa

        self.relative_embedding = RelativeEmbedding(window_size, head_num)
        self.qkv = nn.Linear(emb_size, 3 * emb_size)
        self.proj = nn.Linear(emb_size, emb_size)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        # Lazy per-resolution mask cache (populated on first forward).
        self._mask_cache: dict = {}

    def _apply(self, fn):
        # Clear device-tied caches when model.to()/model.cuda() is invoked.
        self._mask_cache.clear()
        return super()._apply(fn)

    def _build_attn_mask(self, H, W, device):
        """Return ``[num_windows, M, M]`` mask for the cyclic-shift pass."""
        key = (H, W, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]

        shift_size = self.window_size // 2
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        num_h = H // self.window_size
        num_w = W // self.window_size
        mask_windows = img_mask.view(1, num_h, self.window_size, num_w, self.window_size, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        self._mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x):
        B, L, _ = x.shape
        H = W = int(math.sqrt(L))
        assert H * W == L, 'Input sequence length must be a perfect square'
        assert H % self.window_size == 0 and W % self.window_size == 0, (
            f'H={H}/W={W} must be divisible by window_size={self.window_size}'
        )

        shift_size = self.window_size // 2
        if self.shift:
            x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
            x = rearrange(x, 'b h w c -> b (h w) c')

        qkv = self.qkv(x)

        qkv = rearrange(
            qkv,
            'b (num_h nH num_w nW) c -> b (num_h num_w) (nH nW) c',
            num_h=H // self.window_size,
            num_w=W // self.window_size,
            nH=self.window_size,
            nW=self.window_size,
        )

        qkv = qkv.view(B, -1, self.window_size * self.window_size,
                       3, self.head_num, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_windows, head_num, M, head_dim]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        M = q.shape[-2]
        num_windows = q.shape[1]

        # Per-head relative-position bias: [num_heads, M, M] -> broadcast [1, 1, H, M, M]
        rel_bias = self.relative_embedding.bias().view(1, 1, self.head_num, M, M)
        if self.shift:
            # [num_windows, M, M] -> [1, num_windows, 1, M, M]
            attn_mask = self._build_attn_mask(H, W, x.device)
            attn_bias = attn_mask.unsqueeze(0).unsqueeze(2) + rel_bias
        else:
            attn_bias = rel_bias

        if self.use_sdpa:
            # SDPA expects [*, head, N, d]; flatten (B, num_windows) into one batch dim.
            BW = B * num_windows
            q_ = q.reshape(BW, self.head_num, M, self.head_dim)
            k_ = k.reshape(BW, self.head_num, M, self.head_dim)
            v_ = v.reshape(BW, self.head_num, M, self.head_dim)
            # Broadcast bias [1, W, H, M, M] -> [B, W, H, M, M] -> [B*W, H, M, M]
            mask_ = attn_bias.expand(B, num_windows, self.head_num, M, M).reshape(BW, self.head_num, M, M)
            out = F.scaled_dot_product_attention(
                q_, k_, v_,
                attn_mask=mask_,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=False,
            )
            out = out.view(B, num_windows, self.head_num, M, self.head_dim)
        else:
            att_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            att_scores = att_scores + attn_bias
            att_weights = F.softmax(att_scores, dim=-1)
            out = att_weights @ v

        att_out = rearrange(out, 'b w h m d -> b w m (h d)')
        att_out = rearrange(
            att_out,
            'b (num_h num_w) (nH nW) c -> b (num_h nH) (num_w nW) c',
            num_h=H // self.window_size,
            num_w=W // self.window_size,
            nH=self.window_size,
            nW=self.window_size,
        )

        if self.shift:
            att_out = torch.roll(att_out, shifts=(shift_size, shift_size), dims=(1, 2))

        att_out = rearrange(att_out, 'b h w c -> b (h w) c')
        return self.proj(att_out)


__all__ = ['MultiHeadAttention', 'RelativeEmbedding', 'ShiftedWindowAttention']
