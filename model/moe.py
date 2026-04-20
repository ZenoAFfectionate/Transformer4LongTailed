"""Mixture-of-Experts (MoE) building blocks shared across MoE backbones.

Extracted from ``model/MoE4ViT.py`` so that ``MoE4ViT`` and ``MoE4SwT`` can
share a single source of truth for the router, sparse expert pool, and
DeepSeek-style shared+routed expert mixture.

Design notes
------------
This module implements the **DeepSeek-V3 aux-loss-free** load-balancing
recipe (Liu et al. 2024) rather than the coefficient-of-variation auxiliary
loss used in the original MoE4ViT. The differences:

* **Selection vs. weighting are decoupled.** Top-K experts are chosen by
  ``logits + expert_bias`` (a non-trainable buffer). The token→expert mixing
  weight, however, is the *unbiased* ``sigmoid(logits)``. Biases therefore
  nudge *which* experts are used without distorting the representation.
* **Bias update is a direct controller.** After each optimizer step, the
  training loop invokes :meth:`update_expert_bias`; overloaded experts get
  their bias decreased by ``bias_update_speed``, underloaded ones increased.
  No gradient pressure flows from balance back into the features.
* **Sequence-wise auxiliary signal is optional and weak.** A DeepSeek-V3
  style per-sequence balance loss (α ≈ 1e-3) is returned by
  :meth:`get_auxiliary_loss` as a safety net; set ``seq_aux_alpha=0.0`` to
  disable it entirely.
* **Sort-based fixed-capacity dispatch.** Tokens are sorted by expert id and
  routed into an ``[num_experts, capacity, d]`` padded buffer; this keeps
  tensor shapes static (friendly to ``torch.compile``) and scales better
  than the per-expert Python loop in the original implementation.

Classes
-------

* :class:`MoEGate` — aux-loss-free sigmoid-gated Top-K router.
* :class:`SparseMoE` — routed-expert pool with fixed-capacity dispatch.
* :class:`ShareExpertMoE` — DeepSeek-MoE layer: always-on shared expert(s) +
  sparse routed pool.
* :class:`TransformerEncoderBlock_MoE` — pre-norm attention + MoE block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .utils import MLP


class MoEGate(nn.Module):
    """Aux-loss-free sigmoid-gated Top-K router (DeepSeek-V3 style).

    Parameters
    ----------
    emb_size : int
        Token embedding dim.
    num_experts : int
        Number of routed experts.
    top_k : int
        Number of experts each token is dispatched to.
    bias_update_speed : float
        Per-step increment applied to ``expert_bias`` during rebalancing.
        DeepSeek-V3 uses 1e-3; too large causes oscillation, too small is slow.
    seq_aux_alpha : float
        Weight on the sequence-wise balance loss. Set to 0 to disable.
    """

    def __init__(self, emb_size, num_experts, top_k,
                 bias_update_speed=1e-3, seq_aux_alpha=1e-3):
        super().__init__()
        self.emb_size = emb_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_speed = bias_update_speed
        self.seq_aux_alpha = seq_aux_alpha

        self.weight = nn.Parameter(torch.empty(num_experts, emb_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Non-trainable buffers; updated directly after each training step.
        self.register_buffer('expert_bias',
                             torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer('expert_load',
                             torch.zeros(num_experts, dtype=torch.long))

    def forward(self, x):
        """Route tokens of ``x`` [B, N, C] to Top-K experts.

        Returns
        -------
        topk_idx : LongTensor [B*N, top_k]
            Selected expert ids per token.
        topk_weights : FloatTensor [B*N, top_k]
            Renormalized sigmoid scores of the selected experts; used as the
            token→expert mixing coefficients downstream.
        aux_seq_loss : Tensor
            Sequence-wise balance loss (weighted by ``seq_aux_alpha``). Zero
            tensor on the device when disabled or in eval mode.
        """
        bsz, seq_len, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])

        logits = x_flat @ self.weight.t()        # [B*N, E]
        scores = torch.sigmoid(logits)           # unbiased routing prob

        # Biased score is used ONLY for Top-K selection, not for weighting.
        biased = logits + self.expert_bias

        _, topk_idx = torch.topk(biased, k=self.top_k, dim=-1, sorted=False)
        topk_weights = torch.gather(scores, dim=-1, index=topk_idx)
        if self.top_k > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-10)

        if self.training:
            # Record load for next bias update; detach to avoid grad flow.
            expert_counts = F.one_hot(
                topk_idx.reshape(-1), num_classes=self.num_experts
            ).sum(dim=0)
            self.expert_load.copy_(expert_counts.detach())

            if self.seq_aux_alpha > 0:
                aux_seq_loss = self._sequence_balance_loss(
                    scores, topk_idx, bsz, seq_len
                )
            else:
                aux_seq_loss = x.new_zeros(())
        else:
            aux_seq_loss = x.new_zeros(())

        return topk_idx, topk_weights, aux_seq_loss

    def _sequence_balance_loss(self, scores, topk_idx, bsz, seq_len):
        """DeepSeek-V3 Eq. 18-20: L_bal = α Σᵢ fᵢ Pᵢ, averaged per-sequence."""
        scores_seq = scores.view(bsz, seq_len, self.num_experts)
        topk_seq = topk_idx.view(bsz, seq_len, self.top_k)

        expert_mask = F.one_hot(
            topk_seq.reshape(bsz, -1), num_classes=self.num_experts
        )  # [B, N*K, E]
        f_i = expert_mask.sum(dim=1).to(scores.dtype) / (self.top_k * seq_len)

        norm = scores_seq.sum(dim=-1, keepdim=True) + 1e-10
        p_i = (scores_seq / norm).mean(dim=1)          # [B, E]

        loss = (f_i * p_i).sum(dim=-1).mean()
        return self.seq_aux_alpha * loss

    @torch.no_grad()
    def update_bias(self):
        """Nudge ``expert_bias`` toward a uniform load distribution.

        Overloaded experts → bias decreases → less likely to be Top-K.
        Underloaded experts → bias increases → more likely to be Top-K.

        The expected uniform load is derived from ``expert_load.sum()`` —
        no external token count is needed, and the bias step is self-calibrating.
        """
        if not self.training:
            return
        total_assignments = self.expert_load.sum()
        if total_assignments == 0:  # no forward yet this epoch
            return
        expected = total_assignments.float() / self.num_experts
        diff = self.expert_load.float() - expected
        self.expert_bias -= torch.sign(diff) * self.bias_update_speed


class SparseMoE(nn.Module):
    """Routed-expert pool with aux-loss-free gating and sort-based dispatch.

    Fixed-capacity padding keeps tensor shapes static (``[E, C, d]``) so the
    expert loop is JIT/compile-friendly and avoids GPU-CPU sync from
    variable-length splits.

    Parameters
    ----------
    capacity_factor : float
        Buffer fraction on top of the ideal load. 1.25 is the DeepSeek-V3
        default; with good balancing 1.0 is already enough. Tokens beyond
        capacity are dropped (their weight contribution to the output is 0).
    """

    def __init__(self, emb_size, num_experts, top_k, expansion,
                 dropout=0.1, capacity_factor=1.25,
                 bias_update_speed=1e-3, seq_aux_alpha=1e-3):
        super().__init__()
        self.emb_size = emb_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.gate = MoEGate(emb_size, num_experts, top_k,
                            bias_update_speed=bias_update_speed,
                            seq_aux_alpha=seq_aux_alpha)
        self.experts = nn.ModuleList(
            [MLP(emb_size, expansion, dropout) for _ in range(num_experts)]
        )

        # Cached at forward for telemetry.
        self._last_aux_loss: torch.Tensor | None = None

    @staticmethod
    def _within_expert_positions(sorted_expert_ids: torch.Tensor) -> torch.Tensor:
        """Return each token's rank within its expert segment, fully on GPU.

        Given sorted expert ids ``[0,0,2,2,2,5]``, returns ``[0,1,0,1,2,0]``
        via a cummax trick on segment-boundary positions — no GPU-CPU sync.
        """
        n = sorted_expert_ids.shape[0]
        device = sorted_expert_ids.device
        global_pos = torch.arange(n, device=device)
        boundary_markers = torch.zeros(n, device=device, dtype=torch.long)
        if n > 1:
            mask = sorted_expert_ids[1:] != sorted_expert_ids[:-1]
            boundary_markers[1:] = mask * global_pos[1:]
        seg_starts = torch.cummax(boundary_markers, dim=0).values
        return global_pos - seg_starts

    def forward(self, x):
        B, N, C = x.shape
        n_total = B * N
        x_flat = x.reshape(n_total, C)

        topk_idx, topk_weights, aux_seq_loss = self.gate(x)
        self._last_aux_loss = aux_seq_loss

        flat_idx = topk_idx.reshape(-1)                  # [n_total * top_k]
        flat_weights = topk_weights.reshape(-1)
        token_indices = torch.arange(
            n_total, device=x.device
        ).repeat_interleave(self.top_k)

        # Sort by expert id so tokens targeting the same expert are contiguous.
        # ``stable=True`` guarantees identical permutations on repeat calls,
        # which matters when capacity overflow drops tokens — otherwise the
        # same forward on the same input can drop *different* tokens on CUDA
        # and the model appears non-deterministic in eval.
        sorted_expert_ids, perm = torch.sort(flat_idx, stable=True)
        sorted_target = token_indices[perm]
        sorted_tokens = x_flat[sorted_target]
        sorted_weights = flat_weights[perm]

        within_pos = self._within_expert_positions(sorted_expert_ids)

        capacity = math.ceil(
            n_total * self.top_k / self.num_experts * self.capacity_factor
        )
        capacity = max(capacity, 1)

        # Tokens beyond capacity are routed to a dedicated "trash" slot at
        # index ``capacity`` that is never read by any kept token. This is
        # crucial on CUDA: if overflow tokens were clamped to ``capacity-1``,
        # their scatter writes would non-deterministically race with kept
        # tokens' writes, breaking eval reproducibility.
        kept_mask = (within_pos < capacity).to(x_flat.dtype)  # [n_total*top_k]
        within_pos_safe = torch.where(
            within_pos < capacity,
            within_pos,
            torch.full_like(within_pos, capacity),  # trash slot
        )

        padded_in = x_flat.new_zeros(self.num_experts, capacity + 1, C)
        padded_in[sorted_expert_ids, within_pos_safe] = sorted_tokens

        # Use torch.stack so each expert's output is a fresh tensor on the
        # autograd graph — avoids in-place aliasing on a single buffer.
        padded_out = torch.stack(
            [self.experts[eid](padded_in[eid]) for eid in range(self.num_experts)],
            dim=0,
        )

        y_sorted = padded_out[sorted_expert_ids, within_pos_safe]
        weighted = y_sorted * (sorted_weights * kept_mask).unsqueeze(-1)

        # Inverse permutation reorders the sorted assignments back into
        # ``[token_0_k0, token_0_k1, ..., token_(N-1)_k0, token_(N-1)_k1]``
        # so we can sum the top_k contributions along a contiguous axis.
        # This replaces ``index_add_`` (non-deterministic on CUDA due to
        # atomic-add ordering) with a deterministic reduction.
        inv_perm = torch.argsort(perm, stable=True)
        unsorted_weighted = weighted[inv_perm].view(n_total, self.top_k, C)
        output = unsorted_weighted.sum(dim=1)

        return output.view(B, N, C)

    def get_aux_loss(self) -> torch.Tensor:
        if self._last_aux_loss is None:
            return torch.tensor(0.0)
        return self._last_aux_loss

    def get_expert_load(self):
        return self.gate.expert_load.detach().cpu().numpy()

    def update_expert_bias(self):
        self.gate.update_bias()


class ShareExpertMoE(nn.Module):
    """DeepSeek-MoE layer: ``share_experts`` always-on expert(s) + routed pool.

    The shared expert sees every token and captures global knowledge that
    doesn't need to be specialized; routed experts capture the specialized
    modes. Output is ``shared(x) + routed(x)`` (sum, not mean — the original
    implementation averaged the shared pool, which under-weighted it).

    Parameters
    ----------
    total_experts : int
        Total expert count (shared + routed).
    share_experts : int
        Number of always-on experts. DeepSeek-V3 uses 1.
    """

    def __init__(self, emb_size, total_experts, share_experts, top_k, expansion,
                 dropout=0.1, capacity_factor=1.25,
                 bias_update_speed=1e-3, seq_aux_alpha=1e-3):
        super().__init__()
        assert 0 < share_experts < total_experts, \
            'share_experts must satisfy 0 < share_experts < total_experts'
        self.shared_experts = nn.ModuleList(
            [MLP(emb_size, expansion, dropout) for _ in range(share_experts)]
        )
        self.routed_moe = SparseMoE(
            emb_size=emb_size,
            num_experts=total_experts - share_experts,
            top_k=top_k,
            expansion=expansion,
            dropout=dropout,
            capacity_factor=capacity_factor,
            bias_update_speed=bias_update_speed,
            seq_aux_alpha=seq_aux_alpha,
        )

    def forward(self, x):
        routed_out = self.routed_moe(x)
        shared_out = sum(e(x) for e in self.shared_experts)
        return routed_out + shared_out

    def get_expert_loads(self):
        return self.routed_moe.get_expert_load()

    def get_auxiliary_loss(self) -> torch.Tensor:
        return self.routed_moe.get_aux_loss()

    def update_expert_bias(self):
        self.routed_moe.update_expert_bias()


class TransformerEncoderBlock_MoE(nn.Module):
    """Pre-norm attention + MoE block (drop-in for the standard ``MLP`` block)."""

    def __init__(self, emb_size, head_num, expert_num, share_experts, top_k,
                 expansion=4, drop_p=0.1, attn_drop=0.0,
                 capacity_factor=1.25, bias_update_speed=1e-3, seq_aux_alpha=1e-3,
                 qk_norm=True):
        super().__init__()
        self.moe_norm = nn.LayerNorm(emb_size)
        self.att_norm = nn.LayerNorm(emb_size)

        self.attention = MultiHeadAttention(emb_size, head_num,
                                            attn_dropout=attn_drop, proj_dropout=drop_p,
                                            qk_norm=qk_norm)
        self.moe = ShareExpertMoE(emb_size, expert_num, share_experts, top_k, expansion,
                                  dropout=drop_p,
                                  capacity_factor=capacity_factor,
                                  bias_update_speed=bias_update_speed,
                                  seq_aux_alpha=seq_aux_alpha)

    def forward(self, x):
        x = x + self.attention(self.att_norm(x))
        x = x + self.moe(self.moe_norm(x))
        return x

    def get_expert_loads(self):
        return self.moe.get_expert_loads()

    def get_auxiliary_loss(self) -> torch.Tensor:
        return self.moe.get_auxiliary_loss()

    def update_expert_bias(self):
        self.moe.update_expert_bias()


__all__ = ['MoEGate', 'SparseMoE', 'ShareExpertMoE', 'TransformerEncoderBlock_MoE']
