from typing import Optional

import torch
import torch.nn as nn

from scratch_llm.config import ModelConfig
from scratch_llm.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """A pre-norm attention+FFN block.

    Attention and FFN modules are passed in rather than constructed here, so
    a single module instance can be reused across multiple blocks (weight
    sharing) without any change to this class — see
    `scratch_llm.model.layer_assignment` for where sharing is decided.
    attn_norm/ffn_norm are always created fresh per block instance (never
    shared), so LayerNorm stays independent per layer even when attn/ffn are
    shared (PRD variant 3).
    """

    def __init__(self, attn: nn.Module, ffn: nn.Module, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = attn
        self.ffn = ffn

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
