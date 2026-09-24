from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from scratch_llm.config import ModelConfig
from scratch_llm.model.layer_assignment import build_layers
from scratch_llm.model.norm import RMSNorm
from scratch_llm.model.packed_mask import build_packed_attn_mask
from scratch_llm.model.rope import RotaryEmbedding


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(build_layers(config))
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.config.max_seq_len}")

        x = self.tok_embeddings(input_ids)
        cos, sin = self.rope(T)
        cos = cos.to(device=x.device)
        sin = sin.to(device=x.device)

        attn_mask = None
        if self.config.eos_token_id is not None:
            attn_mask = build_packed_attn_mask(input_ids, self.config.eos_token_id)

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss
