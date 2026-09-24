from typing import Iterable

import torch.nn as nn

from scratch_llm.model.transformer import Transformer


def _count_unique(params: Iterable[nn.Parameter]) -> int:
    seen = set()
    total = 0
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def count_parameters(model: Transformer) -> dict:
    """Total parameter count plus a breakdown by component.

    Parameters are de-duplicated by identity, so this reports the correct
    total even when attention/FFN modules are shared across layers (future
    weight-sharing variants) or embeddings are tied to the LM head.
    """
    attn_params = []
    ffn_params = []
    norm_params = []
    for layer in model.layers:
        attn_params += list(layer.attn.parameters())
        ffn_params += list(layer.ffn.parameters())
        norm_params += list(layer.attn_norm.parameters())
        norm_params += list(layer.ffn_norm.parameters())
    norm_params += list(model.norm.parameters())

    tied = model.lm_head.weight is model.tok_embeddings.weight

    breakdown = {
        "tok_embeddings": _count_unique(model.tok_embeddings.parameters()),
        "attention": _count_unique(attn_params),
        "ffn": _count_unique(ffn_params),
        "norms": _count_unique(norm_params),
        "lm_head": 0 if tied else _count_unique(model.lm_head.parameters()),
        "tied_embeddings": tied,
        "total": _count_unique(model.parameters()),
    }
    return breakdown


def format_breakdown(breakdown: dict) -> str:
    lines = ["Parameter breakdown:"]
    for key in ("tok_embeddings", "attention", "ffn", "norms", "lm_head"):
        lines.append(f"  {key:15s}: {breakdown[key]:>14,}")
    lines.append(f"  {'-' * 32}")
    lines.append(f"  {'total':15s}: {breakdown['total']:>14,}")
    lines.append(f"  (tied_embeddings={breakdown['tied_embeddings']})")
    return "\n".join(lines)
