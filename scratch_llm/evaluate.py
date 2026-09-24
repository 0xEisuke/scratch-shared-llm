import math
from typing import Optional

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    max_batches: Optional[int] = None,
) -> dict:
    """Token-weighted cross-entropy loss and perplexity over a data loader.

    Weighting by token count (rather than averaging per-batch means) keeps
    the estimate correct even if batches end up with different token counts.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            _, loss = model(x, y)
        n_tokens = y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    model.train()
    mean_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(min(mean_loss, 50.0))
    return {"loss": mean_loss, "perplexity": perplexity, "tokens": total_tokens}
