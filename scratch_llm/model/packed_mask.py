import torch


def build_packed_attn_mask(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Causal mask that additionally blocks attention across EOS boundaries.

    Packed chunks concatenate multiple documents separated by `eos_token_id`.
    Without this, a token in document N could attend into document N-1
    through the ordinary causal mask alone, since a packed chunk has no
    padding between documents. Each token is assigned a segment id (an
    exclusive running count of EOS tokens seen so far), so the EOS token
    itself belongs to the segment it closes, and the token right after it
    starts a new segment; two positions may attend to each other only if
    they share a segment id and satisfy the usual causal (j <= i) order.

    Args:
        input_ids: (B, T) token ids.
        eos_token_id: token id marking a document/sentence boundary.

    Returns:
        (B, 1, T, T) bool tensor, True where attention is allowed.
    """
    B, T = input_ids.shape
    is_eos = (input_ids == eos_token_id).to(torch.int32)
    segment_id = torch.cumsum(is_eos, dim=-1) - is_eos  # (B, T)

    same_segment = segment_id.unsqueeze(-1) == segment_id.unsqueeze(-2)  # (B, T, T)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=input_ids.device))
    mask = same_segment & causal.unsqueeze(0)
    return mask.unsqueeze(1)  # (B, 1, T, T)
