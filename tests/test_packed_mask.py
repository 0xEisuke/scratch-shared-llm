import torch

from scratch_llm.model.packed_mask import build_packed_attn_mask


def test_single_document_matches_plain_causal_mask():
    """No EOS in the sequence at all -> should reduce to an ordinary causal
    mask (every earlier position is attendable)."""
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    eos_id = 999
    mask = build_packed_attn_mask(ids, eos_id)[0, 0]

    expected = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    assert torch.equal(mask, expected)


def test_blocks_attention_across_eos_boundary():
    """tokens: [a, b, EOS, c, d, EOS, e]. Position of the EOS token itself
    stays part of the segment it closes; the token right after starts a new
    segment that cannot see back across the boundary."""
    eos_id = 0
    ids = torch.tensor([[5, 6, eos_id, 7, 8, eos_id, 9]])
    mask = build_packed_attn_mask(ids, eos_id)[0, 0]

    # segment ids: [0,0,0, 1,1,1, 2]
    # position 3 ('c', first token of segment 1) must NOT attend to
    # positions 0/1/2 (segment 0), only to itself.
    assert mask[3, 0].item() is False
    assert mask[3, 1].item() is False
    assert mask[3, 2].item() is False
    assert mask[3, 3].item() is True

    # position 6 ('e', segment 2) must only attend to itself.
    assert mask[6, :6].any().item() is False
    assert mask[6, 6].item() is True

    # the EOS token at position 2 is still part of segment 0, so position 4
    # ('d', segment 1) may attend to position 3 ('c') and itself, but not
    # to segment 0 (positions 0-2).
    assert mask[4, 3].item() is True
    assert mask[4, 4].item() is True
    assert mask[4, 0].item() is False
    assert mask[4, 2].item() is False

    # causality is still enforced within a segment: position 3 cannot see
    # position 4 (future).
    assert mask[3, 4].item() is False


def test_batch_dimension_is_independent():
    eos_id = 0
    ids = torch.tensor(
        [
            [5, eos_id, 6, 7],
            [5, 6, 7, eos_id],
        ]
    )
    mask = build_packed_attn_mask(ids, eos_id)

    # batch 0: position 2 (new segment after EOS at 1) cannot see position 0
    assert mask[0, 0, 2, 0].item() is False
    # batch 1: no EOS until the very last position, so this is plain causal
    expected_causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(mask[1, 0], expected_causal)
