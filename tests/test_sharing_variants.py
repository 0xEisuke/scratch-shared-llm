import torch

from scratch_llm.checkpoint import load_checkpoint, save_checkpoint
from scratch_llm.config import ModelConfig
from scratch_llm.model.transformer import Transformer

BASE_KWARGS = dict(
    vocab_size=64,
    d_model=32,
    n_layers=4,
    n_heads=4,
    d_ff=64,
    max_seq_len=16,
    tie_embeddings=True,
    eos_token_id=1,
)


def _config(attn_sharing: str, ffn_sharing: str) -> ModelConfig:
    return ModelConfig(attn_sharing=attn_sharing, ffn_sharing=ffn_sharing, **BASE_KWARGS)


def _forward_backward(model: Transformer):
    torch.manual_seed(0)
    x = torch.randint(0, BASE_KWARGS["vocab_size"], (2, 10))
    y = torch.randint(0, BASE_KWARGS["vocab_size"], (2, 10))
    logits, loss = model(x, y)
    loss.backward()
    return logits, loss


def test_variant1_standard_independent_layers():
    model = _config("none", "none")
    m = Transformer(model)
    assert m.layers[0].attn is not m.layers[1].attn
    assert m.layers[0].ffn is not m.layers[1].ffn

    logits, loss = _forward_backward(m)
    assert torch.isfinite(loss)
    assert m.layers[0].attn.wq.weight.grad is not None
    assert m.layers[1].attn.wq.weight.grad is not None


def test_variant2_full_layer_sharing():
    cfg = _config("full", "full")
    m = Transformer(cfg)
    for i in range(1, cfg.n_layers):
        assert m.layers[0].attn is m.layers[i].attn
        assert m.layers[0].ffn is m.layers[i].ffn
        # LayerNorm must stay independent even when attn/ffn are shared.
        assert m.layers[0].attn_norm is not m.layers[i].attn_norm
        assert m.layers[0].ffn_norm is not m.layers[i].ffn_norm

    logits, loss = _forward_backward(m)
    assert torch.isfinite(loss)
    assert m.layers[0].attn.wq.weight.grad is not None


def test_variant3_attention_only_sharing():
    cfg = _config("full", "none")
    m = Transformer(cfg)
    for i in range(1, cfg.n_layers):
        assert m.layers[0].attn is m.layers[i].attn
        assert m.layers[0].ffn is not m.layers[i].ffn

    logits, loss = _forward_backward(m)
    assert torch.isfinite(loss)
    assert m.layers[0].attn.wq.weight.grad is not None
    assert m.layers[0].ffn.gate_proj.weight.grad is not None
    assert m.layers[1].ffn.gate_proj.weight.grad is not None


def test_checkpoint_roundtrip_preserves_sharing_and_forward_output(tmp_path):
    cfg = _config("full", "full")
    m1 = Transformer(cfg)
    optimizer = torch.optim.AdamW(m1.parameters(), lr=1e-3)

    ckpt_path = str(tmp_path / "ckpt.pt")
    from scratch_llm.config import TrainConfig

    save_checkpoint(ckpt_path, m1, optimizer, step=1, tokens_seen=100, model_config=cfg, train_config=TrainConfig())

    m2 = Transformer(cfg)
    load_checkpoint(ckpt_path, m2, map_location="cpu")

    # sharing must survive a fresh build_layers() + load_state_dict() cycle
    for i in range(1, cfg.n_layers):
        assert m2.layers[0].attn is m2.layers[i].attn
        assert m2.layers[0].ffn is m2.layers[i].ffn

    m1.eval()
    m2.eval()
    x = torch.randint(0, BASE_KWARGS["vocab_size"], (2, 10))
    with torch.no_grad():
        out1, _ = m1(x)
        out2, _ = m2(x)
    assert torch.allclose(out1, out2)
