from typing import Dict, List, Union

import torch.nn as nn

from scratch_llm.config import ModelConfig
from scratch_llm.model.attention import CausalSelfAttention
from scratch_llm.model.block import TransformerBlock
from scratch_llm.model.ffn import SwiGLUFFN

# Registry of supported sharing modes:
#   "none"    - every layer gets its own module instance (variant 1).
#   "full"    - a single module instance is reused by every layer (variant 2
#               when applied to both attn and ffn; variant 3 when applied to
#               attn only, leaving ffn on "none").
#   int N     - "stage" grouping: n_layers is split into N contiguous blocks
#               (n_layers // N layers each), e.g. N=2 is A*8 B*8; every layer
#               in a block shares one module instance. N=n_layers is
#               equivalent to "none", N=1 is equivalent to "full".
#   "cyclicN" - "cyclic" grouping: N unique instances assigned round-robin by
#               layer_idx % N, e.g. N=2 is (AB)*8 -- same unique-instance
#               count as int N but interleaved instead of blocked.
# No change to Attention/FFN/TransformerBlock/Transformer is needed to
# support any of this.
def _sharing_key(mode: Union[str, int], layer_idx: int, n_layers: int) -> str:
    if mode == "none":
        return f"layer{layer_idx}"
    if mode == "full":
        return "shared"
    if isinstance(mode, str) and mode.startswith("cyclic"):
        n = int(mode[len("cyclic"):])
        if n <= 0 or n_layers % n != 0:
            raise ValueError(
                f"Cyclic sharing group count {n!r} must be a positive divisor of n_layers ({n_layers})"
            )
        return f"group{layer_idx % n}"
    if isinstance(mode, int):
        if mode <= 0 or n_layers % mode != 0:
            raise ValueError(
                f"Sharing group count {mode!r} must be a positive divisor of n_layers ({n_layers})"
            )
        group_size = n_layers // mode
        return f"group{layer_idx // group_size}"
    raise ValueError(f"Unsupported sharing mode: {mode!r}")


def build_layers(config: ModelConfig) -> List[TransformerBlock]:
    attn_modules: Dict[str, nn.Module] = {}
    ffn_modules: Dict[str, nn.Module] = {}
    layers: List[TransformerBlock] = []

    for layer_idx in range(config.n_layers):
        attn_key = _sharing_key(config.attn_sharing, layer_idx, config.n_layers)
        ffn_key = _sharing_key(config.ffn_sharing, layer_idx, config.n_layers)

        if attn_key not in attn_modules:
            attn_modules[attn_key] = CausalSelfAttention(config)
        if ffn_key not in ffn_modules:
            ffn_modules[ffn_key] = SwiGLUFFN(config)

        layers.append(TransformerBlock(attn_modules[attn_key], ffn_modules[ffn_key], config))

    return layers
