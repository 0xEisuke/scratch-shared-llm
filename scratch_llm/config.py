from dataclasses import asdict, dataclass
from typing import Optional, Union


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = False

    # Sharing mode for attention/FFN modules across layers: "none" (every
    # layer gets its own module instance), "full" (a single shared instance
    # is reused by every layer), an int N (2 <= N < n_layers, must evenly
    # divide n_layers) that splits the n_layers into N contiguous "stage"
    # groups (e.g. N=2 is A*8 B*8), or a string "cyclicN" that assigns N
    # unique instances round-robin by layer index (e.g. "cyclic2" is
    # (AB)*8) -- same unique-instance count as int N but interleaved. See
    # `scratch_llm.model.layer_assignment.build_layers` for how a mode maps
    # to layer->module assignment.
    attn_sharing: Union[str, int] = "none"
    ffn_sharing: Union[str, int] = "none"

    # Token id that marks a document/sentence boundary in packed sequences.
    # When set, the model builds an attention mask that additionally blocks
    # attention across this boundary (on top of the causal mask), so a
    # packed chunk containing multiple concatenated documents doesn't let
    # later documents attend into earlier ones. None keeps the plain causal
    # mask (e.g. for unpacked / single-document training).
    eos_token_id: Optional[int] = None

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 1000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bf16"  # "bf16" | "fp32"
    device: str = "cuda"
    seed: int = 1337
    log_interval: int = 10
    eval_interval: int = 100
    eval_iters: int = 20
    save_interval: int = 200
    out_dir: str = "checkpoints/run"

    # If set, a JSON line with this run's summary metrics (params, val loss/
    # perplexity, wall-clock time, peak VRAM, tokens processed) is appended
    # to `results_path` at the end of training, for cross-model comparison.
    run_name: str = ""
    results_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**d)


def load_yaml_config(path: str, cls):
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return cls.from_dict(data)
