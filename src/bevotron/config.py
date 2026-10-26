"""Typed run configuration: load, validate, count parameters and bytes.

A run config is the baseline YAML with dotted-key overrides applied. Everything
that defines a run lives here, so a run ID can be a hash of this object.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Presets fill the knobs an ablation file does not mention.
PRESETS: dict[str, dict[str, Any]] = {
    "bevo": {"norm": "layernorm", "pos_encoding": "learned", "mlp_act": "gelu"},
    "llama": {"norm": "rmsnorm", "pos_encoding": "rope", "mlp_act": "swiglu"},
}

# Bytes per value for (weights, activations, KV cache) under each scheme.
QUANT_BYTES: dict[str, tuple[float, float, float]] = {
    "none": (4.0, 4.0, 4.0),
    "int8": (1.0, 1.0, 1.0),
    "w4a8": (0.5, 1.0, 1.0),
    "w4a4": (0.5, 0.5, 0.5),  # KV cache at 4 bits too; same as w4a4kv4
    "w4a4kv4": (0.5, 0.5, 0.5),
    "w4a8kv4": (0.5, 1.0, 0.5),
}


def parse_count(value: int | str) -> int:
    """Accept 504000000, '504M', '2.1B', '16K'."""
    if isinstance(value, int):
        return value
    text = str(value).strip().upper()
    scale = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(text[-1], None)
    return int(float(text[:-1]) * scale) if scale else int(float(text))


@dataclass
class ModelConfig:
    preset: str = "bevo"
    n_layer: int = 8
    d_model: int = 512
    n_head: int = 16
    n_kv_head: int = 16
    mlp_ratio: float = 4
    context_len: int = 512
    norm: str | None = None  # layernorm | rmsnorm (None = preset default)
    pos_encoding: str | None = None  # learned | rope
    mlp_act: str | None = None  # gelu | swiglu
    attention_window: int | None = None  # local-attention window, None = full
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


@dataclass
class TokenizerConfig:
    vocab_size: int = 50000
    path: str | None = None  # trained tokenizer JSON; default derived from vocab_size


@dataclass
class DataConfig:
    dataset: str = "tinystories"


@dataclass
class TrainConfig:
    tokens: int | str = "504M"
    batch_tokens: int = 16384
    lr: float = 6.0e-4
    warmup_frac: float = 0.05
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    seed: int = 1337
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 50  # quick val estimate during training; final eval is the full split
    device: str = "auto"  # auto | cuda | cpu

    @property
    def total_tokens(self) -> int:
        return parse_count(self.tokens)

    @property
    def steps(self) -> int:
        return self.total_tokens // self.batch_tokens


@dataclass
class QuantConfig:
    scheme: str = "int8"  # none | int8 | w4a8 | w4a4kv4 | w4a8kv4
    mode: str = "ptq"  # ptq: quantize the trained checkpoint; qat: fine-tune with fake quant (STE)
    # QAT starts from the trained full-precision checkpoint (PTQ comes first), then fine-tunes:
    qat_tokens: int | str = "50M"
    qat_lr: float = 1.0e-4

    @property
    def qat_total_tokens(self) -> int:
        return parse_count(self.qat_tokens)


@dataclass
class EvalConfig:
    metrics: list[str] = field(default_factory=lambda: ["perplexity", "bpb", "memory", "judge"])
    # LLM-as-judge: the model continues the opening of validation stories; a judge grades them.
    judge_model: str = "claude-sonnet-4-6"
    judge_samples: int = 100
    judge_prompt_tokens: int = 16
    judge_max_new_tokens: int = 200
    judge_temperature: float = 0.7


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        self._apply_preset()
        self.validate()

    def _apply_preset(self) -> None:
        if self.model.preset not in PRESETS:
            raise ValueError(f"Unknown preset {self.model.preset!r}; choose {list(PRESETS)}")
        for key, value in PRESETS[self.model.preset].items():
            if getattr(self.model, key) is None:
                setattr(self.model, key, value)

    def validate(self) -> None:
        m = self.model
        problems = []
        if m.d_model % m.n_head:
            problems.append(f"d_model {m.d_model} not divisible by n_head {m.n_head}")
        if m.n_head % m.n_kv_head:
            problems.append(f"n_head {m.n_head} not divisible by n_kv_head {m.n_kv_head}")
        if m.norm not in ("layernorm", "rmsnorm"):
            problems.append(f"norm must be layernorm or rmsnorm, got {m.norm!r}")
        if m.pos_encoding not in ("learned", "rope"):
            problems.append(f"pos_encoding must be learned or rope, got {m.pos_encoding!r}")
        if m.mlp_act not in ("gelu", "swiglu"):
            problems.append(f"mlp_act must be gelu or swiglu, got {m.mlp_act!r}")
        if m.attention_window is not None and m.attention_window > m.context_len:
            problems.append("attention_window larger than context_len")
        if self.tokenizer.vocab_size < 257:
            problems.append("vocab_size must hold 256 bytes plus <|endoftext|>")
        if self.quant.scheme not in QUANT_BYTES:
            problems.append(f"quant.scheme must be one of {list(QUANT_BYTES)}")
        if self.quant.mode not in ("ptq", "qat"):
            problems.append("quant.mode must be ptq or qat")
        if self.quant.mode == "qat" and self.quant.scheme == "none":
            problems.append("quant.mode qat needs a scheme")
        if self.train.total_tokens < self.train.batch_tokens:
            problems.append("train.tokens must be at least one batch")
        if problems:
            raise ValueError("Invalid config: " + "; ".join(problems))

    # ---- parameter counts (biases and norms ignored; <0.1%) ----
    def params_per_layer(self) -> int:
        m = self.model
        kv_dim = m.n_kv_head * m.head_dim
        attention = 2 * m.d_model * m.d_model + 2 * m.d_model * kv_dim  # Q, O + K, V
        hidden = int(m.mlp_ratio * m.d_model)
        mlp = (3 if m.mlp_act == "swiglu" else 2) * m.d_model * hidden
        return attention + mlp

    def non_embedding_params(self) -> int:
        return self.model.n_layer * self.params_per_layer()

    def embedding_params(self) -> int:
        m = self.model
        table = self.tokenizer.vocab_size * m.d_model * (1 if m.tie_embeddings else 2)
        positions = m.context_len * m.d_model if m.pos_encoding == "learned" else 0
        return table + positions

    def total_params(self) -> int:
        return self.non_embedding_params() + self.embedding_params()

    # ---- serialisation ----
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def run_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:10]

    def training_key(self) -> str:
        """Hash of everything that affects the trained weights. PTQ runs that differ only
        in quant.scheme share a key, so one checkpoint serves every PTQ variant."""
        raw = self.to_dict()
        if self.quant.mode == "ptq":
            raw["quant"] = {}
        raw.pop("eval", None)
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:10]

    def full_precision_twin(self) -> RunConfig:
        """The same run with quantization off: the checkpoint a QAT run fine-tunes from."""
        raw = self.to_dict()
        raw["quant"] = {"scheme": "none", "mode": "ptq"}
        return RunConfig.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        sections = {f.name: f.type for f in fields(cls)}
        unknown = set(raw) - set(sections)
        if unknown:
            raise ValueError(f"Unknown config section(s): {sorted(unknown)}")
        built = {}
        for name, section_cls in (
            ("model", ModelConfig),
            ("tokenizer", TokenizerConfig),
            ("data", DataConfig),
            ("train", TrainConfig),
            ("quant", QuantConfig),
            ("eval", EvalConfig),
        ):
            values = raw.get(name, {}) or {}
            allowed = {f.name for f in fields(section_cls)}
            bad = set(values) - allowed
            if bad:
                raise ValueError(f"Unknown key(s) in {name}: {sorted(bad)}")
            built[name] = section_cls(**values)
        return cls(**built)

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        return cls.from_dict(load_raw(path))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def load_raw(path: str | Path) -> dict[str, Any]:
    """The YAML as written. Apply overrides to this, not to a resolved config:
    a preset only fills knobs the file leaves unset, so switching presets on a
    resolved dict (where norm/pos_encoding/mlp_act are already filled) is a no-op."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of base with dotted keys ('model.n_layer') set."""
    result = copy.deepcopy(base)
    for dotted, value in overrides.items():
        node = result
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return result
