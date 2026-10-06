"""What a checkpoint folder is, from its files: the formats the studio trains and exports.

A family (families.py) is a way of building a decision model; a kind is a checkpoint format
the studio has a trainer for. Each kind keeps its maker's own files, so a fine-tune loads in
the maker's own runtime as the base model did:

    laya     Laya (Convai Innovations): rl_agent_config.json, encoder/, tokenizer/,
             model.safetensors. Trained by engine.py (MLX) and torch_engine.py (PyTorch).
    julia    Julia 1 (Supersonic Labs): julia_config.json, inference-policy.json, encoder/,
             tokenizer/, model.safetensors in float32. The same network as Laya with its own
             prompt, budgets and file layout (julia.py), trained by the same two trainers.
    decider  Decider (Mapika): a Qwen3.5 causal model with decider_config.json, read by its
             answer letters. Trained with LoRA by decider_engine.py (PyTorch + PEFT) and
             decider_mlx.py (MLX-LM).
"""

from pathlib import Path

LAYA, JULIA, DECIDER = "laya", "julia", "decider"
KINDS = (LAYA, JULIA, DECIDER)

# The files a checkpoint of each kind cannot do without. Decider's weights may be sharded,
# so its safetensors are checked separately.
REQUIRED = {
    LAYA: (
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
    ),
    JULIA: (
        "model.safetensors",
        "julia_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ),
    DECIDER: ("config.json", "decider_config.json", "tokenizer.json", "tokenizer_config.json"),
}

# What to fetch from Hugging Face for each kind: the checkpoint, never the makers' extra
# formats (GGUF, ONNX), their benchmark files or images.
DOWNLOAD = {
    LAYA: ("model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"),
    JULIA: (
        "model.safetensors",
        "julia_config.json",
        "inference-policy.json",
        "config.json",
        "encoder/*",
        "tokenizer/*",
    ),
    DECIDER: (
        "*.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "decider_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "decider/*.py",
    ),
}

# The family each kind belongs to (families.FAMILIES) and the NoulXP profile its packages use.
FAMILY = {LAYA: "laya", JULIA: "laya", DECIDER: "letter"}
PROFILE = {LAYA: "encoder-markers", JULIA: "encoder-markers", DECIDER: "causal-letters"}
NAME = {LAYA: "Laya", JULIA: "Julia 1", DECIDER: "Decider"}


def detect(path):
    """The kind of a checkpoint folder, from the config file that names it; None if unknown."""
    path = Path(path)
    if (path / "rl_agent_config.json").is_file():
        return LAYA
    if (path / "julia_config.json").is_file():
        return JULIA
    if (path / "decider_config.json").is_file():
        return DECIDER
    return None


def weights(path, kind):
    """The safetensors files of a checkpoint, in order."""
    path = Path(path)
    if kind == DECIDER:
        return sorted(path.glob("*.safetensors"))
    return [path / "model.safetensors"]


def check(path, kind=None):
    """The kind of a complete checkpoint, or FileNotFoundError naming what is missing."""
    path = Path(path)
    kind = kind or detect(path)
    if kind is None:
        raise FileNotFoundError(
            f"Not a checkpoint the studio can train: {path} has no rl_agent_config.json (Laya), "
            "julia_config.json (Julia 1) or decider_config.json (Decider)"
        )
    for name in REQUIRED[kind]:
        if not (path / name).is_file():
            label = "Laya checkpoint" if kind == LAYA else f"{NAME[kind]} checkpoint"
            raise FileNotFoundError(f"Not a complete {label}: {path / name} is missing")
    if not weights(path, kind) or not all(f.is_file() for f in weights(path, kind)):
        raise FileNotFoundError(f"Not a complete {NAME[kind]} checkpoint: {path} has no weights")
    return kind


def of_repo(repo):
    """The kind a Hugging Face repository holds, from the catalogue (Laya when unknown)."""
    from .families import find

    known = find(repo or "")
    return (known.kind if known and known.kind else None) or LAYA
