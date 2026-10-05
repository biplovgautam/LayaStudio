"""Laya fine-tuning engine: datasets, training, calibration, evaluation, export.

Training runs on MLX on Apple silicon and on PyTorch everywhere else (torch_engine.py);
runtime.py picks. Everything else here — datasets, tokenization, calibration, metrics —
is shared by both.

The web server runs every heavy task as a child process of this module, so a crash, a
cancel or an out-of-memory error never takes the UI down, and GPU memory is returned to
the system as soon as a job ends:

    python -m layastudio.engine run <job_dir>     # reads spec.json, appends events.jsonl

Nothing here sends data anywhere. Training and evaluation jobs run with HF_HUB_OFFLINE=1;
only explicit model downloads and the optional public example datasets use the network.
"""

import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import struct
import sys
import time
import traceback
from pathlib import Path

from . import datasets
from .datasets import (  # noqa: F401 - engine.<name> is how the trainers and the server reach them
    SPLITS,
    STATE_KEYS,
    answer_target,
    argmax,
    assign_splits,
    internal_question,
    label_counts,
    option_labels,
    option_names,
    parse_rows,
    resolve_label,
    validate_questions,
)
from .laya_mlx_free import laya_mlx_module

# laya_mlx's plain-Python helpers: imported normally on Apple silicon, from their files elsewhere.
_common = laya_mlx_module("common")
QTYPES = _common.QTYPES
build_prefix = _common.build_prefix
build_sequence = _common.build_sequence
render_options = _common.render_options
serialize_state = _common.serialize_state
temp_bucket = _common.temp_bucket

PACKAGE = Path(__file__).resolve().parent


def default_workspace():
    """Where datasets, runs and checkpoints live.

    $LAYASTUDIO_HOME wins; a git checkout keeps its workspace beside the code; an installed
    copy uses ~/.layastudio so nothing is written into site-packages.
    """
    if os.environ.get("LAYASTUDIO_HOME"):
        return Path(os.environ["LAYASTUDIO_HOME"]).expanduser().resolve()
    checkout = PACKAGE.parent
    if (checkout / "pyproject.toml").exists():
        return checkout / "workspace"
    return Path.home() / ".layastudio" / "workspace"


WORKSPACE = default_workspace()

BASE_MODELS = {
    "convaiinnovations/laya": "English · ModernBERT-large · 421M · 512 tokens · original PyTorch release",
    "aac6fef/laya-mlx": "English · ModernBERT-large · 421M · 512 tokens · MLX port",
    "aac6fef/laya-multilingual-mlx": "Multilingual · mmBERT-base · 322M · 1,024 tokens",
    "aac6fef/laya-typed-decisions-mlx": "Typed-decisions · ModernBERT-large · 421M · 1,024 tokens",
}
# Fine-tuned checkpoints published from System One Studio runs, fetched at startup so the arena
# works before anyone has trained anything. Override with $LAYASTUDIO_DEMO_MODELS
# ("" disables them, or a comma-separated list of repositories).
DEMO_MODELS = {
    "madhavbiplov/laya-snake-mlx": "Snake · fine-tuned in System One Studio (322M, multilingual base)",
}
if os.environ.get("LAYASTUDIO_DEMO_MODELS") is not None:
    DEMO_MODELS = {
        repo.strip(): "Fine-tuned demo"
        for repo in os.environ["LAYASTUDIO_DEMO_MODELS"].split(",")
        if repo.strip()
    }

CHECKPOINT_FILES = ("model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*")
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")

HYPERPARAMETERS = {
    "method": "lora",  # lora | head | full
    "objective": "proper",  # proper | rlcd | ce
    "epochs": 4,
    "batch_size": 8,
    "grad_accum": 2,
    "lr": 2e-4,  # LoRA adapters, or the unfrozen encoder layers when method == "full"
    "head_lr": 1e-4,  # decision head, scorer and question-type embedding
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # LoRA variants, combinable: DoRA learns each output row's magnitude separately from
    # the adapted direction; rsLoRA scales by alpha / sqrt(rank) instead of alpha / rank,
    # which keeps high ranks training; LoRA+ trains the B matrices at loraplus_ratio times
    # the learning rate (1 = off; the paper suggests 16).
    "dora": False,
    "rslora": False,
    "loraplus_ratio": 1.0,
    "lora_layers": 0,  # 0 = every encoder layer, otherwise the top N
    "full_layers": 4,  # method == "full": unfreeze the top N encoder layers
    "head_dropout": 0.1,  # upstream trains its head with PyTorch's default dropout 0.1
    "weight_decay": 0.01,
    "warmup": 0.06,
    "max_grad_norm": 1.0,
    "shuffle_options": True,
    "class_weighting": "none",  # none | balanced
    "patience": 2,
    "grad_checkpoint": "auto",  # auto | on | off
    "precision": "bfloat16",  # frozen encoder weights; trainable weights always stay float32
    "seed": 13,
}


# Julia 1 trains with Laya's recipe. Its frozen weights stay in float32 by default: the model
# is small (144M, most of it the vocabulary), and the checkpoint is float32 like the release.
JULIA_HYPERPARAMETERS = {**HYPERPARAMETERS, "precision": "float32"}


def hyperparameters(kind="laya"):
    """The default hyperparameters of the trainer for a checkpoint kind (kinds.py)."""
    if kind == "decider":
        from .decider import HYPERPARAMETERS as DECIDER

        return dict(DECIDER)
    if kind == "julia":
        return dict(JULIA_HYPERPARAMETERS)
    return dict(HYPERPARAMETERS)


class Cancelled(Exception):
    pass


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def finite(value):
    """JSON-safe copy: NaN/inf become null so browsers can parse every report."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def read_json(path, default=None):
    """Read JSON, or return the default. Missing, unreadable and half-written files all
    count as absent: the workspace is a folder people (and Finder) poke at."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(finite(value), indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def slugify(text, fallback="item"):
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:40]
    return slug or fallback


def check_id(value):
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        raise ValueError(f"Invalid id: {value!r}")
    return value


# ----------------------------------------------------------------------------- datasets


def create_dataset(
    name,
    questions,
    train_text,
    train_name,
    test_text=None,
    test_name=None,
    seed=13,
    workspace=WORKSPACE,
    example=None,
):
    """Check, split and save a dataset: datasets.validate, then save_dataset."""
    questions, rows, report = datasets.validate(
        questions, train_text, train_name, test_text, test_name, seed
    )
    files = {"train": train_name, "test": test_name}
    return save_dataset(name, questions, rows, report, files, workspace, example)


def save_dataset(
    name, questions, rows, report, files, workspace=WORKSPACE, example=None, reuse=False
):
    """Save a checked dataset (datasets.validate) as datasets/<id>: questions.json,
    rows.jsonl and meta.json. reuse: the same rows saved before are that dataset, not an error."""
    dataset_id = f"{slugify(name, 'dataset')}-{report['sha256'][:8]}"
    path = workspace / "datasets" / dataset_id
    if path.exists():
        meta = read_json(path / "meta.json")
        if reuse and meta and meta.get("sha256") == report["sha256"]:
            return meta
        raise ValueError(f"This exact dataset already exists as {dataset_id}")
    path.mkdir(parents=True)
    write_json(path / "questions.json", questions)
    with open(path / "rows.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "id": dataset_id,
        "name": name,
        "example": example,
        "created": now(),
        "files": files,
        **report,
    }
    write_json(path / "meta.json", meta)
    return meta


def load_dataset(dataset_id, workspace=WORKSPACE):
    path = workspace / "datasets" / check_id(dataset_id)
    questions = read_json(path / "questions.json")
    if questions is None:
        raise FileNotFoundError(f"Dataset {dataset_id} does not exist")
    with open(path / "rows.jsonl") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return questions, rows, read_json(path / "meta.json", {})


# ----------------------------------------------------------------------------- models


def hub_parts(value):
    """(repo, revision) of a hub reference's value: <repo>, or <repo>@<revision> for a pinned
    commit, tag or branch. revision is None for the main branch, as cached."""
    repo, at, revision = str(value).partition("@")
    if at and not (repo and revision):
        raise ValueError(f"Unknown model reference 'hub:{value}'")
    return repo, revision or None


def resolve_model_ref(ref, workspace=WORKSPACE, allow_download=False):
    """'hub:<repo>[@<revision>]', 'run:<id>' or 'path:<dir>' -> local checkpoint directory.

    The folder must be a complete checkpoint of a kind the studio trains (kinds.py): Laya,
    Julia 1 or Decider. Hub repositories are fetched with the files their kind needs, at the
    revision the reference pins (a download by commit leaves no main branch in the cache,
    so a pinned model is found only by its revision)."""
    from . import kinds

    kind, _, value = str(ref).partition(":")
    if kind == "run":
        path = workspace / "runs" / check_id(value) / "model"
    elif kind == "path":
        path = Path(value).expanduser()
    elif kind == "hub":
        from huggingface_hub import snapshot_download

        repo, revision = hub_parts(value)
        try:
            path = Path(
                snapshot_download(
                    repo,
                    revision=revision,
                    allow_patterns=list(kinds.DOWNLOAD[kinds.of_repo(repo)]),
                    local_files_only=not allow_download,
                )
            )
        except Exception as error:
            raise FileNotFoundError(f"{value} is not downloaded yet. Download it first.") from error
    else:
        raise ValueError(f"Unknown model reference {ref!r}")
    kinds.check(path)
    return path


def model_kind(ref, workspace=WORKSPACE):
    """The kind (kinds.py) of the checkpoint a reference points at."""
    from . import kinds

    return kinds.check(resolve_model_ref(ref, workspace))


def hub_cached(repo_id):
    try:
        resolve_model_ref(f"hub:{repo_id}")
        return True
    except (FileNotFoundError, ValueError):
        return False


def safetensors_header(path):
    """Tensor names, shapes and dtypes, read without loading any weights."""
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header


def upstream_name(name):
    """Inverse of laya_mlx.model.sanitize_weights: MLX parameter name -> PyTorch name."""
    name = name.replace(".in_proj.weight", ".in_proj_weight")
    name = name.replace(".in_proj.bias", ".in_proj_bias")
    for prefix in ("scorer", "act_head"):
        if name.startswith(prefix + ".layers."):
            name = prefix + "." + name[len(prefix) + len(".layers.") :]
    return name


# ----------------------------------------------------------------------------- analysis


def percentile(values, q):
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def analyze_dataset(dataset_id, model_dir, workspace=WORKSPACE):
    """Token budget report: truncated states, clipped option labels, thin classes."""
    Tokenizer = laya_mlx_module("tokenizer").Tokenizer

    questions, rows, meta = load_dataset(dataset_id, workspace)
    tok = Tokenizer(Path(model_dir) / "tokenizer")
    cfg = read_json(Path(model_dir) / "rl_agent_config.json")
    max_len, head_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
    state_tokens = [
        len(tok(serialize_state(r["state"]).replace(tok.mask_token, " "))["input_ids"])
        for r in rows
    ]
    report = {
        "model_dir": str(model_dir),
        "max_len": max_len,
        "head_max_len": head_len,
        "state_tokens": {
            "p50": percentile(state_tokens, 0.5),
            "p95": percentile(state_tokens, 0.95),
            "max": max(state_tokens),
        },
        "questions": {},
        "warnings": [],
    }
    for qid, qdef in questions.items():
        q = internal_question(qdef)
        options = render_options(q)
        ids, markers = build_prefix(tok, q, head_len)
        room = max(0, max_len - len(ids) - 1)
        full = [1 + min(48, len(tok(" " + o)["input_ids"])) for o in options]
        used = [b - a for a, b in zip(markers, markers[1:] + [len(ids) - 1])]
        clipped = [option_labels(qdef)[i] for i, (f, u) in enumerate(zip(full, used)) if u < f]
        lost = sum(m >= max_len for m in markers)
        labeled = [n for r, n in zip(rows, state_tokens) if qid in r["targets"]]
        truncated = sum(n > room for n in labeled)
        report["questions"][qid] = {
            "prefix_tokens": len(ids),
            "state_room": room,
            "labeled_rows": len(labeled),
            "truncated_rows": truncated,
            "options": len(options),
            "clipped_options": clipped,
            "lost_options": lost,
            "tokens_per_option": round(sum(used) / max(1, len(used)), 1),
        }
        if lost:
            report["warnings"].append(
                f"{qid}: {lost} options do not fit the {max_len}-token window at all. "
                "Reduce the number of labels or shorten their descriptions."
            )
        if clipped:
            report["warnings"].append(
                f"{qid}: {len(clipped)}/{len(options)} option texts are clipped to about "
                f"{min(used)} tokens by the {head_len}-token option budget, so similar labels "
                "can become indistinguishable. Fine-tuning helps; shorter criteria help more."
            )
        if labeled and truncated / len(labeled) > 0.02:
            report["warnings"].append(
                f"{qid}: {truncated} of {len(labeled)} rows ({truncated / len(labeled):.0%}) are "
                f"longer than the {room} state tokens left after the question; their endings are "
                "silently cut. Put the decisive text first, shorten states, or use the 1,024-token "
                "multilingual model."
            )
        train_counts = meta.get("labels", {}).get(qid, {}).get("train", {})
        thin = [label for label, n in train_counts.items() if 0 < n < 8]
        missing = [label for label, n in train_counts.items() if n == 0]
        if thin:
            report["warnings"].append(
                f"{qid}: {len(thin)} labels have fewer than 8 training examples "
                f"({', '.join(thin[:6])}{' ...' if len(thin) > 6 else ''})."
            )
        if missing and qdef["type"] == "choice":
            report["warnings"].append(
                f"{qid}: {len(missing)} labels never appear in training "
                f"({', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''})."
            )
        total = sum(train_counts.values())
        if total and max(train_counts.values()) / total > 0.7 and qdef["type"] != "score":
            report["warnings"].append(
                f"{qid}: the most common label covers {max(train_counts.values()) / total:.0%} "
                "of training rows. Consider class_weighting = balanced."
            )
    test_decisions = meta.get("decisions", {}).get("test", 0)
    if test_decisions < 200:
        half = 1.96 * math.sqrt(0.25 / max(1, test_decisions))
        report["warnings"].append(
            f"The test split has {test_decisions} decisions, so accuracy is only known to about "
            f"±{half:.0%}. Add more labeled test rows for decisions you rely on."
        )
    return report


# ----------------------------------------------------------------------------- training


def encode_items(tok, cfg, rows, questions, rng=None, shuffle=False, weights=None):
    """Tokenize (row, question) pairs; shuffled choice options keep targets aligned."""
    max_len, head_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
    internal = {qid: internal_question(q) for qid, q in questions.items()}
    items, skipped = [], 0
    for index, row in enumerate(rows):
        for qid, target in row["targets"].items():
            q, k = internal[qid], len(target)
            order = list(range(k))
            if shuffle and q["t"] == "choice" and rng is not None:
                rng.shuffle(order)
            ids, markers = build_sequence(tok, row["state"], q, max_len, head_len, order)
            if len(markers) != k:
                skipped += 1
                continue
            label = argmax(target)
            items.append(
                {
                    "ids": ids,
                    "markers": markers,
                    "qtype": QTYPES[q["t"]],
                    "target": [target[i] for i in order],
                    "weight": weights.get((qid, label), 1.0) if weights else 1.0,
                    "key": (index, qid),
                }
            )
    return items, skipped


def class_weights(rows, questions):
    weights = {}
    for qid in questions:
        counts = {}
        for row in rows:
            if qid in row["targets"]:
                label = argmax(row["targets"][qid])
                counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        for label, n in counts.items():
            weights[(qid, label)] = min(5.0, max(0.2, total / (len(counts) * n)))
    return weights


def make_batches(items, batch_size, rng, shuffle=True):
    """Length-bucketed batches: sort within large windows, then shuffle the batches."""
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    window = batch_size * 32
    batches = []
    for start in range(0, len(order), window):
        part = sorted(order[start : start + window], key=lambda i: len(items[i]["ids"]))
        batches += [part[j : j + batch_size] for j in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(items, pad_id, multiple=16):
    import mlx.core as mx
    import numpy as np

    n = len(items)
    length = max(len(it["ids"]) for it in items)
    length = ((length + multiple - 1) // multiple) * multiple
    k = max(2, max(len(it["markers"]) for it in items))
    ids = np.full((n, length), pad_id, dtype=np.int32)
    att = np.zeros((n, length), dtype=np.bool_)
    pos = np.zeros((n, k), dtype=np.int32)
    mask = np.zeros((n, k), dtype=np.bool_)
    target = np.zeros((n, k), dtype=np.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = it["ids"]
        att[i, : len(it["ids"])] = True
        pos[i, : len(it["markers"])] = it["markers"]
        mask[i, : len(it["markers"])] = True
        target[i, : len(it["target"])] = it["target"]
    return {
        "input_ids": mx.array(ids),
        "attention_mask": mx.array(att),
        "marker_pos": mx.array(pos),
        "marker_mask": mx.array(mask),
        "qtype": mx.array(np.array([it["qtype"] for it in items], dtype=np.int32)),
        "target": mx.array(target),
        "weight": mx.array(np.array([it["weight"] for it in items], dtype=np.float32)),
    }


def lora_scale(alpha, rank, rslora=False):
    """alpha / rank, or alpha / sqrt(rank) for rsLoRA."""
    return alpha / math.sqrt(rank) if rslora else alpha / rank


def check_lora_variants(hp):
    """Reject values the LoRA variants cannot use, before a run starts."""
    for key in ("dora", "rslora"):
        if key in hp and not isinstance(hp[key], bool):
            raise ValueError(f"{key} must be true or false")
    ratio = hp.get("loraplus_ratio", 1.0)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 1 <= ratio <= 64:
        raise ValueError("loraplus_ratio must be a number from 1 (off) to 64")


def lora_variants(hp):
    """The LoRA variants a run used, for its record and its model card: e.g. ['DoRA']."""
    if hp.get("method") != "lora":
        return []
    ratio = float(hp.get("loraplus_ratio") or 1.0)
    return (
        (["DoRA"] if hp.get("dora") else [])
        + (["rsLoRA"] if hp.get("rslora") else [])
        + ([f"LoRA+ x{ratio:g}"] if ratio != 1.0 else [])
    )


def lora_class():
    import mlx.core as mx
    import mlx.nn as nn

    class LoRALinear(nn.Module):
        """y = x W^T + s * x A B, with s = alpha / r (or alpha / sqrt(r) for rsLoRA).

        Only A and B train; the base layer stays frozen. With DoRA a magnitude m per output
        row also trains, and the adapted weight is renormalised row by row:
        W' = m * (W + s (A B)^T) / ||W + s (A B)^T||. The norm is treated as a constant in
        the backward pass, as in the DoRA paper (section 4.3) and PEFT.
        """

        def __init__(self, base, rank, alpha, dropout, dora=False, rslora=False):
            super().__init__()
            out_dims, in_dims = base.weight.shape
            bound = 1 / math.sqrt(in_dims)
            self.base = base
            self.lora_a = mx.random.uniform(-bound, bound, (in_dims, rank)).astype(mx.float32)
            self.lora_b = mx.zeros((rank, out_dims), dtype=mx.float32)
            self.scale = lora_scale(alpha, rank, rslora)
            self.p = dropout
            self.dora = dora
            if dora:
                self.magnitude = mx.linalg.norm(base.weight.astype(mx.float32), axis=1)

        def _delta(self):
            return self.scale * (self.lora_a @ self.lora_b).T

        def __call__(self, x):
            y = self.base(x)
            h = x.astype(mx.float32)
            if self.training and self.p > 0:
                h = h * mx.random.bernoulli(1 - self.p, h.shape) / (1 - self.p)
            update = self.scale * ((h @ self.lora_a) @ self.lora_b)
            if not self.dora:
                return y + update.astype(y.dtype)
            weight = self.base.weight.astype(mx.float32)
            norm = mx.stop_gradient(mx.linalg.norm(weight + self._delta(), axis=1))
            bias = self.base.bias.astype(mx.float32) if "bias" in self.base else 0.0
            out = (self.magnitude / norm) * (y.astype(mx.float32) - bias + update) + bias
            return out.astype(y.dtype)

        def fused(self, dtype):
            weight = self.base.weight.astype(mx.float32) + self._delta()
            if self.dora:
                weight = (self.magnitude / mx.linalg.norm(weight, axis=1))[:, None] * weight
            layer = nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in self.base)
            layer.weight = weight.astype(dtype)
            if "bias" in self.base:
                layer.bias = self.base.bias.astype(dtype)
            return layer

    return LoRALinear


ENCODER_LINEARS = (("attn", "Wqkv"), ("attn", "Wo"), ("mlp", "Wi"), ("mlp", "Wo"))


# ----------------------------------------------------------------------------- encoder kinds
# Laya and Julia 1 are the same network with different prompts and checkpoint files
# (julia.py). The trainers below take both; these pick the parts that differ.


def encoder_kind(model_dir):
    """'laya' or 'julia' for a checkpoint the encoder trainers take; ValueError otherwise."""
    from . import kinds

    kind = kinds.check(model_dir)
    if kind not in (kinds.LAYA, kinds.JULIA):
        raise ValueError(f"{kinds.NAME[kind]} checkpoints are trained by their own trainer")
    return kind


def encoder_config(model_dir, kind=None):
    """The configuration the trainers read: rl_agent_config.json, or Julia's equivalent."""
    kind = kind or encoder_kind(model_dir)
    if kind == "julia":
        from . import julia

        return julia.config(model_dir)
    return read_json(Path(model_dir) / "rl_agent_config.json")


def encoder_tokenizer(model_dir, kind=None):
    kind = kind or encoder_kind(model_dir)
    if kind == "julia":
        from . import julia

        return julia.tokenizer(model_dir)
    return laya_mlx_module("tokenizer").Tokenizer(Path(model_dir) / "tokenizer")


def encoder_items(kind):
    """The function that encodes (row, question) pairs in this kind's own prompt."""
    if kind == "julia":
        from . import julia

        return julia.encode_items
    return encode_items


def load_training_model(model_dir, hp):
    """Build the decision model with frozen/trainable parts for the chosen method."""
    import mlx.core as mx
    from laya_mlx.model import DecisionModel, EncoderConfig, sanitize_weights
    from mlx.utils import tree_map

    model_dir = Path(model_dir)
    kind = encoder_kind(model_dir)
    cfg = encoder_config(model_dir, kind)
    enc_cfg = EncoderConfig.from_dict(read_json(model_dir / "encoder/config.json"))
    if kind == "julia":
        # laya_mlx sizes the unused action head from act_costs: n_act = len(act_costs) + 1
        acts = {f"action{i}": 0.0 for i in range(cfg["n_act"] - 1)}
        model = DecisionModel(enc_cfg, {"head_layers": cfg["head_layers"], "act_costs": acts})
    else:
        model = DecisionModel(enc_cfg, cfg)
    frozen_dtype = mx.bfloat16 if hp["precision"] == "bfloat16" else mx.float32
    weights = sanitize_weights(mx.load(str(model_dir / "model.safetensors")))
    weights = {
        k: v.astype(frozen_dtype if k.startswith("encoder.") else mx.float32)
        for k, v in weights.items()
    }
    model.load_weights(list(weights.items()), strict=True)
    model.freeze()
    layers = model.encoder.layers
    if hp["method"] == "lora":
        LoRALinear = lora_class()
        chosen = layers if hp["lora_layers"] <= 0 else layers[-hp["lora_layers"] :]
        for layer in chosen:
            for parent, name in ENCODER_LINEARS:
                owner = getattr(layer, parent)
                setattr(
                    owner,
                    name,
                    LoRALinear(
                        getattr(owner, name),
                        hp["lora_rank"],
                        hp["lora_alpha"],
                        hp["lora_dropout"],
                        dora=bool(hp.get("dora")),
                        rslora=bool(hp.get("rslora")),
                    ),
                )
    elif hp["method"] == "full":
        for module in [*layers[-max(1, hp["full_layers"]) :], model.encoder.final_norm]:
            module.update(tree_map(lambda v: v.astype(mx.float32), module.parameters()))
            module.unfreeze()
    elif hp["method"] != "head":
        raise ValueError(f"Unknown method {hp['method']!r}")
    if hp["method"] == "lora":
        for layer in model.encoder.layers:
            for parent, name in ENCODER_LINEARS:
                module = getattr(getattr(layer, parent), name)
                if hasattr(module, "lora_a"):
                    keys = ["lora_a", "lora_b"] + (["magnitude"] if module.dora else [])
                    module.unfreeze(keys=keys, recurse=False)
    model.head.unfreeze()
    model.scorer.unfreeze()
    model.type_emb.unfreeze()
    mx.eval(model.parameters())
    return model, cfg


def decision_logits(
    model, batch, *, training, head_dropout=0.0, checkpoint=False, encoder_frozen=False
):
    """Same computation as DecisionModel.__call__'s option logits, plus training dropout.

    PyTorch's TransformerEncoderLayer (upstream's head) applies dropout after attention,
    inside the feed-forward block and after it; the MLX inference head has none.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from laya_mlx.model import attention_masks

    def drop(x):
        if not training or head_dropout <= 0:
            return x
        return x * mx.random.bernoulli(1 - head_dropout, x.shape) / (1 - head_dropout)

    enc = model.encoder
    x = enc.embeddings(batch["input_ids"])
    masks = attention_masks(batch["attention_mask"], enc.config.local_attention)
    for layer in enc.layers:
        fn = nn.utils.checkpoint(layer) if checkpoint and not encoder_frozen else layer
        x = fn(x, masks[layer.attention_type])
    h = enc.final_norm(x).astype(mx.float32)
    if encoder_frozen:
        h = mx.stop_gradient(h)
    h = h + model.type_emb(batch["qtype"])[:, None, :]
    key_mask = batch["attention_mask"][:, None, None, :]
    for layer in model.head.layers:
        h = h + drop(layer.self_attn(layer.norm1(h), key_mask))
        h = h + drop(layer.linear2(drop(nn.relu(layer.linear1(layer.norm2(h))))))
    rows = mx.arange(h.shape[0])[:, None]
    markers = h[rows, mx.maximum(batch["marker_pos"], 0)]
    logits = model.scorer(markers).squeeze(-1).astype(mx.float32)
    return mx.where(batch["marker_mask"], logits, -1e4)


def proper_scores(q, target, mask, is_score, w_sph=0.75, w_rps=1.0, log_q=None):
    """Upstream RLCD reward: log score + spherical score - ranked probability score."""
    import mlx.core as mx

    maskf = mask.astype(mx.float32)
    q = q * maskf
    if log_q is None:
        log_q = mx.maximum(mx.log(mx.maximum(q, 1e-12)), -9.21)
    log_score = (target * mx.where(mask, log_q, 0.0)).sum(-1)
    sph = (target * q).sum(-1) / mx.maximum(mx.sqrt((q * q).sum(-1)), 1e-9)
    k = mx.maximum(maskf.sum(-1), 2.0)
    rps = (((mx.cumsum(q, -1) - mx.cumsum(target, -1)) ** 2) * maskf).sum(-1) / (k - 1)
    return log_score + w_sph * sph - w_rps * rps * is_score


def batch_loss(model, batch, hp, sigma, training=True, checkpoint=False):
    import mlx.core as mx

    logits = decision_logits(
        model,
        batch,
        training=training,
        head_dropout=hp["head_dropout"],
        checkpoint=checkpoint,
        encoder_frozen=hp["method"] == "head",
    )
    mask, target, weight = batch["marker_mask"], batch["target"], batch["weight"]
    log_p = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    ce = -(target * mx.where(mask, log_p, 0.0)).sum(-1)
    norm = weight.sum()
    ce_mean = (ce * weight).sum() / norm
    is_score = (batch["qtype"] == QTYPES["score"]).astype(mx.float32)
    objective = hp["objective"]
    if objective == "ce" or not training:
        return ce_mean, ce_mean
    if objective == "proper":
        score = proper_scores(mx.exp(log_p), target, mask, is_score, log_q=log_p)
        return -(score * weight).sum() / norm, ce_mean
    # "rlcd": upstream's Gaussian-perturbed policy gradient on the proper-score reward,
    # plus soft cross-entropy guidance, exactly as in the published fine-tuning notebook.
    group = 4
    maskf = mask.astype(mx.float32)
    k = mx.maximum(maskf.sum(-1, keepdims=True), 1.0)
    eps = mx.random.normal((group,) + logits.shape) * sigma * maskf
    eps = (eps - eps.sum(-1, keepdims=True) / k) * maskf
    z = mx.stop_gradient(logits)[None] + eps
    q = mx.softmax(mx.where(mask, z, -1e4), axis=-1)
    reward = mx.stop_gradient(proper_scores(q, target[None], mask, is_score))
    adv = reward - reward.mean(0, keepdims=True)
    adv = adv / (mx.sqrt(((adv - adv.mean()) ** 2).mean()) + 1e-6)
    log_pi = -(((z - logits[None]) ** 2) * maskf).sum(-1) / (2 * sigma**2)
    loss_rl = -((adv * log_pi) * weight).sum() / (norm * group)
    return loss_rl + ce_mean, ce_mean


def evaluate_logits(model, items, pad_id, batch_size=16):
    """Calibration-ready (qtype, logits, target) triples, with dropout disabled."""
    import mlx.core as mx
    import numpy as np

    model.eval()
    out = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch = collate(chunk, pad_id)
        logits = decision_logits(model, batch, training=False)
        mx.eval(logits)
        logits = np.asarray(logits)
        for row, it in enumerate(chunk):
            k = len(it["markers"])
            out.append((it["qtype"], logits[row, :k].astype(np.float64), np.array(it["target"])))
    model.train()
    return out


def nll_at(triples, temperature):
    import numpy as np

    total = 0.0
    for _, z, t in triples:
        z = z / temperature
        z = z - z.max()
        total -= float((t * (z - np.log(np.exp(z).sum()))).sum())
    return total / max(1, len(triples))


def fit_temperature(triples, lo=0.5, hi=5.0):
    """Golden-section search on 1/T (NLL is convex in inverse temperature)."""
    a, b = 1 / hi, 1 / lo
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll_at(triples, 1 / c), nll_at(triples, 1 / d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll_at(triples, 1 / c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll_at(triples, 1 / d)
    return 1 / ((a + b) / 2)


def calibrate(triples, base_cfg, min_bucket=30):
    """Fit per-type and per-(type, option-count) temperatures on validation logits."""
    temperature = list(base_cfg.get("temperature", [1.0, 1.0, 1.0]))
    by_options = dict(base_cfg.get("temperature_by_options", {}))
    fitted_types = set()
    for qtype in range(3):
        sel = [t for t in triples if t[0] == qtype]
        if len(sel) >= 10:
            temperature[qtype] = round(fit_temperature(sel), 4)
            fitted_types.add(qtype)
    names = {v: k for k, v in QTYPES.items()}
    by_options = {
        key: value
        for key, value in by_options.items()
        if QTYPES.get(key.split(":")[0]) not in fitted_types
    }
    buckets = {}
    for triple in triples:
        buckets.setdefault(temp_bucket(triple[0], len(triple[1])), []).append(triple)
    for key, sel in buckets.items():
        if len(sel) >= min_bucket:
            by_options[key] = round(fit_temperature(sel), 4)
    return temperature, by_options, sorted(names[t] for t in fitted_types)


def ece(conf, correct, bins=10):
    if not conf:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if sel:
            total += (
                len(sel)
                / len(conf)
                * abs(
                    sum(conf[i] for i in sel) / len(sel) - sum(correct[i] for i in sel) / len(sel)
                )
            )
    return total


def calibrated_ece(triples, temperature, by_options):
    import numpy as np

    conf, correct = [], []
    for qtype, z, t in triples:
        scale = by_options.get(temp_bucket(qtype, len(z)), temperature[qtype])
        p = np.exp(z / scale - (z / scale).max())
        p /= p.sum()
        conf.append(float(p.max()))
        correct.append(float(int(p.argmax()) == int(np.argmax(t))))
    return ece(conf, correct)


def fuse_lora(model, dtype):
    for layer in model.encoder.layers:
        for parent, name in ENCODER_LINEARS:
            owner = getattr(layer, parent)
            module = getattr(owner, name)
            if hasattr(module, "fused"):
                setattr(owner, name, module.fused(dtype))


def save_checkpoint(model, base_dir, out_dir, cfg, questions, provenance):
    """Write a standard Laya checkpoint (PyTorch parameter names, FP16 weights).

    The same files load in laya-mlx and in the upstream PyTorch `laya` package, which is
    what later exports (CUDA, CPU, ONNX, Core ML, ...) start from.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    base_dir, out_dir = Path(base_dir), Path(out_dir)
    fuse_lora(model, mx.float16)
    tensors = {upstream_name(k): v.astype(mx.float16) for k, v in tree_flatten(model.parameters())}
    expected = {upstream_name(k) for k in safetensors_header(base_dir / "model.safetensors")}
    if set(tensors) != expected:
        missing, extra = expected - set(tensors), set(tensors) - expected
        raise RuntimeError(f"Checkpoint names differ from base: missing {missing}, extra {extra}")
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "model.safetensors"), tensors, metadata={"format": "pt"})
    shutil.copytree(base_dir / "tokenizer", tmp / "tokenizer")
    (tmp / "encoder").mkdir()
    shutil.copy(base_dir / "encoder/config.json", tmp / "encoder/config.json")
    write_json(tmp / "rl_agent_config.json", cfg)
    write_json(tmp / "questions.json", questions)
    write_json(tmp / "laya_finetune.json", provenance)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return out_dir


def save_julia_checkpoint(model, base_dir, out_dir, questions, provenance):
    """A Julia 1 checkpoint from an MLX model: the LoRA updates merged into the base's own
    float32 weights, the trained parts in float32, the fitted temperature folded into the
    scorer (julia.py). Never the bfloat16 training copy of the frozen encoder."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from . import julia

    base = {
        k: v.astype(mx.float32)
        for k, v in mx.load(str(Path(base_dir) / "model.safetensors")).items()
    }
    tensors = dict(base)
    for i, layer in enumerate(model.encoder.layers):
        for parent, name in ENCODER_LINEARS:
            module = getattr(getattr(layer, parent), name)
            if not hasattr(module, "lora_a"):
                continue
            key = f"encoder.layers.{i}.{parent}.{name}.weight"
            weight = base[key] + module._delta().astype(mx.float32)
            if module.dora:
                weight = (module.magnitude / mx.linalg.norm(weight, axis=1))[:, None] * weight
            tensors[key] = weight
    for name, value in tree_flatten(model.trainable_parameters()):
        if name.endswith(("lora_a", "lora_b", "magnitude")):
            continue
        key = upstream_name(name)
        if key not in base:
            raise RuntimeError(f"Trained parameter {key} is not in the base checkpoint")
        tensors[key] = value.astype(mx.float32)
    julia.fold_temperature(tensors, provenance["calibration"]["folded_temperature"])
    mx.eval(tensors)

    def save(path):
        mx.save_safetensors(str(path), tensors, metadata={"format": "pt", "family": "julia"})

    return julia.write_checkpoint(tensors, base_dir, out_dir, questions, provenance, save)


def fit(spec, hp, emit, workspace=WORKSPACE, before_model=None):
    """Train, pick the best epoch, calibrate and save. Returns a training summary.

    Kept separate from train() so the model, optimizer state and gradients are released
    when this returns, before the fine-tuned checkpoint is reloaded for evaluation.
    before_model: called once the rows are encoded, before the model loads (the baseline).
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten

    run_dir = workspace / "runs" / check_id(spec["run_id"])
    questions, rows, meta = load_dataset(spec["dataset"], workspace)
    base_dir = resolve_model_ref(spec["base_model"], workspace)
    kind = encoder_kind(base_dir)
    cfg = encoder_config(base_dir, kind)
    tok = encoder_tokenizer(base_dir, kind)
    encode = encoder_items(kind)
    random.seed(hp["seed"])
    mx.random.seed(hp["seed"])
    rng = random.Random(hp["seed"])

    emit("phase", phase="prepare", message="Tokenizing and building the model")
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    weights = class_weights(train_rows, questions) if hp["class_weighting"] == "balanced" else None
    val_items, _ = encode(tok, cfg, val_rows, questions)
    probe, skipped = encode(tok, cfg, train_rows, questions)
    if not probe:
        raise ValueError("No training decisions fit the model's token budget")
    if before_model:
        before_model()
    model, cfg = load_training_model(base_dir, hp)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    longest = max(len(it["ids"]) for it in probe)
    checkpoint = hp["grad_checkpoint"] == "on" or (
        hp["grad_checkpoint"] == "auto"
        and hp["method"] != "head"
        and hp["batch_size"] * longest > 2048
    )
    per_epoch = math.ceil(len(probe) / hp["batch_size"])
    updates = math.ceil(per_epoch / hp["grad_accum"]) * hp["epochs"]
    emit(
        "info",
        trainable_params=trainable,
        total_params=total,
        train_decisions=len(probe),
        val_decisions=len(val_items),
        skipped_decisions=skipped,
        longest_tokens=longest,
        gradient_checkpointing=checkpoint,
        updates=updates,
        hyperparameters=hp,
    )

    def schedule(peak):
        warm = max(1, int(hp["warmup"] * updates))
        return optim.join_schedules(
            [
                optim.linear_schedule(peak * 0.01, peak, warm),
                optim.cosine_decay(peak, max(1, updates - warm), peak * 0.1),
            ],
            [warm],
        )

    head_opt = optim.AdamW(learning_rate=schedule(hp["head_lr"]), weight_decay=hp["weight_decay"])
    if hp["method"] == "head":
        optimizer = head_opt
    else:
        optimizers = [
            optim.AdamW(learning_rate=schedule(hp["lr"]), weight_decay=hp["weight_decay"])
        ]
        filters = [lambda path, _: path.startswith("encoder.")]
        ratio = float(hp.get("loraplus_ratio") or 1.0)
        if hp["method"] == "lora" and ratio != 1.0:
            # LoRA+: the B matrices get their own, higher learning rate. First match wins.
            optimizers.insert(
                0,
                optim.AdamW(
                    learning_rate=schedule(hp["lr"] * ratio), weight_decay=hp["weight_decay"]
                ),
            )
            filters.insert(
                0, lambda path, _: path.startswith("encoder.") and path.endswith("lora_b")
            )
        optimizer = optim.MultiOptimizer([*optimizers, head_opt], filters)
    sigma = [0.4]
    loss_and_grad = nn.value_and_grad(
        model, lambda m, b: batch_loss(m, b, hp, sigma[0], training=True, checkpoint=checkpoint)
    )

    def val_metrics():
        triples = evaluate_logits(model, val_items, tok.pad_token_id)
        loss = nll_at(triples, 1.0) if triples else float("nan")
        hits = sum(int(z.argmax()) == int(t.argmax()) for _, z, t in triples)
        return loss, hits / max(1, len(triples))

    emit("phase", phase="train", message=f"Training {hp['epochs']} epochs, {updates} updates")
    model.train()
    mx.reset_peak_memory()
    loss0, acc0 = val_metrics()
    emit("epoch", epoch=0, val_loss=loss0, val_accuracy=acc0)
    best = {"loss": loss0, "epoch": 0, "params": dict(tree_flatten(model.trainable_parameters()))}
    step, started, seen, stale, history = 0, time.perf_counter(), 0, 0, []
    for epoch in range(1, hp["epochs"] + 1):
        # upstream anneals the exploration noise from 0.4 to 0.1 across epochs
        sigma[0] = 0.4 + (0.1 - 0.4) * ((epoch - 1) / max(1, hp["epochs"] - 1))
        items, _ = encode(tok, cfg, train_rows, questions, rng, hp["shuffle_options"], weights)
        batches = make_batches(items, hp["batch_size"], rng)
        accum, count, running = None, 0, []
        for i, indices in enumerate(batches):
            batch = collate([items[j] for j in indices], tok.pad_token_id)
            (loss, ce), grads = loss_and_grad(model, batch)
            accum = grads if accum is None else tree_map(mx.add, accum, grads)
            # Materialize now: a lazy graph spanning several microbatches, the clip and the
            # optimizer update peaked above 12 GB on 512-token batches (under 3 GB this way).
            mx.eval(accum, loss, ce)
            count += 1
            running += [loss, ce]
            seen += len(indices)
            if count == hp["grad_accum"] or i == len(batches) - 1:
                accum = tree_map(lambda g: g / count, accum)
                accum, norm = optim.clip_grad_norm(accum, hp["max_grad_norm"])
                optimizer.update(model, accum)
                mx.eval(model.parameters(), optimizer.state, running)
                step += 1
                elapsed = time.perf_counter() - started
                emit(
                    "step",
                    step=step,
                    updates=updates,
                    epoch=epoch,
                    loss=sum(x.item() for x in running[::2]) / count,
                    ce=sum(x.item() for x in running[1::2]) / count,
                    grad_norm=norm.item(),
                    decisions_per_s=round(seen / elapsed, 2),
                    eta_s=round((updates - step) * elapsed / step),
                    peak_gb=round(mx.get_peak_memory() / 2**30, 2),
                )
                accum, count, running = None, 0, []
        val_loss, val_acc = val_metrics()
        improved = val_loss < best["loss"] - 1e-4
        if improved:
            params = dict(tree_flatten(model.trainable_parameters()))
            best, stale = {"loss": val_loss, "epoch": epoch, "params": params}, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc})
        emit("epoch", epoch=epoch, val_loss=val_loss, val_accuracy=val_acc, best=improved)
        if hp["patience"] and stale >= hp["patience"]:
            emit("log", message=f"Early stop: validation loss has not improved for {stale} epochs")
            break
    model.update(tree_unflatten(list(best["params"].items())))
    emit("log", message=f"Kept epoch {best['epoch']} (validation loss {best['loss']:.4f})")
    seconds = time.perf_counter() - started

    emit("phase", phase="calibrate", message="Fitting temperatures on validation logits")
    triples = evaluate_logits(model, val_items, tok.pad_token_id)
    if kind == "julia":
        from . import julia

        calibration = julia.fit_calibration(triples)
        temperature, by_options = calibration["temperature"], {}
    else:
        temperature, by_options, fitted = calibrate(triples, cfg)
        calibration = {
            "ece_uncalibrated": calibrated_ece(triples, [1.0, 1.0, 1.0], {}),
            "ece_calibrated": calibrated_ece(triples, temperature, by_options),
            "temperature": temperature,
            "temperature_by_options": by_options,
            "fitted_types": fitted,
        }
    emit("calibration", **calibration)

    emit("phase", phase="save", message="Merging adapters and writing the checkpoint")
    summary = {
        "run_id": spec["run_id"],
        "base_model": spec["base_model"],
        "base_model_dir": str(base_dir),
        "kind": kind,
        "dataset": spec["dataset"],
        "dataset_sha256": meta.get("sha256"),
        "hyperparameters": hp,
        "trainable_params": trainable,
        "total_params": total,
        "train_decisions": len(probe),
        "val_decisions": len(val_items),
        "skipped_decisions": skipped,
        "updates": step,
        "best_epoch": best["epoch"],
        "best_val_loss": best["loss"],
        "history": history,
        "train_seconds": round(seconds, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 2),
        "gradient_checkpointing": checkpoint,
        "calibration": calibration,
        "backend": "mlx",
        "created": now(),
    }
    if kind == "julia":
        save_julia_checkpoint(model, base_dir, run_dir / "model", questions, summary)
        write_json(run_dir / "training.json", summary)
        return summary
    new_cfg = {**cfg, "temperature": temperature, "temperature_by_options": by_options}
    new_cfg["fine_tuned"] = {
        key: summary[key] for key in ("base_model", "dataset_sha256", "best_epoch", "created")
    } | {"method": hp["method"], "objective": hp["objective"], "tool": "laya-mlx finetune"}
    if lora_variants(hp):
        new_cfg["fine_tuned"]["lora_variants"] = lora_variants(hp)
    save_checkpoint(model, base_dir, run_dir / "model", new_cfg, questions, summary)
    write_json(run_dir / "training.json", summary)
    return summary


def train(spec, emit, workspace=WORKSPACE):
    from . import kinds, runtime

    kind = model_kind(spec["base_model"], workspace)
    hp = {**hyperparameters(kind), **spec.get("hyperparameters", {})}

    def before_model():
        """The baseline, run by the trainer once it has encoded its rows and before it loads
        the model: a run with nothing it can train on stops before the base model is
        evaluated, not after."""
        if spec.get("baseline", True):
            baseline(spec["base_model"], spec["dataset"], emit, workspace)
            emit("phase", phase="prepare", message="Building the model to fine-tune")

    if kind == kinds.DECIDER:
        from .decider_engine import fit as decider_fit

        decider_fit(spec, hp, emit, workspace, before_model)
    elif runtime.backend() == "mlx":
        fit(spec, hp, emit, workspace, before_model)
    else:
        from .torch_engine import fit as torch_fit

        torch_fit(spec, hp, emit, workspace, before_model)
    runtime.clear_cache()

    emit("phase", phase="evaluate", message="Evaluating the fine-tuned model on the test split")
    run_dir = workspace / "runs" / spec["run_id"]
    finetuned = evaluate(f"run:{spec['run_id']}", spec["dataset"], emit, workspace)
    write_json(run_dir / "eval.json", finetuned)
    base_report = read_json(baseline_path(spec["base_model"], spec["dataset"], workspace))
    if base_report:
        comparison = compare(base_report, finetuned)
        emit("phase", phase="evaluate", message="Timing both models on the same rows")
        refs = [spec["base_model"], f"run:{spec['run_id']}"]
        comparison["latency_paired"] = paired_latency(refs, spec["dataset"], workspace)
        write_json(run_dir / "comparison.json", comparison)
    emit(
        "result",
        run_id=spec["run_id"],
        accuracy=finetuned["overall"]["accuracy"],
        baseline_accuracy=base_report["overall"]["accuracy"] if base_report else None,
    )


# ----------------------------------------------------------------------------- evaluation


def wilson(correct, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = correct / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, center - half), min(1.0, center + half)]


def probabilities(qdef, answer):
    if qdef["type"] == "choice":
        return [answer["probabilities"][label] for label in option_labels(qdef)]
    if qdef["type"] == "score":
        return [answer["probabilities"][str(i)] for i in range(len(qdef["criteria"]))]
    return [1.0 - answer["noul"], answer["noul"]]


THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def metrics(records, qdef=None):
    n = len(records)
    if not n:
        return {"n": 0}
    top = [max(r["p"]) for r in records]
    pred = [argmax(r["p"]) for r in records]
    gold = [argmax(r["gold"]) for r in records]
    correct = [int(a == b) for a, b in zip(pred, gold)]
    result = {
        "n": n,
        "accuracy": sum(correct) / n,
        "accuracy_ci95": wilson(sum(correct), n),
        "nll": sum(
            -sum(g * math.log(max(p, 1e-6)) for p, g in zip(r["p"], r["gold"])) for r in records
        )
        / n,
        "brier": sum(sum((p - g) ** 2 for p, g in zip(r["p"], r["gold"])) for r in records) / n,
        "ece": ece(top, correct),
        "mean_confidence": sum(top) / n,
        "coverage": [
            {
                "threshold": t,
                "coverage": sum(c >= t for c in top) / n,
                "accuracy": (
                    sum(k for c, k in zip(top, correct) if c >= t)
                    / max(1, sum(c >= t for c in top))
                ),
            }
            for t in THRESHOLDS
        ],
    }
    if qdef is not None:
        names = option_names(qdef)
        k = len(names)
        matrix = [[0] * k for _ in range(k)]
        for a, b in zip(gold, pred):
            matrix[a][b] += 1
        f1s = []
        for c in range(k):
            tp = matrix[c][c]
            fp = sum(matrix[r][c] for r in range(k)) - tp
            fn = sum(matrix[c]) - tp
            if tp + fp + fn:
                f1s.append(2 * tp / (2 * tp + fp + fn))
        result.update(labels=names, confusion=matrix, macro_f1=sum(f1s) / max(1, len(f1s)))
        if qdef["type"] == "score":
            levels = range(k)
            result["mae"] = (
                sum(
                    abs(
                        sum(i * p for i, p in zip(levels, r["p"]))
                        - sum(i * g for i, g in zip(levels, r["gold"]))
                    )
                    for r in records
                )
                / n
            )
            result["within_one"] = sum(abs(a - b) <= 1 for a, b in zip(pred, gold)) / n
    return result


def evaluate(model_ref, dataset_id, emit, workspace=WORKSPACE, split="test"):
    import warnings

    from . import runtime

    questions, rows, meta = load_dataset(dataset_id, workspace)
    rows = [r for r in rows if r["split"] == split]
    if not rows:
        raise ValueError(f"Dataset {dataset_id} has no {split} rows")
    model_dir = resolve_model_ref(model_ref, workspace)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = runtime.load_agent(model_dir, batch_size=16)
    notes = [str(w.message) for w in caught]
    first = rows[0]
    for _ in range(3):
        agent.predict(first["state"], {q: questions[q] for q in first["targets"]})
    records, latencies = [], []
    started = time.perf_counter()
    for i, row in enumerate(rows):
        qs = {q: questions[q] for q in row["targets"]}
        t0 = time.perf_counter()
        out = agent.predict(row["state"], qs)
        latencies.append((time.perf_counter() - t0) * 1000)
        for qid, target in row["targets"].items():
            records.append(
                {
                    "row": row["id"],
                    "qid": qid,
                    "p": probabilities(questions[qid], out["answers"][qid]),
                    "gold": [round(g, 4) for g in target],
                }
            )
        if i % 25 == 0 or i == len(rows) - 1:
            emit(
                "progress",
                done=i + 1,
                total=len(rows),
                model=model_ref,
                elapsed_s=round(time.perf_counter() - started, 1),
            )
    del agent
    runtime.clear_cache()
    per_question = {
        qid: metrics([r for r in records if r["qid"] == qid], qdef)
        for qid, qdef in questions.items()
        if any(r["qid"] == qid for r in records)
    }
    return {
        "model": model_ref,
        "model_dir": str(model_dir),
        "dataset": dataset_id,
        "dataset_sha256": meta.get("sha256"),
        "split": split,
        "created": now(),
        "rows": len(rows),
        "overall": metrics(records),
        "questions": per_question,
        "latency_ms": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "per_decision_mean": sum(latencies) / len(records),
        },
        "notes": notes,
        "records": records,
    }


def paired_latency(refs, dataset_id, workspace=WORKSPACE, rows=60):
    """Latency of several models measured interleaved on the same rows.

    Separate evaluation passes run minutes apart, and a Mac that has just trained for ten
    minutes is warmer and slower than one that has not; alternating models row by row makes
    the comparison fair.
    """
    from . import runtime

    questions, data, _ = load_dataset(dataset_id, workspace)
    data = [r for r in data if r["split"] == "test"][:rows]
    agents = [runtime.load_agent(resolve_model_ref(ref, workspace)) for ref in refs]
    times = [[] for _ in refs]
    for index, row in enumerate(data):
        qs = {q: questions[q] for q in row["targets"]}
        for agent, samples in zip(agents, times):
            started = time.perf_counter()
            agent.predict(row["state"], qs)
            if index >= 3:  # the first calls compile kernels (Metal, CUDA graphs)
                samples.append((time.perf_counter() - started) * 1000)
    del agents
    runtime.clear_cache()
    return {
        ref: {"p50": percentile(t, 0.5), "p95": percentile(t, 0.95), "rows": len(t)}
        for ref, t in zip(refs, times)
    }


def baseline_path(model_ref, dataset_id, workspace=WORKSPACE):
    key = hashlib.sha256(f"{model_ref}|{dataset_id}".encode()).hexdigest()[:12]
    return workspace / "evals" / f"{check_id(dataset_id)}--{slugify(model_ref)}-{key}.json"


def baseline(model_ref, dataset_id, emit, workspace=WORKSPACE):
    path = baseline_path(model_ref, dataset_id, workspace)
    report = read_json(path)
    if report:
        emit("log", message=f"Reusing the cached baseline evaluation of {model_ref}")
        return report
    emit("phase", phase="baseline", message=f"Evaluating {model_ref} before fine-tuning")
    report = evaluate(model_ref, dataset_id, emit, workspace)
    write_json(path, report)
    return report


def mcnemar(b, c):
    """Exact two-sided McNemar test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(base, tuned):
    def paired(qid=None):
        index = {(r["row"], r["qid"]): r for r in base["records"]}
        b = c = 0
        for r in tuned["records"]:
            if qid and r["qid"] != qid:
                continue
            other = index.get((r["row"], r["qid"]))
            if other is None:
                continue
            gold = argmax(r["gold"])
            was, now_ = argmax(other["p"]) == gold, argmax(r["p"]) == gold
            b += was and not now_
            c += now_ and not was
        return {"fixed": c, "broken": b, "p_value": mcnemar(b, c)}

    def summary(report):
        keys = (
            "n",
            "accuracy",
            "accuracy_ci95",
            "macro_f1",
            "nll",
            "brier",
            "ece",
            "mae",
            "within_one",
        )
        return {
            "overall": {k: report["overall"].get(k) for k in keys if k in report["overall"]},
            "questions": {
                q: {k: m.get(k) for k in keys if k in m} for q, m in report["questions"].items()
            },
            "latency_ms": report["latency_ms"],
        }

    return {
        "base": {"model": base["model"], **summary(base)},
        "finetuned": {"model": tuned["model"], **summary(tuned)},
        "paired": {"overall": paired(), **{q: paired(q) for q in tuned["questions"]}},
    }


# ----------------------------------------------------------------------------- job runner


def download(spec, emit):
    from huggingface_hub import snapshot_download

    repo = spec["repo_id"]
    emit("phase", phase="download", message=f"Downloading {repo} from Hugging Face")
    path = snapshot_download(repo)
    emit("result", path=path)


def limit_mlx_cache():
    """Cap MLX's buffer cache. Batches of varying length leave freed buffers behind, and an
    uncapped cache grew past 10 GB in a few dozen steps on a 16 GB Mac, pushing it into swap.
    Nothing to do on PyTorch."""
    from . import runtime

    runtime.limit_cache()


def write_job(workspace, job_id, kind, spec, title):
    """jobs/<id>: the job's spec.json, which run_job reads, and job.json, which lists it."""
    path = workspace / "jobs" / check_id(job_id)
    path.mkdir(parents=True)
    write_json(path / "spec.json", {**spec, "kind": kind, "workspace": str(workspace)})
    write_json(path / "job.json", {"id": job_id, "kind": kind, "title": title, "created": now()})
    return path


def start_job(job_dir, kind, log, env=None):
    """`python -m layastudio.engine run <job_dir>` as a child process, the way every job runs,
    from the UI or headless (cloud.py): a crash, a cancel or an out-of-memory error ends the
    child, never its caller, and the GPU's memory goes back when it exits. Training and
    evaluation run offline. env: more variables for the child."""
    import subprocess

    env = {**os.environ, **(env or {}), "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1"}
    if kind in ("train", "evaluate"):
        env["HF_HUB_OFFLINE"] = "1"
    return subprocess.Popen(
        [sys.executable, "-m", "layastudio.engine", "run", str(job_dir)],
        cwd=str(PACKAGE.parent),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def run_job(job_dir):
    job_dir = Path(job_dir)
    spec = read_json(job_dir / "spec.json")
    events = open(job_dir / "events.jsonl", "a", buffering=1)

    def emit(kind, **data):
        event = finite({"t": round(time.time(), 3), "type": kind, **data})
        events.write(json.dumps(event, ensure_ascii=False, default=float) + "\n")

    def on_term(*_):
        raise Cancelled()

    signal.signal(signal.SIGTERM, on_term)
    workspace = Path(spec.get("workspace", WORKSPACE))
    try:
        limit_mlx_cache()
        kind = spec["kind"]
        if kind == "download":
            download(spec, emit)
        elif kind == "example":
            from .examples import fetch_example

            meta = fetch_example(spec["name"], emit, workspace)
            emit("result", dataset=meta["id"])
        elif kind == "evaluate":
            report = evaluate(spec["model"], spec["dataset"], emit, workspace)
            write_json(baseline_path(spec["model"], spec["dataset"], workspace), report)
            emit("result", accuracy=report["overall"]["accuracy"])
        elif kind == "train":
            train(spec, emit, workspace)
        elif kind == "export":
            from .export import export

            export(
                spec["model"],
                spec["target"],
                workspace,
                emit,
                precision=spec.get("precision", "float"),
                test_rows=spec.get("test_rows"),
                gguf=spec.get("gguf"),  # the GGUF a Decider package carries; bf16 unless asked
            )
        elif kind == "import":
            from .families import import_from_registry

            entry = import_from_registry(spec["repo"], emit, workspace, spec.get("version"))
            emit("result", ref=entry["ref"], path=entry["path"])
        elif kind == "publish":
            from .publish_systemone import publish

            publish(
                spec["model"],
                spec.get("repo"),
                workspace,
                emit,
                private=bool(spec.get("private")),
                noulxp=bool(spec.get("noulxp", True)),
            )
        else:
            raise ValueError(f"Unknown job kind {kind!r}")
        emit("done")
        return 0
    except (Cancelled, KeyboardInterrupt):
        emit("cancelled")
        return 143
    except BaseException as error:  # noqa: BLE001 - every failure must reach the UI
        emit("error", message=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        sys.exit("usage: python -m layastudio.engine run <job_dir>")
    sys.exit(run_job(sys.argv[2]))
