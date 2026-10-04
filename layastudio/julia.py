"""Julia 1 (Supersonic Labs) in the studio: its prompt, its checkpoint and its inference.

Julia 1 (huggingface.co/SupersonicLabs/Julia-1, Apache-2.0) is the network Laya is: a
ModernBERT encoder (mmBERT-small: 22 layers, width 384, a 256k-token vocabulary), a
question-type embedding, a two-layer pre-norm decision transformer and a scorer read at one
marker token per option. Its parameter names are Laya's, so the studio's two Laya trainers
(engine.py on MLX, torch_engine.py on PyTorch) train it unchanged. Everything around the
network is Julia's own, adapted here from julia/data.py (sequence), julia/typed.py and
julia/model.py at revision a85b127 (Apache-2.0); Supersonic Labs publish no trainer.

- **Prompt.** "<type> question: <instructions>", one marker per option followed by the
  option's description (its name when it has none; a score level's own text; for noul the
  false and true descriptions, or the words false and true), then the state. Laya instead
  writes "name: description", "level i: ...".
- **Budgets.** Its published inference policy: 8,192 tokens in all, 512 for the question and
  options, 48 per option. Julia's runtime and NoulXP packages refuse what does not fit
  (strict); training and the studio's own evaluation cut the state instead, as julia/data.py
  does without strict, and the dataset analysis counts the rows strict would refuse.
- **Options.** Julia answers 2 to 20 options per question.
- **Checkpoint.** julia_config.json (float32 weights), inference-policy.json, config.json,
  encoder/, tokenizer/ and model.safetensors, written in float32: LoRA updates are merged into
  the base's own float32 weights, never into a lower-precision training copy.
- **Calibration.** Julia's runtime applies no temperature, so the one temperature a fine-tune
  fits on its validation split is folded into the scorer's output layer (its weight and bias
  divided by T). Every runtime that reads the weights, Julia's own and NoulXP's included, then
  gives the calibrated probabilities.
"""

import json
import math
from pathlib import Path

QTYPES = {"choice": 0, "score": 1, "noul": 2}
OPTION_TOKENS = 48  # julia/data.py's per-option contract
MAX_OPTIONS = 20  # julia/data.py validate_row: 2 to 20 options
POLICY = {"max_length": 8192, "head_length": 512}  # inference-policy.json of the release
SCORER_OUT = "scorer.3"  # the scorer's last Linear: logits = W h + b
SOURCE = "SupersonicLabs/Julia-1 at a85b127321d580d65176c89ced8273f305745d85 (Apache-2.0)"


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def text(value):
    """A criterion or instruction as text: strings as they are, anything else as JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def instructions(qdef):
    return text(qdef.get("instructions", ""))


def option_texts(qdef):
    """Julia's option texts, in the order of engine.option_labels (criteria keys, level
    indices, false/true). The System One adapter of noulxp.native.julia: a choice with no
    description is its name; a noul without descriptions is the words false and true."""
    kind, criteria = qdef["type"], qdef.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            return [str(c) for c in criteria]
        return [k if v in (None, "") else text(v) for k, v in criteria.items()]
    if kind == "score":
        return [text(c) for c in criteria]
    given = criteria if isinstance(criteria, dict) else {}
    return [k if given.get(k) in (None, "") else text(given[k]) for k in ("false", "true")]


# ----------------------------------------------------------------------------- tokens


class Tokens:
    """`tokenizer(text, add_special_tokens=False)["input_ids"]` from tokenizer.json alone.

    The `tokenizers` library gives the same ids as Julia's transformers tokenizer (noulxp
    checked 641 strings), so neither training nor inference needs transformers' tokenizer."""

    def __init__(self, directory):
        from tokenizers import Tokenizer

        directory = Path(directory)
        self.backend = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.backend.no_truncation()
        self.backend.no_padding()
        config = _read_json(directory / "tokenizer_config.json", {}) or {}
        ident = self.backend.token_to_id
        self.mask_token = config.get("mask_token", "<mask>")
        self.mask_token_id = ident(self.mask_token)
        self.cls_token_id = ident(config.get("cls_token", "<bos>"))
        self.sep_token_id = ident(config.get("sep_token", "<eos>"))
        self.pad_token_id = ident(config.get("pad_token", "<pad>"))
        if None in (self.mask_token_id, self.cls_token_id, self.sep_token_id, self.pad_token_id):
            raise ValueError("Julia's tokenizer must define mask, cls, sep and pad tokens")

    def __call__(self, value):
        return self.backend.encode(value, add_special_tokens=False).ids


def sequence(tok, state, kind, question, options, max_length=8192, head_length=512, strict=False):
    """julia/data.py sequence(): (ids, markers, truncated). Raises ValueError where Julia does.

    strict refuses what does not fit (Julia's runtime and NoulXP packages); without it the
    state is cut to the room left, as julia/data.py does for training."""
    if head_length + 4 >= max_length:
        raise ValueError("max_length must leave room beyond the question head")
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    if strict and any(tok.mask_token in t for t in [state, question, *options]):
        raise ValueError("the request contains the model's reserved marker token")

    def clean(value):
        return value.replace(tok.mask_token, " ")

    head = tok(f"{kind} question: {clean(question)}")
    option_ids = [tok(" " + clean(x)) for x in options]
    if strict and any(len(x) > OPTION_TOKENS for x in option_ids):
        raise ValueError(f"an option is longer than the model's {OPTION_TOKENS}-token limit")
    marked = [[tok.mask_token_id] + x[:OPTION_TOKENS] for x in option_ids]
    budget = head_length - sum(map(len, marked))
    if budget < 16:
        per_option = max(4, (head_length - 16) // len(marked))
        marked = [x[:per_option] for x in marked]
        budget = head_length - sum(map(len, marked))
    if strict and (
        len(head) > budget or any(len(x) != len(y) + 1 for x, y in zip(marked, option_ids))
    ):
        raise ValueError("the question and options are too long for the model; shorten them")
    ids = [tok.cls_token_id, *head[: max(8, budget)], tok.sep_token_id]
    markers = []
    for option in marked:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tok.sep_token_id)
    state_ids = tok(clean(state))
    room = max_length - len(ids) - 1
    if room < 1:
        raise ValueError("the question and options exceed the sequence budget")
    if strict and len(state_ids) > room:
        raise ValueError(f"the state is too long: this model reads up to {max_length} tokens")
    return ids + state_ids[:room] + [tok.sep_token_id], markers, len(state_ids) > room


def refusal(tok, state, qdef, cfg):
    """Why Julia's own runtime (strict) would refuse this question, or None."""
    opts = option_texts(qdef)
    if not 2 <= len(opts) <= MAX_OPTIONS:
        return f"{len(opts)} options; Julia answers 2 to {MAX_OPTIONS}"
    try:
        sequence(
            tok,
            state,
            qdef["type"],
            instructions(qdef),
            opts,
            cfg["max_len"],
            cfg["head_max_len"],
            strict=True,
        )
    except ValueError as error:
        return str(error)
    return None


# ----------------------------------------------------------------------------- checkpoint


def config(model_dir):
    """The parts of a Julia checkpoint's configuration the trainers read, named as Laya's
    rl_agent_config.json names them (max_len, head_max_len, head_layers)."""
    model_dir = Path(model_dir)
    julia = _read_json(model_dir / "julia_config.json", {}) or {}
    found = (julia.get("format_version"), julia.get("architecture"))
    if found != (1, "JuliaDecisionModel"):
        raise ValueError(f"not a Julia 1 checkpoint: {found}")
    policy = _read_json(model_dir / "inference-policy.json", {}) or {}
    return {
        "max_len": int(policy.get("max_length", POLICY["max_length"])),
        "head_max_len": int(policy.get("head_length", POLICY["head_length"])),
        "head_layers": int(julia.get("head_layers", 2)),
        "n_act": int(julia.get("n_act", 2)),
        "dropout": float(julia.get("dropout", 0.1)),
        "weight_dtype": julia.get("weight_dtype", "float32"),
        "max_options": MAX_OPTIONS,
    }


def tokenizer(model_dir):
    return Tokens(Path(model_dir) / "tokenizer")


def encode_items(tok, cfg, rows, questions, rng=None, shuffle=False, weights=None):
    """(row, question) pairs in Julia's prompt, as engine.encode_items encodes Laya's: the
    same item fields, so the trainers' batching, losses and metrics are shared. Questions
    with more options than Julia answers, or a question too long for its budget, are
    skipped and counted."""
    from .engine import argmax

    items, skipped = [], 0
    for index, row in enumerate(rows):
        for qid, target in row["targets"].items():
            qdef = questions[qid]
            k = len(target)
            texts = option_texts(qdef)
            if not 2 <= k <= MAX_OPTIONS or len(texts) != k:
                skipped += 1
                continue
            order = list(range(k))
            if shuffle and qdef["type"] == "choice" and rng is not None:
                rng.shuffle(order)
            try:
                ids, markers, _ = sequence(
                    tok,
                    row["state"],
                    qdef["type"],
                    instructions(qdef),
                    [texts[i] for i in order],
                    cfg["max_len"],
                    cfg["head_max_len"],
                )
            except ValueError:
                skipped += 1
                continue
            label = argmax(target)
            items.append(
                {
                    "ids": ids,
                    "markers": markers,
                    "qtype": QTYPES[qdef["type"]],
                    "target": [target[i] for i in order],
                    "weight": weights.get((qid, label), 1.0) if weights else 1.0,
                    "key": (index, qid),
                }
            )
    return items, skipped


def torch_model(model_dir):
    """Julia's network in PyTorch, float32, weights loaded strictly, on the CPU.

    julia/model.py's JuliaDecisionModel, with the forward of laya.common.DecisionModel (the
    same parameters): it returns (logits, None) and can stop gradients at the encoder."""
    import torch
    from safetensors.torch import load_file
    from torch import nn
    from transformers import AutoConfig, AutoModel

    cfg = config(model_dir)

    class JuliaDecisionModel(nn.Module):
        def __init__(self, encoder, head_layers, n_act, dropout):
            super().__init__()
            self.encoder = encoder
            width = encoder.config.hidden_size
            layer = nn.TransformerEncoderLayer(
                width, max(1, width // 64), 4 * width, dropout, batch_first=True, norm_first=True
            )
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            self.type_emb = nn.Embedding(3, width)
            self.scorer = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
            )
            self.act_head = nn.Sequential(
                nn.Linear(width + 4, 256), nn.GELU(), nn.Linear(256, n_act)
            )
            self.register_buffer("temperature", torch.ones(3))

        def forward(
            self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False
        ):
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            if detach_encoder:
                hidden = hidden.detach()
            hidden = hidden + self.type_emb(qtype)[:, None, :]
            padding = ~attention_mask.bool()
            for layer in self.head.layers:
                hidden = layer(hidden, src_key_padding_mask=padding)
            index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.shape[-1])
            scores = self.scorer(hidden.gather(1, index)).squeeze(-1).float()
            return scores.masked_fill(~marker_mask, -1e4), None

    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # older transformers
        from transformers.modeling_utils import no_init_weights
    encoder_config = AutoConfig.from_pretrained(Path(model_dir) / "encoder")
    encoder_config.reference_compile = False
    with no_init_weights():
        encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
        model = JuliaDecisionModel(encoder, cfg["head_layers"], cfg["n_act"], cfg["dropout"])
    state = load_file(str(Path(model_dir) / "model.safetensors"))
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    return model.float()


def mlx_model(model_dir):
    """Julia's network in MLX (laya_mlx's ModernBERT and decision head), float32."""
    import mlx.core as mx
    from laya_mlx.model import DecisionModel, EncoderConfig, sanitize_weights

    cfg = config(model_dir)
    encoder = EncoderConfig.from_dict(_read_json(Path(model_dir) / "encoder/config.json"))
    # laya_mlx sizes the (unused) action head from act_costs: n_act = len(act_costs) + 1
    acts = {f"action{i}": 0.0 for i in range(cfg["n_act"] - 1)}
    model = DecisionModel(encoder, {"head_layers": cfg["head_layers"], "act_costs": acts})
    weights = sanitize_weights(mx.load(str(Path(model_dir) / "model.safetensors")))
    model.load_weights([(k, v.astype(mx.float32)) for k, v in weights.items()], strict=True)
    return model


def fold_temperature(tensors, temperature):
    """The scorer's output layer divided by T, so the logits are already calibrated."""
    for name in (f"{SCORER_OUT}.weight", f"{SCORER_OUT}.bias"):
        tensors[name] = tensors[name] / temperature
    return tensors


def fit_calibration(triples):
    """One temperature for every question type, fitted by NLL on validation logits (needs
    10 decisions; T = 1 below that). Per-type fits are recorded for reading only: Julia's
    runtime has one place for a temperature, the scorer's weights."""
    from .engine import QTYPES as NAMES
    from .engine import calibrated_ece, fit_temperature

    temperature = round(fit_temperature(triples), 4) if len(triples) >= 10 else 1.0
    per_type = {}
    for name, code in NAMES.items():
        chosen = [t for t in triples if t[0] == code]
        if len(chosen) >= 10:
            per_type[name] = round(fit_temperature(chosen), 4)
    return {
        "ece_uncalibrated": calibrated_ece(triples, [1.0, 1.0, 1.0], {}),
        "ece_calibrated": calibrated_ece(triples, [temperature] * 3, {}),
        "temperature": [temperature] * 3,
        "temperature_by_options": {},
        "fitted_types": sorted(per_type),
        "folded_temperature": temperature,
        "per_type_temperature": per_type,
        "how": f"one temperature, folded into {SCORER_OUT} (logits / T): Julia reads none",
    }


def write_checkpoint(tensors, base_dir, out_dir, questions, provenance, save):
    """A Julia checkpoint: the base's files with new float32 weights, written by `save`
    (a function of a path); the folder appears only once complete."""
    import hashlib
    import shutil

    from .engine import write_json

    base_dir, out_dir = Path(base_dir), Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save(tmp / "model.safetensors")
    shutil.copytree(base_dir / "encoder", tmp / "encoder")
    shutil.copytree(base_dir / "tokenizer", tmp / "tokenizer")
    julia = _read_json(base_dir / "julia_config.json", {}) or {}
    write_json(tmp / "julia_config.json", {**julia, "weight_dtype": "float32"})
    if (base_dir / "config.json").is_file():
        shutil.copy(base_dir / "config.json", tmp / "config.json")
    digest = hashlib.sha256((tmp / "model.safetensors").read_bytes()).hexdigest()
    policy = _read_json(base_dir / "inference-policy.json", {}) or {}
    calibration = provenance.get("calibration") or {}
    write_json(
        tmp / "inference-policy.json",
        {
            **POLICY,
            "strict_encoding": True,
            **{k: v for k, v in policy.items() if k not in ("step", "replacement_qualified")},
            "weights_sha256": digest,
            "calibration": {
                "temperature": calibration.get("folded_temperature", 1.0),
                "folded_into": SCORER_OUT,
            },
            "fine_tuned": {
                "tool": "System One Studio",
                "base_model": provenance.get("base_model"),
                "dataset_sha256": provenance.get("dataset_sha256"),
                "best_epoch": provenance.get("best_epoch"),
                "created": provenance.get("created"),
            },
        },
    )
    write_json(tmp / "questions.json", questions)
    write_json(tmp / "finetune.json", provenance)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return out_dir


# ----------------------------------------------------------------------------- inference


def answer(qdef, logits):
    """One question's answer in the System One format, from its option logits."""
    from .engine import option_labels

    top = max(logits)
    e = [math.exp(x - top) for x in logits]
    total = sum(e)
    p = [x / total for x in e]
    keys = option_labels(qdef)
    best = max(range(len(p)), key=p.__getitem__)
    if qdef["type"] == "choice":
        return {
            "type": "choice",
            "choice": keys[best],
            "probabilities": dict(zip(keys, p)),
            "confidence": p[best],
        }
    if qdef["type"] == "score":
        return {
            "type": "score",
            "score": sum(i * x for i, x in enumerate(p)),
            "legend": dict(zip(keys, option_texts(qdef))),
            "probabilities": dict(zip(keys, p)),
            "confidence": p[best],
        }
    return {"type": "noul", "noul": p[1], "confidence": max(p)}


class Agent:
    """A Julia checkpoint answering System One questions: `predict(state, questions)`.

    MLX on Apple silicon (backend "mlx"), PyTorch elsewhere, float32 either way; the
    studio's evaluation, playground and arena use it as they use laya's agents."""

    def __init__(self, model_dir, backend="torch", device=None):
        self.model_dir = Path(model_dir)
        self.cfg = config(self.model_dir)
        self.tok = tokenizer(self.model_dir)
        self.backend = backend
        if backend == "mlx":
            self.model = mlx_model(self.model_dir)
            self.model.eval()
        else:
            import torch

            self.torch = torch
            self.device = device or torch.device("cpu")
            self.model = torch_model(self.model_dir).to(self.device).eval()

    def logits(self, items):
        """Option logits for encoded items (ids, markers, qtype), one list per item."""
        n = len(items)
        length = max(len(it["ids"]) for it in items)
        length = ((length + 7) // 8) * 8
        count = max(2, max(len(it["markers"]) for it in items))
        if self.backend == "mlx":
            import mlx.core as mx
            import numpy as np

            from .engine import collate, decision_logits

            batch = collate(
                [{**it, "target": [0.0] * len(it["markers"]), "weight": 1.0} for it in items],
                self.tok.pad_token_id,
                multiple=8,
            )
            out = decision_logits(self.model, batch, training=False)
            mx.eval(out)
            out = np.asarray(out)
        else:
            torch = self.torch
            ids = torch.full((n, length), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((n, length), dtype=torch.long)
            pos = torch.zeros((n, count), dtype=torch.long)
            mask = torch.zeros((n, count), dtype=torch.bool)
            for i, it in enumerate(items):
                ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
                att[i, : len(it["ids"])] = 1
                pos[i, : len(it["markers"])] = torch.tensor(it["markers"])
                mask[i, : len(it["markers"])] = True
            qtype = torch.tensor([it["qtype"] for it in items], dtype=torch.long)
            with torch.inference_mode():
                out, _ = self.model(
                    ids.to(self.device),
                    att.to(self.device),
                    pos.to(self.device),
                    mask.to(self.device),
                    qtype.to(self.device),
                )
            out = out.float().cpu().numpy()
        return [out[i, : len(it["markers"])].astype(float).tolist() for i, it in enumerate(items)]

    def predict(self, state, questions):
        items, kept = [], []
        for qid, qdef in questions.items():
            ids, markers, _ = sequence(
                self.tok,
                state,
                qdef["type"],
                instructions(qdef),
                option_texts(qdef),
                self.cfg["max_len"],
                self.cfg["head_max_len"],
            )
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[qdef["type"]]})
            kept.append(qid)
        logits = self.logits(items)
        answers = {qid: answer(questions[qid], z) for qid, z in zip(kept, logits)}
        tokens = sum(len(it["ids"]) for it in items)
        return {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}


# ----------------------------------------------------------------------------- analysis


def analyze(dataset_id, model_dir, workspace):
    """The token-budget report for a Julia checkpoint: the rows its own runtime would refuse
    and why, option counts over 20, states cut by the 8,192-token window, thin classes."""
    from .engine import load_dataset, percentile

    questions, rows, meta = load_dataset(dataset_id, workspace)
    cfg = config(model_dir)
    tok = tokenizer(model_dir)
    state_tokens = [
        len(tok(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])))
        for r in rows
    ]
    report = {
        "model_dir": str(model_dir),
        "kind": "julia",
        "max_len": cfg["max_len"],
        "head_max_len": cfg["head_max_len"],
        "state_tokens": {
            "p50": percentile(state_tokens, 0.5),
            "p95": percentile(state_tokens, 0.95),
            "max": max(state_tokens),
        },
        "questions": {},
        "warnings": [],
    }
    for qid, qdef in questions.items():
        labeled = [r for r in rows if qid in r["targets"]]
        refused = {}
        for row in labeled:
            why = refusal(tok, row["state"], qdef, cfg)
            if why:
                refused[why] = refused.get(why, 0) + 1
        opts = option_texts(qdef)
        report["questions"][qid] = {
            "options": len(opts),
            "labeled_rows": len(labeled),
            "refused_by_strict": sum(refused.values()),
            "refusals": refused,
            "longest_option_tokens": max(len(tok(" " + o)) for o in opts),
        }
        if len(opts) > MAX_OPTIONS:
            report["warnings"].append(
                f"{qid}: {len(opts)} options. Julia answers 2 to {MAX_OPTIONS}: the studio trains "
                "and scores it anyway, but Julia's own runtime and NoulXP packages refuse it."
            )
        elif refused:
            top = max(refused, key=refused.get)
            report["warnings"].append(
                f"{qid}: Julia's own runtime would refuse {sum(refused.values())} of "
                f"{len(labeled)} rows ({top}). Training and the studio's evaluation cut instead."
            )
        counts = meta.get("labels", {}).get(qid, {}).get("train", {})
        thin = [label for label, n in counts.items() if 0 < n < 8]
        if thin:
            report["warnings"].append(
                f"{qid}: {len(thin)} labels have fewer than 8 training examples "
                f"({', '.join(thin[:6])}{' ...' if len(thin) > 6 else ''})."
            )
    return report
