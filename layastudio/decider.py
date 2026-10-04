"""Decider (Mapika) in the studio: its prompt, answers, calibration and inference.

Decider (huggingface.co/Mapika/decider-2b, github.com/Mapika/decider, Apache-2.0) is a
Qwen3.5 causal language model that does not generate. For each question it reads a prompt
that ends at an answer slot, "Answer: (", and the answer is the softmax of the logits of the
option letters at that slot, divided by a temperature per answer type. The studio fine-tunes
it as Decider v11 itself was made: LoRA on the attention and MLP projections, cross-entropy
over the option letters at the slot (decider_engine.py on PyTorch, decider_mlx.py on MLX).

Adapted from decider-ai at github.com/Mapika/decider 45024082 (Apache-2.0): prompt.py
(`build`, `label_table`: the plain state-first layout), systemone.py (`render_question`,
`render_state`, isolated levels, `format_answer`), temperature.py (one temperature per type)
and calibrate.py's rule (a type's temperature is fitted through that type's own readout).

The plain layout, one question per row, the context and the question tokenised apart:

    Context:\\n<state>
    \\n\\nQuestion: <instructions>\\nOptions:\\n(A) <option>\\n(B) <option>\\nAnswer: (

Up to 10 options the question is one encoded string; above 10 each option's label is one
token (A to Z, then the two-letter labels that are single tokens, 255 in all). Choice options
are "name: description" (the name alone without one); noul is "no"/"yes" ("no: <false>",
"yes: <true>" with descriptions); a score question with isolated levels (every released
Decider) is one yes/no row per level, "<instructions>\\nProposed answer: <level>\\nDoes the
proposed answer fit?", and its distribution is each row's P(yes), normalised.
"""

import json
import math
import re
import string
from pathlib import Path

LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)
MAX_OPTIONS = 255
MAX_LEVELS = 10
ANNOTATE_MIN = 8
ISOLATED = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?"
NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"
TYPES = ("choice", "noul", "score")
SOURCE = "github.com/Mapika/decider at 45024082 (decider-ai, Apache-2.0)"

# What the trainer adapts (decider_engine, decider_mlx): the attention and MLP projections of
# Qwen3.5's full-attention and Gated DeltaNet layers. in_proj_a and in_proj_b (one value per
# head) and the convolution stay as they are.
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Decider v11's own LoRA stage was rank 64, alpha 128, learning rate 1e-4, 5% warm-up and a
# cosine decay. A studio dataset is far smaller, so the default is the same shape at rank 16.
# The objective defaults to the proper scoring rule the Laya trainer uses (log + spherical
# score on each row's letters): on the Emotion example (A40, 600 test decisions) it beat
# Decider's own cross-entropy, 90.7% against 89.2%, log loss 0.33 against 0.48, ECE 0.029
# against 0.053. Cross-entropy stays available as "ce".
HYPERPARAMETERS = {
    "method": "lora",
    "objective": "proper",  # proper | ce
    "epochs": 2,
    "batch_size": 8,  # rows per micro-batch (fewer when batch_tokens says so)
    "batch_tokens": 8192,  # padded tokens per micro-batch
    "grad_accum": 2,
    "lr": 1e-4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "dora": False,
    "rslora": False,
    "loraplus_ratio": 1.0,
    "weight_decay": 0.0,
    "warmup": 0.05,
    "max_grad_norm": 1.0,
    "shuffle_options": True,  # choice options in a new order every epoch
    "max_train_options": 10,  # Decider's training protocol: at most 10 options, the gold kept
    "max_state_tokens": 2048,  # the context is cut to this many tokens in training
    "class_weighting": "none",
    "patience": 2,
    "grad_checkpoint": "auto",  # auto (on for GPUs) | on | off
    "precision": "bfloat16",  # the frozen base: bfloat16 | float32
    "quantization": "none",  # none | 4bit (QLoRA: the frozen base in 4 bits, CUDA or MLX)
    "seed": 13,
}


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def config(model_dir):
    """decider_config.json, refused unless it is the plain layout the studio trains."""
    cfg = _read_json(Path(model_dir) / "decider_config.json", {}) or {}
    layout = cfg.get("layout") or ("chat" if cfg.get("chat_template") is True else "plain")
    if layout != "plain":
        raise ValueError(f"only Decider's plain layout is trained here, not {layout!r}")
    return cfg


def temperature(cfg, qtype):
    """decider.temperature: temperature_by_type[type], else temperature (state-first)."""
    default = float(cfg.get("temperature", 1.0))
    return float((cfg.get("temperature_by_type") or {}).get(qtype, default))


# ----------------------------------------------------------------------------- tokens


class Tokens:
    """Decider's tokenizer from tokenizer.json alone, `encode(text)` without special tokens.

    The `tokenizers` library gives the ids transformers' tokenizer gives (noulxp built all 149
    rows of its request set both ways: 0 differences)."""

    def __init__(self, directory):
        from tokenizers import Tokenizer

        directory = Path(directory)
        self.backend = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.backend.no_truncation()
        self.backend.no_padding()
        cfg = _read_json(directory / "tokenizer_config.json", {}) or {}
        pad = cfg.get("pad_token") or cfg.get("eos_token")
        if isinstance(pad, dict):
            pad = pad.get("content")
        self.pad_token_id = self.backend.token_to_id(pad) if pad else None
        if self.pad_token_id is None:
            self.pad_token_id = 0

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids


def _txt(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _annotate(x):
    if isinstance(x, list):
        if len(x) >= ANNOTATE_MIN:
            return [
                {"_index": i, **_annotate(v)}
                if isinstance(v, dict)
                else {"_index": i, "value": _annotate(v)}
                for i, v in enumerate(x)
            ]
        return [_annotate(v) for v in x]
    if isinstance(x, dict):
        return {k: _annotate(v) for k, v in x.items()}
    return x


def render_state(state):
    """systemone.render_state: JSON states compact, long arrays with their indices written."""
    return state if isinstance(state, str) else json.dumps(_annotate(state), ensure_ascii=False)


def render_question(qdef):
    """systemone.render_question for a validated System One question: its text, option texts
    (in engine.option_labels order), type and level legend."""
    kind, crit = qdef["type"], qdef.get("criteria")
    raw = qdef.get("instructions", "")
    text = NOUL_WITHOUT_INSTRUCTIONS if kind == "noul" and raw in (None, "") else _txt(raw)
    if not text:
        raise ValueError("question without instructions")
    legend = None
    if kind == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(str(c) for c in crit)
        if not 2 <= len(crit) <= MAX_OPTIONS:
            raise ValueError(f"choice criteria: 2..{MAX_OPTIONS} options")
        options = [n if crit[n] in (None, "") else f"{n}: {_txt(crit[n])}" for n in crit]
    elif kind == "score":
        if not 2 <= len(crit) <= MAX_LEVELS:
            raise ValueError(f"score criteria: 2..{MAX_LEVELS} levels")
        options = [f"{i}: {_txt(c)}" for i, c in enumerate(crit)]
        legend = [_txt(c) for c in crit]
    else:
        c = crit if isinstance(crit, dict) else {}
        f, t = c.get("false"), c.get("true")
        options = [
            "no" if f in (None, "") else f"no: {_txt(f)}",
            "yes" if t in (None, "") else f"yes: {_txt(t)}",
        ]
    return {"question": text, "options": options, "type": kind, "legend": legend}


def strip_level_number(text):
    return re.sub(r"^\s*-?\d+\s*:\s*", "", text)


class Prompter:
    """Token ids exactly as decider.prompt builds the plain layout, from Tokens."""

    def __init__(self, tok, cfg=None):
        self.tok = tok
        self.cfg = cfg or {}
        self.isolated = bool(self.cfg.get("isolated_levels", False))
        upper = string.ascii_uppercase
        ids = []
        for label in list(upper) + [a + b for a in upper for b in upper]:
            found = tok.encode(label)
            if len(found) == 1:
                ids.append(found[0])
            if len(ids) == MAX_OPTIONS:
                break
        if len(ids) != MAX_OPTIONS or len(set(ids)) != MAX_OPTIONS:
            raise ValueError("the tokenizer has no 255 single-token option labels")
        for j, letter in enumerate(LETTERS):
            if tok.encode(letter) != [ids[j]]:
                raise ValueError("the tokenizer's letters do not match Decider's labels")
        self.label_ids = ids
        self.open_ids = tok.encode("\n(")

    def context(self, state, max_tokens=None):
        ids = self.tok.encode("Context:\n" + render_state(state))
        return ids[:max_tokens] if max_tokens else ids

    def question(self, text, options):
        head, tail = f"\n\nQuestion: {text}\nOptions:", "\nAnswer: ("
        if len(options) <= NARROW:
            body = "".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options))
            return self.tok.encode(head + body + tail)
        piece = self.tok.encode(head)
        for j, option in enumerate(options):
            piece += self.open_ids + [self.label_ids[j]] + self.tok.encode(f") {option}")
        return piece + self.tok.encode(tail)

    def rows(self, rendered):
        """The scoring rows of one rendered question: [(text, options)], and their kind."""
        if self.isolated and rendered["type"] == "score":
            return "iso", [
                (
                    ISOLATED.format(q=rendered["question"], level=strip_level_number(level)),
                    ["no", "yes"],
                )
                for level in rendered["legend"]
            ]
        return "list", [(rendered["question"], list(rendered["options"]))]


# ----------------------------------------------------------------------------- training rows


def _subsample(target, order, limit, rng):
    """Decider's training protocol: at most `limit` options, the gold always kept."""
    if not limit or len(order) <= limit:
        return order
    gold = max(range(len(target)), key=target.__getitem__)
    others = [i for i in order if i != gold]
    keep = (rng.sample(others, limit - 1) if rng else others[: limit - 1]) + [gold]
    if rng:
        rng.shuffle(keep)
    return keep


def encode_items(prompter, rows, questions, rng=None, shuffle=False, weights=None, hp=None):
    """Training and validation rows: one item per scoring row (an isolated score question
    gives one yes/no item per level), each with its token ids, option count, soft target,
    weight and answer type. Choice options are shuffled (and sub-sampled) when asked."""
    from .engine import argmax

    hp = hp or HYPERPARAMETERS
    limit = int(hp.get("max_train_options") or 0) if shuffle else 0
    items, skipped = [], 0
    rendered = {}
    for qid, qdef in questions.items():
        try:
            rendered[qid] = render_question(qdef)
        except ValueError:
            rendered[qid] = None
    for index, row in enumerate(rows):
        context = prompter.context(row["state"], hp.get("max_state_tokens"))
        for qid, target in row["targets"].items():
            rq = rendered.get(qid)
            if rq is None:
                skipped += 1
                continue
            label = argmax(target)
            weight = weights.get((qid, label), 1.0) if weights else 1.0
            kind, plan = prompter.rows(rq)
            if kind == "iso":
                for level, (text, options) in enumerate(plan):
                    yes = float(target[level])
                    items.append(
                        {
                            "ids": context + prompter.question(text, options),
                            "n": 2,
                            "target": [1.0 - yes, yes],
                            "weight": weight,
                            "type": "score",
                            "key": (index, qid, level),
                        }
                    )
                continue
            text, options = plan[0]
            order = list(range(len(options)))
            if shuffle and rq["type"] == "choice" and rng is not None:
                rng.shuffle(order)
                order = _subsample(target, order, limit, rng)
            items.append(
                {
                    "ids": context + prompter.question(text, [options[i] for i in order]),
                    "n": len(order),
                    "target": _renormalised([target[i] for i in order]),
                    "weight": weight,
                    "type": rq["type"],
                    "key": (index, qid, None),
                }
            )
    return items, skipped


def _renormalised(target):
    total = sum(target)
    return [t / total for t in target] if total > 0 else target


def token_batches(items, batch_size, batch_tokens, rng, shuffle=True):
    """Length-bucketed batches of at most batch_size rows and batch_tokens padded tokens."""
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    window = max(1, batch_size) * 32
    batches = []
    for start in range(0, len(order), window):
        part = sorted(order[start : start + window], key=lambda i: len(items[i]["ids"]))
        batch, longest = [], 0
        for i in part:
            n = len(items[i]["ids"])
            wider = max(longest, n)
            if batch and (len(batch) >= batch_size or wider * (len(batch) + 1) > batch_tokens):
                batches.append(batch)
                batch, longest = [], 0
            batch.append(i)
            longest = max(longest, n)
        if batch:
            batches.append(batch)
    if shuffle:
        rng.shuffle(batches)
    return batches


# ----------------------------------------------------------------------------- answers


def _norm(p):
    total = sum(p)
    return [1.0 / len(p)] * len(p) if total == 0 else [x / total for x in p]


def _certainty(p):
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, 1.0 - h / math.log(len(p))) if len(p) > 1 else 1.0


def softmax(z, temperature=1.0):
    scaled = [x / temperature for x in z]
    top = max(scaled)
    e = [math.exp(x - top) for x in scaled]
    total = sum(e)
    return [x / total for x in e]


def assemble(qdef, kind, row_probs):
    """One answer in the System One format (systemone.format_answer and assemble), keyed as
    engine.option_labels keys it: choice names, level indices, noul."""
    from .engine import option_labels

    if kind == "iso":
        fit = [float(p[1]) for p in row_probs]
        p = _norm(fit)
    else:
        p = _norm([float(x) for x in row_probs[0]])
    keys = option_labels(qdef)
    best = max(range(len(p)), key=p.__getitem__)
    if qdef["type"] == "noul":
        return {"type": "noul", "noul": p[1], "confidence": max(p)}
    answer = {
        "type": qdef["type"],
        "x_p_max": p[best],
        "confidence": p[best],
        "certainty": _certainty(p),
        "probabilities": dict(zip(keys, p)),
    }
    if qdef["type"] == "choice":
        answer["choice"] = keys[best]
    else:
        answer["score"] = sum(i * x for i, x in enumerate(p))
        answer["legend"] = {str(i): _txt(c) for i, c in enumerate(qdef["criteria"])}
        if kind == "iso":
            answer["level_fit"] = {str(j): x for j, x in enumerate(fit)}
            answer["fit_mass"] = sum(fit)
    return answer


# ----------------------------------------------------------------------------- calibration


def _golden(nll, lo=0.5, hi=5.0):
    """Golden-section search on 1/T, as engine.fit_temperature."""
    a, b = 1 / hi, 1 / lo
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(1 / c), nll(1 / d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(1 / c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(1 / d)
    return 1 / ((a + b) / 2)


def answer_distribution(kind, rows_logits, T):
    """The answer's distribution at temperature T from its rows' letter logits."""
    if kind == "iso":
        return _norm([softmax(z, T)[1] for z in rows_logits])
    return softmax(rows_logits[0], T)


def fit_temperatures(answers, base_cfg, minimum=10):
    """One temperature per answer type, fitted by NLL through that type's own readout (an
    isolated score through its normalised level fits), on validation answers:
    [(type, kind, rows_logits, target)]. A type with fewer than `minimum` answers keeps
    the base model's temperature."""

    def nll(chosen, T):
        total = 0.0
        for _, kind, logits, target in chosen:
            p = answer_distribution(kind, logits, T)
            total -= sum(t * math.log(max(x, 1e-12)) for t, x in zip(target, p))
        return total / max(1, len(chosen))

    fitted, by_type = [], {}
    for qtype in TYPES:
        chosen = [a for a in answers if a[0] == qtype]
        if len(chosen) >= minimum:
            # The Laya trainer's range (engine.fit_temperature), which laya-mlx also enforces:
            # a small validation split answered perfectly would otherwise drive T towards 0.
            by_type[qtype] = round(_golden(lambda T, c=chosen: nll(c, T), lo=0.5, hi=5.0), 4)
            fitted.append(qtype)
        else:
            by_type[qtype] = temperature(base_cfg, qtype)
    return by_type, fitted


def answers_ece(answers, by_type):
    from .engine import ece

    conf, correct = [], []
    for qtype, kind, logits, target in answers:
        p = answer_distribution(kind, logits, by_type.get(qtype, 1.0))
        conf.append(max(p))
        correct.append(float(p.index(max(p)) == target.index(max(target))))
    return ece(conf, correct)


def calibrated_config(base_cfg, by_type, provenance):
    """decider_config.json of a fine-tune: its fitted temperatures per type, its lineage.

    temperature_by_options (decider-ai 1.8) is dropped: decider-ai refuses a config with
    both maps, and the fine-tune's temperatures are the ones it fitted per type."""
    cfg = {k: v for k, v in base_cfg.items() if k != "temperature_by_options"}
    cfg["temperature_by_type"] = {t: float(by_type[t]) for t in TYPES}
    cfg["layout"] = "plain"
    base = provenance.get("base_model") or ""
    cfg["parent"] = f"{base} ({base_cfg.get('version', 'unversioned')})"
    cfg["version"] = f"{base_cfg.get('version', 'decider')}+studio-{provenance.get('run_id', '')}"
    hp = provenance.get("hyperparameters") or {}
    cfg["stage"] = (
        f"System One Studio: LoRA rank {hp.get('lora_rank')} (alpha {hp.get('lora_alpha')}) on "
        f"{', '.join(LORA_TARGETS)}, learning rate {hp.get('lr')}, best epoch "
        f"{provenance.get('best_epoch')} of {hp.get('epochs')}, "
        f"{provenance.get('train_decisions')} training rows in the plain state-first layout, "
        "cross-entropy on the option letters at the answer slot, merged into the weights; "
        "temperature_by_type fitted by NLL on the run's validation split"
    )
    cfg["release_date"] = (provenance.get("created") or "")[:10]
    cfg["fine_tuned"] = {
        "tool": "System One Studio",
        "base_model": base,
        "dataset_sha256": provenance.get("dataset_sha256"),
        "best_epoch": provenance.get("best_epoch"),
        "created": provenance.get("created"),
    }
    return cfg


# ----------------------------------------------------------------------------- inference


class Agent:
    """A Decider checkpoint answering System One questions: `predict(state, questions)`.

    PyTorch (transformers) on CUDA in bfloat16, as Decider's own engine runs, and in float32
    on a CPU; or MLX on Apple silicon (decider_mlx). Every row is read in full."""

    def __init__(self, model_dir, backend="torch", device=None, dtype=None, max_rows=16):
        self.model_dir = Path(model_dir)
        self.cfg = config(self.model_dir)
        self.tok = Tokens(self.model_dir)
        self.prompter = Prompter(self.tok, self.cfg)
        self.max_state_tokens = int(self.cfg.get("max_state_tokens", 32768))
        self.backend = backend
        self.max_rows = max_rows
        if backend == "mlx":
            from . import decider_mlx

            self.model = decider_mlx.load(self.model_dir)
            self.letters_fn = decider_mlx.letter_logits
        else:
            import torch
            from transformers import AutoModelForCausalLM

            self.torch = torch
            self.device = device or torch.device("cpu")
            kind = getattr(self.device, "type", "cpu")
            if dtype is None:
                bf16 = kind == "cuda" and torch.cuda.is_bf16_supported()
                dtype = torch.bfloat16 if bf16 else torch.float32
            portable_kernels(self.device)
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_dir), dtype=dtype)
            self.model.to(self.device).eval()

    def row_logits(self, rows):
        """Letter logits at the slot of every row (a list of (ids, n)), float lists."""
        out = []
        for start in range(0, len(rows), self.max_rows):
            chunk = rows[start : start + self.max_rows]
            if self.backend == "mlx":
                values = self.letters_fn(self.model, chunk, self.prompter, self.tok.pad_token_id)
            else:
                values = slot_letter_logits(
                    self.model, chunk, self.prompter.label_ids, self.tok.pad_token_id, self.device
                )
            out.extend(values)
        return out

    def predict(self, state, questions):
        context = self.prompter.context(state, self.max_state_tokens)
        plan, rows, used = [], [], len(context)
        for qid, qdef in questions.items():
            rq = render_question(qdef)
            kind, pieces = self.prompter.rows(rq)
            first = len(rows)
            for text, options in pieces:
                piece = self.prompter.question(text, options)
                rows.append((context + piece, len(options)))
                used += len(piece)
            plan.append((qid, qdef, kind, first, len(pieces)))
        logits = self.row_logits(rows)
        answers = {}
        for qid, qdef, kind, first, count in plan:
            T = temperature(self.cfg, qdef["type"])
            probs = [softmax(z, T) for z in logits[first : first + count]]
            answers[qid] = assemble(qdef, kind, probs)
        return {"answers": answers, "usage": {"input_tokens": used, "output_tokens": 0}}


# transformers binds flash-linear-attention's Triton kernels for Qwen3.5's linear attention when
# the package is installed, whatever device a model is on, and Triton takes only CUDA tensors.
PORTABLE = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)


def portable_kernels(device):
    """On any device but CUDA, Qwen3.5's linear attention through transformers' own PyTorch
    functions (the kernels it would otherwise call need CUDA tensors). Process-wide: a studio
    job runs on one device."""
    if getattr(device, "type", str(device)) == "cuda":
        return
    import importlib
    import inspect

    try:
        module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    except ImportError:
        return
    for name in PORTABLE:
        found = getattr(module, name, None)
        if found is not None:
            setattr(module, name, inspect.unwrap(found))


def slot_letter_logits(model, rows, label_ids, pad_id, device):
    """PyTorch: right-padded rows through the causal model; at each row's last token, its
    hidden state on the option-letter rows of the LM head (decider.model.slot_logits)."""
    import torch

    width = max(len(ids) for ids, _ in rows)
    width = ((width + 63) // 64) * 64
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (row, _) in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row)
        mask[i, : len(row)] = 1
    slots = torch.tensor([len(row) - 1 for row, _ in rows])
    letters = torch.tensor(label_ids[: max(n for _, n in rows)])
    with torch.inference_mode():
        backbone, head = backbone_and_head(model)
        hidden = backbone(
            input_ids=ids.to(device), attention_mask=mask.to(device)
        ).last_hidden_state
        picked = hidden[torch.arange(len(rows), device=hidden.device), slots.to(hidden.device)]
        weight = head.weight[letters.to(hidden.device)]
        values = torch.nn.functional.linear(picked.float(), weight.float()).cpu()
    return [values[i, :n].double().tolist() for i, (_, n) in enumerate(rows)]


def backbone_and_head(model):
    """The decoder stack and the LM head of a causal LM, under a PEFT wrapper or not."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return inner.model, inner.get_output_embeddings()


# ----------------------------------------------------------------------------- analysis


def analyze(dataset_id, model_dir, workspace):
    """The token report for a Decider checkpoint: state lengths against the training cap and
    the model's 32k context, rows per decision (isolated levels), wide option lists."""
    from .engine import load_dataset, percentile

    questions, rows, meta = load_dataset(dataset_id, workspace)
    cfg = config(model_dir)
    tok = Tokens(model_dir)
    prompter = Prompter(tok, cfg)
    states = [len(prompter.context(r["state"])) for r in rows]
    cap = HYPERPARAMETERS["max_state_tokens"]
    report = {
        "model_dir": str(model_dir),
        "kind": "decider",
        "max_len": int(cfg.get("max_state_tokens", 32768)),
        "train_state_cap": cap,
        "state_tokens": {
            "p50": percentile(states, 0.5),
            "p95": percentile(states, 0.95),
            "max": max(states),
        },
        "questions": {},
        "warnings": [],
    }
    for qid, qdef in questions.items():
        rq = render_question(qdef)
        kind, pieces = prompter.rows(rq)
        report["questions"][qid] = {
            "options": len(rq["options"]),
            "rows_per_decision": len(pieces),
            "question_tokens": max(len(prompter.question(t, o)) for t, o in pieces),
            "readout": "isolated yes/no per level" if kind == "iso" else "letters",
        }
        if len(rq["options"]) > NARROW:
            report["warnings"].append(
                f"{qid}: {len(rq['options'])} options, read with one label token each (Decider "
                f"trains on at most {HYPERPARAMETERS['max_train_options']} per example, gold kept)."
            )
        counts = meta.get("labels", {}).get(qid, {}).get("train", {})
        thin = [label for label, n in counts.items() if 0 < n < 8]
        if thin:
            report["warnings"].append(
                f"{qid}: {len(thin)} labels have fewer than 8 training examples "
                f"({', '.join(thin[:6])}{' ...' if len(thin) > 6 else ''})."
            )
    over = sum(n > cap for n in states)
    if over:
        report["warnings"].append(
            f"{over} of {len(rows)} states are longer than the {cap:,} tokens training reads; "
            "their ends are cut in training (raise max_state_tokens if memory allows)."
        )
    return report
