"""Datasets: reading labeled rows, checking them and splitting them, with no ML library.

Everything here is the standard library, so a dataset gets the same answer wherever it is
checked: on this machine when the studio makes one (engine.create_dataset), on a server that
checks an upload before any GPU is rented, and on the GPU before training starts
(cloud.prepare_run). The same file, questions and seed give the same rows, the same split and
the same digest everywhere.

    questions, rows, report = validate(questions, text, "train.jsonl", seed=13)

Rows come as JSONL, a JSON array, {"rows": [...]}, CSV or TSV (README, "Your data"). A row
without a "split" is placed by assign_splits: 80/10/10, stratified by the first question's
label, from the seed. The report has the rows and decisions per split, the labels, the options
per question and the checkpoint kinds (kinds.py) whose trainers answer every question.
"""

import csv
import hashlib
import io
import json
import math
import random
import re

from . import decider, julia

# The question types of laya_mlx's common.QTYPES, in its order (tests/test_datasets.py checks).
QUESTION_TYPES = ("choice", "score", "noul")
STATE_KEYS = ("state", "text", "input", "message", "content")
SPLITS = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
MIN_ROWS = 10


# ----------------------------------------------------------------------------- questions


def internal_question(qdef):
    """A question in the model's internal form ({"t", "ins", "crit"}), or ValueError.

    The rules and messages of laya_mlx's Agent._to_internal (laya-mlx, Apache-2.0,
    github.com/mizorewww/laya-mlx), adapted here because that module imports MLX, which
    only Apple silicon has, and datasets are made on every machine. tests/test_engine.py
    checks the two agree wherever laya_mlx can be imported.
    """
    if not isinstance(qdef, dict):
        raise ValueError("Each question must be a dictionary")
    kind = qdef.get("type")
    if kind not in QUESTION_TYPES:
        raise ValueError(f"Unknown question type {kind!r}; expected choice, score, or noul")
    if "instructions" not in qdef:
        raise ValueError("Question is missing instructions")
    criteria = qdef.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            if not all(isinstance(c, str) for c in criteria):
                raise ValueError("Choice labels must be strings")
            if len(set(criteria)) != len(criteria):
                raise ValueError("Choice labels must be unique")
            criteria = dict.fromkeys(criteria)
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("Choice criteria must be a nonempty dictionary or list")
        if not all(isinstance(k, str) for k in criteria):
            raise ValueError("Choice labels must be strings")
    elif kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("Score criteria must be a nonempty list")
    elif criteria is not None and not isinstance(criteria, dict):
        raise ValueError("Noul criteria must be a dictionary with false/true descriptions")
    instructions = qdef["instructions"]
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions)
    return {"t": kind, "ins": instructions, "crit": criteria}


def option_labels(qdef):
    """Answer names in the model's option order: criteria keys, level indices, false/true."""
    if qdef["type"] == "choice":
        return list(qdef["criteria"])
    if qdef["type"] == "score":
        return [str(i) for i in range(len(qdef["criteria"]))]
    return ["false", "true"]


def option_names(qdef):
    """Human-readable names for reports: level text for score questions."""
    if qdef["type"] == "score":
        return [
            f"{i}: {c}" if isinstance(c, str) else str(i) for i, c in enumerate(qdef["criteria"])
        ]
    return option_labels(qdef)


def validate_questions(questions):
    if isinstance(questions, str):
        questions = json.loads(questions)
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty JSON object keyed by question id")
    for qid, qdef in questions.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", qid):
            raise ValueError(f"Question id {qid!r} must use letters, digits, '_', '.' or '-'")
        try:
            internal_question(qdef)
        except ValueError as error:
            raise ValueError(f"Question {qid!r}: {error}") from None
    return questions


def resolve_label(qdef, value):
    """Map one answer value to an option index, accepting the forgiving spellings people use."""
    kind, labels = qdef["type"], option_labels(qdef)
    if kind == "choice":
        text = str(value)
        if text in labels:
            return labels.index(text)
        folded = [
            i for i, label in enumerate(labels) if label.strip().lower() == text.strip().lower()
        ]
        if len(folded) == 1:
            return folded[0]
        shown = ", ".join(labels[:8]) + (" ..." if len(labels) > 8 else "")
        raise ValueError(f"unknown label {text!r} (expected one of: {shown})")
    if kind == "score":
        if isinstance(value, bool):
            raise ValueError("score answers must be a level index, not true/false")
        if isinstance(value, (int, float)) and float(value).is_integer():
            index = int(value)
        else:
            text = str(value).strip()
            match = re.fullmatch(r"(?:level\s*)?(\d+)", text, re.IGNORECASE)
            levels = [str(c).strip().lower() for c in qdef["criteria"]]
            if match:
                index = int(match.group(1))
            elif text.lower() in levels:
                index = levels.index(text.lower())
            else:
                raise ValueError(f"unknown score level {value!r}")
        if not 0 <= index < len(labels):
            raise ValueError(f"score level {index} is outside 0..{len(labels) - 1}")
        return index
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in ("true", "yes", "y", "t", "1"):
        return 1
    if text in ("false", "no", "n", "f", "0"):
        return 0
    raise ValueError(f"noul answers must be true/false or a probability, got {value!r}")


def answer_target(qdef, value):
    """Return the gold distribution over options. Dicts are soft labels (e.g. from annotators)."""
    n = len(option_labels(qdef))
    if isinstance(value, str) and value.strip().startswith("{"):
        value = json.loads(value)
    if isinstance(value, dict):
        probs = [0.0] * n
        for key, p in value.items():
            p = float(p)
            if not math.isfinite(p) or p < 0:
                raise ValueError(f"soft label probability for {key!r} must be >= 0")
            probs[resolve_label(qdef, key)] += p
        total = sum(probs)
        if total <= 0:
            raise ValueError("soft label probabilities sum to zero")
        return [p / total for p in probs]
    if qdef["type"] == "noul" and not isinstance(value, bool):
        try:
            p = float(value)
        except (TypeError, ValueError):
            p = None
        if p is not None and not (isinstance(value, str) and value.strip() in ("0", "1")):
            if not 0.0 <= p <= 1.0:
                raise ValueError("noul probability must be between 0 and 1")
            return [1.0 - p, p]
    target = [0.0] * n
    target[resolve_label(qdef, value)] = 1.0
    return target


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def fits(questions):
    """Why each checkpoint kind's trainer would leave a question out: {kind: [reasons]}.

    The trainers' own rules: Julia 1 answers 2 to 20 options (julia.encode_items), Decider 2 to
    255 choice options and 2 to 10 score levels (decider.render_question). Laya has no count
    limit; its options share a token budget, which only its tokenizer can measure (the
    dataset analysis does)."""
    out = {"laya": [], "julia": [], "decider": []}
    for qid, qdef in questions.items():
        count = len(julia.option_texts(qdef))
        if not 2 <= count <= julia.MAX_OPTIONS:
            out["julia"].append(f"{qid}: {count} options; Julia 1 answers 2 to {julia.MAX_OPTIONS}")
        try:
            decider.render_question(qdef)
        except ValueError as error:
            out["decider"].append(f"{qid}: {error}")
    return out


# ----------------------------------------------------------------------------- rows


def _records(text, filename):
    """Yield (line, record) from JSONL, a JSON array, {"rows": [...]}, CSV or TSV."""
    name = (filename or "").lower()
    stripped = text.lstrip("﻿").strip()
    if name.endswith((".csv", ".tsv")):
        delimiter = "\t" if name.endswith(".tsv") else ","
        reader = csv.DictReader(io.StringIO(stripped), delimiter=delimiter)
        for i, record in enumerate(reader, start=2):
            yield i, {k.strip(): v for k, v in record.items() if k is not None}
        return
    if stripped.startswith("[") or (stripped.startswith("{") and name.endswith(".json")):
        data = json.loads(stripped)
        rows = data.get("rows") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError('JSON files must contain a list of rows or {"rows": [...]}')
        yield from enumerate(rows, start=1)
        return
    for i, line in enumerate(stripped.splitlines(), start=1):
        if line.strip():
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as error:
                yield i, ValueError(f"invalid JSON: {error.msg}")


def _cell_state(value):
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def normalize_record(record, questions):
    if isinstance(record, Exception):
        raise record
    if not isinstance(record, dict):
        raise ValueError("each row must be an object")
    key = next((k for k in STATE_KEYS if k in record), None)
    if key is None:
        raise ValueError(f"missing state (use one of: {', '.join(STATE_KEYS)})")
    state = _cell_state(record[key])
    if state is None or (isinstance(state, str) and not state.strip()):
        raise ValueError("state is empty")
    answers = record.get("answers")
    if answers is None:
        answers = {k: v for k, v in record.items() if k in questions}
    if not isinstance(answers, dict):
        raise ValueError("answers must be an object keyed by question id")
    targets = {}
    for qid, value in answers.items():
        if qid not in questions:
            raise ValueError(f"answer for unknown question {qid!r}")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        try:
            targets[qid] = answer_target(questions[qid], value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{qid}: {error}") from None
    if not targets:
        raise ValueError("row has no labeled questions")
    row = {"state": state, "targets": targets}
    split = str(record.get("split", "")).strip().lower()
    if split:
        if split not in SPLITS:
            raise ValueError(f"split must be train, val or test, got {split!r}")
        row["split"] = SPLITS[split]
    if record.get("id") not in (None, ""):
        row["id"] = str(record["id"])
    return row


def parse_rows(text, filename, questions):
    rows, errors = [], []
    try:
        for line, record in _records(text, filename):
            try:
                rows.append(normalize_record(record, questions))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                errors.append({"file": filename, "line": line, "error": str(error)})
    except (ValueError, csv.Error, json.JSONDecodeError) as error:
        errors.append({"file": filename, "line": 0, "error": str(error)})
    return rows, errors


# ----------------------------------------------------------------------------- the split


def assign_splits(rows, questions, seed, has_test_file, val_frac=0.1, test_frac=0.1):
    """Stratified, deterministic split for rows without an explicit split field."""
    free = [r for r in rows if "split" not in r]
    has_test = has_test_file or any(r.get("split") == "test" for r in rows)
    first = next(iter(questions))
    rng = random.Random(seed)
    rng.shuffle(free)
    free.sort(key=lambda r: str(argmax(r["targets"][first])) if first in r["targets"] else "~")
    golden = (math.sqrt(5) - 1) / 2
    for j, row in enumerate(free):
        u = (j * golden) % 1.0
        if not has_test and u < test_frac:
            row["split"] = "test"
        elif u < (0 if has_test else test_frac) + val_frac:
            row["split"] = "val"
        else:
            row["split"] = "train"
    for needed in ("val", "test"):
        if not any(r["split"] == needed for r in rows):
            train = [r for r in rows if r["split"] == "train"]
            if len(train) < 3:
                raise ValueError("Not enough rows to create train, validation and test splits")
            train[-1]["split"] = needed


def split_counts(rows):
    return {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}


def label_counts(rows, questions):
    counts = {}
    for qid, qdef in questions.items():
        labels = option_labels(qdef)
        per = {s: {label: 0 for label in labels} for s in ("train", "val", "test")}
        for row in rows:
            if qid in row["targets"]:
                per[row["split"]][labels[argmax(row["targets"][qid])]] += 1
        counts[qid] = per
    return counts


def digest(questions, rows):
    """The dataset's identity: its questions and its split rows, as canonical JSON."""
    return hashlib.sha256(
        json.dumps([questions, rows], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


# ----------------------------------------------------------------------------- the check


def validate(
    questions,
    train_text,
    train_name,
    test_text=None,
    test_name=None,
    seed=13,
    max_bytes=None,
    max_rows=None,
):
    """Read, check and split a dataset without writing anything. (questions, rows, report)

    Raises ValueError for what cannot train: bad questions, a file over max_bytes, more than
    max_rows rows, or fewer than MIN_ROWS valid labeled rows. Rows that cannot be read are
    reported (the first 100, and how many) and left out, as the studio has always done."""
    questions = validate_questions(questions)
    size = sum(len(text.encode()) for text in (train_text, test_text) if text)
    if max_bytes is not None and size > max_bytes:
        raise ValueError(
            f"The dataset is {size / 2**20:.1f} MB; the limit is {max_bytes / 2**20:g} MB"
        )
    rows, errors = parse_rows(train_text, train_name, questions)
    if test_text:
        test_rows, test_errors = parse_rows(test_text, test_name, questions)
        for row in test_rows:
            row["split"] = "test"
        rows += test_rows
        errors += test_errors
    if max_rows is not None and len(rows) + len(errors) > max_rows:
        raise ValueError(
            f"The dataset has {len(rows) + len(errors):,} rows; the limit is {max_rows:,}"
        )
    if len(rows) < MIN_ROWS:
        detail = f" First error: line {errors[0]['line']}: {errors[0]['error']}" if errors else ""
        raise ValueError(f"Need at least {MIN_ROWS} valid labeled rows, found {len(rows)}.{detail}")
    assign_splits(rows, questions, seed, bool(test_text))
    for i, row in enumerate(rows):
        row.setdefault("id", str(i))
    counts = split_counts(rows)
    problems = fits(questions)
    report = {
        "seed": seed,
        "sha256": digest(questions, rows),
        "size_bytes": size,
        "rows": counts,
        "decisions": {s: sum(len(r["targets"]) for r in rows if r["split"] == s) for s in counts},
        "labels": label_counts(rows, questions),
        "questions": {
            qid: {"type": qdef["type"], "options": len(option_labels(qdef))}
            for qid, qdef in questions.items()
        },
        "max_options": max(len(option_labels(qdef)) for qdef in questions.values()),
        "fits": problems,
        "kinds": [kind for kind, reasons in problems.items() if not reasons],
        "errors": errors[:100],
        "error_count": len(errors),
    }
    return questions, rows, report
