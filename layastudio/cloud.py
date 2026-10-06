"""Fine-tunes without the UI: the checks every run starts with, and a whole run from a file.

A run in the cloud studio trains on a rented GPU that nobody watches, so it has to be refused,
or started, exactly as the Train button here refuses or starts it. Both call prepare_run, the
one place those rules live:

- the base model is a complete checkpoint of a kind the studio trains (kinds.py), and its
  licence allows a fine-tune (families.trainer_status);
- the hyperparameters keep only the keys that kind's trainer has; each value has its default's
  type and is one of the trainer's choices, within the platform's bounds when it sends any;
  the LoRA variants' rules (engine.check_lora_variants) and Decider's own
  (decider_engine.check_hyperparameters) apply;
- the dataset is one this workspace has, or files read, checked and split by
  datasets.validate: the same split the platform computed when it checked the upload, which
  the run can be told to expect; and the kind's trainer has decisions to learn from in it;
- the exports are ones the kind has, at a precision it takes, with the tooling installed
  (check_export, which the Export button calls too).

Then it writes the run's record, runs/<id>/run.json, and returns the jobs to start.

    layastudio train --config run.json         # or: python -m layastudio.cloud --config run.json

runs one fine-tune end to end with no UI: training (baseline, fit, evaluation, comparison),
then each export, NoulXP packages included. Each is the child process the UI starts
(python -m layastudio.engine run <job_dir>), so a run here is the run the studio makes. Every
event is printed to stdout as one JSON line, and a result file says how the run ended, with
its measurements and its output files (size and SHA-256). It is what a cloud GPU runs.

The config, with paths relative to its own folder:

    {
      "name": "emotion",                              optional
      "base_model": "hub:Mapika/decider-2b@<commit>", hub:<repo>[@<revision>] (downloaded),
                                                      path:<dir>, run:<id>
      "dataset": {                                    or the id of a dataset in the workspace
        "questions": "questions.json",                a file, or the questions themselves
        "train": "train.jsonl", "test": "test.jsonl", test optional; JSONL, JSON, CSV or TSV
        "seed": 13,                                   the split's seed
        "expected": {"rows": {"train": 1079, "val": 121, "test": 600}, "digest": "..."}
      },
      "hyperparameters": {"epochs": 2},               over the kind's defaults
      "bounds": {"epochs": {"min": 1, "max": 5}},     optional: what the run may change
      "limits": {"max_bytes": 26214400, "max_rows": 200000},
      "exports": [{"target": "noulxp", "gguf": "bf16"}],
      "keep_checkpoint": false,                       optional: model/ is among what is kept
      "template": "decider-2b.lora",                  optional: the platform's name for the run
      "run_id": "...", "baseline": true, "workspace": "..."
    }

- base_model: a model downloaded at a pinned revision is hub:<repo>@<revision>, which is how
  its licence is checked and its packages name it (source.model, source.revision). A path:
  folder has a published name only when it was imported from the registry (imports.json);
  any other is trained with a warning, unchecked, and its packages name no base model.
- A dataset file is a path, or {"path", "name", "sha256"}: name, the file's own name, whose
  extension chooses the reader (when it was saved under another); sha256, its bytes', which
  must match. expected: what the dataset was when it was checked: its rows per split, and its
  digest, the report's sha256 (its questions and split rows, not a file's bytes).
- bounds: with any, the run may change only the hyperparameters they name, each within its
  {"min", "max"} or its {"choices"}.
- keep_checkpoint (default false): whether the checkpoint in its maker's format, model/, is
  among the outputs a cloud run keeps, besides its NoulXP package and its card. The run's
  folder holds it either way (the package and the card are made from it) and the result's
  manifest lists every output: the agent uploads what the switch says. The result and the
  "prepared" event repeat it; a run that keeps no checkpoint and makes no NoulXP package is
  told so in its warnings, since its card would be all it keeps.
- template: recorded as it is, in the result and in card/finetune.json.
- The config is the agent's, never a user's: its paths are read as given (absolute ones and
  "..", too). A GPU image sets $LAYASTUDIO_TOOLS to the llama.cpp converter it bakes in
  (gguf.py), or Decider's GGUF fetches it beside the workspace.

What a run makes, in runs/<id>/, each group listed in the result's outputs with every file's
size and SHA-256: model/ (the checkpoint), noulxp/ (its NoulXP package, only one that passed),
exports/ (GGUF, MLX-LM) and card/, written once it trained (whether or not an export failed):

- card/README.md: the model card the registry shows, written from the run's measurements as
  publish_systemone writes it, NoulXP line included. It names the model <namespace>/<name>
  (CARD_REPO) in its examples and the run's name in its title: publishing puts the model's.
- card/finetune.json: the version's record of the fine-tune: the base model at its revision,
  the template, the hyperparameters, the dataset's digest and size, the training and the
  measurements before and after, and the NoulXP check. A kept checkpoint's own finetune.json
  (Julia 1, Decider) is the trainers' record of the same run, with paths on the GPU: at a
  version's root the card's replaces it.

Neither holds a row of the dataset or a path on the machine that trained.

Training's "step" events (every trainer) carry the update, the epoch of epochs, and
fraction: the share of the training done, which reaches 1 at the last update, with eta_s
the seconds left at that pace.

The exit code, also the result's exit_code: 0 when it trained and made every export; 3
(partial) when it trained but an export failed, which the result's failed_exports names (the
checkpoint and the other exports are there); 1 when it failed; 2 when it was refused before
anything ran; 143 when it was cancelled (SIGTERM or Ctrl+C).
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import datasets, engine, kinds

# The values a text hyperparameter takes (the trainers' own branches; Decider narrows method
# and objective further in decider_engine.check_hyperparameters).
CHOICES = {
    "method": ("lora", "head", "full"),
    "objective": ("proper", "rlcd", "ce"),
    "class_weighting": ("none", "balanced"),
    "grad_checkpoint": ("auto", "on", "off"),
    "precision": ("bfloat16", "float32"),
    "quantization": ("none", "4bit"),
}
# Hyperparameters that count something: whole numbers, and some of them at least 1.
COUNTS = (
    "epochs",
    "batch_size",
    "grad_accum",
    "lora_rank",
    "lora_layers",
    "full_layers",
    "patience",
    "seed",
    "batch_tokens",
    "max_train_options",
    "max_state_tokens",
)
AT_LEAST_ONE = ("epochs", "batch_size", "grad_accum", "lora_rank", "batch_tokens")
# Fractions: a dropout is below 1 (PyTorch and MLX refuse 1 and more, after the baseline), and
# warm-up is a share of the updates.
BELOW_ONE = ("lora_dropout", "head_dropout")
AT_MOST_ONE = ("warmup",)
# The precision an export gets when the run does not say (as `python -m layastudio.export`).
DEFAULT_PRECISION = {"gguf": "bf16", "mlx": "int4"}
# Outputs of a run, in the result's manifest: the checkpoint, its NoulXP package (only one that
# passed its check), the exports (GGUF, MLX-LM) and the card (README.md, finetune.json).
CARD = "card"
OUTPUTS = ("model", "noulxp", "exports", CARD)
# The model's name in the card's examples, until publishing names it.
CARD_REPO = "<namespace>/<name>"
# What card/finetune.json keeps of the training summary (training.json): never a path.
TRAINING_KEYS = (
    "train_decisions",
    "val_decisions",
    "skipped_decisions",
    "trainable_params",
    "total_params",
    "updates",
    "best_epoch",
    "best_val_loss",
    "history",
    "train_seconds",
    "peak_memory_gb",
    "lora_variants",
    "backend",
    "device",
    "created",
)
# What it keeps of the NoulXP check (noulxp_package.describe).
NOULXP_KEYS = (
    "standard",
    "profile",
    "noulxp",
    "gguf",
    "cases",
    "cases_passed",
    "max_abs_dp",
    "mean_abs_dp",
    "argmax",
    "test_rows",
    "runtime_precision",
    "checked_at",
    "size_mb",
)


class Refused(ValueError):
    """A run that does not start, and why. The server answers 400 with the message."""


class Unavailable(Refused):
    """What this machine cannot make until something is installed. The server answers 409."""


# ----------------------------------------------------------------------------- the checks


def check_licence(ref, workspace):
    """Refuse a base model whose licence does not allow derivatives (families.py): a hub
    repository at any revision, a folder imported from the registry, or a run's own base."""
    from . import families, noulxp_package

    known = families.find(noulxp_package.base_model(ref, workspace) or "")
    if known and not families.trainer_status(known)["ready"]:
        raise Refused(families.trainer_status(known)["reason"])


def base_model(ref, workspace):
    """(checkpoint folder, kind) of a base model the studio may fine-tune, or Refused."""
    if not isinstance(ref, str) or not ref:
        raise Refused("Choose a base model: hub:<repo>[@<revision>], path:<folder> or run:<id>")
    try:
        model_dir = engine.resolve_model_ref(ref, workspace)
        kind = kinds.check(model_dir)
    except (FileNotFoundError, ValueError) as error:
        raise Refused(str(error)) from None
    check_licence(ref, workspace)
    return model_dir, kind


def check_value(key, value, default):
    """A hyperparameter's value, of its default's type and within the trainer's choices; counts
    come back as whole numbers."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise Refused(f"{key} must be true or false")
        return value
    if isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Refused(f"{key} must be a number")
        if not math.isfinite(value):
            raise Refused(f"{key} must be a finite number")
        if key in COUNTS:
            if not float(value).is_integer():
                raise Refused(f"{key} must be a whole number")
            value = int(value)
        least = 1 if key in AT_LEAST_ONE else 0
        if value < least:
            raise Refused(f"{key} must be at least {least}")
        if key in BELOW_ONE and value >= 1:
            raise Refused(f"{key} must be below 1")
        if key in AT_MOST_ONE and value > 1:
            raise Refused(f"{key} must be at most 1")
        return value
    allowed = CHOICES.get(key)
    if not isinstance(value, str) or (allowed and value not in allowed):
        raise Refused(f"{key} is one of: {', '.join(allowed)}" if allowed else f"{key} is text")
    return value


def check_bounds(hp, overrides, bounds):
    """The platform's limits for a template, on the run's final values:
    {key: {"min": a, "max": b} or {"choices": [...]}}. With bounds, the run changes only the
    keys they name: a template's own settings come bounded too ({"choices": [one]})."""
    if bounds is None:
        return
    if not isinstance(bounds, dict):
        raise Refused("bounds must be an object of {key: {min, max} or {choices}}")
    unbounded = sorted(k for k in overrides if k not in bounds)
    if unbounded:
        raise Refused(
            f"This run may change {', '.join(sorted(bounds)) or 'no hyperparameter'}, "
            f"not {', '.join(unbounded)}"
        )
    for key, bound in bounds.items():
        if key not in hp or not isinstance(bound, dict):
            raise Refused(f"bounds for {key!r}: not a hyperparameter of this trainer")
        value = hp[key]
        try:
            if "choices" in bound and value not in bound["choices"]:
                shown = ", ".join(str(c) for c in bound["choices"])
                raise Refused(f"{key} must be one of: {shown}")
            if "min" in bound and value < bound["min"]:
                raise Refused(f"{key} must be at least {bound['min']}")
            if "max" in bound and value > bound["max"]:
                raise Refused(f"{key} must be at most {bound['max']}")
        except TypeError:
            raise Refused(f"bounds for {key!r} do not fit its value, {value!r}") from None


def hyperparameters(kind, given, bounds=None):
    """(merged, overrides, ignored): the kind's defaults with the given values for the keys its
    trainer has, the overrides alone (what the job spec carries), and the keys left out."""
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise Refused("hyperparameters must be an object")
    defaults = engine.hyperparameters(kind)
    overrides = {k: check_value(k, v, defaults[k]) for k, v in given.items() if k in defaults}
    ignored = sorted(k for k in given if k not in defaults)
    try:
        engine.check_lora_variants(overrides)
        if kind == kinds.DECIDER:
            from .decider_engine import check_hyperparameters

            check_hyperparameters({**defaults, **overrides})
    except ValueError as error:
        raise Refused(str(error)) from None
    merged = {**defaults, **overrides}
    check_bounds(merged, overrides, bounds)
    return merged, overrides, ignored


def _read(path, base, max_bytes=None):
    """A file's bytes, exactly: the platform checks the bytes uploaded, and reading the file as
    text would turn a quoted CSV cell's CRLF into LF, which makes it another dataset."""
    path = Path(path)
    path = path if path.is_absolute() else Path(base) / path
    try:
        size = path.stat().st_size
        if max_bytes is not None and size > max_bytes:
            raise Refused(
                f"{path.name} is {size / 2**20:.1f} MB; the limit is {max_bytes / 2**20:g} MB"
            )
        return path.read_bytes()
    except OSError as error:
        raise Refused(f"Cannot read {path.name}: {error}") from None


def _text(data, name):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Refused(f"Cannot read {name}: {error}") from None


def dataset_file(given, base, max_bytes=None, field="train"):
    """(name, text) of one of a run's dataset files: a path, or {"path", "name", "sha256"}.
    name is the file's own name, whose extension chooses the reader, when it was saved under
    another; sha256 is its bytes', and a file that differs is refused."""
    entry = given if isinstance(given, dict) else {"path": given}
    path, name, sha256 = entry.get("path"), entry.get("name"), entry.get("sha256")
    if not isinstance(path, str) or not path:
        raise Refused(f'The {field} file is a path, or {{"path", "name", "sha256"}}')
    if name is not None and (not isinstance(name, str) or not name):
        raise Refused(f"The {field} file's name is text: the file's own name")
    name = name or Path(path).name
    data = _read(path, base, max_bytes)
    if sha256 is not None:
        found = hashlib.sha256(data).hexdigest()
        if found != sha256:
            raise Refused(
                f"The {field} file is not the one uploaded: its sha256 is {found}, "
                f"expected {sha256}"
            )
    return name, _text(data, name)


def _limit(limits, key):
    value = limits.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise Refused(f"limits.{key} is a whole number")
    return value


def dataset(given, workspace, base=None, limits=None):
    """(questions, rows, report, meta) of the run's dataset: one the workspace has (its id), or,
    when the run comes from a file (base: that file's folder), files to read, check and split
    now: {"questions", "train", "test", "name", "seed", "expected"}. report is None for a
    saved dataset; meta is None until a checked one is saved (save_dataset)."""
    if isinstance(given, str):
        try:
            questions, rows, meta = engine.load_dataset(given, workspace)
        except (FileNotFoundError, ValueError) as error:
            raise Refused(str(error)) from None
        return questions, rows, None, meta
    if base is None or not isinstance(given, dict):
        raise Refused("Choose a dataset: the id of one in this workspace")
    limits = {} if limits is None else limits
    if not isinstance(limits, dict):
        raise Refused("limits is {max_bytes, max_rows}")
    max_bytes, max_rows = _limit(limits, "max_bytes"), _limit(limits, "max_rows")
    if "train" not in given or "questions" not in given:
        raise Refused("A dataset names its questions and its train file")
    questions = given["questions"]
    if isinstance(questions, str):
        try:
            questions = json.loads(_text(_read(questions, base), questions))
        except json.JSONDecodeError as error:
            raise Refused(f"The questions are not JSON: {error}") from None
    elif not isinstance(questions, dict):
        raise Refused("The questions are a file, or the questions themselves")
    train_name, train_text = dataset_file(given["train"], base, max_bytes, "train")
    test_name, test_text = (
        dataset_file(given["test"], base, max_bytes, "test") if given.get("test") else (None, None)
    )
    seed = given.get("seed", 13)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Refused("The split's seed is a whole number")
    try:
        questions, rows, report = datasets.validate(
            questions, train_text, train_name, test_text, test_name, seed, max_bytes, max_rows
        )
    except (ValueError, TypeError) as error:
        raise Refused(str(error)) from None
    expected = given.get("expected") or {}
    if not isinstance(expected, dict):
        raise Refused("expected is {rows, digest}: what the checked dataset was")
    for key in expected:
        if key not in ("rows", "digest"):
            raise Refused(
                f"expected has rows and digest (the checked dataset's sha256), not {key!r}; "
                "a file's own sha256 goes with the file"
            )
    for key, found in (("rows", report["rows"]), ("digest", report["sha256"])):
        if key in expected and expected[key] != found:
            raise Refused(
                f"The dataset does not match the one that was checked: {key} is "
                f"{found}, expected {expected[key]}"
            )
    report["files"] = {"train": train_name, "test": test_name}
    report["name"] = str(given.get("name") or Path(train_name).stem)
    return questions, rows, report, None


def check_export(kind, item, model_ref):
    """One export of a checkpoint of a kind (kinds.py), as its job's spec: {"model", "target",
    "precision", "gguf", "test_rows"}. The Export button's rules and a run's, in one place.

    item: {"target", "precision", "gguf", "test_rows"}, or a target alone. precision defaults
    to the export's own (bf16 GGUF, int4 MLX, float otherwise); gguf chooses the GGUF a
    Decider NoulXP package carries; test_rows, the test rows its conformance file holds.
    A NoulXP package needs its tooling here (Unavailable)."""
    from .export import KIND_TARGETS, PRECISIONS

    if isinstance(item, str):
        item = {"target": item}
    if not isinstance(item, dict):
        raise Refused('An export is {"target": ...}')
    target = item.get("target")
    allowed = KIND_TARGETS[kind]
    if target not in allowed:
        raise Refused(f"A {kinds.NAME[kind]} model exports to {', '.join(allowed)}, not {target!r}")
    precision = item.get("precision") or DEFAULT_PRECISION.get(target, "float")
    if precision not in PRECISIONS[target]:
        raise Refused(f"{target} exports can be {', '.join(PRECISIONS[target])}, not {precision!r}")
    spec = {"model": model_ref, "target": target, "precision": precision}
    if item.get("gguf") is not None:
        if target != "noulxp" or kind != kinds.DECIDER:
            raise Refused("gguf chooses the GGUF a Decider NoulXP package carries")
        if item["gguf"] not in PRECISIONS["gguf"]:
            raise Refused(f"gguf is one of: {', '.join(PRECISIONS['gguf'])}")
        spec["gguf"] = item["gguf"]
    if item.get("test_rows") is not None:
        rows = item["test_rows"]
        if target != "noulxp" or isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise Refused("test_rows is a whole number of test rows for a NoulXP package")
        spec["test_rows"] = rows
    if target == "noulxp":
        from . import noulxp_package

        missing = noulxp_package.missing_tooling(kind)
        if missing:
            raise Unavailable(noulxp_package.tooling_message(missing, kind))
    return spec


def check_decisions(kind, questions, rows):
    """Refuse a dataset a kind's trainer has nothing to learn from: no training or validation
    decision on the questions it answers (it picks its best epoch and calibrates on the
    validation split). Returns the warnings for the questions it leaves out."""
    name = kinds.NAME[kind]
    left_out = datasets.left_out(questions)[kind]
    reasons = [f"{qid}: {why}" for qid, why in left_out.items()]
    if len(left_out) == len(questions):
        raise Refused(f"{name} cannot train on any of these questions: {reasons[0]}")
    answered = ", ".join(qid for qid in questions if qid not in left_out)
    counts = datasets.decisions(rows, questions, kind)
    for split, label in (("train", "training"), ("val", "validation")):
        if not counts[split]:
            raise Refused(
                f"{name} has nothing to learn from: no {label} row answers {answered}"
                + (f" (it leaves out {'; '.join(reasons)})" if reasons else "")
            )
    return [f"{name} leaves out {reason}" for reason in reasons]


def run_id_for(spec, name, workspace, stamp=None):
    run_id = spec.get("run_id") or (
        f"{engine.slugify(name, 'run')[:40]}-{stamp or time.strftime('%m%d-%H%M%S')}"
    )
    try:
        engine.check_id(run_id)
    except ValueError as error:
        raise Refused(str(error)) from None
    # The train job is named after its run, here and in the studio.
    if (workspace / "runs" / run_id).exists() or (workspace / "jobs" / run_id).exists():
        raise Refused(f"A run named {run_id} exists already")
    return run_id


def prepare_run(spec, workspace=engine.WORKSPACE, stamp=None, base=None):
    """Check one fine-tune and write its record, runs/<id>/run.json. Nothing is written for a
    run that is refused.

    spec is what the Train button sends ({"dataset", "base_model", "name", "hyperparameters",
    "baseline"}) or a run's config (this module's docstring). base: the folder a config's
    paths are relative to; only a run from a file may bring its own dataset files.

    Returns {"run_id", "name", "kind", "job" (the train job's spec), "exports" (the export
    jobs' specs), "dataset" (its meta), "hyperparameters", "ignored", "warnings", "run"}.
    Raises Refused, saying why, for anything that would not train."""
    if not isinstance(spec, dict):
        raise Refused("A run is a JSON object")
    workspace = Path(workspace)
    questions, rows, report, meta = dataset(
        spec.get("dataset"), workspace, base, spec.get("limits")
    )
    ref = spec.get("base_model")
    _, kind = base_model(ref, workspace)
    hp, overrides, ignored = hyperparameters(kind, spec.get("hyperparameters"), spec.get("bounds"))
    warnings = check_decisions(kind, questions, rows)
    if ignored:
        warnings.append(
            f"Not hyperparameters of the {kinds.NAME[kind]} trainer: {', '.join(ignored)}"
        )
    if ref.startswith("path:"):
        from . import noulxp_package

        if not noulxp_package.base_model(ref, workspace):
            warnings.append(
                "The base model is a folder with no published name: its licence is not "
                "checked, and its packages name no base model (hub:<repo>@<revision> does both)"
            )

    name = spec.get("name")
    if name is not None and not isinstance(name, str):
        raise Refused("A run's name is text")
    name = (name or "").strip() or f"{meta['name'] if meta else report['name']} · {hp['method']}"
    run_id = run_id_for(spec, name, workspace, stamp)
    given = spec.get("exports") or []
    if not isinstance(given, list):
        raise Refused("exports is a list")
    exports = [check_export(kind, item, f"run:{run_id}") for item in given]
    for export in exports:
        if export["target"] in ("onnx", "coreml"):  # into the run's folder, as GGUF and MLX go
            suffix = "" if export["precision"] == "float" else f"-{export['precision']}"
            export["out_dir"] = f"runs/{run_id}/exports/{export['target']}{suffix}"

    if meta is None:  # checked above, saved only now that the run starts
        files = report.pop("files")
        meta = engine.save_dataset(
            report.pop("name"), questions, rows, report, files, workspace, reuse=True
        )
    job = {
        "run_id": run_id,
        "dataset": meta["id"],
        "base_model": ref,
        "hyperparameters": overrides,
        "baseline": bool(spec.get("baseline", True)),
    }
    record = {
        "id": run_id,
        "name": name,
        "dataset": meta["id"],
        "dataset_name": meta["name"],
        "base_model": ref,
        "kind": kind,
        "hyperparameters": hp,
        "created": engine.now(),
        "questions": list(questions),
    }
    if exports:
        record["exports"] = exports
    engine.write_json(workspace / "runs" / run_id / "run.json", record)
    return {
        "run_id": run_id,
        "name": name,
        "kind": kind,
        "job": job,
        "exports": exports,
        "dataset": meta,
        "hyperparameters": hp,
        "ignored": ignored,
        "warnings": warnings,
        "run": record,
    }


# ----------------------------------------------------------------------------- headless runs


def _sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def manifest(run_dir):
    """The run's output files, by kind: {"model": [{"path", "size", "sha256"}], ...}, with
    paths relative to the run's folder."""
    out = {}
    for group in OUTPUTS:
        folder = run_dir / group
        if not folder.is_dir():
            continue
        out[group] = [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(folder.rglob("*"))
            if path.is_file() and not path.name.endswith(".tmp")
        ]
    return out


class Printer:
    """Events as JSON lines on a stream, each written whole and flushed at once."""

    def __init__(self, stream):
        self.stream = stream

    def line(self, event):
        self.stream.write(json.dumps(engine.finite(event), ensure_ascii=False, default=str) + "\n")
        self.stream.flush()

    def __call__(self, type_, /, **data):  # positional: events carry a "kind" field too
        self.line({"t": round(time.time(), 3), "type": type_, **data})


def _free_job_id(workspace, job_id):
    job_id = job_id[:81].rstrip("-")
    found, n = job_id, 1
    while (workspace / "jobs" / found).exists():
        n += 1
        found = f"{job_id[:76].rstrip('-')}-{n}"
    return found


def _tail(path, lines=50):
    try:
        return path.read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return []


class Headless:
    """One prepared run's jobs, one after another, each a child process whose events are
    forwarded as they are written. SIGTERM or Ctrl+C cancels the job that is running (SIGTERM,
    then SIGKILL after 30 s) and starts no other. A child is never left running: whatever
    ends the following of its events, it is stopped and waited for."""

    KILL_AFTER = 30

    def __init__(self, workspace, emit):
        self.workspace = workspace
        self.emit = emit
        self.process = None
        self.cancelled = False

    @property
    def env(self):
        # The jobs' workspace is the run's, for the files they keep beside it too.
        return {"LAYASTUDIO_HOME": str(self.workspace)}

    def cancel(self, *_):
        self.cancelled = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def stage(self, stage, kind, spec, job_id, title):
        """Run one job. {"stage", "job", "state", "seconds", "result", "error"}"""
        job_id = _free_job_id(self.workspace, job_id)
        path = engine.write_job(self.workspace, job_id, kind, spec, title)
        self.emit("stage", stage=stage, job=job_id, state="running")
        started = time.monotonic()
        result = error = None
        with open(path / "output.log", "w") as log:
            self.process = engine.start_job(path, kind, log, self.env)
            try:
                if self.cancelled:
                    self.process.terminate()
                for event in self._follow(path / "events.jsonl"):
                    self.emit.line({**event, "stage": stage})
                    if event.get("type") == "result":
                        result = {k: v for k, v in event.items() if k not in ("t", "type")}
                    elif event.get("type") == "error":
                        error = event
            finally:
                self._stop()
        code, self.process = self.process.returncode, None
        state = "done" if code == 0 else "cancelled" if code == 143 or self.cancelled else "failed"
        outcome = {
            "stage": stage,
            "job": job_id,
            "state": state,
            "seconds": round(time.monotonic() - started, 1),
            "result": result,
        }
        if state == "failed":
            error = error or {}
            outcome["error"] = {
                "message": error.get("message") or f"The job ended with exit code {code}",
                "traceback": (error.get("traceback") or "").splitlines()[-50:],
                "log_tail": _tail(path / "output.log"),
            }
        self.emit("stage", **{k: v for k, v in outcome.items() if k != "result"})
        return outcome

    def _stop(self):
        """The child, ended: at once when it has exited, else SIGTERM, then SIGKILL."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(self.KILL_AFTER)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process.wait()

    def _follow(self, events_path):
        """The child's events as they are written, until it exits. Read as bytes and decoded a
        whole line at a time: a read can end inside a character the child is still writing."""
        position, pending, stop_at = 0, b"", None
        while True:
            done = self.process.poll() is not None
            if events_path.exists():
                with open(events_path, "rb") as handle:
                    handle.seek(position)
                    pending += handle.read()
                    position = handle.tell()
                *complete, pending = pending.split(b"\n")
                for raw in complete:
                    line = raw.decode("utf-8", errors="replace")
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            yield {"t": round(time.time(), 3), "type": "log", "message": line}
            if done:
                return
            if self.cancelled:
                stop_at = stop_at or time.monotonic() + self.KILL_AFTER
                if time.monotonic() > stop_at:
                    self.process.kill()
            time.sleep(0.2)


def _summaries(run_dir):
    evaluation = engine.read_json(run_dir / "eval.json")
    return {
        "training": engine.read_json(run_dir / "training.json"),
        "comparison": engine.read_json(run_dir / "comparison.json"),
        "eval": {k: v for k, v in evaluation.items() if k != "records"} if evaluation else None,
    }


def _measured(comparison):
    """The before and after measurements, without the models' references (a base given as a
    folder is a path on this machine)."""
    if not comparison:
        return None
    return {
        "base": {k: v for k, v in comparison["base"].items() if k != "model"},
        "finetuned": {k: v for k, v in comparison["finetuned"].items() if k != "model"},
        "paired": comparison.get("paired"),
    }


def write_card(run_id, workspace, dataset=None, template=None):
    """runs/<id>/card from the run's own files: README.md, the model card, and finetune.json,
    the fine-tune's record (this module's docstring). dataset: its meta, of which only the
    digest and the counts are kept. Returns the record."""
    from . import kinds, noulxp_package
    from .families import find
    from .publish import build_card
    from .publish_systemone import registry_card, with_noulxp

    run_dir = workspace / "runs" / engine.check_id(run_id)
    model_dir = run_dir / "model"
    run = engine.read_json(run_dir / "run.json") or {}
    training = engine.read_json(run_dir / "training.json") or {}
    comparison = engine.read_json(run_dir / "comparison.json")
    questions = engine.read_json(model_dir / "questions.json") or {}
    kind = run.get("kind") or kinds.detect(model_dir) or kinds.LAYA
    repo, revision = noulxp_package.base_reference(run.get("base_model"), workspace)
    known = find(repo) if repo else None
    # The package this run's export just built and checked (describe: only one that passed),
    # not hashed again: the manifest hashes every file once.
    package = noulxp_package.describe(run_dir / noulxp_package.PACKAGE)

    named = {**run, "base_model": f"hub:{repo or 'unknown'}"}  # a published name, never a path
    card = build_card(named, training, comparison, CARD_REPO, questions, kind)
    card = with_noulxp(registry_card(card, CARD_REPO, kind), noulxp_package.card_line(package))
    hp = training.get("hyperparameters") or run.get("hyperparameters") or {}
    dataset = dataset or {}
    record = {
        "tool": "System One Studio",
        "run": run_id,
        "name": run.get("name"),
        "template": template,
        "kind": kind,
        "base_model": {
            "repo": repo,
            "revision": revision,
            "license": known.licence if known else None,
        },
        "method": hp.get("method"),
        "hyperparameters": hp,
        "dataset": {
            "sha256": training.get("dataset_sha256") or dataset.get("sha256"),
            "seed": dataset.get("seed"),
            "rows": dataset.get("rows"),
            "decisions": dataset.get("decisions"),
            "questions": list(questions),
        },
        "training": {k: training[k] for k in TRAINING_KEYS if k in training},
        "calibration": training.get("calibration"),
        "metrics": _measured(comparison),
        "noulxp": {k: package.get(k) for k in NOULXP_KEYS} if package else None,
        "created": engine.now(),
    }
    folder = run_dir / CARD
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "README.md").write_text(card, encoding="utf-8")
    engine.write_json(folder / "finetune.json", record)
    return record


def run(config, workspace=None, result=None, stream=None):
    """Run the fine-tune a config file describes, end to end, and return its exit code (this
    module's docstring). The result file is written whatever happens, and the last event,
    "finished", says how the run ended: its state, exit code and result file, and the error
    (its stage and message) when it did not succeed. SIGTERM and Ctrl+C cancel it from the
    start: during the checks they stop it before anything trains."""
    emit = Printer(stream or sys.stdout)
    config = Path(config).resolve()
    result = Path(result).resolve() if result else config.parent / "result.json"
    started = time.time()
    outcome = {"state": "failed", "config": str(config), "started": engine.now()}

    def finish(state, code, **fields):
        outcome.update(fields, state=state, exit_code=code, finished=engine.now())
        outcome["seconds"] = round(time.time() - started, 1)
        engine.write_json(result, outcome)
        why = {k: outcome[k] for k in ("error", "failed_exports") if k in outcome}
        emit("finished", state=state, exit_code=code, result=str(result), **why)
        return code

    runner = Headless(None, emit)
    previous = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, runner.cancel)
    except ValueError:  # not the main thread: the caller handles signals
        pass
    try:
        return _run(config, workspace, runner, emit, outcome, finish)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _run(config, workspace, runner, emit, outcome, finish):
    def refuse(message):
        emit("refused", message=message)
        return finish("refused", 2, error={"stage": "prepare", "message": message})

    try:
        spec = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return refuse(f"Cannot read the config: {error}")
    if not isinstance(spec, dict):
        return refuse("The config is a JSON object")
    where = workspace or spec.get("workspace")
    if where is not None and not isinstance(where, (str, Path)):
        return refuse("workspace is a folder's path")
    workspace = (config.parent / where).resolve() if where else engine.WORKSPACE
    outcome["workspace"] = str(workspace)
    runner.workspace = workspace
    keep = spec.get("keep_checkpoint", False)
    if not isinstance(keep, bool):
        return refuse("keep_checkpoint is true or false: whether the run keeps its checkpoint")
    template = spec.get("template")
    if template is not None and not isinstance(template, str):
        return refuse("A run's template is text: the platform's name for it")
    ref = spec.get("base_model")
    if isinstance(ref, str) and ref.startswith("path:"):  # a folder beside the config, too
        spec["base_model"] = f"path:{(config.parent / Path(ref[5:]).expanduser()).resolve()}"
    try:
        prepared = prepare_run(spec, workspace, base=config.parent)
    except Refused as error:
        return refuse(str(error))
    except KeyboardInterrupt:
        return finish("cancelled", 143, error={"stage": "prepare", "message": "Cancelled"})
    except Exception as error:  # noqa: BLE001 - the result file says what went wrong
        message = f"{type(error).__name__}: {error}"
        emit("error", message=message)
        return finish("failed", 1, error={"stage": "prepare", "message": message})

    run_id = prepared["run_id"]
    run_dir = workspace / "runs" / run_id
    if runner.cancelled:  # during the checks: nothing ran, and the run's record goes
        shutil.rmtree(run_dir, ignore_errors=True)
        return finish("cancelled", 143, error={"stage": "prepare", "message": "Cancelled"})
    meta = prepared["dataset"]
    if not keep and not any(e["target"] == "noulxp" for e in prepared["exports"]):
        prepared["warnings"].append(
            "This run keeps no checkpoint (keep_checkpoint) and makes no NoulXP package: "
            "its card is all it keeps"
        )
    outcome.update(
        run_id=run_id,
        name=prepared["name"],
        kind=prepared["kind"],
        base_model=prepared["job"]["base_model"],
        run_dir=str(run_dir),
        dataset={
            k: meta.get(k) for k in ("id", "name", "sha256", "seed", "rows", "decisions", "kinds")
        },
        hyperparameters=prepared["hyperparameters"],
        ignored=prepared["ignored"],
        warnings=prepared["warnings"],
        keep_checkpoint=keep,
        template=template,
        stages=[],
        exports=[],
    )
    emit(
        "prepared",
        **{k: outcome[k] for k in ("run_id", "name", "kind", "base_model", "dataset")},
        hyperparameters=prepared["hyperparameters"],
        exports=prepared["exports"],
        keep_checkpoint=keep,
        warnings=prepared["warnings"],
    )

    try:
        train = runner.stage(
            "train", "train", prepared["job"], run_id, f"Fine-tune: {prepared['name']}"
        )
        outcome["stages"].append(train)
        outcome.update(_summaries(run_dir))
        if train["state"] != "done":
            error = {"stage": "train", **train.get("error", {"message": "Cancelled"})}
            return finish(train["state"], 143 if train["state"] == "cancelled" else 1, error=error)
        for export in prepared["exports"]:
            target = export["target"]
            if runner.cancelled:
                outcome["exports"].append({**export, "state": "cancelled"})
                continue
            done = runner.stage(
                f"export:{target}",
                "export",
                export,
                f"export-{target}-{run_id}",
                f"Export {run_id}",
            )
            outcome["stages"].append(done)
            outcome["exports"].append(
                {**export, "state": done["state"], "result": done["result"]}
                | ({"error": done["error"]} if "error" in done else {})
            )
        if not runner.cancelled:
            _card(run_id, workspace, meta, template, emit, outcome)
        outcome["outputs"] = manifest(run_dir)
        if runner.cancelled:
            return finish("cancelled", 143, error={"stage": "export", "message": "Cancelled"})
        failed = [e for e in outcome["exports"] if e["state"] == "failed"]
        if failed:  # trained, and the other outputs are there: the caller decides what to keep
            first = failed[0]
            return finish(
                "partial",
                3,
                failed_exports=[e["target"] for e in failed],
                error={"stage": f"export:{first['target']}", "message": first["error"]["message"]},
            )
        return finish("succeeded", 0)
    except Exception as error:  # noqa: BLE001 - the result file says what went wrong
        return finish(
            "failed", 1, error={"stage": "run", "message": f"{type(error).__name__}: {error}"}
        )


def _card(run_id, workspace, meta, template, emit, outcome):
    """The run's card, written once it trained. A card that cannot be written is a warning:
    the training and the exports it would describe are there, and publishing can do without."""
    emit("phase", phase="card", message="Writing the model card from the run's measurements")
    try:
        write_card(run_id, workspace, meta, template)
    except Exception as error:  # noqa: BLE001 - the run's outputs stand without a card
        message = f"No model card: {type(error).__name__}: {error}"
        outcome["warnings"].append(message)
        emit("log", message=message)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="layastudio train",
        description="Run one fine-tune with no UI: training, then its exports. Events go to "
        "stdout as JSON lines; the result file says how it ended.",
    )
    parser.add_argument("--config", required=True, type=Path, help="The run, as a JSON file")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Datasets, runs and checkpoints (default: the config's, else the studio's)",
    )
    parser.add_argument(
        "--result", type=Path, help="Where the result goes (default: result.json beside the config)"
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    workspace = args.workspace.expanduser().resolve() if args.workspace else None
    return run(args.config, workspace, args.result)


if __name__ == "__main__":
    sys.exit(main())
