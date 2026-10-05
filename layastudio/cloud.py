"""Fine-tunes without the UI: the checks every run starts with, here and in the cloud.

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
  the run can be told to expect;
- the exports are ones the kind has, at a precision it takes, with the tooling installed.

Then it writes the run's record, runs/<id>/run.json, and returns the jobs to start.

A run's config, with paths relative to its own folder:

    {
      "name": "emotion",                              optional
      "base_model": "hub:Mapika/decider-2b",          hub:<repo> (downloaded), path:<dir>, run:<id>
      "dataset": {                                    or the id of a dataset in the workspace
        "questions": "questions.json",                a file, or the questions themselves
        "train": "train.jsonl", "test": "test.jsonl", test optional; JSONL, JSON, CSV or TSV
        "seed": 13,                                   the split's seed
        "expected": {"rows": {"train": 1079, "val": 121, "test": 600}, "sha256": "..."}
      },
      "hyperparameters": {"epochs": 2},               over the kind's defaults
      "bounds": {"epochs": {"min": 1, "max": 5}},     optional limits on the final values
      "limits": {"max_bytes": 26214400, "max_rows": 200000},
      "exports": [{"target": "noulxp", "gguf": "bf16"}],
      "run_id": "...", "baseline": true, "workspace": "..."
    }
"""

import json
import math
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
# The precision an export gets when the run does not say (as `python -m layastudio.export`).
DEFAULT_PRECISION = {"gguf": "bf16", "mlx": "int4"}


class Refused(ValueError):
    """A run that does not start, and why. The server answers 400 with the message."""


# ----------------------------------------------------------------------------- the checks


def check_licence(ref, workspace):
    """Refuse a base model whose licence does not allow derivatives (families.py)."""
    from . import families, noulxp_package

    repo = (
        ref.split(":", 1)[1]
        if ref.startswith("hub:")
        else noulxp_package.base_model(ref, workspace)
    )
    known = families.find(repo or "")
    if known and not families.trainer_status(known)["ready"]:
        raise Refused(families.trainer_status(known)["reason"])


def base_model(ref, workspace):
    """(checkpoint folder, kind) of a base model the studio may fine-tune, or Refused."""
    if not isinstance(ref, str) or not ref:
        raise Refused("Choose a base model: hub:<repo>, path:<folder> or run:<id>")
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
        return value
    allowed = CHOICES.get(key)
    if not isinstance(value, str) or (allowed and value not in allowed):
        raise Refused(f"{key} is one of: {', '.join(allowed)}" if allowed else f"{key} is text")
    return value


def check_bounds(hp, bounds):
    """The platform's limits for a template, on the run's final values:
    {key: {"min": a, "max": b} or {"choices": [...]}}."""
    if bounds is None:
        return
    if not isinstance(bounds, dict):
        raise Refused("bounds must be an object of {key: {min, max} or {choices}}")
    for key, bound in bounds.items():
        if key not in hp or not isinstance(bound, dict):
            raise Refused(f"bounds for {key!r}: not a hyperparameter of this trainer")
        value = hp[key]
        if "choices" in bound and value not in bound["choices"]:
            shown = ", ".join(str(c) for c in bound["choices"])
            raise Refused(f"{key} must be one of: {shown}")
        if "min" in bound and value < bound["min"]:
            raise Refused(f"{key} must be at least {bound['min']}")
        if "max" in bound and value > bound["max"]:
            raise Refused(f"{key} must be at most {bound['max']}")


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
    check_bounds(merged, bounds)
    return merged, overrides, ignored


def _read(path, base, max_bytes=None):
    path = Path(path)
    path = path if path.is_absolute() else Path(base) / path
    try:
        size = path.stat().st_size
        if max_bytes is not None and size > max_bytes:
            raise Refused(
                f"{path.name} is {size / 2**20:.1f} MB; the limit is {max_bytes / 2**20:g} MB"
            )
        return path.name, path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise Refused(f"Cannot read {path.name}: {error}") from None


def dataset(given, workspace, base=None, limits=None):
    """(questions, rows, report, meta) of the run's dataset: one the workspace has (its id), or,
    when the run comes from a file (base: that file's folder), files to read, check and split
    now: {"questions", "train", "test", "name", "seed", "expected"}. rows and report are None
    for a saved dataset; meta is None until a checked one is saved (save_dataset)."""
    if isinstance(given, str):
        try:
            questions, _, meta = engine.load_dataset(given, workspace)
        except (FileNotFoundError, ValueError) as error:
            raise Refused(str(error)) from None
        return questions, None, None, meta
    if base is None or not isinstance(given, dict):
        raise Refused("Choose a dataset: the id of one in this workspace")
    limits = limits or {}
    max_bytes, max_rows = limits.get("max_bytes"), limits.get("max_rows")
    if "train" not in given or "questions" not in given:
        raise Refused("A dataset names its questions and its train file")
    questions = given["questions"]
    if isinstance(questions, str):
        try:
            questions = json.loads(_read(questions, base)[1])
        except json.JSONDecodeError as error:
            raise Refused(f"The questions are not JSON: {error}") from None
    train_name, train_text = _read(given["train"], base, max_bytes)
    test_name, test_text = (
        _read(given["test"], base, max_bytes) if given.get("test") else (None, None)
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
    for key in ("rows", "sha256"):
        if key in expected and expected[key] != report[key]:
            raise Refused(
                f"The dataset does not match the one that was checked: {key} is "
                f"{report[key]}, expected {expected[key]}"
            )
    report["files"] = {"train": train_name, "test": test_name}
    report["name"] = str(given.get("name") or Path(train_name).stem)
    return questions, rows, report, None


def check_export(kind, item, run_id):
    """One export the run asks for, as its job's spec: {"model", "target", "precision", ...}."""
    from .export import KIND_TARGETS, PRECISIONS

    if isinstance(item, str):
        item = {"target": item}
    if not isinstance(item, dict):
        raise Refused('An export is {"target": ...}')
    target = item.get("target")
    allowed = KIND_TARGETS[kind]
    if target not in allowed:
        raise Refused(
            f"A {kinds.NAME[kind]} fine-tune exports to {', '.join(allowed)}, not {target!r}"
        )
    precision = item.get("precision") or DEFAULT_PRECISION.get(target, "float")
    if precision not in PRECISIONS[target]:
        raise Refused(f"{target} exports can be {', '.join(PRECISIONS[target])}, not {precision!r}")
    spec = {"model": f"run:{run_id}", "target": target, "precision": precision}
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
            raise Refused(noulxp_package.tooling_message(missing, kind))
    return spec


def run_id_for(spec, name, workspace, stamp=None):
    run_id = spec.get("run_id") or (
        f"{engine.slugify(name, 'run')[:40]}-{stamp or time.strftime('%m%d-%H%M%S')}"
    )
    try:
        engine.check_id(run_id)
    except ValueError as error:
        raise Refused(str(error)) from None
    if (workspace / "runs" / run_id).exists():
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

    left_out = datasets.fits(questions)[kind]
    if len(left_out) == len(questions):
        raise Refused(f"{kinds.NAME[kind]} cannot train on any of these questions: {left_out[0]}")
    warnings = [f"{kinds.NAME[kind]} leaves out {reason}" for reason in left_out]
    if ignored:
        warnings.append(
            f"Not hyperparameters of the {kinds.NAME[kind]} trainer: {', '.join(ignored)}"
        )

    name = spec.get("name")
    if name is not None and not isinstance(name, str):
        raise Refused("A run's name is text")
    name = (name or "").strip() or f"{meta['name'] if meta else report['name']} · {hp['method']}"
    run_id = run_id_for(spec, name, workspace, stamp)
    given = spec.get("exports") or []
    if not isinstance(given, list):
        raise Refused("exports is a list")
    exports = [check_export(kind, item, run_id) for item in given]

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
