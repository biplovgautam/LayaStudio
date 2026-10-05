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
  the run can be told to expect;
- the exports are ones the kind has, at a precision it takes, with the tooling installed.

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
        "expected": {"rows": {"train": 1079, "val": 121, "test": 600}, "sha256": "..."}
      },
      "hyperparameters": {"epochs": 2},               over the kind's defaults
      "bounds": {"epochs": {"min": 1, "max": 5}},     optional limits on the final values
      "limits": {"max_bytes": 26214400, "max_rows": 200000},
      "exports": [{"target": "noulxp", "gguf": "bf16"}],
      "run_id": "...", "baseline": true, "workspace": "..."
    }
"""

import argparse
import hashlib
import json
import math
import os
import signal
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
# The precision an export gets when the run does not say (as `python -m layastudio.export`).
DEFAULT_PRECISION = {"gguf": "bf16", "mlx": "int4"}
# Outputs of a run, in the result's manifest: the checkpoint, its NoulXP package (only one that
# passed its check), and the exports (GGUF, MLX-LM).
OUTPUTS = ("model", "noulxp", "exports")


class Refused(ValueError):
    """A run that does not start, and why. The server answers 400 with the message."""


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
    if not isinstance(limits, dict):
        raise Refused("limits is {max_bytes, max_rows}")
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
    if not isinstance(expected, dict):
        raise Refused("expected is {rows, sha256}: what the checked dataset was")
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
    then SIGKILL after 30 s) and starts no other."""

    KILL_AFTER = 30

    def __init__(self, workspace, emit):
        self.workspace = workspace
        self.emit = emit
        # The jobs' workspace is the run's, for the files they keep beside it too.
        self.env = {"LAYASTUDIO_HOME": str(workspace)}
        self.process = None
        self.cancelled = False

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
            if self.cancelled:
                self.process.terminate()
            for event in self._follow(path / "events.jsonl"):
                self.emit.line({**event, "stage": stage})
                if event.get("type") == "result":
                    result = {k: v for k, v in event.items() if k not in ("t", "type")}
                elif event.get("type") == "error":
                    error = event
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

    def _follow(self, events_path):
        """The child's events as they are written, until it exits."""
        position, pending, stop_at = 0, "", None
        while True:
            done = self.process.poll() is not None
            if events_path.exists():
                with open(events_path, encoding="utf-8") as handle:
                    handle.seek(position)
                    pending += handle.read()
                    position = handle.tell()
                *complete, pending = pending.split("\n")
                for line in complete:
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


def run(config, workspace=None, result=None, stream=None):
    """Run the fine-tune a config file describes, end to end. Returns the exit code: 0 when it
    trained (each export's own state is in the result), 1 when it failed, 2 when it was
    refused, 143 when it was cancelled. The result file is written whatever happens."""
    emit = Printer(stream or sys.stdout)
    config = Path(config).resolve()
    result = Path(result).resolve() if result else config.parent / "result.json"
    started = time.time()
    outcome = {"state": "failed", "config": str(config), "started": engine.now()}

    def finish(state, code, **fields):
        outcome.update(fields, state=state, finished=engine.now())
        outcome["seconds"] = round(time.time() - started, 1)
        engine.write_json(result, outcome)
        emit("finished", state=state, result=str(result))
        return code

    try:
        spec = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        emit("refused", message=f"Cannot read the config: {error}")
        return finish("refused", 2, error={"stage": "prepare", "message": str(error)})
    if not isinstance(spec, dict):
        emit("refused", message="The config is a JSON object")
        return finish("refused", 2, error={"stage": "prepare", "message": "not a JSON object"})
    where = workspace or spec.get("workspace")
    workspace = (config.parent / where).resolve() if where else engine.WORKSPACE
    outcome["workspace"] = str(workspace)
    ref = spec.get("base_model")
    if isinstance(ref, str) and ref.startswith("path:"):  # a folder beside the config, too
        spec["base_model"] = f"path:{(config.parent / Path(ref[5:]).expanduser()).resolve()}"
    try:
        prepared = prepare_run(spec, workspace, base=config.parent)
    except Refused as error:
        emit("refused", message=str(error))
        return finish("refused", 2, error={"stage": "prepare", "message": str(error)})
    except Exception as error:  # noqa: BLE001 - the result file says what went wrong
        message = f"{type(error).__name__}: {error}"
        emit("error", message=message)
        return finish("failed", 1, error={"stage": "prepare", "message": message})

    run_id = prepared["run_id"]
    run_dir = workspace / "runs" / run_id
    meta = prepared["dataset"]
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
        stages=[],
        exports=[],
    )
    emit(
        "prepared",
        **{k: outcome[k] for k in ("run_id", "name", "kind", "base_model", "dataset")},
        hyperparameters=prepared["hyperparameters"],
        exports=prepared["exports"],
        warnings=prepared["warnings"],
    )

    runner = Headless(workspace, emit)
    previous = {sig: signal.signal(sig, runner.cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
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
        outcome["outputs"] = manifest(run_dir)
        if runner.cancelled:
            return finish("cancelled", 143, error={"stage": "export", "message": "Cancelled"})
        return finish("succeeded", 0)
    except Exception as error:  # noqa: BLE001 - the result file says what went wrong
        return finish(
            "failed", 1, error={"stage": "run", "message": f"{type(error).__name__}: {error}"}
        )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


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
