"""NoulXP packages for fine-tunes: built, checked, kept with the run, published with it.

NoulXP (https://github.com/systemonemodels/noulxp) is the open standard that lets any engine
run a System One model from a package of files, with no code written for that model. When a
model version on systemonemodels.tech carries a package in a `noulxp/` folder, the registry's
engine checks it against the package's own conformance file, and a pass shows the model as
NoulXP compatible.

    python -m layastudio.export run:<id> --target noulxp

builds that package for one fine-tune with noulxp 0.4, the release the registry checks with,
in the way the standard asks (SPEC.md, section 9):

1. `noulxp export laya` writes it from the run's checkpoint: an ONNX graph over the
   checkpoint's own weights file, the tokenizer, template.json and calibration.json.
2. `noulxp conformance generate` records the fine-tune's own answers, from the `laya`
   package in float32 on the CPU, to NoulXP's request set (52 requests in 11 languages, the
   coverage the standard asks for) and to up to 100 rows of the run's test split, asked the
   questions the run was tuned for. Those rows are published inside the package.
3. `noulxp validate`, then `noulxp check` on the CPU: the reference runtime has to reproduce
   every case (each probability within 0.01, and the same leading option).

Each step runs in a process of its own, so memory goes back between them. The package is built
in runs/<id>/noulxp.partial/noulxp and becomes runs/<id>/noulxp only once it passes, with the
check's report inside it (check-cpu.json). A package that fails is kept as
runs/<id>/noulxp-failed for reading and is never published. runs/<id>/noulxp-report.json
describes the last attempt.

Publishing (publish_systemone.py) places the passing package next to the checkpoint's own
files, as the version's `noulxp/` folder, for as long as the upload takes.

Needs the optional extra:  uv sync --extra export
"""

import collections
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

from .engine import WORKSPACE, check_id, load_dataset, now, read_json, resolve_model_ref, write_json

REQUIREMENT = "noulxp[export,laya,onnx]>=0.4,<0.5"
RELEASE = (0, 4)  # the noulxp release line systemonemodels.tech checks packages with
TEST_ROWS = 100  # rows of the run's test split recorded next to NoulXP's request set

PACKAGE = "noulxp"  # runs/<id>/noulxp: a package that passed its check, and nothing else
BUILDING = "noulxp.partial"
FAILED = "noulxp-failed"
REPORT = "noulxp-report.json"
MANIFEST = "noulxp.json"
CHECK = "check-cpu.json"
TEST_TAG = "studio:test-split"

# What `noulxp export laya`, the `laya` runtime and `noulxp check` import, besides noulxp.
NEEDED = (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("laya", "laya"),
    ("onnx", "onnx"),
    ("onnxscript", "onnxscript"),
    ("onnx_ir", "onnx-ir"),
    ("onnxruntime", "onnxruntime"),
)
# noulxp refuses to export an encoder with an older transformers: older releases compute
# ModernBERT differently, and the package would not give the model's own answers.
MIN_TRANSFORMERS = (5, 2)
# The file entries a manifest names (SPEC.md 4.2); weights also list their external data.
FILE_KEYS = ("weights", "tokenizer", "template", "prompt", "calibration", "conformance")

# `noulxp export laya`, called as its command calls it (hard-linked weights, opset 18), with
# the fine-tune's own provenance in the manifest's `source` instead of upstream Laya's.
EXPORT = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "from noulxp.export import laya\n"
    "laya.export(Path(sys.argv[1]), Path(sys.argv[2]), name=sys.argv[3],"
    " source=json.loads(sys.argv[4]))\n"
)
CASES = re.compile(r"^\s*(\d+)/(\d+) cases\b")
GLOG = re.compile(r"^[WIEF]\d{4} ")  # torch's own warnings, e.g. "W1004 15:17:08 ..."


# ----------------------------------------------------------------------------- tooling


def _release(version, parts=2):
    numbers = [int(n) for n in re.findall(r"\d+", str(version))[:parts]]
    return tuple(numbers + [0] * (parts - len(numbers)))


def _importable(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing_tooling():
    """Why this environment cannot build a NoulXP package, or None when it can."""
    try:
        version = importlib.metadata.version("noulxp")
    except importlib.metadata.PackageNotFoundError:
        return "noulxp is not installed"
    if _release(version) != RELEASE:
        return (
            f"noulxp {version} is installed, and the studio builds packages with noulxp 0.4, "
            "the release systemonemodels.tech checks them with"
        )
    for module, name in NEEDED:
        if not _importable(module):
            return f"{name} is not installed"
    try:
        found = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return "transformers is not installed"
    if _release(found) < MIN_TRANSFORMERS:
        return f"transformers {found} is installed; noulxp exports Laya with 5.2 or later"
    return None


def tooling_message(missing):
    return (
        f"{missing}. Building a NoulXP package needs the optional extra:\n"
        "    uv sync --extra export\n"
        f'or, outside a checkout:  pip install "{REQUIREMENT}"'
    )


# ----------------------------------------------------------------------------- the run


def locate(model_ref, workspace=WORKSPACE):
    """(run id, run folder, checkpoint folder, run.json) of a `run:<id>` reference."""
    kind, _, value = str(model_ref).partition(":")
    if kind != "run":
        raise ValueError(
            "NoulXP packages are built for fine-tuned runs (run:<id>): the package is kept with "
            "its run and published with it. For a base Laya checkpoint, run `noulxp export laya`."
        )
    run_id = check_id(value)
    run_dir = workspace / "runs" / run_id
    model_dir = resolve_model_ref(f"run:{run_id}", workspace)
    run = read_json(run_dir / "run.json") or {"id": run_id, "base_model": ""}
    return run_id, run_dir, model_dir, run


def base_model(ref, workspace=WORKSPACE, depth=0):
    """The published name of the model a run started from; never a path on this machine."""
    kind, _, value = str(ref or "").partition(":")
    if kind == "hub":
        return value
    if kind == "path":
        from .families import imports

        return next((e.get("repo") for e in imports(workspace) if e.get("ref") == ref), None)
    if kind == "run" and depth < 4:
        try:
            parent = read_json(workspace / "runs" / check_id(value) / "run.json") or {}
        except ValueError:
            return None
        return base_model(parent.get("base_model"), workspace, depth + 1)
    return None


def support(run, workspace=WORKSPACE):
    """(family, the family's NoulXP entry) for a run, from the model it was tuned from."""
    from .families import NOULXP, find

    base = base_model(run.get("base_model"), workspace)
    known = find(base) if base else None
    # Only the laya trainer exists, so a run's checkpoint is a Laya checkpoint unless the
    # catalogue knows its base belongs to another family.
    family = known.family if known else "laya"
    return family, NOULXP[family]


def provenance(run, run_id, workspace=WORKSPACE):
    """The manifest's `source`: what this package is a conversion of. No local paths."""
    from .families import find

    base = base_model(run.get("base_model"), workspace)
    known = find(base) if base else None
    training = read_json(workspace / "runs" / run_id / "training.json") or {}
    return {
        "model": base,
        "license": known.licence if known else None,
        "fine_tune": {
            "tool": "System One Studio",
            "run": run_id,
            "method": (run.get("hyperparameters") or {}).get("method"),
            "dataset_sha256": training.get("dataset_sha256"),
            "trained_at": training.get("created"),
        },
    }


def own_requests(questions, rows, limits, count=TEST_ROWS):
    """Requests from the run's test split, each asking the questions the run was tuned for.

    A question the package cannot be asked (not a valid System One question, or over its
    option or level limits) is left out, with the reason: the reference runtime would refuse
    it where the model's own code answers, and the case could never pass.
    """
    from noulxp.errors import RequestError
    from noulxp.request import parse_question

    asked, left_out = {}, {}
    for qid, qdef in questions.items():
        try:
            question = parse_question(qid, qdef)
        except RequestError as error:
            left_out[qid] = str(error)
            continue
        options = len(question.options)
        most, levels = limits.get("max_options"), limits.get("max_levels")
        if most and options > most:
            left_out[qid] = f"{options} options; NoulXP packages answer up to {most}"
        elif levels and question.type == "score" and options > levels:
            left_out[qid] = f"{options} levels; NoulXP packages answer up to {levels}"
        else:
            asked[qid] = {
                key: qdef[key] for key in ("type", "instructions", "criteria") if key in qdef
            }
    requests = []
    if asked and count > 0:
        test = [row for row in rows if row.get("split") == "test"][:count]
        for i, row in enumerate(test, 1):
            state = row["state"]
            if not isinstance(state, str):  # as laya.common.serialize_state renders it
                state = json.dumps(state, ensure_ascii=False)
            requests.append(
                {
                    "id": f"test-{i:03d}",
                    "tags": [TEST_TAG, f"row:{row.get('id', i)}"],
                    "request": {"state": state, "questions": asked},
                }
            )
    return requests, {"rows": len(requests), "asked": list(asked), "left_out": left_out}


def checkpoint_view(model_dir, dest):
    """A copy of the checkpoint for the `laya` package to read.

    laya rewrites tokenizer/tokenizer_config.json in place when it names no tokenizer class;
    the copy keeps that away from the run's own files. The weights are hard-linked: laya
    only reads them."""
    dest.mkdir(parents=True)
    shutil.copyfile(model_dir / "rl_agent_config.json", dest / "rl_agent_config.json")
    for folder in ("encoder", "tokenizer"):
        shutil.copytree(model_dir / folder, dest / folder)
    _place(model_dir / "model.safetensors", dest / "model.safetensors")
    return dest


def _place(source, target):
    """A hard link where the file system allows one, else a copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(os.path.realpath(source), target)
    except OSError:
        shutil.copyfile(source, target)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- the steps


def _python(args, emit, progress=False):
    """One step in a process of its own, its output into the job's log.

    Returns (exit code, the last lines it printed). Cancelling the job stops the step."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        [sys.executable, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
    )
    tail = collections.deque(maxlen=20)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail.append(line)
            counted = CASES.match(line) if progress else None
            if counted:
                emit("progress", done=int(counted[1]), total=int(counted[2]))
            else:
                emit("log", message=line)
        code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    return code, list(tail)


def _last(tail):
    useful = [line for line in tail if not GLOG.match(line) and "warnings.warn" not in line]
    return (useful or tail or ["no output"])[-1]


def export_package(model_dir, out_dir, name, source, emit):
    """Step 1: `noulxp export laya` from the run's checkpoint."""
    code, tail = _python(
        ["-c", EXPORT, str(model_dir), str(out_dir), name, json.dumps(source)], emit
    )
    if code:
        raise RuntimeError(f"noulxp export laya failed: {_last(tail)}")


def record_conformance(package_dir, checkpoint, requests, emit):
    """Step 2: the fine-tune's own answers (laya, float32, CPU) into conformance.jsonl."""
    code, tail = _python(
        [
            "-m",
            "noulxp",
            "conformance",
            "generate",
            str(package_dir),
            "--native",
            str(checkpoint),
            "--runtime",
            "laya",
            "--requests",
            str(requests),
        ],
        emit,
        progress=True,
    )
    if code:
        raise RuntimeError(f"noulxp conformance generate failed: {_last(tail)}")


def validate_package(package_dir, emit):
    """Step 3a: schemas, hashes and coverage, without running anything. Returns the problems."""
    code, tail = _python(["-m", "noulxp", "validate", str(package_dir)], emit)
    if code == 0:
        return []
    return [line for line in tail if not line.endswith("problem(s)")] or [_last(tail)]


def check_package(package_dir, emit):
    """Step 3b: the conformance file through the reference runtime on the CPU. The report."""
    report_path = package_dir / CHECK
    code, tail = _python(
        [
            "-m",
            "noulxp",
            "check",
            str(package_dir),
            "--device",
            "cpu",
            "--report",
            str(report_path),
        ],
        emit,
    )
    report = read_json(report_path)
    if not isinstance(report, dict):
        raise RuntimeError(f"noulxp check failed (exit {code}): {_last(tail)}")
    return report


# ----------------------------------------------------------------------------- build


def build(model_ref, workspace=WORKSPACE, emit=None, test_rows=TEST_ROWS):
    """Build, record, validate and check one run's package. Returns the report.

    Raises when the package does not pass, after keeping it as runs/<id>/noulxp-failed: a
    failing package never replaces the run's package, and is never published."""
    emit = emit or (lambda *a, **k: None)
    run_id, run_dir, model_dir, run = locate(model_ref, workspace)
    family, entry = support(run, workspace)
    if entry["status"] != "ready":
        raise ValueError(refusal(family))
    missing = missing_tooling()
    if missing:
        raise RuntimeError(tooling_message(missing))

    started = time.perf_counter()
    staging, kept, failed = run_dir / BUILDING, run_dir / PACKAGE, run_dir / FAILED
    shutil.rmtree(staging, ignore_errors=True)
    # Built in a folder named as it is published, so the check's report names it so too.
    building = staging / PACKAGE
    staging.mkdir()
    report = {
        "model": f"run:{run_id}",
        "target": "noulxp",
        "state": "failed",
        "created": now(),
        "noulxp": importlib.metadata.version("noulxp"),
        "family": family,
        "profile": entry["profile"],
    }
    try:
        with tempfile.TemporaryDirectory(prefix=".noulxp-", dir=run_dir) as scratch:
            scratch = Path(scratch)
            emit("phase", phase="export", message="Writing the NoulXP package (noulxp export laya)")
            source = provenance(run, run_id, workspace)
            export_package(model_dir, building, f"studio:{run_id}", source, emit)
            limits = (read_json(building / MANIFEST) or {}).get("limits") or {}
            questions = read_json(model_dir / "questions.json") or {}
            try:
                _, rows, _ = load_dataset(run.get("dataset", ""), workspace)
            except (FileNotFoundError, ValueError):
                rows = []  # the run's dataset was deleted: NoulXP's request set alone
            own, asked = own_requests(questions, rows, limits, test_rows)
            from noulxp.conformance import DEFAULT_REQUESTS, read_jsonl

            requests = read_jsonl(DEFAULT_REQUESTS) + own
            _write_jsonl(scratch / "requests.jsonl", requests)
            report["conformance"] = {
                "noulxp_requests": len(requests) - len(own),
                "test_rows": len(own),
                "questions_asked": asked["asked"],
                "questions_left_out": asked["left_out"],
            }
            for qid, why in asked["left_out"].items():
                emit("log", message=f"Question {qid!r} is not in the conformance file: {why}")
            emit(
                "phase",
                phase="conformance",
                message=f"Recording the fine-tune's own answers to {len(requests)} requests "
                "(laya, float32, CPU)",
            )
            view = checkpoint_view(model_dir, scratch / "checkpoint")
            record_conformance(building, view, scratch / "requests.jsonl", emit)
        emit("phase", phase="validate", message="Validating the package: schemas, hashes, coverage")
        problems = validate_package(building, emit)
        emit("phase", phase="check", message="Checking the package on the CPU (noulxp check)")
        check = check_package(building, emit)
    except BaseException as error:
        shutil.rmtree(staging, ignore_errors=True)
        report["error"] = f"{type(error).__name__}: {error}"
        report["seconds"] = round(time.perf_counter() - started, 1)
        write_json(run_dir / REPORT, report)
        raise

    passed = bool(check.get("passed") and check.get("compatible")) and not problems
    report.update(
        state="passed" if passed else "failed",
        seconds=round(time.perf_counter() - started, 1),
        problems=problems,
        check=_check_summary(check),
    )
    if passed:
        shutil.rmtree(kept, ignore_errors=True)
        building.rename(kept)
        shutil.rmtree(failed, ignore_errors=True)
        report["path"] = PACKAGE
    else:
        shutil.rmtree(failed, ignore_errors=True)
        building.rename(failed)
        report["path"] = FAILED
        report["error"] = failure(check, problems)
    shutil.rmtree(staging, ignore_errors=True)
    report["size_mb"] = round(_size(run_dir / report["path"]) / 2**20, 1)
    write_json(run_dir / REPORT, report)
    emit("result", **result_fields(report))
    if not passed:
        raise RuntimeError(
            f"{report['error']} The package is kept in runs/{run_id}/{FAILED} for reading; it "
            "is not the run's package and is never published."
        )
    return report


def _check_summary(check):
    agree = check.get("argmax_agreement") or {}
    runtime = check.get("runtime") or {}
    return {
        "passed": bool(check.get("passed")),
        "compatible": bool(check.get("compatible")),
        "cases": check.get("cases"),
        "cases_passed": check.get("cases_passed"),
        "questions": check.get("questions"),
        "max_abs_dp": check.get("max_abs_dp"),
        "mean_abs_dp": check.get("mean_abs_dp"),
        "argmax": [agree.get("agree"), agree.get("total")],
        "errors": check.get("errors"),
        "coverage_ok": bool((check.get("coverage") or {}).get("ok")),
        "device": runtime.get("device"),
        "backend": runtime.get("backend"),
        "runtime_precision": runtime.get("precision"),
        "package_problems": check.get("package_problems") or [],
        "failures": (check.get("failures") or [])[:5],
        "checked_at": check.get("checked_at"),
    }


def failure(check, problems):
    """Why a package did not pass, in one sentence."""
    if problems:
        return f"The package does not validate: {problems[0]}"
    if check.get("package_problems"):
        return f"The package's files do not check out: {check['package_problems'][0]}"
    if check.get("passed") and not check.get("compatible"):
        return "Every case passed, but the conformance file covers less than the standard asks."
    return (
        f"The NoulXP reference runtime reproduced {check.get('cases_passed')} of "
        f"{check.get('cases')} cases (max |Δp| {check.get('max_abs_dp', 0):.2g})."
    )


def result_fields(report):
    """The job's result event: what the UI shows for a NoulXP export."""
    check = report.get("check") or {}
    return {
        "target": "noulxp",
        "model": report["model"],
        "state": report["state"],
        "cases": check.get("cases"),
        "cases_passed": check.get("cases_passed"),
        "max_abs_dp": check.get("max_abs_dp"),
        "compatible": check.get("compatible"),
        "test_rows": (report.get("conformance") or {}).get("test_rows"),
        "questions_left_out": (report.get("conformance") or {}).get("questions_left_out"),
        "path": report.get("path"),
        "size_mb": report.get("size_mb"),
        "seconds": report.get("seconds"),
    }


def _size(folder):
    return sum(path.stat().st_size for path in Path(folder).rglob("*") if path.is_file())


# ----------------------------------------------------------------------------- reading


def describe(package_dir):
    """A package that passed its check, as the UI and the model card show it; else None."""
    package_dir = Path(package_dir)
    manifest = read_json(package_dir / MANIFEST)
    check = read_json(package_dir / CHECK)
    if not isinstance(manifest, dict) or not isinstance(check, dict):
        return None
    if not (check.get("passed") and check.get("compatible")):
        return None
    conformance = manifest.get("conformance") or {}
    relative = str(conformance.get("path", "conformance.jsonl"))
    own, tag = 0, json.dumps(TEST_TAG)
    if _safe(relative) and (package_dir / relative).is_file():
        with open(package_dir / relative, encoding="utf-8") as handle:
            own = sum(tag in line for line in handle)  # cases from the run's test split
    summary = _check_summary(check)
    return {
        **summary,
        "state": "passed",
        "name": manifest.get("name"),
        "standard": manifest.get("standard"),
        "profile": manifest.get("profile"),
        "noulxp": (manifest.get("source") or {}).get("converted_by"),
        "generated_by": conformance.get("generated_by") or {},
        "test_rows": own,
        "created": summary["checked_at"],
        "size_mb": round(_size(package_dir) / 2**20, 1),
    }


def listing(run_dir, shown=str):
    """The run's NoulXP export for the Models page: the package it keeps, or the attempt
    that failed. None when the run has neither."""
    run_dir = Path(run_dir)
    report = read_json(run_dir / REPORT) or {}
    package = describe(run_dir / PACKAGE)
    if package is None and not report:
        return None
    entry = {"model": f"run:{run_dir.name}", "target": "noulxp", "precision": "float"}
    if package:
        entry.update(package, path=shown(run_dir / PACKAGE))
        if report.get("state") == "failed":
            entry["last_attempt"] = {"created": report.get("created"), "error": report.get("error")}
    else:
        check = report.get("check") or {}
        entry.update(
            state="failed",
            created=report.get("created"),
            error=report.get("error"),
            cases=check.get("cases"),
            cases_passed=check.get("cases_passed"),
            max_abs_dp=check.get("max_abs_dp"),
            path=shown(run_dir / FAILED) if (run_dir / FAILED).is_dir() else None,
        )
    entry["questions_left_out"] = (report.get("conformance") or {}).get("questions_left_out")
    return entry


def _safe(relative):
    parts = PurePosixPath(relative).parts
    return (
        bool(parts)
        and not relative.startswith("/")
        and ".." not in parts
        and "\\" not in relative
        and ":" not in relative
    )


def _sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _entries(manifest):
    for key in FILE_KEYS:
        value = manifest.get(key)
        if isinstance(value, dict) and "path" in value:
            yield value
            yield from value.get("data") or []


def passing_package(run_dir, model_dir):
    """The run's package, if it passed its check and is the package of this very checkpoint.

    Every file the manifest names is hashed again; the report must be the check of this
    conformance file and these weights; and the weights must be the checkpoint's own."""
    run_dir, model_dir = Path(run_dir), Path(model_dir)
    root = run_dir / PACKAGE
    info = describe(root)
    if info is None:
        return None
    manifest = read_json(root / MANIFEST)
    checked = (read_json(root / CHECK) or {}).get("package") or {}
    if checked.get("conformance_sha256") != (manifest.get("conformance") or {}).get("sha256"):
        return None
    if checked.get("weights_sha256") != (manifest.get("weights") or {}).get("sha256"):
        return None
    weights_data = None
    for entry in _entries(manifest):
        relative = str(entry.get("path", ""))
        if not _safe(relative) or not (root / relative).is_file():
            return None
        if not entry.get("sha256") or _sha256(root / relative) != entry["sha256"]:
            return None
        if relative.endswith(".safetensors"):
            weights_data = root / relative
    own = model_dir / "model.safetensors"
    if weights_data is None or not own.is_file():
        return None
    if not os.path.samefile(weights_data, own) and _sha256(weights_data) != _sha256(own):
        return None
    return info


# ----------------------------------------------------------------------------- publishing


def for_publish(model_ref, workspace=WORKSPACE, emit=None):
    """The run's passing package, built and checked first when it has none.

    Raises (and nothing is uploaded) when there is none and none passes: a NoulXP package
    that did not pass its check is never published."""
    emit = emit or (lambda *a, **k: None)
    run_id, run_dir, model_dir, _ = locate(model_ref, workspace)
    found = passing_package(run_dir, model_dir)
    if found:
        emit(
            "log",
            message=f"Including the run's NoulXP package: {found['cases_passed']}/{found['cases']}"
            f" cases reproduced on the CPU, max |Δp| {found['max_abs_dp']:.2g}",
        )
        return found
    emit("phase", phase="noulxp", message="Building the run's NoulXP package before publishing")
    hint = (
        " Nothing was uploaded. Publish without a NoulXP package with --skip-noulxp (in the "
        "studio, untick Include a NoulXP package)."
    )
    try:
        build(f"run:{run_id}", workspace, emit)
    except Exception as error:  # noqa: BLE001 - every reason ends the publish the same way
        raise RuntimeError(f"{error}{hint}") from None
    found = passing_package(run_dir, model_dir)
    if found is None:
        raise RuntimeError(f"The NoulXP package passed its check but does not verify.{hint}")
    return found


@contextlib.contextmanager
def staged(run_dir, model_dir, include):
    """The run's passing package as the checkpoint folder's `noulxp/` while an upload runs.

    The folder belongs to publishing: one left over from an interrupted upload is removed
    first, so a publish without NoulXP never carries a package, and one with it carries the
    package that passed."""
    target = Path(model_dir) / PACKAGE
    shutil.rmtree(target, ignore_errors=True)
    if not include:
        yield None
        return
    source = Path(run_dir) / PACKAGE
    try:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                _place(path, target / path.relative_to(source))
        yield target
    finally:
        shutil.rmtree(target, ignore_errors=True)


def card_line(info):
    """The model card's line on NoulXP: whether the version carries a package, and how it
    was checked."""
    if not info:
        return (
            "**NoulXP:** this version was published without a NoulXP package, so "
            "systemonemodels.tech does not check it for NoulXP compatibility."
        )
    own = int(info.get("test_rows") or 0)
    cases = int(info.get("cases") or 0)
    asked = f"NoulXP's {cases - own} conformance requests" + (
        f" and {own} rows of this model's test split" if own else ""
    )
    laya = (info.get("generated_by") or {}).get("laya")
    native = f"the `laya` package {laya}" if laya else "the `laya` package"
    return (
        f"**NoulXP:** this version carries a NoulXP package in `noulxp/` "
        f"({info.get('standard')}, {info.get('profile')} profile: an ONNX graph over these "
        f"weights). Before publishing, System One Studio recorded this fine-tune's own answers to "
        f"{asked} with {native} (float32, CPU), and the NoulXP reference runtime "
        f"({info.get('noulxp')}) reproduced {info.get('cases_passed')} of {cases} cases on the CPU "
        f"(max |Δp| {float(info.get('max_abs_dp') or 0):.1e}, tolerance 0.01). "
        "systemonemodels.tech runs the same check before it shows the model as NoulXP compatible."
    )


def refusal(family):
    """What a fine-tune of a family without NoulXP export is told, instead of a badge."""
    from .families import FAMILIES, NOULXP

    entry = NOULXP[family]
    name = FAMILIES[family]["name"]
    if entry["profile"]:
        return (
            f"NoulXP export arrives with the trainer for {name} models: {entry['note']} Until "
            "then their fine-tunes publish without a NoulXP package."
        )
    return (
        f"NoulXP export arrives with the trainer for {name} models, and needs a NoulXP profile "
        f"first: {entry['note']} Until then their fine-tunes publish without a NoulXP package."
    )
