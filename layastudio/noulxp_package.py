"""NoulXP packages for fine-tunes: built, checked, kept with the run, published with it.

NoulXP (https://github.com/systemonemodels/noulxp) is the open standard that lets any engine
run a System One model from a package of files, with no code written for that model. When a
model version on systemonemodels.tech carries a package in a `noulxp/` folder, the registry's
engine checks it against the package's own conformance file, and a pass shows the model as
NoulXP compatible.

    python -m layastudio.export run:<id> --target noulxp

builds that package for one fine-tune with noulxp 0.4, the release the registry checks with,
in the way the standard asks (SPEC.md, section 9), with the converter and the model's own
runtime that noulxp has for the run's checkpoint kind (kinds.py):

    laya     noulxp export laya     the `laya` package               encoder-markers (ONNX)
    julia    noulxp export julia    noulxp.native.julia (Julia's own) encoder-markers (ONNX)
    decider  noulxp export decider  noulxp.native.decider (Decider's  causal-letters (GGUF)
                                    GGUF readout, llama.cpp)

1. `noulxp export <kind>` writes it from the run's checkpoint: for the encoders an ONNX graph
   over the checkpoint's own weights file, the tokenizer, template.json and calibration.json;
   for Decider the run's GGUF (gguf.py: converted with a pinned llama.cpp; bf16 by default, the
   merged weights exactly; q8_0, as Decider's own package ships, on request), its tokenizer,
   prompt.json and calibration.json.
2. `noulxp conformance generate` records the fine-tune's own answers, from the model's own
   runtime on the CPU, to NoulXP's request set (52 requests in 11 languages, the coverage the
   standard asks for). Rows of the run's test split, asked the questions the run was tuned for,
   are added only on request (test_rows > 0): they are published inside the package.
3. `noulxp validate` (schemas, parsers, coverage), then `noulxp check` on the CPU: every
   file's SHA-256 against the manifest, then the reference runtime has to reproduce every case
   (each probability within 0.01, and the same leading option). The check is the one pass
   that hashes the package, on the final files, right before it decides.

Each step runs in a process of its own, so memory goes back between them, with as many CPU
threads as the machine's quota allows (runtime.cpu_threads(): a pod's cgroup cpu.max), the same
count for the recording and the check. The package is built in runs/<id>/noulxp.partial/noulxp
and becomes runs/<id>/noulxp only once it passes, with the check's report inside it
(check-cpu.json). A package that fails is kept as runs/<id>/noulxp-failed for reading and is
never published. runs/<id>/noulxp-report.json
describes the last attempt, with what each step took ("steps": seconds, CPU, threads, the
container's throttling and memory; telemetry.py): it explains a measurement, it certifies
nothing, and none of it goes into the package.

Publishing (publish_systemone.py) places the passing package next to the checkpoint's own
files, as the version's `noulxp/` folder, for as long as the upload takes.

Needs the optional extra:  uv sync --extra export   (Decider: --extra export --extra gguf)
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

from .engine import (
    WORKSPACE,
    check_id,
    hub_parts,
    load_dataset,
    now,
    read_json,
    resolve_model_ref,
    write_json,
)

REQUIREMENT = "noulxp[export,laya,onnx]>=0.4,<0.5"
RELEASE = (0, 4)  # the noulxp release line systemonemodels.tech checks packages with
# Rows of the run's test split recorded next to NoulXP's request set. 0 by default: the
# conformance file is published with the package, and a dataset can be private. Opt in per run.
TEST_ROWS = 0

PACKAGE = "noulxp"  # runs/<id>/noulxp: a package that passed its check, and nothing else
BUILDING = "noulxp.partial"
FAILED = "noulxp-failed"
REPORT = "noulxp-report.json"
MANIFEST = "noulxp.json"
CHECK = "check-cpu.json"
TEST_TAG = "studio:test-split"

# What each kind's `noulxp export`, the model's own runtime and `noulxp check` import, besides
# noulxp. Decider's also covers the GGUF conversion (gguf.py) and llama.cpp.
ENCODER_NEEDS = (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("onnx", "onnx"),
    ("onnxscript", "onnxscript"),
    ("onnx_ir", "onnx-ir"),
    ("onnxruntime", "onnxruntime"),
)
NEEDS = {
    "laya": (*ENCODER_NEEDS[:3], ("laya", "laya"), *ENCODER_NEEDS[3:]),
    "julia": ENCODER_NEEDS,
    "decider": (
        ("llama_cpp", "llama-cpp-python"),
        ("transformers", "transformers"),
        ("tokenizers", "tokenizers"),
        ("torch", "torch"),
        ("safetensors", "safetensors"),
        ("sentencepiece", "sentencepiece"),
        ("yaml", "pyyaml"),
    ),
}
NEEDED = NEEDS["laya"]
EXTRAS = {"laya": "export", "julia": "export", "decider": "export --extra gguf"}
RUNTIME = {"laya": "the `laya` package", "julia": "Julia's own inference", "decider": "Decider's"}
GGUF_PRECISION = "bf16"  # the GGUF in a Decider package: the merged weights exactly. q8_0 is
# half the size but changes answers (21 of 2,000 typed-decisions questions on Decider 2B, 16
# beyond ties), so it is an opt-in (--gguf q8_0).
# noulxp refuses to export an encoder with an older transformers: older releases compute
# ModernBERT differently, and the package would not give the model's own answers.
MIN_TRANSFORMERS = (5, 2)
# The file entries a manifest names (SPEC.md 4.2); weights also list their external data.
FILE_KEYS = ("weights", "tokenizer", "template", "prompt", "calibration", "conformance")

# ONNX Runtime's threads in the exporter's own process, for its informative onnxruntime-versus-
# torch comparison (source.export.verification) only: noulxp 0.4's compare_with_torch opens its
# session with no thread count, which is one thread per host core on a pod. Installed before
# the exporter is imported, and nothing unless LAYASTUDIO_THREADS (which the step's process
# gets: _child_env) is a positive integer and noulxp 0.4's providers.ort_session exists. Only
# the install is guarded: an error of the export or of a real session still fails the step.
ORT_THREADS = (
    "def _cap_ort_threads():\n"
    "    import os\n"
    "    try:\n"
    "        n = int(os.environ.get('LAYASTUDIO_THREADS', ''))\n"
    "    except ValueError:\n"
    "        return\n"
    "    if n <= 0:\n"
    "        return\n"
    "    try:\n"
    "        import noulxp\n"
    "        import noulxp.providers as providers\n"
    "    except ImportError:\n"
    "        return\n"
    "    original = getattr(providers, 'ort_session', None)\n"
    "    if not callable(original) or not str(getattr(noulxp, '__version__', '')).startswith('0.4'):\n"
    "        return\n"
    "    def ort_session(path, *args, threads=None, **kwargs):\n"
    "        return original(path, *args, threads=threads or n, **kwargs)\n"
    "    ort_session.__wrapped__ = original\n"
    "    providers.ort_session = ort_session\n"
    "_cap_ort_threads()\n"
)
# `noulxp export <kind>`, called as its command calls it (hard-linked weights, opset 18), with
# the fine-tune's own provenance in the manifest's `source` instead of the base model's. It caps
# ONNX Runtime's threads for the exporter's informative comparison first (ORT_THREADS).
EXPORT = ORT_THREADS + (
    "import importlib, json, sys\n"
    "from pathlib import Path\n"
    "exporter = importlib.import_module('noulxp.export.' + sys.argv[1])\n"
    "exporter.export(Path(sys.argv[2]), Path(sys.argv[3]), name=sys.argv[4],"
    " source=json.loads(sys.argv[5]), **json.loads(sys.argv[6]))\n"
)
CASES = re.compile(r"^\s*(\d+)/(\d+) cases\b")
# The variables that cap a step process's BLAS and OpenMP threads (torch reads them at import),
# and LAYASTUDIO_THREADS, which the export's ORT_THREADS reads.
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LAYASTUDIO_THREADS")
# Where the package's hashes are verified: once, by the check, on the final files.
HASHES = "verified by noulxp check (package_problems)"
# Laya's own runtime takes no thread count, and noulxp 0.4 records none for it.
LAYA_THREADS = "OMP_NUM_THREADS and MKL_NUM_THREADS (laya's runtime takes no thread count)"
# How long a step process may take to end after SIGTERM before it is killed (the local server
# kills a job 15 s after asking, a cloud run 30 s after).
END_AFTER = 10
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


def missing_tooling(kind="laya"):
    """Why this environment cannot build a NoulXP package of this kind, or None when it can."""
    try:
        version = importlib.metadata.version("noulxp")
    except importlib.metadata.PackageNotFoundError:
        return "noulxp is not installed"
    if _release(version) != RELEASE:
        return (
            f"noulxp {version} is installed, and the studio builds packages with noulxp 0.4, "
            "the release systemonemodels.tech checks them with"
        )
    for module, name in NEEDS.get(kind, NEEDED):
        if not _importable(module):
            return f"{name} is not installed"
    try:
        found = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return "transformers is not installed"
    if _release(found) < MIN_TRANSFORMERS:
        return f"transformers {found} is installed; noulxp exports with 5.2 or later"
    return None


def tooling_message(missing, kind="laya"):
    extra = EXTRAS.get(kind, "export")
    gguf = ' "llama-cpp-python==0.3.35"' if kind == "decider" else ""
    return (
        f"{missing}. Building a NoulXP package needs the optional extra:\n"
        f"    uv sync --extra {extra}\n"
        f'or, outside a checkout:  pip install "{REQUIREMENT}"{gguf}'
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


def base_model(ref, workspace=WORKSPACE):
    """The published name of the model a run started from; never a path on this machine."""
    return base_reference(ref, workspace)[0]


def base_reference(ref, workspace=WORKSPACE, depth=0):
    """(repository, revision) of the model a run started from: a hub reference's own (its
    revision when it pins one), a path imported from the registry (imports.json), or a run's
    base, followed back. (None, None) for a folder the studio knows no published name of."""
    kind, _, value = str(ref or "").partition(":")
    if kind == "hub":
        try:
            return hub_parts(value)
        except ValueError:
            return None, None
    if kind == "path":
        from .families import imports

        repo = next((e.get("repo") for e in imports(workspace) if e.get("ref") == ref), None)
        return repo, None
    if kind == "run" and depth < 4:
        try:
            parent = read_json(workspace / "runs" / check_id(value) / "run.json") or {}
        except ValueError:
            return None, None
        return base_reference(parent.get("base_model"), workspace, depth + 1)
    return None, None


def support(run, workspace=WORKSPACE, model_dir=None):
    """(family, the family's NoulXP entry) for a run: from its checkpoint's kind when the
    studio trained it (every kind it trains has a NoulXP exporter), else from the catalogue."""
    from . import kinds
    from .families import NOULXP, find

    if model_dir is not None and kinds.detect(model_dir):
        kind = kinds.detect(model_dir)
        family = kinds.FAMILY[kind]
        return family, {**NOULXP[family], "profile": kinds.PROFILE[kind], "status": "ready"}
    base = base_model(run.get("base_model"), workspace)
    known = find(base) if base else None
    family = known.family if known else "laya"
    return family, NOULXP[family]


def provenance(run, run_id, workspace=WORKSPACE):
    """The manifest's `source`: what this package is a conversion of. No local paths."""
    from .families import find

    base, revision = base_reference(run.get("base_model"), workspace)
    known = find(base) if base else None
    training = read_json(workspace / "runs" / run_id / "training.json") or {}
    return {
        "model": base,
        **({"revision": revision} if revision else {}),
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


def checkpoint_view(model_dir, dest, kind="laya", gguf=None):
    """A copy of the checkpoint for the model's own runtime (and Decider's exporter) to read.

    laya rewrites tokenizer/tokenizer_config.json in place when it names no tokenizer class;
    the copy keeps that away from the run's own files. Decider's runtime and exporter read a
    folder holding one GGUF: the run's export, as model.gguf. Weights are hard-linked: the
    runtimes only read them."""
    dest.mkdir(parents=True)
    if kind == "decider":
        for name in ("tokenizer.json", "tokenizer_config.json", "decider_config.json"):
            shutil.copyfile(model_dir / name, dest / name)
        if (model_dir / "chat_template.jinja").is_file():
            shutil.copyfile(model_dir / "chat_template.jinja", dest / "chat_template.jinja")
        _place(gguf, dest / "model.gguf")
        return dest
    if kind == "julia":
        for name in ("julia_config.json", "inference-policy.json", "config.json"):
            if (model_dir / name).is_file():
                shutil.copyfile(model_dir / name, dest / name)
    else:
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


def _child_env(threads=None):
    """A step process's environment: offline, unbuffered, and, with a thread count, the
    BLAS/OpenMP caps (THREAD_VARS) at that count. A cap the machine already sets wins
    (setdefault): report the effective values (thread_env), not the count asked for. Only step
    processes get these; the job's own process (training, the GPU) keeps its own."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    if threads:
        for name in THREAD_VARS:
            env.setdefault(name, str(int(threads)))
    return env


def thread_env(threads=None):
    """The thread caps a step process started with this count sees."""
    env = _child_env(threads)
    return {name: env.get(name) for name in THREAD_VARS}


def _end(processes, timeout=END_AFTER):
    """Every process still running: SIGTERM to all at once, one shared deadline, then SIGKILL.
    Each is reaped before this returns."""
    alive = [p for p in processes if p is not None and p.poll() is None]
    for process in alive:
        with contextlib.suppress(OSError):
            process.terminate()
    deadline = time.monotonic() + timeout
    for process in alive:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                process.kill()
            process.wait()


def _python(args, emit, progress=False, threads=None):
    """One step in a process of its own, its output into the job's log.

    threads: the BLAS/OpenMP caps of its environment (_child_env). Returns (exit code, the
    last lines it printed). Cancelling the job stops the step."""
    from .telemetry import Sampler

    process = subprocess.Popen(
        [sys.executable, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=_child_env(threads),
    )
    tail = collections.deque(maxlen=20)
    try:
        sampler = Sampler(process.pid)
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
        finally:
            sampler.stop()
        code = process.wait()
    except BaseException:
        _end([process])
        raise
    return code, list(tail)


def _last(tail):
    useful = [line for line in tail if not GLOG.match(line) and "warnings.warn" not in line]
    return (useful or tail or ["no output"])[-1]


def export_package(model_dir, out_dir, name, source, emit, kind="laya", options=None, threads=None):
    """Step 1: `noulxp export <kind>` from the run's checkpoint."""
    code, tail = _python(
        [
            "-c",
            EXPORT,
            kind,
            str(model_dir),
            str(out_dir),
            name,
            json.dumps(source),
            json.dumps(options or {}),
        ],
        emit,
        threads=threads,
    )
    if code:
        raise RuntimeError(f"noulxp export {kind} failed: {_last(tail)}")


def _generate_args(package_dir, checkpoint, requests, runtime, threads=None):
    """`noulxp conformance generate`'s arguments: every request (no --limit), the model's own
    runtime on the CPU, with this many threads when given."""
    args = [
        "-m",
        "noulxp",
        "conformance",
        "generate",
        str(package_dir),
        "--native",
        str(checkpoint),
        "--runtime",
        runtime,
        "--requests",
        str(requests),
    ]
    if threads:
        args += ["--threads", str(int(threads))]
    return args


def record_conformance(package_dir, checkpoint, requests, emit, runtime="laya", threads=None):
    """Step 2: the fine-tune's own answers (its own runtime, CPU) into conformance.jsonl."""
    code, tail = _python(
        _generate_args(package_dir, checkpoint, requests, runtime, threads),
        emit,
        progress=True,
        threads=threads,
    )
    if code:
        raise RuntimeError(f"noulxp conformance generate failed: {_last(tail)}")


def validate_package(package_dir, emit):
    """Step 3a: schemas, parsers, coverage, without running anything. Returns the problems.

    It leaves the files' SHA-256 to the check (--no-hashes), which runs right after on the same
    unchanged files and hashes every one of them before it decides (check_package): one hash
    pass, in the step that decides "passed". It still finds a file that is missing or outside
    the package."""
    code, tail = _python(["-m", "noulxp", "validate", str(package_dir), "--no-hashes"], emit)
    if code == 0:
        return []
    return [line for line in tail if not line.endswith("problem(s)")] or [_last(tail)]


def check_package(package_dir, emit, threads=None):
    """Step 3b: every file's SHA-256 against the manifest, then the conformance file through
    the reference runtime on the CPU, with this many threads when given. The report.

    The package's only hash pass (validate_package skips it): `noulxp check` with its default
    hash verification, whose mismatches are the report's package_problems, which build() refuses.
    Never add --no-hashes here. A check moved into this process must call
    conformance.check(hashes=True), or pass package.verify(hashes=True)'s problems to replay():
    never problems=[] while validate skips the hashes."""
    report_path = package_dir / CHECK
    args = [
        "-m",
        "noulxp",
        "check",
        str(package_dir),
        "--device",
        "cpu",
        "--report",
        str(report_path),
    ]
    if threads:
        args += ["--threads", str(int(threads))]
    code, tail = _python(args, emit, threads=threads)
    report = read_json(report_path)
    if not isinstance(report, dict):
        raise RuntimeError(f"noulxp check failed (exit {code}): {_last(tail)}")
    return report


def _reported(described):
    """The thread count a step says it ran with (generated_by.threads; a check's
    runtime.threads, which noulxp 0.4.0 leaves null and 0.4.1 fills), else None: never
    inferred."""
    value = described.get("threads") if isinstance(described, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# ----------------------------------------------------------------------------- build


def ensure_gguf(model_ref, workspace, emit, precision=GGUF_PRECISION, threads=None, stats=None):
    """The run's GGUF export at this precision, converted (gguf.py) unless one exists that
    was converted from the checkpoint's current weights. (path, export report)

    Reused only when all of these hold: the file is there, its report names this llama.cpp
    commit and the weights it was converted from, those are the checkpoint's weights now, and
    the file is the one the report hashed. Nothing is hashed for a run without one (a fresh
    pod): gguf.export reads the weights' digests while the converter runs; a stale one hands
    over the digests computed here. These hashes are the studio's cache keys, not the
    package's certification (the check hashes the package). stats, when given, gets
    {"cached": bool}."""
    from . import gguf

    _, run_dir, model_dir, _ = locate(model_ref, workspace)
    target = run_dir / "exports" / gguf.FILE.format(precision=precision)
    report = read_json(run_dir / "exports" / f"gguf-{precision}.json") or {}
    weights = None
    if (
        target.is_file()
        and report.get("llama_cpp") == gguf.LLAMA_CPP_COMMIT
        and report.get("source_sha256")
    ):
        weights = {p.name: _sha256(p) for p in sorted(model_dir.glob("*.safetensors"))}
        if report.get("source_sha256") == weights and report.get("sha256") == _sha256(target):
            emit("log", message=f"Using the run's GGUF export ({precision})")
            if stats is not None:
                stats["cached"] = True
            return target, report
    if stats is not None:
        stats["cached"] = False
    report = gguf.export(
        model_ref, workspace, emit, precision=precision, threads=threads, source_sha256=weights
    )
    return target, report


def build(model_ref, workspace=WORKSPACE, emit=None, test_rows=TEST_ROWS, gguf=GGUF_PRECISION):
    """Build, record, validate and check one run's package. Returns the report.

    Raises when the package does not pass, after keeping it as runs/<id>/noulxp-failed: a
    failing package never replaces the run's package, and is never published."""
    from . import kinds, telemetry
    from .runtime import cpu_budget

    emit = emit or (lambda *a, **k: None)
    run_id, run_dir, model_dir, run = locate(model_ref, workspace)
    kind = kinds.detect(model_dir) or "laya"
    family, entry = support(run, workspace, model_dir)
    if entry["status"] != "ready":
        raise ValueError(refusal(family))
    missing = missing_tooling(kind)
    if missing:
        raise RuntimeError(tooling_message(missing, kind))
    from noulxp.conformance import DEFAULT_REQUESTS, read_jsonl

    started = time.perf_counter()
    # One thread count for every CPU step here, the recording and the check alike: the
    # machine's quota (runtime.cpu_budget).
    budget = cpu_budget()
    threads = budget["threads"]
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
        "kind": kind,
        "profile": entry["profile"],
        "threads": threads,
        "cpu_quota": budget["quota"],
        "threads_source": budget["source"],
        # What the step processes run with (a cap the machine sets already wins).
        "thread_env": thread_env(threads),
        "hashes": HASHES,
    }
    if kind == "laya":
        report["laya_threads"] = LAYA_THREADS
    report["machine"] = telemetry.machine(kind)
    report["memory"] = {"start": telemetry.memory()}
    steps = telemetry.Steps(report, emit=emit)
    try:
        with tempfile.TemporaryDirectory(prefix=".noulxp-", dir=run_dir) as scratch:
            scratch = Path(scratch)
            source = provenance(run, run_id, workspace)
            view = None
            if kind == "decider":
                cached = {}
                with steps("gguf", threads=threads, device="cpu") as record:
                    gguf_file, converted = ensure_gguf(
                        f"run:{run_id}", workspace, emit, gguf, threads=threads, stats=cached
                    )
                    if cached.get("cached"):
                        record["cached"] = True
                    else:  # this conversion's own timings, never a stored report's
                        record["timings"] = converted.get("timings")
                report["gguf"] = {
                    k: converted.get(k)
                    for k in ("precision", "sha256", "size_mb", "llama_cpp", "verification")
                }
                source = {
                    **source,
                    "gguf": {k: converted.get(k) for k in ("precision", "llama_cpp")},
                }
                view = checkpoint_view(model_dir, scratch / "checkpoint", kind, gguf_file)
                emit(
                    "phase",
                    phase="export",
                    message="Writing the NoulXP package (noulxp export decider)",
                    threads=threads,
                    device="cpu",
                )
                with steps("export", threads=threads, device="cpu", env=thread_env(threads)):
                    export_package(
                        view,
                        building,
                        f"studio:{run_id}",
                        source,
                        emit,
                        kind,
                        {"gguf": "model.gguf"},
                        threads=threads,
                    )
            else:
                options = None
                if kind == "julia":
                    # The package's budgets are the checkpoint's own inference policy.
                    from . import julia

                    policy = julia.config(model_dir)
                    options = {
                        "max_tokens": policy["max_len"],
                        "head_tokens": policy["head_max_len"],
                    }
                emit(
                    "phase",
                    phase="export",
                    message=f"Writing the NoulXP package (noulxp export {kind})",
                    threads=threads,
                    device="cpu",
                )
                with steps("export", threads=threads, device="cpu", env=thread_env(threads)):
                    export_package(
                        model_dir,
                        building,
                        f"studio:{run_id}",
                        source,
                        emit,
                        kind,
                        options,
                        threads=threads,
                    )
            limits = (read_json(building / MANIFEST) or {}).get("limits") or {}
            questions = read_json(model_dir / "questions.json") or {}
            try:
                _, rows, _ = load_dataset(run.get("dataset", ""), workspace)
            except (FileNotFoundError, ValueError):
                rows = []  # the run's dataset was deleted: NoulXP's request set alone
            own, asked = own_requests(questions, rows, limits, test_rows)
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
                f"({RUNTIME.get(kind, kind)} runtime, CPU)",
                threads=threads,
                device="cpu",
            )
            if view is None:
                view = checkpoint_view(model_dir, scratch / "checkpoint", kind)
            with steps(
                "conformance", threads=threads, device="cpu", env=thread_env(threads)
            ) as record:
                record_conformance(
                    building,
                    view,
                    scratch / "requests.jsonl",
                    emit,
                    runtime=kind,
                    threads=threads,
                )
                record["threads_reported"] = _reported(
                    ((read_json(building / MANIFEST) or {}).get("conformance") or {}).get(
                        "generated_by"
                    )
                )
        emit(
            "phase",
            phase="validate",
            message="Validating the package: schemas, parsers, coverage (the check verifies "
            "every file's hash)",
        )
        with steps("validate"):
            problems = validate_package(building, emit)
        emit(
            "phase",
            phase="check",
            message="Checking the package (noulxp check): every file's SHA-256, then the cases "
            "on the CPU",
            threads=threads,
            device="cpu",
        )
        with steps("check", threads=threads, device="cpu", env=thread_env(threads)) as record:
            check = check_package(building, emit, threads=threads)
            record["threads_reported"] = _reported(check.get("runtime"))
    except BaseException as error:
        shutil.rmtree(staging, ignore_errors=True)
        report["error"] = f"{type(error).__name__}: {error}"
        report["seconds"] = round(time.perf_counter() - started, 1)
        report["memory"]["end"] = telemetry.memory()
        write_json(run_dir / REPORT, report)
        raise

    # A file that does not match its hash fails the check (package_problems); refused here too,
    # whatever a later noulxp decides "passed" means.
    passed = (
        bool(check.get("passed") and check.get("compatible"))
        and not problems
        and not check.get("package_problems")
    )
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
    report["memory"]["end"] = telemetry.memory()
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
        "gguf": (manifest.get("source") or {}).get("gguf"),
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
    weights_data = gguf_file = None
    for entry in _entries(manifest):
        relative = str(entry.get("path", ""))
        if not _safe(relative) or not (root / relative).is_file():
            return None
        if not entry.get("sha256") or _sha256(root / relative) != entry["sha256"]:
            return None
        if relative.endswith(".safetensors"):
            weights_data = root / relative
        elif relative.endswith(".gguf"):
            gguf_file = root / relative
    if gguf_file is not None:
        return info if _gguf_of_checkpoint(run_dir, model_dir, gguf_file) else None
    own = model_dir / "model.safetensors"
    if weights_data is None or not own.is_file():
        return None
    if not os.path.samefile(weights_data, own) and _sha256(weights_data) != _sha256(own):
        return None
    return info


def _gguf_of_checkpoint(run_dir, model_dir, gguf_file):
    """A Decider package's GGUF is the run's export, converted from these very weights."""
    digest = _sha256(gguf_file)
    weights = {p.name: _sha256(p) for p in sorted(Path(model_dir).glob("*.safetensors"))}
    for path in sorted((Path(run_dir) / "exports").glob("gguf-*.json")):
        report = read_json(path) or {}
        if report.get("sha256") == digest and report.get("source_sha256") == weights:
            return True
    return False


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
    by = info.get("generated_by") or {}
    runtime = by.get("runtime") or "laya"
    if runtime == "decider":
        what = (
            f"this fine-tune as a {(info.get('gguf') or {}).get('precision', 'GGUF')} GGUF, "
            "converted with llama.cpp"
        )
        native = "Decider's own GGUF readout (llama.cpp, every row decoded in full, CPU)"
    elif runtime == "julia":
        what = "an ONNX graph over these weights"
        native = "Julia 1's own inference (float32, CPU)"
    else:
        what = "an ONNX graph over these weights"
        native = f"the `laya` package {by['laya']}" if by.get("laya") else "the `laya` package"
        native += " (float32, CPU)"
    return (
        f"**NoulXP:** this version carries a NoulXP package in `noulxp/` "
        f"({info.get('standard')}, {info.get('profile')} profile: {what}). Before publishing, "
        f"System One Studio recorded this fine-tune's own answers to {asked} with {native}, and "
        f"the NoulXP reference runtime ({info.get('noulxp')}) reproduced "
        f"{info.get('cases_passed')} of {cases} cases on the CPU "
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
