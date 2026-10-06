"""GGUF exports of Decider fine-tunes, through llama.cpp's own converter at a pinned commit.

    python -m systemone_studio.export run:<id> --target gguf     # bf16, the merged weights
    python -m systemone_studio.export run:<id> --target gguf --precision q8_0

The converter is convert_hf_to_gguf.py from ggml-org/llama.cpp at the commit llama-cpp-python
0.3.35 vendors (LLAMA_CPP_COMMIT), so the files are written by the same llama.cpp that later
reads them (the studio, `noulxp check`, Decider's own GGUF engine). It is downloaded once into
the tools folder beside the workspace, checked against pinned SHA-256s, and run in a process of
its own with that commit's gguf-py.

Every export is measured, not assumed: the fine-tune's test rows go through the GGUF on the CPU
(llama.cpp, every row decoded from empty memory) and through the merged safetensors in
PyTorch in float32, and the report records the same-answer rate and the largest probability
difference.
bf16 holds the merged weights exactly; q8_0 is half the size and a little further off.

The measurement is System One Studio's own report on the export (exports/gguf-<p>.json), not
part of a NoulXP package. On a CUDA machine with memory to spare, its float32 half runs in a
process of its own (gguf_reference.py), started before the converter, so the GPU reads the rows
while the converter and llama.cpp use the CPU: the same rows, the same reference, the same
comparison, in a different order. SYSTEMONE_STUDIO_SERIAL_VERIFY=1 (or its old name,
LAYASTUDIO_SERIAL_VERIFY=1) keeps everything in this process, one after the other.
"""

import ctypes
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

from . import children, environment
from .engine import PACKAGE, WORKSPACE, now, read_json, write_json

LLAMA_CPP_COMMIT = "4df29be4f4c3673f428170fda944a5b19f743bb8"  # vendored by llama-cpp-python 0.3.35
ARCHIVE = f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{LLAMA_CPP_COMMIT}"
# Files of that commit checked after download: the converter and Qwen3.5's conversion code.
PINNED = {
    "convert_hf_to_gguf.py": "e38975e1c68d98ac1664dfd530616eb35c72294382a4dd873d4746b23f27779f",
    "conversion/qwen.py": "65c61155458078232dd3f9d23284710fa39c29cf7ecc341194c825cf5334f43f",
}
WANTED = ("convert_hf_to_gguf.py", "conversion/", "gguf-py/", "LICENSE")
PRECISIONS = ("bf16", "q8_0", "f16")
FILE = "model-{precision}.gguf"
N_CTX = 8192
# Held-out test rows a GGUF is measured on. FULL_VERIFY_ROWS for every precision by default;
# VERIFY_ROWS is the founder's call for bf16 only (the merged weights exactly, so the measure
# there is float32-versus-bf16 rounding): below FULL_VERIFY_ROWS it caps the decodes and
# spreads them over the whole test split, and the report says it is a sample. q8_0 and f16
# always get FULL_VERIFY_ROWS: there the measure is the only evidence of fidelity.
FULL_VERIFY_ROWS = 60
VERIFY_ROWS = FULL_VERIFY_ROWS
# The float32 reference in a process of its own (gguf_reference.py), and how long it may take.
REFERENCE = ("-m", "systemone_studio.gguf_reference")
REFERENCE_WAIT = 20 * 60
# Free memory the overlap needs: the converter's and the float32 load's (about 4 + 8 GB for a 2B
# Decider) beside what the job already holds.
OVERLAP_MEMORY = 24 * 2**30


def tools_dir(workspace=WORKSPACE):
    """Where downloaded tools live: beside the workspace, never inside a run."""
    return Path(environment.get("TOOLS") or Path(workspace).parent / "tools")


def _sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def converter(workspace=WORKSPACE, emit=None):
    """llama.cpp's converter at LLAMA_CPP_COMMIT, downloaded and verified once."""
    emit = emit or (lambda *a, **k: None)
    root = tools_dir(workspace) / f"llama.cpp-{LLAMA_CPP_COMMIT[:12]}"
    if all((root / name).is_file() for name in PINNED) and (root / ".verified").is_file():
        return root
    emit(
        "phase",
        phase="download",
        message=f"Fetching llama.cpp's converter ({LLAMA_CPP_COMMIT[:7]})",
    )
    partial = root.with_name(root.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    with urllib.request.urlopen(ARCHIVE, timeout=300) as response:
        data = io.BytesIO(response.read())
    with tarfile.open(fileobj=data, mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) != 2 or not member.isfile():
                continue
            relative = parts[1]
            if not relative.startswith(WANTED) or ".." in relative.split("/"):
                continue
            target = partial / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)
    for name, digest in PINNED.items():
        found = _sha256(partial / name) if (partial / name).is_file() else None
        if found != digest:
            shutil.rmtree(partial, ignore_errors=True)
            raise RuntimeError(f"llama.cpp's {name} does not match the pinned commit ({found})")
    (partial / ".verified").write_text(LLAMA_CPP_COMMIT + "\n")
    shutil.rmtree(root, ignore_errors=True)
    partial.rename(root)
    return root


def _run_converter(args, env):
    """The converter (`python <script> args`) behind children.WATCH, so that it ends with this
    job even when the job is killed outright: its stdin is a pipe nobody writes to, held until
    it has ended. Its stdout and stderr go to files, never pipes, so nothing waits on a full
    pipe while stdin stays open (subprocess.run's communicate() would close it at once). Any
    exception here, a cancel included, kills it first, as subprocess.run did. Returns (exit
    code, stdout, stderr)."""
    import tempfile

    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        process = subprocess.Popen(
            children.command(args),
            stdin=subprocess.PIPE,  # children.WATCH: never written to
            stdout=out,
            stderr=err,
            env=children.child_env(env),
        )
        try:
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            children.close_stdin(process)
        texts = []
        for handle in (out, err):
            handle.seek(0)
            texts.append(handle.read().decode(errors="replace"))
    return code, texts[0], texts[1]


def convert(model_dir, out_file, precision="bf16", workspace=WORKSPACE, emit=None):
    """convert_hf_to_gguf.py on a Decider checkpoint folder, in a process of its own that ends
    with this job (_run_converter)."""
    emit = emit or (lambda *a, **k: None)
    if precision not in PRECISIONS:
        raise ValueError(f"GGUF precision is one of {PRECISIONS}")
    tool = converter(workspace, emit)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # The converter imports its own conversion/ package and gguf-py: both on PYTHONPATH, since
    # its folder is on the path only when Python adds a script's own, which -P and
    # PYTHONSAFEPATH (the trainer image's) turn off. No empty entry: Python reads one as the
    # current folder.
    path = [str(tool), str(tool / "gguf-py"), *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(entry for entry in path if entry),
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "NO_LOCAL_GGUF": "",
    }
    args = [
        str(tool / "convert_hf_to_gguf.py"),
        str(model_dir),
        "--outfile",
        str(out_file),
        "--outtype",
        precision,
        # No multi-token-prediction (NextN) draft layer: Decider ships none, and a fine-tune
        # neither trains nor carries one, though Qwen3.5's config names it.
        "--no-mtp",
    ]
    emit(
        "phase",
        phase="convert",
        message=f"Converting to GGUF ({precision}) with llama.cpp",
        device="cpu",
    )
    code, stdout, stderr = _run_converter(args, env)
    for line in (stdout + stderr).splitlines()[-12:]:
        if line.strip():
            emit("log", message=line.strip())
    if code or not out_file.is_file():
        tail = (stderr or stdout).strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"convert_hf_to_gguf.py failed: {tail[0]}")
    with open(out_file, "rb") as handle:
        if handle.read(4) != b"GGUF":
            raise RuntimeError(f"{out_file.name} is not a GGUF file")
    return out_file


# ----------------------------------------------------------------------------- reading it


class Reader:
    """A GGUF read on the CPU with llama.cpp: every row decoded in full from empty memory,
    the letter logits at its last token (decider's engine_gguf, one row per decode)."""

    def __init__(self, gguf_file, threads=None):
        import llama_cpp as lc
        import numpy as np

        self.lc, self.np = lc, np
        lc.llama_log_set(_quiet(lc), ctypes.c_void_p(0))
        lc.llama_backend_init()
        params = lc.llama_model_default_params()
        params.n_gpu_layers = 0
        self.model = lc.llama_model_load_from_file(str(gguf_file).encode(), params)
        if not self.model:
            raise RuntimeError(f"llama.cpp could not load {gguf_file}")
        context = lc.llama_context_default_params()
        context.n_ctx = context.n_batch = N_CTX
        context.n_ubatch = 2048
        if threads:
            context.n_threads = context.n_threads_batch = threads
        self.ctx = lc.llama_init_from_model(self.model, context)
        if not self.ctx:
            raise RuntimeError("llama.cpp could not create a context")
        self.threads = self.n_threads()
        self.memory = lc.llama_get_memory(self.ctx)
        self.n_vocab = lc.llama_vocab_n_tokens(lc.llama_model_get_vocab(self.model))
        self.batch = lc.llama_batch_init(N_CTX, 0, 1)

    def n_threads(self):
        """The threads llama.cpp decodes with (its own count, not the one asked for)."""
        try:
            return int(self.lc.llama_n_threads(self.ctx))
        except (AttributeError, TypeError, ValueError):
            return None

    def set_threads(self, threads):
        """Decode with this many threads from the next row on."""
        self.lc.llama_set_n_threads(self.ctx, int(threads), int(threads))
        self.threads = self.n_threads()

    def logits(self, ids, letters):
        lc, b = self.lc, self.batch
        if len(ids) > N_CTX:
            raise ValueError(f"a row of {len(ids)} tokens is longer than {N_CTX}")
        lc.llama_memory_clear(self.memory, True)
        for i, token in enumerate(ids):
            b.token[i] = token
            b.pos[i] = i
            b.n_seq_id[i] = 1
            b.seq_id[i][0] = 0
            b.logits[i] = i == len(ids) - 1
        b.n_tokens = len(ids)
        if lc.llama_decode(self.ctx, b) != 0:
            raise RuntimeError("llama_decode failed")
        pointer = ctypes.cast(
            lc.llama_get_logits_ith(self.ctx, len(ids) - 1), ctypes.POINTER(ctypes.c_float)
        )
        row = self.np.ctypeslib.as_array(pointer, shape=(self.n_vocab,))
        return [float(row[t]) for t in letters]

    def close(self):
        lc = self.lc
        if getattr(self, "batch", None) is not None:
            lc.llama_batch_free(self.batch)
            self.batch = None
        if getattr(self, "ctx", None):
            lc.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            lc.llama_model_free(self.model)
            self.model = None


_LOG = None


def _quiet(lc):
    """llama.cpp's log, silenced process-wide (the callback must outlive every call)."""
    global _LOG
    if _LOG is None:
        _LOG = lc.llama_log_callback(lambda level, text, data: None)
    return _LOG


# ----------------------------------------------------------------------------- export


def test_rows(model_ref, workspace, limit=100):
    """The run's test rows as Decider prompt rows: [(ids, n, type, answer key)], with the
    questions and dataset rows, for measuring an export on held-out data."""
    from . import decider
    from .engine import check_id, load_dataset, resolve_model_ref

    model_dir = resolve_model_ref(model_ref, workspace)
    run = read_json(workspace / "runs" / check_id(model_ref.split(":", 1)[1]) / "run.json") or {}
    try:
        questions, rows, _ = load_dataset(run.get("dataset", ""), workspace)
    except (FileNotFoundError, ValueError):
        return [], {}, model_dir
    cfg = decider.config(model_dir)
    prompter = decider.Prompter(decider.Tokens(model_dir), cfg)
    test = [r for r in rows if r["split"] == "test"][:limit]
    items, _ = decider.encode_items(prompter, test, questions)
    return items, questions, model_dir


def compare(reference, exported, items):
    """Same answer rate and largest probability difference between two sets of letter logits."""
    from . import decider

    same, worst = 0, 0.0
    for a, b, item in zip(reference, exported, items):
        p, q = decider.softmax(a), decider.softmax(b)
        same += p.index(max(p)) == q.index(max(q))
        worst = max(worst, max(abs(x - y) for x, y in zip(p, q)))
    return {"rows": len(items), "same_answer": same, "max_probability_difference": worst}


def verify_rows(precision):
    """The held-out rows a GGUF of this precision is measured on (VERIFY_ROWS: bf16 only)."""
    return VERIFY_ROWS if precision == "bf16" else FULL_VERIFY_ROWS


def spread(items, budget):
    """At most `budget` items, spread over the whole list and over its answer types: each type
    gets its share (at least one while the budget allows), taken at an even stride within
    that type. Deterministic, and in the items' own order."""
    if budget <= 0:
        return []
    if len(items) <= budget:
        return list(items)
    groups = {}
    for index, item in enumerate(items):
        groups.setdefault(item.get("type"), []).append(index)
    types = list(groups)  # in order of first appearance
    total = len(items)
    quota = {t: min(len(groups[t]), max(1, budget * len(groups[t]) // total)) for t in types}
    while sum(quota.values()) < budget:  # what flooring left: to the least represented type
        open_ = [t for t in types if quota[t] < len(groups[t])]
        if not open_:
            break
        quota[max(open_, key=lambda t: len(groups[t]) / quota[t])] += 1
    while sum(quota.values()) > budget:  # more types than the budget: the smallest go first
        quota[max(types, key=lambda t: (quota[t], -len(groups[t])))] -= 1
    picked = []
    for t in types:
        indices, k = groups[t], quota[t]
        picked += [indices[(2 * j + 1) * len(indices) // (2 * k)] for j in range(k)]
    return [items[i] for i in sorted(picked)]


def _test_split(model_ref, workspace):
    """How many rows the run's dataset holds in its test split (0 when it is gone)."""
    from .engine import check_id, load_dataset

    run = read_json(workspace / "runs" / check_id(model_ref.split(":", 1)[1]) / "run.json") or {}
    try:
        _, rows, _ = load_dataset(run.get("dataset", ""), workspace)
    except (FileNotFoundError, ValueError):
        return 0
    return sum(row.get("split") == "test" for row in rows)


def verify_items(model_ref, workspace, rows):
    """The items (decodes) a GGUF is measured on, and, for a sample, what the report says of
    it ({"test_rows", "of_test_rows"}), else None. At FULL_VERIFY_ROWS or more: the first
    `rows` rows of the test split, every decode they make (as always). Fewer: at most `rows`
    decodes, spread over the whole split (spread())."""
    if rows >= FULL_VERIFY_ROWS:
        items, _, _ = test_rows(model_ref, workspace, limit=rows)
        return items, None
    every, _, _ = test_rows(model_ref, workspace, limit=None)
    picked = spread(every, rows)
    return picked, {
        "test_rows": len({item["key"][0] for item in picked}),
        "of_test_rows": _test_split(model_ref, workspace),
    }


class Hasher:
    """The SHA-256 of the checkpoint's weights, read in a daemon thread while the converter
    reads the same files (hashlib lets go of the GIL). stop() ends it at its next block;
    result() waits for it and raises what it raised."""

    def __init__(self, paths, chunk=1 << 22):
        self.paths = list(paths)
        self.chunk = chunk
        self.digests = self.error = self.seconds = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gguf-source-sha256", daemon=True)
        self._thread.start()

    def _run(self):
        started = time.perf_counter()
        try:
            found = {}
            for path in self.paths:
                digest = hashlib.sha256()
                with open(path, "rb") as handle:
                    while block := handle.read(self.chunk):
                        if self._stop.is_set():
                            return
                        digest.update(block)
                found[path.name] = digest.hexdigest()
            self.digests = found
        except BaseException as error:  # noqa: BLE001 - raised again by result()
            self.error = error
        finally:
            self.seconds = round(time.perf_counter() - started, 2)

    def stop(self):
        self._stop.set()

    def result(self):
        self._thread.join()
        if self.error is not None:
            raise self.error
        if self.digests is None:
            raise RuntimeError("the weights' SHA-256 was stopped before it ended")
        return self.digests


PARENT = children.PARENT  # the pid a reference process belongs to
SCRATCH = ".gguf-verify-"  # a measurement's scratch folders in the run's folder (children.scratch)


class Reference:
    """The float32 half of a GGUF's measurement in a process of its own (gguf_reference.py):
    decider.Agent on the merged safetensors, row_logits over the same rows in the same order.

    Its rows, its answer and its log live in `folder` (a scratch folder of the run, never one
    of its outputs). Its output goes to a file, never a pipe. It ends with this process: its
    stdin is a pipe nobody writes to, and it exits when that pipe closes (and, on Linux, on
    PR_SET_PDEATHSIG). end() stops it (SIGTERM, then SIGKILL) and is safe to call twice."""

    def __init__(self, model_dir, items, folder):
        self.folder = Path(folder)
        self.out = self.folder / "logits.json"
        self.log_path = self.folder / "reference.log"
        rows = self.folder / "rows.json"
        rows.write_text(json.dumps([[list(item["ids"]), int(item["n"])] for item in items]))
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            # Its CPU work is small (the GPU computes); the readout has the CPU.
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            PARENT: str(os.getpid()),  # children.WATCH
        }
        self.waited = None
        self.log = open(self.log_path, "wb")
        try:
            self.process = subprocess.Popen(
                [sys.executable, *REFERENCE, str(model_dir), str(rows), str(self.out)],
                stdin=subprocess.PIPE,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(PACKAGE.parent),
            )
        except BaseException:
            self.log.close()
            raise

    def alive(self):
        return self.process.poll() is None

    def check(self, emit):
        """Raises at once when the process has already failed."""
        code = self.process.poll()
        if code not in (None, 0):
            self._fail(emit, f"exit {code}")

    def result(self, items, emit, timeout=REFERENCE_WAIT):
        """{"device", "logits", ...} once the process has ended, with a row of logits for
        every item (each as many as its options); raises otherwise."""
        waiting = time.perf_counter()
        try:
            code = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._fail(emit, f"no answer after {timeout // 60} minutes")
        self.waited = round(time.perf_counter() - waiting, 2)
        if code != 0:
            self._fail(emit, f"exit {code}")
        try:
            data = json.loads(self.out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._fail(emit, "it wrote no logits")
        logits = data.get("logits") if isinstance(data, dict) else None
        if not isinstance(logits, list) or len(logits) != len(items):
            got = len(logits) if isinstance(logits, list) else "no"
            self._fail(emit, f"{got} rows of logits for {len(items)} rows")
        for row, item in zip(logits, items):
            if (
                not isinstance(row, list)
                or len(row) != item["n"]
                or not all(isinstance(x, (int, float)) for x in row)
            ):
                self._fail(emit, "a row of logits does not match its row's options")
        return data

    def tail(self, lines=20):
        try:
            text = self.log_path.read_text(errors="replace")
        except OSError:
            return []
        return [line for line in text.splitlines() if line.strip()][-lines:]

    def _fail(self, emit, why):
        tail = self.tail()
        for line in tail:
            emit("log", message=line)
        raise RuntimeError(
            f"The GGUF's float32 reference failed ({why}): {tail[-1] if tail else 'no output'}"
        )

    def end(self, timeout=10):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for handle in (self.process.stdin, self.log):
            try:
                if handle is not None:
                    handle.close()
            except OSError:
                pass


def verify_gate():
    """Whether the float32 reference runs in a process of its own beside the converter, and
    what that was decided on: {"overlap", "reason", "device", "free_bytes", "needed_bytes",
    "memory"} (the last four when they were read). On a CUDA device (ROCm included) with
    OVERLAP_MEMORY free (telemetry.available_memory: the page cache counts as free) it does
    ("reason": "gates"); anywhere else the measurement runs in this process after the
    conversion (today's order): "device" or "memory" says which gate kept it there.
    SYSTEMONE_STUDIO_SERIAL_VERIFY=1 asks for that too; SYSTEMONE_STUDIO_PARALLEL_VERIFY=1 skips
    both gates (tests); either under its old name, LAYASTUDIO_..., as well, and "reason" names
    the variable that decided. The report records it (timings.overlap_gate); it certifies
    nothing."""
    for key, overlap in (("SERIAL_VERIFY", False), ("PARALLEL_VERIFY", True)):
        if environment.get(key, "").strip() == "1":
            return {"overlap": overlap, "reason": environment.source(key)}
    from .runtime import torch_device

    try:
        device = str(getattr(torch_device(), "type", "cpu"))
    except (ImportError, RuntimeError):
        device = None
    gate = {"overlap": False, "reason": "device", "device": device}
    if device != "cuda":
        return gate
    from .telemetry import available_memory

    memory = available_memory()
    free = memory.get("free")
    gate.update(free_bytes=free, needed_bytes=OVERLAP_MEMORY, memory=memory)
    if free is None or free < OVERLAP_MEMORY:
        gate["reason"] = "memory"
        return gate
    gate.update(overlap=True, reason="gates")
    return gate


def overlap_verify():
    """verify_gate()'s decision alone."""
    return verify_gate()["overlap"]


def export(
    model_ref,
    workspace=WORKSPACE,
    emit=None,
    precision="bf16",
    rows=None,
    *,
    threads=None,
    source_sha256=None,
):
    """runs/<id>/exports/model-<precision>.gguf, measured against the merged checkpoint.

    rows: held-out test rows to measure on (verify_rows(precision) by default). threads:
    llama.cpp's for the readout (runtime.cpu_threads() by default). source_sha256: the
    checkpoint's weights' digests when the caller has them already; else they are read while
    the converter runs. The report (exports/gguf-<precision>.json) also says what each part
    took ("timings"); the result event leaves that and the digests out."""
    from . import decider, kinds
    from .engine import check_id, resolve_model_ref
    from .runtime import cpu_threads, torch_device
    from .telemetry import maxrss_kb

    emit = emit or (lambda *a, **k: None)
    kind, _, value = str(model_ref).partition(":")
    if kind != "run":
        raise ValueError("GGUF exports are made of fine-tuned runs (run:<id>)")
    run_dir = workspace / "runs" / check_id(value)
    model_dir = resolve_model_ref(model_ref, workspace)
    if kinds.detect(model_dir) != kinds.DECIDER:
        raise ValueError("GGUF exports are for decoder fine-tunes (Decider)")
    out_dir = run_dir / "exports"
    target = out_dir / FILE.format(precision=precision)
    children.sweep(run_dir, SCRATCH)  # those of a job killed outright (SIGKILL)
    rows = verify_rows(precision) if rows is None else int(rows)
    threads = int(threads) if threads else cpu_threads()
    started = time.perf_counter()
    gate = verify_gate()
    overlap = gate["overlap"]
    timings = {"overlap": overlap, "overlap_gate": gate, "threads": threads}
    hasher = reference = scratch = None
    items = sample = None
    try:
        if overlap:
            # The rows first (they need only the checkpoint and the dataset), so the float32
            # reference can read them on the GPU while the converter runs.
            timed = time.perf_counter()
            items, sample = verify_items(model_ref, workspace, rows)
            timings["test_rows_s"] = round(time.perf_counter() - timed, 2)
            if items:
                scratch = children.scratch(run_dir, SCRATCH)
                reference = Reference(model_dir, items, scratch.name)
        if source_sha256 is None:
            hasher = Hasher(sorted(model_dir.glob("*.safetensors")))
        timed = time.perf_counter()
        convert(model_dir, target, precision, workspace, emit)
        timings["convert_s"] = round(time.perf_counter() - timed, 2)
        if reference is not None:
            reference.check(emit)  # one that has failed already ends the export now
        timed = time.perf_counter()
        digest = _sha256(target)
        timings["gguf_sha256_s"] = round(time.perf_counter() - timed, 2)
        if hasher is not None:
            timed = time.perf_counter()
            source_sha256 = hasher.result()
            timings["source_sha256_s"] = hasher.seconds
            timings["source_sha256_wait_s"] = round(time.perf_counter() - timed, 2)
        report = {
            "model": model_ref,
            "target": "gguf",
            "precision": precision,
            "path": str(target),
            "created": now(),
            "llama_cpp": LLAMA_CPP_COMMIT,
            "sha256": digest,
            "size_mb": round(target.stat().st_size / 2**20, 1),
            "source_sha256": source_sha256,
            "runs_on": "llama.cpp everywhere (CPU, CUDA, Metal, Vulkan), Ollama, LM Studio, "
            "decider-ai's GGUF engine",
        }
        if not overlap:
            timed = time.perf_counter()
            items, sample = verify_items(model_ref, workspace, rows)
            timings["test_rows_s"] = round(time.perf_counter() - timed, 2)
        if items:
            emit(
                "phase",
                phase="verify",
                message=f"Reading {len(items)} test rows both ways",
                threads=threads,
                device="cpu",
                reference="process" if reference is not None else "in-process",
            )
            prompter = decider.Prompter(decider.Tokens(model_dir), decider.config(model_dir))
            if reference is None:
                exported, ms, used = _read_rows(target, items, prompter, threads, timings)
                import torch

                # The reference is the merged weights in float32: what a GGUF should hold,
                # without bfloat16's own rounding (which alone moves Decider's answers by up to
                # 0.03).
                timed = time.perf_counter()
                agent = decider.Agent(
                    model_dir, backend="torch", device=torch_device(), dtype=torch.float32
                )
                timings["reference_load_s"] = round(time.perf_counter() - timed, 2)
                timed = time.perf_counter()
                expected = agent.row_logits([(it["ids"], it["n"]) for it in items])
                timings["reference_rows_s"] = round(time.perf_counter() - timed, 2)
                del agent
                device = str(getattr(torch_device(), "type", "cpu"))
                if device == "cuda":
                    timings["reference_cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
            else:
                exported, ms, used = _read_rows(
                    target, items, prompter, threads, timings, reference=reference
                )
                answer = reference.result(items, emit)
                timings["reference_wait_s"] = reference.waited
                for key in ("load_s", "rows_s", "cuda_peak_bytes"):
                    if key in answer:
                        timings[f"reference_{key}"] = answer[key]
                expected = answer["logits"]
                device = str(answer.get("device") or "cpu")
            report["verification"] = {
                **compare(expected, exported, items),
                "reference": "the merged safetensors in float32, PyTorch on " + device,
                "ms_per_row_cpu": round(ms, 1),
                "threads": used,
                **(sample or {}),
            }
            emit(
                "log",
                message=f"GGUF {precision} vs the merged weights on {len(items)} test rows: "
                f"{report['verification']['same_answer']}/{len(items)} same answers, max |Δp| "
                f"{report['verification']['max_probability_difference']:.2g}",
            )
    except BaseException:
        if hasher is not None:
            hasher.stop()  # not waited for: a daemon thread, stopped at its next block
        raise
    finally:
        if reference is not None:
            reference.end()
        if scratch is not None:
            scratch.cleanup()
    report["seconds"] = round(time.perf_counter() - started, 1)
    timings["maxrss_kb"] = maxrss_kb()
    report["timings"] = timings
    emit("log", message=_timings_line(report), seconds=report["seconds"], timings=timings)
    write_json(out_dir / f"gguf-{precision}.json", report)
    emit("result", **{k: v for k, v in report.items() if k not in ("source_sha256", "timings")})
    return report


def _read_rows(target, items, prompter, threads, timings, reference=None):
    """The GGUF's letter logits for every item, on the CPU: (logits, ms per row, threads it
    started with). Beside a running reference process the readout leaves it two threads, and
    takes them back once it has ended (checked between rows)."""
    budget = max(1, threads - 2) if reference is not None and reference.alive() else threads
    timed = time.perf_counter()
    reader = Reader(target, threads=budget)
    timings["reader_load_s"] = round(time.perf_counter() - timed, 2)
    used = reader.threads or budget
    timings["reader_threads"] = {"start": used}
    try:
        timed = time.perf_counter()
        exported = []
        for item in items:
            if budget != threads and not reference.alive():
                reader.set_threads(threads)
                budget = threads
                timings["reader_threads"].update(raised_to=reader.threads, at_row=len(exported))
            exported.append(reader.logits(item["ids"], prompter.label_ids[: item["n"]]))
        seconds = time.perf_counter() - timed
    finally:
        reader.close()
    timings["reader_rows_s"] = round(seconds, 2)
    return exported, seconds * 1000 / len(items), used


def _timings_line(report):
    t = report.get("timings") or {}
    parts = [f"convert {t.get('convert_s')} s"]
    if "reader_rows_s" in t:
        parts.append(
            f"readout {t['reader_rows_s']} s ({(t.get('reader_threads') or {}).get('start')} threads)"
        )
    if "reference_rows_s" in t:
        where = "its own process" if t.get("overlap") else "this process"
        parts.append(
            f"float32 reference {t.get('reference_load_s')} + {t['reference_rows_s']} s ({where})"
        )
    return f"GGUF export: {report['seconds']} s: " + ", ".join(parts)
