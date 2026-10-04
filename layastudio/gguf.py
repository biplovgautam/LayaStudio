"""GGUF exports of Decider fine-tunes, through llama.cpp's own converter at a pinned commit.

    python -m layastudio.export run:<id> --target gguf                 # bf16, the merged weights
    python -m layastudio.export run:<id> --target gguf --precision q8_0

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
"""

import ctypes
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from .engine import WORKSPACE, now, read_json, write_json

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


def tools_dir(workspace=WORKSPACE):
    """Where downloaded tools live: beside the workspace, never inside a run."""
    return Path(os.environ.get("LAYASTUDIO_TOOLS") or Path(workspace).parent / "tools")


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


def convert(model_dir, out_file, precision="bf16", workspace=WORKSPACE, emit=None):
    """convert_hf_to_gguf.py on a Decider checkpoint folder, in a process of its own."""
    emit = emit or (lambda *a, **k: None)
    if precision not in PRECISIONS:
        raise ValueError(f"GGUF precision is one of {PRECISIONS}")
    tool = converter(workspace, emit)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tool / "gguf-py"), os.environ.get("PYTHONPATH", "")]),
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "NO_LOCAL_GGUF": "",
    }
    command = [
        sys.executable,
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
    emit("phase", phase="convert", message=f"Converting to GGUF ({precision}) with llama.cpp")
    process = subprocess.run(command, env=env, capture_output=True, text=True, errors="replace")
    for line in (process.stdout + process.stderr).splitlines()[-12:]:
        if line.strip():
            emit("log", message=line.strip())
    if process.returncode or not out_file.is_file():
        tail = (process.stderr or process.stdout).strip().splitlines()[-1:] or ["no output"]
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
        self.memory = lc.llama_get_memory(self.ctx)
        self.n_vocab = lc.llama_vocab_n_tokens(lc.llama_model_get_vocab(self.model))
        self.batch = lc.llama_batch_init(N_CTX, 0, 1)

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


def export(model_ref, workspace=WORKSPACE, emit=None, precision="bf16", rows=60):
    """runs/<id>/exports/model-<precision>.gguf, measured against the merged checkpoint."""
    from . import decider, kinds
    from .engine import check_id, resolve_model_ref
    from .runtime import torch_device

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
    started = time.perf_counter()
    convert(model_dir, target, precision, workspace, emit)
    report = {
        "model": model_ref,
        "target": "gguf",
        "precision": precision,
        "path": str(target),
        "created": now(),
        "llama_cpp": LLAMA_CPP_COMMIT,
        "sha256": _sha256(target),
        "size_mb": round(target.stat().st_size / 2**20, 1),
        "source_sha256": {p.name: _sha256(p) for p in sorted(model_dir.glob("*.safetensors"))},
        "runs_on": "llama.cpp everywhere (CPU, CUDA, Metal, Vulkan), Ollama, LM Studio, "
        "decider-ai's GGUF engine",
    }
    items, _, _ = test_rows(model_ref, workspace, limit=rows)
    if items:
        emit("phase", phase="verify", message=f"Reading {len(items)} test rows both ways")
        prompter = decider.Prompter(decider.Tokens(model_dir), decider.config(model_dir))
        reader = Reader(target)
        try:
            timed = time.perf_counter()
            exported = [reader.logits(it["ids"], prompter.label_ids[: it["n"]]) for it in items]
            ms = (time.perf_counter() - timed) * 1000 / len(items)
        finally:
            reader.close()
        import torch

        # The reference is the merged weights in float32: what a GGUF should hold, without
        # bfloat16's own rounding (which alone moves Decider's answers by up to 0.03).
        agent = decider.Agent(
            model_dir, backend="torch", device=torch_device(), dtype=torch.float32
        )
        reference = agent.row_logits([(it["ids"], it["n"]) for it in items])
        del agent
        report["verification"] = {
            **compare(reference, exported, items),
            "reference": "the merged safetensors in float32, PyTorch on "
            + str(getattr(torch_device(), "type", "cpu")),
            "ms_per_row_cpu": round(ms, 1),
        }
        emit(
            "log",
            message=f"GGUF {precision} vs the merged weights on {len(items)} test rows: "
            f"{report['verification']['same_answer']}/{len(items)} same answers, max |Δp| "
            f"{report['verification']['max_probability_difference']:.2g}",
        )
    report["seconds"] = round(time.perf_counter() - started, 1)
    write_json(out_dir / f"gguf-{precision}.json", report)
    emit("result", **{k: v for k, v in report.items() if k != "source_sha256"})
    return report
