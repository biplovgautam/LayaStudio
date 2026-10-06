"""The float32 reference of a GGUF export's measurement, in a process of its own.

    python -m layastudio.gguf_reference <model_dir> <rows.json> <out.json>

gguf.py starts it before the converter, on a CUDA machine, so the GPU reads the rows while the
converter and llama.cpp use the CPU. It is the reference the measurement always had: the merged
safetensors in float32 (decider.Agent, PyTorch, this machine's device), row_logits over the
same (ids, n) rows in the same order and chunks. It writes {"device", "logits", ...} to
out.json (written whole, then renamed), its logits as Python floats, unrounded.

It ends with the job that started it: when its stdin (a pipe the job never writes to) closes,
and on Linux when its parent dies (PR_SET_PDEATHSIG), so a job killed outright leaves no float32
model on the GPU. SIGTERM ends it as it ends any Python process.
"""

import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

PARENT = "LAYASTUDIO_REFERENCE_PARENT"
PR_SET_PDEATHSIG = 1
SIGKILL = 9


def watch_parent():
    """Exit as soon as the process that started this one is gone."""
    parent = os.environ.get(PARENT, "")
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, SIGKILL)
        except (OSError, AttributeError):
            pass
    if parent.isdigit() and os.getppid() != int(parent):
        os._exit(1)  # the parent ended before this process was watching

    try:
        stdin = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        return

    def wait_for_eof():
        # The file descriptor, not sys.stdin: a daemon thread blocked in a buffered read holds
        # its lock, and the interpreter cannot shut down past that.
        try:
            while os.read(stdin, 4096):
                pass
        except OSError:
            pass
        os._exit(1)

    threading.Thread(target=wait_for_eof, name="parent-watch", daemon=True).start()


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: python -m layastudio.gguf_reference <model_dir> <rows.json> <out.json>")
    watch_parent()
    model_dir, rows_path, out_path = (Path(a) for a in argv)
    import torch

    from . import decider
    from .runtime import torch_device

    rows = [(list(ids), int(n)) for ids, n in json.loads(rows_path.read_text())]
    device = torch_device()
    started = time.perf_counter()
    agent = decider.Agent(model_dir, backend="torch", device=device, dtype=torch.float32)
    loaded = time.perf_counter()
    logits = agent.row_logits(rows)
    done = time.perf_counter()
    kind = str(getattr(device, "type", "cpu"))
    answer = {
        "device": kind,
        "logits": [[float(x) for x in row] for row in logits],
        "load_s": round(loaded - started, 2),
        "rows_s": round(done - loaded, 2),
    }
    if kind == "cuda":
        answer["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
    partial = out_path.with_name(out_path.name + ".partial")
    partial.write_text(json.dumps(answer), encoding="utf-8")
    os.replace(partial, out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
