"""The float32 reference of a GGUF export's measurement, in a process of its own.

    python -m systemone_studio.gguf_reference <model_dir> <rows.json> <out.json>

gguf.py starts it before the converter, on a CUDA machine, so the GPU reads the rows while the
converter and llama.cpp use the CPU. It is the reference the measurement always had: the merged
safetensors in float32 (decider.Agent, PyTorch, this machine's device), row_logits over the
same (ids, n) rows in the same order and chunks. It writes {"device", "logits", ...} to
out.json (written whole, then renamed), its logits as Python floats, unrounded.

It ends with the job that started it (children.WATCH): when its stdin (a pipe the job never
writes to) closes, and on Linux when its parent dies (PR_SET_PDEATHSIG), so a job killed
outright leaves no float32 model on the GPU. SIGTERM ends it as it ends any Python process.
"""

import json
import os
import sys
import time
from pathlib import Path

from .children import watch_parent


def main(argv):
    if len(argv) != 3:
        sys.exit(
            "usage: python -m systemone_studio.gguf_reference <model_dir> <rows.json> <out.json>"
        )
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
