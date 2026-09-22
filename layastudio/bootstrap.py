"""Everything that has to be true before the first fine-tuning run, done in the background.

The server answers requests immediately; this module runs beside it and reports progress to
the page, so `layastudio` is the only command anyone has to type:

  1. machine   - what this Mac is (chip, cores, memory, macOS) and what it can train
  2. runtime   - MLX and the laya-mlx runtime are importable, with their versions
  3. workspace - the folders for datasets, runs and checkpoints exist, with free space
  4. model     - a base checkpoint is in the Hugging Face cache, downloading it if not
  5. examples  - public example datasets are fetched from their source URLs (nothing
                 ships in git), so the Datasets page is not empty on a fresh clone

Steps never raise: a failure is recorded with a message the page can show, and the parts
that still work stay usable.
"""

import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import engine, examples

DEFAULT_MODEL = "aac6fef/laya-mlx"


@dataclass
class Step:
    key: str
    title: str
    state: str = "pending"  # pending | running | done | warning | failed | skipped
    detail: str = ""
    progress: float | None = None
    started: float | None = None
    seconds: float | None = None


@dataclass
class Machine:
    """What this particular Mac is, detected at startup - never hard-coded."""

    ok: bool = False
    chip: str = "unknown"
    cores: int = 0
    gpu_cores: int | None = None
    memory_gb: float = 0.0
    usable_gpu_gb: float = 0.0
    os: str = ""
    python: str = ""
    mlx: str | None = None
    laya_mlx: str | None = None
    disk_free_gb: float = 0.0
    note: str = ""
    recommended: dict = field(default_factory=dict)


def _sysctl(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_machine(workspace):
    machine = Machine(
        os=(
            f"macOS {platform.mac_ver()[0]}"
            if platform.system() == "Darwin" and platform.mac_ver()[0]
            else f"{platform.system()} {platform.release()}"
        ),
        python=platform.python_version(),
        cores=os.cpu_count() or 0,
        disk_free_gb=round(shutil.disk_usage(workspace).free / 2**30, 1),
    )
    if platform.system() == "Darwin":
        machine.chip = _sysctl("machdep.cpu.brand_string").removeprefix("Apple ") or "Mac"
        memory = _sysctl("hw.memsize")
        machine.memory_gb = round(int(memory) / 2**30) if memory.isdigit() else 0
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        machine.note = (
            "LayaStudio trains on Apple silicon (M1 or newer). Checkpoints it produces already "
            "run elsewhere through the upstream PyTorch runtime; training on NVIDIA and Linux "
            "is planned."
        )
        return machine
    try:
        import mlx.core as mx

        info = mx.device_info()
        machine.mlx = mx.__version__
        machine.chip = info.get("device_name") or machine.chip
        machine.memory_gb = round(info.get("memory_size", 0) / 2**30) or machine.memory_gb
        machine.usable_gpu_gb = round(info.get("max_recommended_working_set_size", 0) / 2**30, 1)
        machine.ok = True
    except Exception as error:  # noqa: BLE001 - reported to the page, never fatal
        machine.note = f"MLX is not usable here: {error}"
        return machine
    try:
        import laya_mlx

        machine.laya_mlx = laya_mlx.__version__
    except Exception as error:  # noqa: BLE001
        machine.ok = False
        machine.note = f"The laya-mlx runtime is missing: {error}. Run: uv sync"
        return machine
    memory = machine.memory_gb
    machine.recommended = {
        "batch_size": 4 if memory <= 8 else 8 if memory < 32 else 16,
        "method": "lora",
        "lora_layers": 8 if memory <= 8 else 0,
        "note": (
            f"Defaults tuned for {memory} GB: "
            + ("small batches and top-layer adapters." if memory <= 8 else "full LoRA adapters.")
        ),
    }
    return machine


def repo_size(repo_id):
    """Total download size of a checkpoint, for an honest progress bar."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    wanted = ("model.safetensors", "rl_agent_config.json")
    return sum(
        f.size or 0
        for f in info.siblings
        if f.rfilename in wanted or f.rfilename.startswith(("encoder/", "tokenizer/"))
    )


def cache_bytes(repo_id):
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"))
    total = 0
    for root, _, files in os.walk(folder):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class Bootstrap:
    """Runs the startup steps on a background thread and exposes their progress."""

    def __init__(self, workspace, *, model=DEFAULT_MODEL, download=True, fetch_examples=True):
        self.workspace = workspace
        self.model = model
        self.download = download
        self.fetch_examples = fetch_examples
        self.machine = Machine()
        self.lock = threading.Lock()
        self.started = time.time()
        self.steps = [
            Step("machine", "Checking this Mac"),
            Step("runtime", "Loading the MLX runtime"),
            Step("workspace", "Preparing the workspace"),
            Step("model", "Getting a base model"),
            Step("demos", "Fetching a ready-made fine-tune"),
            Step("examples", "Fetching example datasets"),
        ]

    # --- reporting

    def step(self, key):
        return next(s for s in self.steps if s.key == key)

    def set(self, key, state=None, detail=None, progress=None):
        with self.lock:
            step = self.step(key)
            if state:
                if state == "running" and step.started is None:
                    step.started = time.time()
                if state not in ("pending", "running") and step.started:
                    step.seconds = round(time.time() - step.started, 1)
                step.state = state
            if detail is not None:
                step.detail = detail
            step.progress = progress

    def snapshot(self):
        with self.lock:
            steps = [asdict(s) for s in self.steps]
            machine = asdict(self.machine)
        done = all(s["state"] in ("done", "warning", "skipped") for s in steps)
        failed = any(s["state"] == "failed" for s in steps)
        return {
            "state": "failed" if failed else "ready" if done else "running",
            "ready": done and not failed,
            "steps": steps,
            "machine": machine,
            "seconds": round(time.time() - self.started, 1),
        }

    @property
    def ready(self):
        return self.snapshot()["ready"]

    # --- the steps

    def start(self):
        threading.Thread(target=self.run, name="bootstrap", daemon=True).start()
        return self

    def run(self):
        try:
            self._machine()
            self._runtime()
            self._workspace()
            self._model()
            self._demos()
            self._examples()
        except Exception as error:  # noqa: BLE001 - the page shows what went wrong
            self.set("machine", "failed", f"Setup stopped: {type(error).__name__}: {error}")

    def _machine(self):
        self.set("machine", "running")
        (self.workspace).mkdir(parents=True, exist_ok=True)
        self.machine = detect_machine(self.workspace)
        m = self.machine
        if not m.ok and m.note and "laya-mlx" not in m.note:
            self.set("machine", "failed", m.note)
            for key in ("runtime", "workspace", "model", "examples"):
                self.set(key, "skipped", "Needs an Apple silicon Mac")
            raise SystemExit
        self.set(
            "machine", "done", f"{m.chip} · {m.cores} cores · {m.memory_gb} GB memory · {m.os}"
        )

    def _runtime(self):
        self.set("runtime", "running")
        m = self.machine
        if not m.laya_mlx:
            self.set("runtime", "failed", m.note or "laya-mlx is not installed. Run: uv sync")
            for key in ("model", "examples"):
                self.set(key, "skipped", "Runtime missing")
            raise SystemExit
        self.set(
            "runtime",
            "done",
            f"laya-mlx {m.laya_mlx} · MLX {m.mlx} · Python {m.python} · "
            f"{m.usable_gpu_gb} GB usable by the GPU",
        )

    def _workspace(self):
        self.set("workspace", "running")
        for name in ("datasets", "runs", "evals", "jobs"):
            (self.workspace / name).mkdir(parents=True, exist_ok=True)
        datasets = len(list((self.workspace / "datasets").glob("*/meta.json")))
        runs = len(list((self.workspace / "runs").glob("*/run.json")))
        free = round(shutil.disk_usage(self.workspace).free / 2**30, 1)
        home = str(self.workspace).replace(str(Path.home()), "~", 1)
        detail = f"{home} · {datasets} datasets · {runs} runs · {free} GB free"
        self.set(
            "workspace",
            "warning" if free < 5 else "done",
            detail + (" · low disk space" if free < 5 else ""),
        )

    def _model(self):
        self.set("model", "running", f"Looking for {self.model}")
        if engine.hub_cached(self.model):
            self.set("model", "done", f"{self.model} is ready")
            return
        if not self.download:
            self.set(
                "model", "warning", f"{self.model} is not downloaded. Get it on the Models page."
            )
            return
        try:
            total = repo_size(self.model)
        except Exception as error:  # noqa: BLE001
            self.set(
                "model",
                "warning",
                f"Cannot reach Hugging Face ({error}). Download the model later "
                "from the Models page.",
            )
            return
        done = threading.Event()
        result = {}

        def fetch():
            from huggingface_hub import snapshot_download

            try:
                snapshot_download(self.model, allow_patterns=list(engine.CHECKPOINT_FILES))
            except Exception as error:  # noqa: BLE001
                result["error"] = error
            finally:
                done.set()

        threading.Thread(target=fetch, name="model-download", daemon=True).start()
        start = cache_bytes(self.model)
        while not done.wait(1.0):
            got = max(0, cache_bytes(self.model) - start)
            self.set(
                "model",
                "running",
                f"Downloading {self.model} — {got / 2**20:.0f} of {total / 2**20:.0f} MB",
                min(0.99, got / total) if total else None,
            )
        if result.get("error"):
            self.set(
                "model", "warning", f"Download failed: {result['error']}. Retry on the Models page."
            )
            return
        self.set("model", "done", f"{self.model} downloaded ({total / 2**20:.0f} MB)")

    def _demos(self):
        """A published fine-tune, so the Snake arena has something to show immediately."""
        if not engine.DEMO_MODELS:
            self.set("demos", "skipped", "No demo models configured")
            return
        self.set("demos", "running")
        ready, missing = [], []
        for repo in engine.DEMO_MODELS:
            if engine.hub_cached(repo):
                ready.append(repo)
                continue
            if not self.download:
                missing.append(repo)
                continue
            try:
                from huggingface_hub import snapshot_download

                self.set("demos", "running", f"Downloading {repo}")
                snapshot_download(repo, allow_patterns=list(engine.CHECKPOINT_FILES))
                ready.append(repo)
            except Exception as error:  # noqa: BLE001 - not published yet, or offline
                missing.append(f"{repo} ({type(error).__name__})")
        if ready and not missing:
            self.set("demos", "done", f"{', '.join(ready)} ready — try the Snake arena")
        elif ready:
            self.set(
                "demos",
                "warning",
                f"{', '.join(ready)} ready · could not fetch " + ", ".join(missing),
            )
        else:
            self.set(
                "demos",
                "warning",
                "Could not fetch "
                + ", ".join(missing)
                + ". Fine-tune your own, or publish one with layastudio.publish.",
            )

    def _examples(self):
        if not self.fetch_examples:
            self.set("examples", "skipped", "Disabled with --no-examples")
            return
        self.set("examples", "running")
        have = {
            (engine.read_json(p / "meta.json") or {}).get("example")
            for p in (self.workspace / "datasets").glob("*")
        }
        added, failed = [], []
        catalog = examples.catalog()
        for index, (name, spec) in enumerate(catalog.items(), start=1):
            if name in have:
                continue
            self.set("examples", "running", f"Fetching {spec['title']}", (index - 1) / len(catalog))
            try:
                examples.fetch_example(name, lambda *a, **k: None, self.workspace)
                added.append(spec["title"])
            except Exception as error:  # noqa: BLE001 - offline is not fatal
                failed.append(f"{spec['title']} ({type(error).__name__})")
        detail = f"{len(have & set(catalog)) + len(added)} of {len(catalog)} example datasets ready"
        if failed:
            self.set("examples", "warning", detail + " · could not fetch " + ", ".join(failed))
        else:
            self.set("examples", "done", detail + " · your own data stays local")
