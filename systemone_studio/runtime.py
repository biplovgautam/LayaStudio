"""Which machine-learning stack this studio runs on, and the few calls that differ.

Two backends train and evaluate the same checkpoints (Laya, Julia 1 and Decider):

- **mlx** on Apple silicon: the native laya-mlx runtime, fastest on a Mac.
- **torch** everywhere else: the upstream PyTorch `laya` package on NVIDIA (CUDA),
  AMD (ROCm on Linux, DirectML on Windows), Intel (XPU) or the CPU. Also usable on a
  Mac, through Metal (MPS), when asked for.

Checkpoints are interchangeable: a model trained on one backend loads on the other,
because both write the upstream PyTorch parameter names.

$SYSTEMONE_STUDIO_BACKEND (mlx | torch) and $SYSTEMONE_STUDIO_DEVICE (cuda, mps, xpu, cpu,
...) override the choice, as their old names $LAYASTUDIO_BACKEND and $LAYASTUDIO_DEVICE still do
(environment.py: the new name wins).

cpu_threads() is how many CPU threads a CPU-bound step (a NoulXP package's steps, a GGUF's
readout) should use here: a container's CPU quota (cgroup cpu.max) when it has one, so the
steps neither oversubscribe the quota nor leave it idle. $SYSTEMONE_STUDIO_THREADS (or
$LAYASTUDIO_THREADS) overrides it.
"""

import importlib.util
import math
import os
import platform
import subprocess
import sys
from pathlib import Path, PurePosixPath

from . import environment


def _has(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def apple_silicon():
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def backend():
    """'mlx' or 'torch' — the stack jobs on this machine use."""
    wanted = environment.get("BACKEND", "").strip().lower()
    if wanted in ("mlx", "torch"):
        return wanted
    if apple_silicon() and _has("mlx") and _has("laya_mlx"):
        return "mlx"
    return "torch"


def torch_device():
    """The best PyTorch device here: cuda (NVIDIA, and AMD ROCm builds), xpu, mps,
    DirectML, or cpu."""
    import torch

    wanted = environment.get("DEVICE", "").strip().lower()
    if wanted == "directml" or (not wanted and _directml_only(torch)):
        import torch_directml  # type: ignore[import-not-found]

        return torch_directml.device()
    if wanted:
        return torch.device(wanted)
    if torch.cuda.is_available():  # ROCm builds of PyTorch answer here too
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _directml_only(torch):
    return (
        not torch.cuda.is_available()
        and not (hasattr(torch, "xpu") and torch.xpu.is_available())
        and _has("torch_directml")
    )


def device_label(device):
    kind = getattr(device, "type", str(device))
    if kind == "cuda":
        import torch

        name = torch.cuda.get_device_name(device)
        return f"{'ROCm' if getattr(torch.version, 'hip', None) else 'CUDA'} · {name}"
    return {
        "xpu": "Intel XPU",
        "mps": "Apple Metal (MPS)",
        "cpu": "CPU",
        "privateuseone": "DirectML",
    }.get(kind, kind)


def torch_checkpoint(model_dir):
    """The checkpoint as the upstream PyTorch package expects it.

    MLX ports store MLX parameter names (scorer.layers.0.weight, ...in_proj.weight).
    Those are renamed once into a cached copy beside the workspace; checkpoints already
    in upstream names — every studio run, convaiinnovations/laya — are used as they are.
    """
    import hashlib
    import shutil
    from pathlib import Path

    from .engine import WORKSPACE, safetensors_header, upstream_name

    model_dir = Path(model_dir)
    weights = model_dir / "model.safetensors"
    names = list(safetensors_header(weights))
    if all(upstream_name(n) == n for n in names):
        return model_dir
    stamp = f"{weights.resolve()}|{weights.stat().st_size}|{weights.stat().st_mtime_ns}"
    target = WORKSPACE / "converted" / hashlib.sha256(stamp.encode()).hexdigest()[:16]
    if (target / "model.safetensors").exists():
        return target
    from safetensors.numpy import load_file, save_file

    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(model_dir, partial, ignore=shutil.ignore_patterns("model.safetensors"))
    tensors = {upstream_name(k): v for k, v in load_file(str(weights)).items()}
    save_file(tensors, str(partial / "model.safetensors"), metadata={"format": "pt"})
    if target.exists():
        shutil.rmtree(target)
    partial.rename(target)
    return target


def load_agent(model_dir, batch_size=16):
    """An inference agent for a local checkpoint, on this machine's backend.

    Laya through laya-mlx or laya, Julia 1 through julia.Agent, Decider through
    decider.Agent; each answers `predict(state, questions)` in the System One format."""
    from . import kinds

    kind = kinds.detect(model_dir)
    if kind == kinds.JULIA:
        from .julia import Agent

        if backend() == "mlx":
            return Agent(model_dir, backend="mlx")
        return Agent(model_dir, backend="torch", device=torch_device())
    if kind == kinds.DECIDER:
        from .decider import Agent

        if backend() == "mlx":
            return Agent(model_dir, backend="mlx")
        return Agent(model_dir, backend="torch", device=torch_device())
    if backend() == "mlx":
        import laya_mlx

        return laya_mlx.load(str(model_dir), batch_size=batch_size)
    import laya

    device = torch_device()
    return laya.load(str(torch_checkpoint(model_dir)), device=str(getattr(device, "type", "cpu")))


def clear_cache():
    """Hand freed accelerator memory back between jobs."""
    import gc

    gc.collect()
    if backend() == "mlx":
        import mlx.core as mx

        mx.clear_cache()
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def limit_cache():
    """Cap MLX's buffer cache (see engine); nothing to do on PyTorch."""
    if backend() != "mlx":
        return
    import mlx.core as mx

    total = mx.device_info()["memory_size"]
    mx.set_cache_limit(int(min(2 * 2**30, 0.1 * total)))


def describe():
    """What the studio will train with, for the machine panel."""
    info = {
        "backend": backend(),
        "mlx": None,
        "laya_mlx": None,
        "torch": None,
        "laya": None,
        "device": None,
    }
    for key, module in (
        ("mlx", "mlx.core"),
        ("laya_mlx", "laya_mlx"),
        ("torch", "torch"),
        ("laya", "laya"),
    ):
        if _has(module.split(".")[0]):
            try:
                imported = importlib.import_module(module)
                info[key] = getattr(imported, "__version__", "installed")
            except Exception as error:  # noqa: BLE001 - a broken install is still worth reporting
                info[key] = f"broken: {error}"
    if info["backend"] == "torch" and info["torch"] and not str(info["torch"]).startswith("broken"):
        try:
            info["device"] = device_label(torch_device())
        except Exception as error:  # noqa: BLE001
            info["device"] = f"unavailable: {error}"
    elif info["backend"] == "mlx":
        info["device"] = "Apple MLX (Metal)"
    return info


# ----------------------------------------------------------------------------- CPU threads

CGROUP = "/sys/fs/cgroup"
PROC_CGROUP = "/proc/self/cgroup"
MAX_THREADS = 32
# Without a quota: today's default for the check (noulxp: min(8, CPUs)). A machine's CPU count
# also counts hyperthreads and, on a Mac, efficiency cores, and a host can hold its quota in a
# parent cgroup a container cannot see: more threads there is no faster, often slower.
UNQUOTED_THREADS = 8
_warned = set()


def _read(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (OSError, ValueError):  # missing, unreadable, or not text
        return None


def _warn_once(message):
    if message not in _warned:
        _warned.add(message)
        print(message, file=sys.stderr)


def _cgroups(proc=PROC_CGROUP):
    """{controller: path} of this process from /proc/self/cgroup; "" for cgroup v2's line
    ("0::<path>"). Empty when there is no such file (macOS, Windows)."""
    found = {}
    for line in (_read(proc) or "").splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        number, controllers, path = parts
        if number == "0" and not controllers:
            found[""] = path
        for controller in filter(None, controllers.split(",")):
            found[controller] = path
    return found


def _inside(root, path):
    """root/<path>, never above root: a path with "..", as a cgroup outside the container's
    namespace shows, is root itself."""
    parts = PurePosixPath(path or "/").parts
    if ".." in parts:
        return Path(root)
    return Path(root).joinpath(*[p for p in parts if p not in ("/", ".")])


def _v2_quota(root, path):
    """The smallest cgroup v2 cpu.max quota, in CPUs, from the process's own cgroup up to the
    root; None when no level limits it ("max")."""
    root = Path(root)
    folder, best = _inside(root, path), None
    while True:
        fields = (_read(folder / "cpu.max") or "").split()
        if fields and fields[0] != "max":
            try:
                quota = int(fields[0])
                period = int(fields[1]) if len(fields) > 1 else 100000
            except ValueError:
                quota = period = 0
            if quota > 0 and period > 0:
                best = quota / period if best is None else min(best, quota / period)
        if folder == root or root not in folder.parents:
            return best
        folder = folder.parent


def _v1_quota(root, path):
    """cgroup v1's cpu.cfs_quota_us / cpu.cfs_period_us, in CPUs; None when unlimited (-1)."""
    best = None
    for mount in ("cpu", "cpu,cpuacct"):
        folders = [Path(root) / mount]
        if path and path != "/":
            folders.insert(0, _inside(Path(root) / mount, path))
        for folder in folders:
            try:
                quota = int((_read(folder / "cpu.cfs_quota_us") or "").strip())
                period = int((_read(folder / "cpu.cfs_period_us") or "").strip())
            except ValueError:
                continue
            if quota > 0 and period > 0:
                best = quota / period if best is None else min(best, quota / period)
    return best


def affinity():
    """CPUs this process may run on."""
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _unquoted():
    """Threads where no quota is found (see UNQUOTED_THREADS): the affinity on Linux, a Mac's
    performance cores, capped at 8."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, min(UNQUOTED_THREADS, affinity()))
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            cores = int(out.stdout.strip())
            if cores > 0:
                return min(UNQUOTED_THREADS, cores)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return max(1, min(UNQUOTED_THREADS, os.cpu_count() or 1))


def cpu_budget(root=CGROUP, proc=PROC_CGROUP, environ=None):
    """{"threads", "quota", "source", "affinity"}: the threads a CPU-bound step uses here.

    In order: $SYSTEMONE_STUDIO_THREADS, else $LAYASTUDIO_THREADS (a positive integer;
    anything else is ignored, said once);
    the container's quota, cgroup v2 cpu.max (the smallest from this process's cgroup up to
    the root) or cgroup v1's cfs quota, floored (8.5 CPUs gives 8, under 1 gives 1), no more
    than the CPUs this process may run on, and at most 32; else min(8, CPUs) (see
    UNQUOTED_THREADS). quota is the raw quota in CPUs, or None. Never raises."""
    environ = os.environ if environ is None else environ
    cpus = affinity()
    name = environment.source("THREADS", environ)
    raw = str(environ.get(name, "") or "").strip() if name else ""
    if raw:
        try:
            wanted = int(raw)
        except ValueError:
            wanted = 0
        if wanted > 0:
            return {"threads": wanted, "quota": None, "source": "env", "affinity": cpus}
        _warn_once(f"{name}={raw!r} is not a positive integer; ignored")
    paths = _cgroups(proc)
    quota, source = _v2_quota(root, paths.get("", "/")), "cgroup2"
    if quota is None:
        quota, source = _v1_quota(root, paths.get("cpu") or paths.get("cpuacct")), "cgroup1"
    if quota is None:
        return {"threads": _unquoted(), "quota": None, "source": "fallback", "affinity": cpus}
    threads = max(1, min(math.floor(quota), cpus, MAX_THREADS))
    return {"threads": threads, "quota": round(quota, 4), "source": source, "affinity": cpus}


def cpu_threads(**kwargs):
    """The threads a CPU-bound step uses here (cpu_budget)."""
    return cpu_budget(**kwargs)["threads"]
