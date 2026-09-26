"""Which machine-learning stack this studio runs on, and the few calls that differ.

Two backends train and evaluate the same Laya checkpoints:

- **mlx** on Apple silicon: the native laya-mlx runtime, fastest on a Mac.
- **torch** everywhere else: the upstream PyTorch `laya` package on NVIDIA (CUDA),
  AMD (ROCm on Linux, DirectML on Windows), Intel (XPU) or the CPU. Also usable on a
  Mac, through Metal (MPS), when asked for.

Checkpoints are interchangeable: a model trained on one backend loads on the other,
because both write the upstream PyTorch parameter names.

$LAYASTUDIO_BACKEND (mlx | torch) and $LAYASTUDIO_DEVICE (cuda, mps, xpu, cpu, ...)
override the choice.
"""

import importlib.util
import os
import platform


def _has(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def apple_silicon():
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def backend():
    """'mlx' or 'torch' — the stack jobs on this machine use."""
    wanted = os.environ.get("LAYASTUDIO_BACKEND", "").strip().lower()
    if wanted in ("mlx", "torch"):
        return wanted
    if apple_silicon() and _has("mlx") and _has("laya_mlx"):
        return "mlx"
    return "torch"


def torch_device():
    """The best PyTorch device here: cuda (NVIDIA, and AMD ROCm builds), xpu, mps,
    DirectML, or cpu."""
    import torch

    wanted = os.environ.get("LAYASTUDIO_DEVICE", "").strip().lower()
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
    """An inference agent for a local Laya checkpoint, on this machine's backend."""
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
