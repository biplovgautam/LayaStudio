"""Laya fine-tuning on PyTorch: the same training as the MLX engine, on any hardware.

A line-for-line port of engine.fit for machines without MLX — NVIDIA (CUDA), AMD (ROCm,
or DirectML on Windows), Intel (XPU), a Mac through Metal (MPS) when asked for, or the
CPU. Same LoRA placement, same losses (proper scoring rule, RLCD, cross-entropy), same
schedule, early stopping, calibration and checkpoint format, so a run trained here loads
in laya-mlx and a run trained on a Mac loads here.

Datasets, tokenization, calibration and metrics are shared with engine.py; only the
tensor code differs.
"""

import math
import random
import shutil
import time
from pathlib import Path

from . import runtime

try:  # not on Windows
    import resource
except ImportError:  # pragma: no cover
    resource = None
from .engine import (
    ENCODER_LINEARS,
    QTYPES,
    WORKSPACE,
    calibrate,
    calibrated_ece,
    check_id,
    class_weights,
    encode_items,
    load_dataset,
    make_batches,
    nll_at,
    now,
    read_json,
    resolve_model_ref,
    safetensors_header,
    upstream_name,
    write_json,
)


def lora_linear():
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """y = x W^T + (alpha / r) * x A B. Only A and B train; the base layer stays frozen."""

        def __init__(self, base, rank, alpha, dropout):
            super().__init__()
            out_dims, in_dims = base.weight.shape
            bound = 1 / math.sqrt(in_dims)
            self.base = base
            device = base.weight.device
            self.lora_a = nn.Parameter(
                torch.empty(in_dims, rank, device=device).uniform_(-bound, bound)
            )
            self.lora_b = nn.Parameter(torch.zeros(rank, out_dims, device=device))
            self.scale = alpha / rank
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            y = self.base(x)
            h = self.dropout(x.float())
            return y + (self.scale * ((h @ self.lora_a) @ self.lora_b)).to(y.dtype)

        def fused(self, dtype):
            with torch.no_grad():
                delta = self.scale * (self.lora_a @ self.lora_b).T
                layer = nn.Linear(delta.shape[1], delta.shape[0], bias=self.base.bias is not None)
                layer.weight.copy_((self.base.weight.float() + delta).to(dtype))
                if self.base.bias is not None:
                    layer.bias.copy_(self.base.bias.to(dtype))
            return layer.to(device=self.base.weight.device, dtype=dtype)

    return LoRALinear


def frozen_dtype(torch, device, precision):
    """The dtype frozen encoder weights are held in. bfloat16 halves memory where the
    hardware does it well; the CPU and DirectML stay in float32."""
    if precision != "bfloat16":
        return torch.float32
    kind = getattr(device, "type", "cpu")
    if kind == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if kind in ("xpu", "mps"):
        return torch.bfloat16
    return torch.float32


def load_training_model(model_dir, hp, device):
    """The PyTorch decision model with frozen and trainable parts for the chosen method."""
    import torch
    from laya.common import build_model
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    cfg = read_json(model_dir / "rl_agent_config.json")
    # The fused TransformerEncoderLayer fast path does not train; the plain path does.
    torch.backends.mha.set_fastpath_enabled(False)
    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    for layer in model.head.layers if model.head is not None else []:
        for module in layer.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = hp["head_dropout"]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dtype = frozen_dtype(torch, device, hp["precision"])
    model.encoder.to(dtype)
    model.to(device)
    layers = model.encoder.layers
    if hp["method"] == "lora":
        LoRALinear = lora_linear()
        chosen = layers if hp["lora_layers"] <= 0 else layers[-hp["lora_layers"] :]
        for layer in chosen:
            for parent, name in ENCODER_LINEARS:
                owner = getattr(layer, parent)
                setattr(
                    owner,
                    name,
                    LoRALinear(
                        getattr(owner, name), hp["lora_rank"], hp["lora_alpha"], hp["lora_dropout"]
                    ),
                )
    elif hp["method"] == "full":
        for module in [*layers[-max(1, hp["full_layers"]) :], model.encoder.final_norm]:
            module.float()
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif hp["method"] != "head":
        raise ValueError(f"Unknown method {hp['method']!r}")
    # The decision head, scorer and question-type embedding always train, in float32.
    for module in (model.head, model.scorer, model.type_emb):
        if module is not None:
            module.float()
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return model, cfg


def collate(items, pad_id, device, multiple=16):
    import torch

    n = len(items)
    length = max(len(it["ids"]) for it in items)
    length = ((length + multiple - 1) // multiple) * multiple
    k = max(2, max(len(it["markers"]) for it in items))
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    pos = torch.zeros((n, k), dtype=torch.long)
    mask = torch.zeros((n, k), dtype=torch.bool)
    target = torch.zeros((n, k), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        pos[i, : len(it["markers"])] = torch.tensor(it["markers"])
        mask[i, : len(it["markers"])] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    batch = {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": pos,
        "marker_mask": mask,
        "qtype": torch.tensor([it["qtype"] for it in items], dtype=torch.long),
        "target": target,
        "weight": torch.tensor([it.get("weight", 1.0) for it in items], dtype=torch.float32),
    }
    return {key: value.to(device) for key, value in batch.items()}


def decision_logits(model, batch, *, encoder_frozen=False):
    logits, _ = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["marker_pos"],
        batch["marker_mask"],
        batch["qtype"],
        detach_encoder=encoder_frozen,
    )
    return logits.float()


def proper_scores(q, target, mask, is_score, w_sph=0.75, w_rps=1.0, log_q=None):
    """Upstream RLCD reward: log score + spherical score - ranked probability score."""
    import torch

    maskf = mask.float()
    q = q * maskf
    if log_q is None:
        log_q = torch.log(q.clamp_min(1e-12)).clamp_min(-9.21)
    log_score = (target * torch.where(mask, log_q, torch.zeros_like(log_q))).sum(-1)
    sph = (target * q).sum(-1) / torch.sqrt((q * q).sum(-1)).clamp_min(1e-9)
    k = maskf.sum(-1).clamp_min(2.0)
    rps = (((torch.cumsum(q, -1) - torch.cumsum(target, -1)) ** 2) * maskf).sum(-1) / (k - 1)
    return log_score + w_sph * sph - w_rps * rps * is_score


def batch_loss(model, batch, hp, sigma, training=True):
    import torch

    logits = decision_logits(model, batch, encoder_frozen=hp["method"] == "head")
    mask, target, weight = batch["marker_mask"], batch["target"], batch["weight"]
    log_p = torch.log_softmax(logits, dim=-1)
    ce = -(target * torch.where(mask, log_p, torch.zeros_like(log_p))).sum(-1)
    norm = weight.sum()
    ce_mean = (ce * weight).sum() / norm
    is_score = (batch["qtype"] == QTYPES["score"]).float()
    objective = hp["objective"]
    if objective == "ce" or not training:
        return ce_mean, ce_mean
    if objective == "proper":
        score = proper_scores(log_p.exp(), target, mask, is_score, log_q=log_p)
        return -(score * weight).sum() / norm, ce_mean
    # "rlcd": Gaussian-perturbed policy gradient on the proper-score reward, plus soft
    # cross-entropy guidance, as in upstream's published fine-tuning notebook.
    group = 4
    maskf = mask.float()
    k = maskf.sum(-1, keepdim=True).clamp_min(1.0)
    eps = torch.randn((group, *logits.shape), device=logits.device) * sigma * maskf
    eps = (eps - eps.sum(-1, keepdim=True) / k) * maskf
    z = logits.detach()[None] + eps
    q = torch.softmax(torch.where(mask, z, torch.full_like(z, -1e4)), dim=-1)
    with torch.no_grad():
        reward = proper_scores(q, target[None], mask, is_score)
        adv = reward - reward.mean(0, keepdim=True)
        adv = adv / (torch.sqrt(((adv - adv.mean()) ** 2).mean()) + 1e-6)
    log_pi = -(((z - logits[None]) ** 2) * maskf).sum(-1) / (2 * sigma**2)
    loss_rl = -((adv * log_pi) * weight).sum() / (norm * group)
    return loss_rl + ce_mean, ce_mean


def evaluate_logits(model, items, pad_id, device, batch_size=16):
    """Calibration-ready (qtype, logits, target) triples, with dropout disabled."""
    import numpy as np
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            logits = decision_logits(model, collate(chunk, pad_id, device)).cpu().numpy()
            for row, it in enumerate(chunk):
                k = len(it["markers"])
                out.append(
                    (it["qtype"], logits[row, :k].astype(np.float64), np.array(it["target"]))
                )
    model.train()
    return out


def peak_memory_gb(torch, device):
    kind = getattr(device, "type", "cpu")
    if kind == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    if kind == "xpu":
        return torch.xpu.max_memory_allocated(device) / 2**30
    if kind == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    # Peak resident memory of this process: ru_maxrss is bytes on macOS, KiB on Linux.
    if resource is None:
        return 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**30 if runtime.platform.system() == "Darwin" else rss / 2**20


def reset_peak(torch, device):
    kind = getattr(device, "type", "cpu")
    if kind == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif kind == "xpu":
        torch.xpu.reset_peak_memory_stats(device)


def fuse_lora(model, dtype):
    for layer in model.encoder.layers:
        for parent, name in ENCODER_LINEARS:
            owner = getattr(layer, parent)
            module = getattr(owner, name)
            if hasattr(module, "fused"):
                setattr(owner, name, module.fused(dtype))


def save_checkpoint(model, base_dir, out_dir, cfg, questions, provenance):
    """Write a standard Laya checkpoint (PyTorch parameter names, FP16 weights) — the same
    files the MLX engine writes, loadable by laya-mlx and by the upstream `laya` package."""
    import torch
    from safetensors.torch import save_file

    base_dir, out_dir = Path(base_dir), Path(out_dir)
    fuse_lora(model, torch.float16)
    tensors = {
        name: tensor.detach().to("cpu", torch.float16).contiguous()
        for name, tensor in model.state_dict().items()
    }
    expected = {upstream_name(k) for k in safetensors_header(base_dir / "model.safetensors")}
    if set(tensors) != expected:
        missing, extra = expected - set(tensors), set(tensors) - expected
        raise RuntimeError(f"Checkpoint names differ from base: missing {missing}, extra {extra}")
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(tensors, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    shutil.copytree(base_dir / "tokenizer", tmp / "tokenizer")
    (tmp / "encoder").mkdir()
    shutil.copy(base_dir / "encoder/config.json", tmp / "encoder/config.json")
    write_json(tmp / "rl_agent_config.json", cfg)
    write_json(tmp / "questions.json", questions)
    write_json(tmp / "laya_finetune.json", provenance)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return out_dir


def schedule_factor(warm, updates):
    """Linear warm-up from 1% to the peak, then cosine decay to 10% of it (engine.fit's schedule)."""

    def factor(step):
        if step < warm:
            return 0.01 + (1 - 0.01) * step / warm
        progress = min(1.0, (step - warm) / max(1, updates - warm))
        return 0.1 + (1 - 0.1) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def fit(spec, hp, emit, workspace=WORKSPACE):
    """Train, pick the best epoch, calibrate and save. Returns a training summary."""
    import torch
    from laya_mlx.tokenizer import Tokenizer

    device = runtime.torch_device()
    run_dir = workspace / "runs" / check_id(spec["run_id"])
    questions, rows, meta = load_dataset(spec["dataset"], workspace)
    base_dir = resolve_model_ref(spec["base_model"], workspace)
    cfg = read_json(base_dir / "rl_agent_config.json")
    tok = Tokenizer(base_dir / "tokenizer")
    random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])
    rng = random.Random(hp["seed"])

    emit(
        "phase",
        phase="prepare",
        message=f"Tokenizing and building the model on {runtime.device_label(device)}",
    )
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    weights = class_weights(train_rows, questions) if hp["class_weighting"] == "balanced" else None
    val_items, _ = encode_items(tok, cfg, val_rows, questions)
    probe, skipped = encode_items(tok, cfg, train_rows, questions)
    if not probe:
        raise ValueError("No training decisions fit the model's token budget")
    model, cfg = load_training_model(base_dir, hp, device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in trainable_params)
    total = sum(p.numel() for p in model.parameters())
    longest = max(len(it["ids"]) for it in probe)
    checkpoint = hp["grad_checkpoint"] == "on" or (
        hp["grad_checkpoint"] == "auto"
        and hp["method"] != "head"
        and hp["batch_size"] * longest > 2048
    )
    if checkpoint and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    per_epoch = math.ceil(len(probe) / hp["batch_size"])
    updates = math.ceil(per_epoch / hp["grad_accum"]) * hp["epochs"]
    emit(
        "info",
        trainable_params=trainable,
        total_params=total,
        train_decisions=len(probe),
        val_decisions=len(val_items),
        skipped_decisions=skipped,
        longest_tokens=longest,
        gradient_checkpointing=checkpoint,
        updates=updates,
        hyperparameters=hp,
        backend="torch",
        device=runtime.device_label(device),
    )

    encoder_params = [
        p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")
    ]
    head_params = [
        p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")
    ]
    groups = [{"params": head_params, "lr": hp["head_lr"]}]
    if encoder_params:
        groups.insert(0, {"params": encoder_params, "lr": hp["lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=hp["weight_decay"])
    warm = max(1, int(hp["warmup"] * updates))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_factor(warm, updates))

    def val_metrics():
        triples = evaluate_logits(model, val_items, tok.pad_token_id, device)
        loss = nll_at(triples, 1.0) if triples else float("nan")
        hits = sum(int(z.argmax()) == int(t.argmax()) for _, z, t in triples)
        return loss, hits / max(1, len(triples))

    def trainable_state():
        return {
            n: p.detach().to("cpu", copy=True)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    emit("phase", phase="train", message=f"Training {hp['epochs']} epochs, {updates} updates")
    model.train()
    reset_peak(torch, device)
    loss0, acc0 = val_metrics()
    emit("epoch", epoch=0, val_loss=loss0, val_accuracy=acc0)
    best = {"loss": loss0, "epoch": 0, "params": trainable_state()}
    step, started, seen, stale, history = 0, time.perf_counter(), 0, 0, []
    for epoch in range(1, hp["epochs"] + 1):
        sigma = 0.4 + (0.1 - 0.4) * ((epoch - 1) / max(1, hp["epochs"] - 1))
        items, _ = encode_items(
            tok, cfg, train_rows, questions, rng, hp["shuffle_options"], weights
        )
        batches = make_batches(items, hp["batch_size"], rng)
        count, running = 0, []
        optimizer.zero_grad(set_to_none=True)
        for i, indices in enumerate(batches):
            batch = collate([items[j] for j in indices], tok.pad_token_id, device)
            loss, ce = batch_loss(model, batch, hp, sigma, training=True)
            loss.backward()
            count += 1
            running += [loss.item(), ce.item()]
            seen += len(indices)
            if count == hp["grad_accum"] or i == len(batches) - 1:
                for parameter in trainable_params:
                    if parameter.grad is not None:
                        parameter.grad /= count
                norm = torch.nn.utils.clip_grad_norm_(trainable_params, hp["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                elapsed = time.perf_counter() - started
                emit(
                    "step",
                    step=step,
                    updates=updates,
                    epoch=epoch,
                    loss=sum(running[::2]) / count,
                    ce=sum(running[1::2]) / count,
                    grad_norm=float(norm),
                    decisions_per_s=round(seen / elapsed, 2),
                    eta_s=round((updates - step) * elapsed / step),
                    peak_gb=round(peak_memory_gb(torch, device), 2),
                )
                count, running = 0, []
        val_loss, val_acc = val_metrics()
        improved = val_loss < best["loss"] - 1e-4
        if improved:
            best, stale = {"loss": val_loss, "epoch": epoch, "params": trainable_state()}, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc})
        emit("epoch", epoch=epoch, val_loss=val_loss, val_accuracy=val_acc, best=improved)
        if hp["patience"] and stale >= hp["patience"]:
            emit("log", message=f"Early stop: validation loss has not improved for {stale} epochs")
            break
    with torch.no_grad():
        parameters = dict(model.named_parameters())
        for name, value in best["params"].items():
            parameters[name].copy_(value.to(parameters[name].device))
    emit("log", message=f"Kept epoch {best['epoch']} (validation loss {best['loss']:.4f})")
    seconds = time.perf_counter() - started

    emit("phase", phase="calibrate", message="Fitting temperatures on validation logits")
    triples = evaluate_logits(model, val_items, tok.pad_token_id, device)
    temperature, by_options, fitted = calibrate(triples, cfg)
    calibration = {
        "ece_uncalibrated": calibrated_ece(triples, [1.0, 1.0, 1.0], {}),
        "ece_calibrated": calibrated_ece(triples, temperature, by_options),
        "temperature": temperature,
        "temperature_by_options": by_options,
        "fitted_types": fitted,
    }
    emit("calibration", **calibration)

    emit("phase", phase="save", message="Merging adapters and writing the checkpoint")
    summary = {
        "run_id": spec["run_id"],
        "base_model": spec["base_model"],
        "base_model_dir": str(base_dir),
        "dataset": spec["dataset"],
        "dataset_sha256": meta.get("sha256"),
        "hyperparameters": hp,
        "trainable_params": trainable,
        "total_params": total,
        "train_decisions": len(probe),
        "val_decisions": len(val_items),
        "updates": step,
        "best_epoch": best["epoch"],
        "best_val_loss": best["loss"],
        "history": history,
        "train_seconds": round(seconds, 1),
        "peak_memory_gb": round(peak_memory_gb(torch, device), 2),
        "gradient_checkpointing": checkpoint,
        "calibration": calibration,
        "backend": "torch",
        "device": runtime.device_label(device),
        "created": now(),
    }
    new_cfg = {**cfg, "temperature": temperature, "temperature_by_options": by_options}
    new_cfg["fine_tuned"] = {
        key: summary[key] for key in ("base_model", "dataset_sha256", "best_epoch", "created")
    } | {"method": hp["method"], "objective": hp["objective"], "tool": "layastudio (pytorch)"}
    model.eval()
    save_checkpoint(model, base_dir, run_dir / "model", new_cfg, questions, summary)
    write_json(run_dir / "training.json", summary)
    return summary
