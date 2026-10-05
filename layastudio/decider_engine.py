"""Decider fine-tuning on PyTorch: LoRA with PEFT, the way Decider v11 itself was made.

    rows -> Decider's plain-layout prompt rows (decider.encode_items: one row per question,
    one yes/no row per level of an isolated score) -> the base model in bfloat16 (or 4-bit,
    QLoRA, on CUDA) with PEFT LoRA adapters on the attention and MLP projections ->
    cross-entropy on the option letters at each row's answer slot -> early stopping on the
    validation rows -> one temperature per answer type fitted on the validation split, through
    each type's own readout -> the adapters merged into the bfloat16 weights -> a Decider
    checkpoint with the base's own files (config, tokenizer, chat template, decider/ code) and a
    new decider_config.json.

The optimizer is MLX's AdamW rule (torch_engine.mlx_adamw), so this trainer and the MLX one
(decider_mlx.py, which Apple silicon uses) take the same steps. The schedule is the studio's:
a linear warm-up, then a cosine decay to a tenth of the peak.

Needs the `decoder` extra (peft; on CUDA, flash-linear-attention makes Qwen3.5's linear
attention several times faster, and bitsandbytes enables 4-bit).
"""

import math
import random
import shutil
import time
from pathlib import Path

from . import decider, runtime
from .engine import (
    WORKSPACE,
    check_id,
    class_weights,
    load_dataset,
    lora_variants,
    now,
    resolve_model_ref,
    write_json,
)

# Files a Decider checkpoint carries besides its weights and config.json, copied as they are.
CARRIED = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)


def fit(spec, hp, emit, workspace=WORKSPACE, before_model=None):
    """Train on this machine's backend: MLX-LM on Apple silicon, PyTorch everywhere else.
    before_model: called once the rows are built, before the model loads (the baseline)."""
    if runtime.backend() == "mlx":
        from . import decider_mlx

        return decider_mlx.fit(spec, hp, emit, workspace, before_model)
    return torch_fit(spec, hp, emit, workspace, before_model)


def check_hyperparameters(hp):
    if hp.get("method", "lora") != "lora":
        raise ValueError("Decider trains with LoRA (method lora)")
    if hp.get("objective") not in ("ce", "proper"):
        raise ValueError("Decider's objective is ce or proper")
    if hp.get("quantization", "none") not in ("none", "4bit"):
        raise ValueError("quantization is none or 4bit")
    for key in ("batch_size", "batch_tokens", "grad_accum", "epochs", "lora_rank"):
        if int(hp.get(key, 1)) < 1:
            raise ValueError(f"{key} must be at least 1")


def check_device(hp, device):
    """Refuse what this PyTorch device cannot train: 4-bit QLoRA needs CUDA (bitsandbytes)."""
    if hp.get("quantization") == "4bit" and getattr(device, "type", "cpu") != "cuda":
        raise ValueError("4-bit QLoRA needs an NVIDIA GPU here (bitsandbytes); use LoRA")


# ----------------------------------------------------------------------------- the model


def base_dtype(torch, device, precision):
    kind = getattr(device, "type", "cpu")
    if precision != "bfloat16" or kind == "cpu":
        return torch.float32
    if kind == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float32
    return torch.bfloat16


def load_base(model_dir, hp, device):
    """The causal LM with frozen weights: bfloat16 (float32 on a CPU), or 4-bit NF4 on CUDA."""
    import torch
    from transformers import AutoModelForCausalLM

    dtype = base_dtype(torch, device, hp["precision"])
    if hp.get("quantization") == "4bit":
        check_device(hp, device)
        from transformers import BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
            quantization_config=quantization,
            device_map={"": device.index or 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype)
        model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def lora_config(hp):
    from peft import LoraConfig

    return LoraConfig(
        r=int(hp["lora_rank"]),
        lora_alpha=float(hp["lora_alpha"]),
        lora_dropout=float(hp["lora_dropout"]),
        target_modules=list(decider.LORA_TARGETS),
        use_dora=bool(hp.get("dora")),
        use_rslora=bool(hp.get("rslora")),
        bias="none",
        task_type="CAUSAL_LM",
    )


def add_adapters(model, hp, checkpoint):
    """PEFT LoRA on the attention and MLP projections; gradient checkpointing if asked."""
    from peft import get_peft_model, prepare_model_for_kbit_training

    if hp.get("quantization") == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpoint)
    elif checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(hp))
    return model


def batch_logits(model, items, label_ids, pad_id, device):
    """Letter logits at every row's slot, [rows, 255] float32, -inf past each row's options.

    Rows are right-padded: the slot is a row's last token, which padding after it cannot
    reach in a causal model (attention or Qwen3.5's linear attention)."""
    import torch

    width = max(len(it["ids"]) for it in items)
    width = ((width + 15) // 16) * 16
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        mask[i, : len(it["ids"])] = 1
    slots = torch.tensor([len(it["ids"]) - 1 for it in items], device=device)
    backbone, head = decider.backbone_and_head(model)
    hidden = backbone(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
    picked = hidden[torch.arange(len(items), device=device), slots]
    letters = torch.tensor(label_ids, device=device)
    logits = torch.nn.functional.linear(picked.float(), head.weight[letters].float())
    count = torch.tensor([it["n"] for it in items], device=device)
    valid = torch.arange(logits.shape[1], device=device)[None, :] < count[:, None]
    return logits.masked_fill(~valid, float("-inf")), valid


def targets(items, width, device):
    import torch

    out = torch.zeros((len(items), width), dtype=torch.float32)
    for i, it in enumerate(items):
        out[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    weight = torch.tensor([it["weight"] for it in items], dtype=torch.float32)
    return out.to(device), weight.to(device)


def row_loss(logits, valid, target, weight, objective):
    """Cross-entropy on the option letters (Decider's own loss), or the proper scoring rule
    the Laya trainer optimises (log + spherical score) on each row's distribution."""
    import torch

    log_p = torch.log_softmax(logits, dim=-1)
    log_p = torch.where(valid, log_p, torch.zeros_like(log_p))
    ce = -(target * log_p).sum(-1)
    norm = weight.sum()
    ce_mean = (ce * weight).sum() / norm
    if objective != "proper":
        return ce_mean, ce_mean
    from .torch_engine import proper_scores

    q = torch.where(valid, log_p.exp(), torch.zeros_like(log_p))
    score = proper_scores(q, target, valid, torch.zeros_like(weight), log_q=log_p)
    return -(score * weight).sum() / norm, ce_mean


# ----------------------------------------------------------------------------- training


def evaluate_rows(model, items, label_ids, pad_id, device, rows_per_batch=16):
    """Validation loss and accuracy per row, and every row's letter logits (dropout off)."""
    import torch

    model.eval()
    total = hits = 0.0
    out = []
    with torch.no_grad():
        for start in range(0, len(items), rows_per_batch):
            chunk = items[start : start + rows_per_batch]
            logits, valid = batch_logits(model, chunk, label_ids, pad_id, device)
            target, _ = targets(chunk, logits.shape[1], device)
            log_p = torch.where(valid, torch.log_softmax(logits, -1), torch.zeros_like(logits))
            total += float(-(target * log_p).sum())
            hits += float((logits.argmax(-1) == target.argmax(-1)).sum())
            for i, it in enumerate(chunk):
                out.append(logits[i, : it["n"]].double().cpu().tolist())
    model.train()
    return total / max(1, len(items)), hits / max(1, len(items)), out


def validation_answers(items, logits, val_rows):
    """Rows grouped back into answers, for calibration: (type, kind, rows' logits, target)."""
    grouped = {}
    for item, z in zip(items, logits):
        index, qid, level = item["key"]
        entry = grouped.setdefault(
            (index, qid),
            [item["type"], "iso" if level is not None else "list", [], None],
        )
        entry[2].append((level if level is not None else 0, z))
        if level is None:
            entry[3] = item["target"]
    answers = []
    for (index, qid), (qtype, kind, rows, target) in grouped.items():
        rows.sort(key=lambda pair: pair[0])
        if kind == "iso":
            target = val_rows[index]["targets"][qid]
        answers.append((qtype, kind, [z for _, z in rows], list(target)))
    return answers


def trainable_state(model):
    return {
        name: parameter.detach().to("cpu", copy=True)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def torch_fit(spec, hp, emit, workspace=WORKSPACE, before_model=None):
    """Train, keep the best epoch, calibrate, merge and save. Returns the training summary."""
    import torch

    from .torch_engine import mlx_adamw, peak_memory_gb, reset_peak, schedule_factor

    check_hyperparameters(hp)
    device = runtime.torch_device()
    check_device(hp, device)  # before the baseline, which a run that cannot train never needs
    decider.portable_kernels(device)
    run_dir = workspace / "runs" / check_id(spec["run_id"])
    questions, rows, meta = load_dataset(spec["dataset"], workspace)
    base_dir = resolve_model_ref(spec["base_model"], workspace)
    cfg = decider.config(base_dir)
    tok = decider.Tokens(base_dir)
    prompter = decider.Prompter(tok, cfg)
    random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])
    rng = random.Random(hp["seed"])

    emit(
        "phase",
        phase="prepare",
        message=f"Building Decider's prompt rows and the model on {runtime.device_label(device)}",
    )
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    weights = class_weights(train_rows, questions) if hp["class_weighting"] == "balanced" else None
    val_items, _ = decider.encode_items(prompter, val_rows, questions, hp=hp)
    probe, skipped = decider.encode_items(prompter, train_rows, questions, hp=hp)
    if not probe:
        raise ValueError("No training rows could be built in Decider's prompt")
    if before_model:
        before_model()
    longest = max(len(it["ids"]) for it in probe)
    kind = getattr(device, "type", "cpu")
    checkpoint = hp["grad_checkpoint"] == "on" or (
        hp["grad_checkpoint"] == "auto" and kind != "cpu"
    )
    model = load_base(base_dir, hp, device)
    model = add_adapters(model, hp, checkpoint)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable = sum(p.numel() for _, p in named)
    total = sum(p.numel() for p in model.parameters())
    per_epoch = len(decider.token_batches(probe, hp["batch_size"], hp["batch_tokens"], rng))
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

    ratio = float(hp.get("loraplus_ratio") or 1.0)
    lora_b = [p for n, p in named if ratio != 1.0 and "lora_B" in n]
    others = [p for n, p in named if not (ratio != 1.0 and "lora_B" in n)]
    groups = [{"params": others, "lr": hp["lr"]}]
    if lora_b:
        groups.insert(0, {"params": lora_b, "lr": hp["lr"] * ratio})
    optimizer = mlx_adamw()(groups, lr=hp["lr"], weight_decay=hp["weight_decay"])
    warm = max(1, int(hp["warmup"] * updates))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_factor(warm, updates))
    label_ids, pad_id = prompter.label_ids, tok.pad_token_id

    emit("phase", phase="train", message=f"Training {hp['epochs']} epochs, {updates} updates")
    model.train()
    reset_peak(torch, device)
    loss0, acc0, _ = evaluate_rows(model, val_items, label_ids, pad_id, device)
    emit("epoch", epoch=0, val_loss=loss0, val_accuracy=acc0)
    best = {"loss": loss0, "epoch": 0, "params": trainable_state(model)}
    step, started, seen, stale, history = 0, time.perf_counter(), 0, 0, []
    params = [p for _, p in named]
    for epoch in range(1, hp["epochs"] + 1):
        items, _ = decider.encode_items(
            prompter, train_rows, questions, rng, hp["shuffle_options"], weights, hp
        )
        batches = decider.token_batches(items, hp["batch_size"], hp["batch_tokens"], rng)
        count, running = 0, []
        optimizer.zero_grad(set_to_none=True)
        for i, indices in enumerate(batches):
            chunk = [items[j] for j in indices]
            logits, valid = batch_logits(model, chunk, label_ids, pad_id, device)
            target, weight = targets(chunk, logits.shape[1], device)
            loss, ce = row_loss(logits, valid, target, weight, hp["objective"])
            loss.backward()
            count += 1
            running += [loss.item(), ce.item()]
            seen += len(indices)
            if count == hp["grad_accum"] or i == len(batches) - 1:
                for parameter in params:
                    if parameter.grad is not None:
                        parameter.grad /= count
                norm = torch.nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
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
        val_loss, val_acc, _ = evaluate_rows(model, val_items, label_ids, pad_id, device)
        improved = val_loss < best["loss"] - 1e-4
        if improved:
            best, stale = {"loss": val_loss, "epoch": epoch, "params": trainable_state(model)}, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc})
        emit("epoch", epoch=epoch, val_loss=val_loss, val_accuracy=val_acc, best=improved)
        if hp["patience"] and stale >= hp["patience"]:
            emit("log", message=f"Early stop: validation loss has not improved for {stale} epochs")
            break
    with torch.no_grad():
        current = dict(model.named_parameters())
        for name, value in best["params"].items():
            current[name].copy_(value.to(current[name].device))
    emit("log", message=f"Kept epoch {best['epoch']} (validation loss {best['loss']:.4f})")
    seconds = time.perf_counter() - started

    emit("phase", phase="calibrate", message="Fitting a temperature per answer type")
    _, _, val_logits = evaluate_rows(model, val_items, label_ids, pad_id, device)
    calibration = calibrate(validation_answers(val_items, val_logits, val_rows), cfg)
    emit("calibration", **calibration)

    emit("phase", phase="save", message="Merging the adapters into the weights")
    summary = {
        "run_id": spec["run_id"],
        "base_model": spec["base_model"],
        "base_model_dir": str(base_dir),
        "kind": "decider",
        "dataset": spec["dataset"],
        "dataset_sha256": meta.get("sha256"),
        "hyperparameters": hp,
        "trainable_params": trainable,
        "total_params": total,
        "train_decisions": len(probe),
        "val_decisions": len(val_items),
        "skipped_decisions": skipped,
        "updates": step,
        "best_epoch": best["epoch"],
        "best_val_loss": best["loss"],
        "history": history,
        "train_seconds": round(seconds, 1),
        "peak_memory_gb": round(peak_memory_gb(torch, device), 2),
        "gradient_checkpointing": checkpoint,
        "calibration": calibration,
        "lora_variants": lora_variants(hp),
        "backend": "torch",
        "device": runtime.device_label(device),
        "created": now(),
    }
    merged = merge(model, base_dir, hp, best["params"], device)
    save_checkpoint(merged, base_dir, run_dir / "model", cfg, questions, summary)
    write_json(run_dir / "training.json", summary)
    return summary


def calibrate(answers, cfg):
    by_type, fitted = decider.fit_temperatures(answers, cfg)
    before = {t: decider.temperature(cfg, t) for t in decider.TYPES}
    return {
        "ece_uncalibrated": decider.answers_ece(answers, {t: 1.0 for t in decider.TYPES}),
        "ece_base_temperatures": decider.answers_ece(answers, before),
        "ece_calibrated": decider.answers_ece(answers, by_type),
        "temperature_by_type": by_type,
        "base_temperature_by_type": before,
        "fitted_types": fitted,
        "answers": len(answers),
    }


def stored_dtype(base_dir):
    """The dtype the base checkpoint stores its weights in (bfloat16 for every Decider)."""
    import torch

    from .engine import safetensors_header

    found = {
        v["dtype"]
        for f in Path(base_dir).glob("*.safetensors")
        for v in safetensors_header(f).values()
    }
    if "BF16" in found:
        return torch.bfloat16
    if "F16" in found:
        return torch.float16
    return torch.float32


def merge(model, base_dir, hp, adapters, device):
    """The fine-tuned weights: the adapters merged into the base, written in the dtype the
    base is stored in. A 4-bit base is not merged into: the base is loaded again in that
    dtype and the adapters applied to it."""
    import torch

    dtype = stored_dtype(base_dir)
    if hp.get("quantization") == "4bit":
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM

        del model
        runtime.clear_cache()
        base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=dtype)
        model = get_peft_model(base, lora_config(hp))
        current = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in adapters.items():
                current[name].copy_(value.to(current[name].dtype))
    merged = model.merge_and_unload()
    merged.to(dtype)
    return merged


def save_checkpoint(model, base_dir, out_dir, cfg, questions, provenance):
    """A Decider checkpoint: the merged weights in bfloat16 (safetensors, as the base ships
    them), the base's tokenizer, chat template, generation config and decider/ inference code,
    and decider_config.json with the fitted temperatures and the fine-tune's lineage."""
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(str(tmp), safe_serialization=True)
    for name in CARRIED:
        if (base_dir / name).is_file():
            shutil.copy(base_dir / name, tmp / name)
    if (base_dir / "decider").is_dir():
        shutil.copytree(
            base_dir / "decider", tmp / "decider", ignore=shutil.ignore_patterns("__pycache__")
        )
    by_type = (provenance.get("calibration") or {}).get("temperature_by_type") or {}
    write_json(tmp / "decider_config.json", decider.calibrated_config(cfg, by_type, provenance))
    write_json(tmp / "questions.json", questions)
    write_json(tmp / "finetune.json", provenance)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return out_dir
