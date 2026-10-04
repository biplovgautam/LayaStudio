"""Decider fine-tuning and inference on Apple silicon, through MLX-LM's Qwen3.5.

The same training as decider_engine.py (PyTorch + PEFT): Decider's plain-layout rows, LoRA on
the attention and MLP projections (engine.lora_class(), the studio's own LoRA with DoRA and
rsLoRA; MLX-LM's for a 4-bit base), cross-entropy on the option letters at the answer slot,
MLX's AdamW with the studio's schedule, early stopping, one temperature per answer type. The
fine-tune is written as the base's own Hugging Face files: every adapter merged into its
weight in the base's dtype, under the base's tensor names, so it loads in transformers,
converts to GGUF and loads back here unchanged.

MLX-LM names Decider's model type qwen3_5 (its config.json says qwen3_5_text); the model is
built from mlx_lm.models.qwen3_5 directly. Needs the `decoder` extra on Apple silicon.
"""

import json
import math
import random
import shutil
import time
from pathlib import Path

from . import decider
from .decider_engine import CARRIED, calibrate, validation_answers
from .engine import (
    WORKSPACE,
    check_id,
    class_weights,
    load_dataset,
    lora_class,
    lora_scale,
    lora_variants,
    now,
    resolve_model_ref,
    write_json,
)


def mlx_name(key):
    """mlx_lm.models.qwen3_5.Model.sanitize's renaming of a Hugging Face tensor name."""
    if key.startswith("model.language_model"):
        return key.replace("model.language_model", "language_model.model", 1)
    if key.startswith("language_model."):
        return key
    return "language_model." + key


def load(model_dir, dtype=None):
    """Decider's weights in MLX-LM's Qwen3.5 (text), in their stored dtype unless asked."""
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    model = Model(ModelArgs.from_dict({**config, "model_type": "qwen3_5"}))
    weights = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        weights.update(mx.load(str(path)))
    weights = model.sanitize(weights)
    if dtype is not None:
        weights = {
            k: v.astype(dtype)
            if mx.issubdtype(v.dtype, mx.floating) and not k.endswith("A_log")
            else v
            for k, v in weights.items()
        }
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def head_rows(model, letters):
    """The LM head's rows for the option letters (the tied embedding when it is tied),
    dequantized when the head is quantized (an MLX-LM int4/int8 export)."""
    import mlx.core as mx

    text = model.language_model
    tied = getattr(text.args, "tie_word_embeddings", False)
    layer = text.model.embed_tokens if tied else text.lm_head
    if "scales" in layer:
        return mx.dequantize(
            layer.weight[letters],
            layer.scales[letters],
            layer.biases[letters] if "biases" in layer else None,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=getattr(layer, "mode", "affine"),
        )
    return layer.weight[letters]


def slot_logits(model, ids, slots, letters):
    """[rows, len(letters)] float32 logits at each row's slot (right-padded ids)."""
    import mlx.core as mx

    hidden = model.language_model.model(ids)
    picked = hidden[mx.arange(ids.shape[0]), slots]
    return picked.astype(mx.float32) @ head_rows(model, letters).astype(mx.float32).T


def pad(rows, pad_id):
    import mlx.core as mx
    import numpy as np

    width = max(len(ids) for ids in rows)
    width = ((width + 15) // 16) * 16
    out = np.full((len(rows), width), pad_id, dtype=np.int32)
    for i, ids in enumerate(rows):
        out[i, : len(ids)] = ids
    return mx.array(out), mx.array(np.array([len(ids) - 1 for ids in rows], dtype=np.int32))


def letter_logits(model, rows, prompter, pad_id):
    """decider.Agent's MLX path: letter logits for (ids, n) rows, as lists of floats."""
    import mlx.core as mx
    import numpy as np

    ids, slots = pad([r for r, _ in rows], pad_id)
    letters = mx.array(prompter.label_ids[: max(n for _, n in rows)])
    values = slot_logits(model, ids, slots, letters)
    mx.eval(values)
    values = np.asarray(values)
    return [values[i, :n].astype(float).tolist() for i, (_, n) in enumerate(rows)]


# ----------------------------------------------------------------------------- training


def add_adapters(model, hp):
    """LoRA on decider.LORA_TARGETS in every layer; everything else frozen."""
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear as QuantizedLoRA

    model.freeze()
    LoRALinear = lora_class()
    scale = lora_scale(hp["lora_alpha"], hp["lora_rank"], bool(hp.get("rslora")))
    adapted = []
    for i, layer in enumerate(model.language_model.model.layers):
        for part in ("self_attn", "linear_attn", "mlp"):
            owner = getattr(layer, part, None)
            if owner is None:
                continue
            for name in decider.LORA_TARGETS:
                base = getattr(owner, name, None)
                if not isinstance(base, (nn.Linear, nn.QuantizedLinear)):
                    continue
                if isinstance(base, nn.QuantizedLinear):
                    if hp.get("dora"):
                        raise ValueError("DoRA is not available on a 4-bit base")
                    module = QuantizedLoRA.from_base(
                        base, r=int(hp["lora_rank"]), dropout=float(hp["lora_dropout"]), scale=scale
                    )
                else:
                    module = LoRALinear(
                        base,
                        int(hp["lora_rank"]),
                        float(hp["lora_alpha"]),
                        float(hp["lora_dropout"]),
                        dora=bool(hp.get("dora")),
                        rslora=bool(hp.get("rslora")),
                    )
                setattr(owner, name, module)
                keys = ["lora_a", "lora_b"] + (
                    ["magnitude"] if getattr(module, "dora", False) else []
                )
                module.unfreeze(keys=keys, recurse=False)
                adapted.append((i, part, name))
    return adapted


def collate(items, pad_id):
    import mlx.core as mx
    import numpy as np

    ids, slots = pad([it["ids"] for it in items], pad_id)
    width = max(it["n"] for it in items)
    target = np.zeros((len(items), width), dtype=np.float32)
    valid = np.zeros((len(items), width), dtype=np.bool_)
    for i, it in enumerate(items):
        target[i, : it["n"]] = it["target"]
        valid[i, : it["n"]] = True
    weight = np.array([it["weight"] for it in items], dtype=np.float32)
    return ids, slots, mx.array(target), mx.array(valid), mx.array(weight)


def batch_loss(model, batch, letters, objective):
    import mlx.core as mx

    ids, slots, target, valid, weight = batch
    logits = slot_logits(model, ids, slots, letters[: target.shape[1]])
    logits = mx.where(valid, logits, -1e9)
    log_p = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    log_p = mx.where(valid, log_p, 0.0)
    ce = -(target * log_p).sum(-1)
    norm = weight.sum()
    ce_mean = (ce * weight).sum() / norm
    if objective != "proper":
        return ce_mean, ce_mean
    from .engine import proper_scores

    q = mx.where(valid, mx.exp(log_p), 0.0)
    score = proper_scores(q, target, valid, mx.zeros_like(weight), log_q=log_p)
    return -(score * weight).sum() / norm, ce_mean


def evaluate_rows(model, items, letters, pad_id, rows_per_batch=16):
    import mlx.core as mx
    import numpy as np

    model.eval()
    total = hits = 0.0
    out = []
    for start in range(0, len(items), rows_per_batch):
        chunk = items[start : start + rows_per_batch]
        ids, slots, target, valid, _ = collate(chunk, pad_id)
        logits = mx.where(valid, slot_logits(model, ids, slots, letters[: target.shape[1]]), -1e9)
        log_p = mx.where(valid, logits - mx.logsumexp(logits, axis=-1, keepdims=True), 0.0)
        mx.eval(logits, log_p)
        total += float(-(target * log_p).sum())
        hits += float((logits.argmax(-1) == target.argmax(-1)).sum())
        values = np.asarray(logits)
        for i, it in enumerate(chunk):
            out.append(values[i, : it["n"]].astype(float).tolist())
    model.train()
    return total / max(1, len(items)), hits / max(1, len(items)), out


def fit(spec, hp, emit, workspace=WORKSPACE):
    """The MLX trainer: decider_engine.torch_fit's steps, on Apple silicon."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten

    from .decider_engine import check_hyperparameters

    check_hyperparameters(hp)
    run_dir = workspace / "runs" / check_id(spec["run_id"])
    questions, rows, meta = load_dataset(spec["dataset"], workspace)
    base_dir = resolve_model_ref(spec["base_model"], workspace)
    cfg = decider.config(base_dir)
    tok = decider.Tokens(base_dir)
    prompter = decider.Prompter(tok, cfg)
    random.seed(hp["seed"])
    mx.random.seed(hp["seed"])
    rng = random.Random(hp["seed"])

    emit("phase", phase="prepare", message="Building Decider's prompt rows and the model (MLX)")
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    weights = class_weights(train_rows, questions) if hp["class_weighting"] == "balanced" else None
    val_items, _ = decider.encode_items(prompter, val_rows, questions, hp=hp)
    probe, skipped = decider.encode_items(prompter, train_rows, questions, hp=hp)
    if not probe:
        raise ValueError("No training rows could be built in Decider's prompt")
    frozen = mx.bfloat16 if hp["precision"] == "bfloat16" else mx.float32
    model = load(base_dir, dtype=frozen)
    if hp.get("quantization") == "4bit":
        nn.quantize(model, group_size=64, bits=4, class_predicate=_quantizable)
    add_adapters(model, hp)
    checkpoint = hp["grad_checkpoint"] in ("on", "auto")
    if checkpoint:
        from mlx_lm.tuner.trainer import grad_checkpoint

        for layer in {type(layer): layer for layer in model.language_model.model.layers}.values():
            grad_checkpoint(layer)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    per_epoch = len(
        decider.token_batches(probe, hp["batch_size"], hp["batch_tokens"], random.Random(0))
    )
    updates = math.ceil(per_epoch / hp["grad_accum"]) * hp["epochs"]
    emit(
        "info",
        trainable_params=trainable,
        total_params=total,
        train_decisions=len(probe),
        val_decisions=len(val_items),
        skipped_decisions=skipped,
        longest_tokens=max(len(it["ids"]) for it in probe),
        gradient_checkpointing=checkpoint,
        updates=updates,
        hyperparameters=hp,
        backend="mlx",
    )

    def schedule(peak):
        warm = max(1, int(hp["warmup"] * updates))
        return optim.join_schedules(
            [
                optim.linear_schedule(peak * 0.01, peak, warm),
                optim.cosine_decay(peak, max(1, updates - warm), peak * 0.1),
            ],
            [warm],
        )

    ratio = float(hp.get("loraplus_ratio") or 1.0)
    main = optim.AdamW(learning_rate=schedule(hp["lr"]), weight_decay=hp["weight_decay"])
    if ratio != 1.0:
        faster = optim.AdamW(
            learning_rate=schedule(hp["lr"] * ratio), weight_decay=hp["weight_decay"]
        )
        optimizer = optim.MultiOptimizer([faster, main], [lambda path, _: path.endswith("lora_b")])
    else:
        optimizer = main
    letters = mx.array(prompter.label_ids)
    pad_id = tok.pad_token_id
    loss_and_grad = nn.value_and_grad(
        model, lambda m, b: batch_loss(m, b, letters, hp["objective"])
    )

    emit("phase", phase="train", message=f"Training {hp['epochs']} epochs, {updates} updates")
    model.train()
    mx.reset_peak_memory()
    loss0, acc0, _ = evaluate_rows(model, val_items, letters, pad_id)
    emit("epoch", epoch=0, val_loss=loss0, val_accuracy=acc0)
    best = {"loss": loss0, "epoch": 0, "params": dict(tree_flatten(model.trainable_parameters()))}
    step, started, seen, stale, history = 0, time.perf_counter(), 0, 0, []
    for epoch in range(1, hp["epochs"] + 1):
        items, _ = decider.encode_items(
            prompter, train_rows, questions, rng, hp["shuffle_options"], weights, hp
        )
        batches = decider.token_batches(items, hp["batch_size"], hp["batch_tokens"], rng)
        accum, count, running = None, 0, []
        for i, indices in enumerate(batches):
            (loss, ce), grads = loss_and_grad(model, collate([items[j] for j in indices], pad_id))
            accum = grads if accum is None else tree_map(mx.add, accum, grads)
            mx.eval(accum, loss, ce)
            count += 1
            running += [loss, ce]
            seen += len(indices)
            if count == hp["grad_accum"] or i == len(batches) - 1:
                accum = tree_map(lambda g: g / count, accum)
                accum, norm = optim.clip_grad_norm(accum, hp["max_grad_norm"])
                optimizer.update(model, accum)
                mx.eval(model.parameters(), optimizer.state, running)
                step += 1
                elapsed = time.perf_counter() - started
                emit(
                    "step",
                    step=step,
                    updates=updates,
                    epoch=epoch,
                    loss=sum(x.item() for x in running[::2]) / count,
                    ce=sum(x.item() for x in running[1::2]) / count,
                    grad_norm=norm.item(),
                    decisions_per_s=round(seen / elapsed, 2),
                    eta_s=round((updates - step) * elapsed / step),
                    peak_gb=round(mx.get_peak_memory() / 2**30, 2),
                )
                accum, count, running = None, 0, []
        val_loss, val_acc, _ = evaluate_rows(model, val_items, letters, pad_id)
        improved = val_loss < best["loss"] - 1e-4
        if improved:
            params = dict(tree_flatten(model.trainable_parameters()))
            best, stale = {"loss": val_loss, "epoch": epoch, "params": params}, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc})
        emit("epoch", epoch=epoch, val_loss=val_loss, val_accuracy=val_acc, best=improved)
        if hp["patience"] and stale >= hp["patience"]:
            emit("log", message=f"Early stop: validation loss has not improved for {stale} epochs")
            break
    model.update(tree_unflatten(list(best["params"].items())))
    emit("log", message=f"Kept epoch {best['epoch']} (validation loss {best['loss']:.4f})")
    seconds = time.perf_counter() - started

    emit("phase", phase="calibrate", message="Fitting a temperature per answer type")
    _, _, val_logits = evaluate_rows(model, val_items, letters, pad_id)
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
        "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 2),
        "gradient_checkpointing": checkpoint,
        "calibration": calibration,
        "lora_variants": lora_variants(hp),
        "backend": "mlx",
        "created": now(),
    }
    save_checkpoint(model, base_dir, run_dir / "model", cfg, questions, summary)
    write_json(run_dir / "training.json", summary)
    return summary


def _quantizable(path, module):
    """4-bit QLoRA quantises the projections LoRA adapts and the MLPs, nothing else."""
    import mlx.nn as nn

    return isinstance(module, nn.Linear) and path.rsplit(".", 1)[-1] in decider.LORA_TARGETS


def deltas(model):
    """{MLX module path: (float32 delta or None, LoRA module)} for every adapted linear."""
    import mlx.core as mx

    out = {}
    for i, layer in enumerate(model.language_model.model.layers):
        for part in ("self_attn", "linear_attn", "mlp"):
            owner = getattr(layer, part, None)
            if owner is None:
                continue
            for name in decider.LORA_TARGETS:
                module = getattr(owner, name, None)
                if module is None or not hasattr(module, "lora_a"):
                    continue
                scale = module.scale
                delta = (scale * (module.lora_a @ module.lora_b).T).astype(mx.float32)
                out[f"language_model.model.layers.{i}.{part}.{name}.weight"] = (delta, module)
    return out


def save_checkpoint(model, base_dir, out_dir, cfg, questions, provenance):
    """The base's own files with every adapter merged into its weight: the base's tensors
    (dtype and names as stored) read again from disk, so nothing passes through the training
    copy's precision or a 4-bit quantisation."""
    import mlx.core as mx

    base_dir, out_dir = Path(base_dir), Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    merged = deltas(model)
    done = set()
    for path in sorted(base_dir.glob("*.safetensors")):
        tensors = mx.load(str(path))
        for key in list(tensors):
            name = mlx_name(key)
            if name not in merged:
                continue
            delta, module = merged[name]
            weight = tensors[key].astype(mx.float32) + delta
            if getattr(module, "dora", False):
                weight = (module.magnitude / mx.linalg.norm(weight, axis=1))[:, None] * weight
            tensors[key] = weight.astype(tensors[key].dtype)
            done.add(name)
        mx.save_safetensors(str(tmp / path.name), tensors, metadata={"format": "pt"})
    missing = set(merged) - done
    if missing:
        raise RuntimeError(
            f"Adapted weights not found in the base checkpoint: {sorted(missing)[:3]}"
        )
    for name in ("config.json", "model.safetensors.index.json", *CARRIED):
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


# ----------------------------------------------------------------------------- export


MLX_PRECISIONS = {"int8": 8, "int4": 4}


def export(model_ref, workspace=WORKSPACE, emit=None, precision="int4", rows=60):
    """runs/<id>/exports/mlx-<precision>: the fine-tune quantized for MLX-LM (group size 64),
    loadable with mlx_lm.load, and measured against the unquantized weights on test rows."""
    import mlx.core as mx
    from mlx_lm.utils import quantize_model, save_config, save_model

    from . import kinds
    from .gguf import compare, test_rows

    emit = emit or (lambda *a, **k: None)
    if precision not in MLX_PRECISIONS:
        raise ValueError(f"MLX exports are {', '.join(MLX_PRECISIONS)}")
    kind, _, value = str(model_ref).partition(":")
    if kind != "run":
        raise ValueError("MLX exports are made of fine-tuned runs (run:<id>)")
    run_dir = workspace / "runs" / check_id(value)
    model_dir = resolve_model_ref(model_ref, workspace)
    if kinds.detect(model_dir) != kinds.DECIDER:
        raise ValueError("MLX-LM exports are for decoder fine-tunes (Decider)")
    started = time.perf_counter()
    out = run_dir / "exports" / f"mlx-{precision}"
    emit("phase", phase="export", message=f"Quantizing to {precision} for MLX-LM")
    items, _, _ = test_rows(model_ref, workspace, limit=rows)
    prompter = decider.Prompter(decider.Tokens(model_dir), decider.config(model_dir))
    pad_id = prompter.tok.pad_token_id
    model = load(model_dir)
    batch = [(it["ids"], it["n"]) for it in items]
    reference = [
        z
        for i in range(0, len(batch), 8)
        for z in letter_logits(model, batch[i : i + 8], prompter, pad_id)
    ]
    config = json.loads((model_dir / "config.json").read_text())
    config = {**config, "model_type": "qwen3_5"}  # MLX-LM's name for Qwen3.5 text
    model, config = quantize_model(model, config, 64, MLX_PRECISIONS[precision])
    quantized = [
        z
        for i in range(0, len(batch), 8)
        for z in letter_logits(model, batch[i : i + 8], prompter, pad_id)
    ]
    shutil.rmtree(out, ignore_errors=True)
    save_model(out, model, donate_model=True)
    save_config(config, out / "config.json")
    for name in ("decider_config.json", "questions.json", *CARRIED):
        if (model_dir / name).is_file():
            shutil.copy(model_dir / name, out / name)
    mx.clear_cache()
    report = {
        "model": model_ref,
        "target": "mlx",
        "precision": precision,
        "path": str(out),
        "created": now(),
        "size_mb": round(sum(p.stat().st_size for p in out.glob("*.safetensors")) / 2**20, 1),
        "runs_on": "Apple silicon: mlx_lm.load(path), or decider's MLX path in the studio",
        "verification": {
            **(compare(reference, quantized, items) if items else {"rows": 0}),
            "reference": "the same fine-tune unquantized, in MLX",
        },
        "seconds": round(time.perf_counter() - started, 1),
    }
    write_json(run_dir / "exports" / f"mlx-{precision}.json", report)
    emit("result", **report)
    return report
