import json
import math

import laya_mlx
import mlx.core as mx
import numpy as np
import pytest
from laya_mlx.model import DecisionModel, EncoderConfig, sanitize_weights
from mlx.utils import tree_flatten
from tokenizers import Tokenizer, models, pre_tokenizers

from layastudio import engine

QUESTIONS = {
    "topic": {"type": "choice", "instructions": "Choose", "criteria": ["alpha", "beta", "gamma"]},
    "level": {"type": "score", "instructions": "Level", "criteria": ["low", "mid", "high"]},
    "flag": {"type": "noul", "instructions": "Is this flagged?"},
}
WORDS = ["alpha", "beta", "gamma", "low", "mid", "high", "red", "green", "blue", "yes", "no"]


@pytest.fixture
def checkpoint(tmp_path):
    cfg = {
        "model_type": "modernbert",
        "vocab_size": 64,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 3,
        "num_attention_heads": 1,
        "local_attention": 16,
        "max_position_embeddings": 256,
    }
    agent_cfg = {
        "encoder": "test/tiny",
        "head_layers": 1,
        "max_len": 96,
        "head_max_len": 40,
        "act_costs": {"escalate": 0.5},
        "temperature": [1.3, 1.1, 2.0],
        "temperature_by_options": {"choice:3-5": 0.1},
    }
    path = tmp_path / "base"
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    (path / "encoder/config.json").write_text(json.dumps(cfg))
    (path / "rl_agent_config.json").write_text(json.dumps(agent_cfg))
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {t: i for i, t in enumerate(specials + WORDS + [":", "?", "level", "0", "1", "2"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer/tokenizer.json"))
    names = ("pad", "cls", "sep", "mask")
    tokens = ("[PAD]", "[CLS]", "[SEP]", "[MASK]")
    (path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps({f"{n}_token": t for n, t in zip(names, tokens)})
    )
    mx.random.seed(3)
    model = DecisionModel(EncoderConfig.from_dict(cfg), agent_cfg)
    upstream = {engine.upstream_name(k): v for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(path / "model.safetensors"), upstream)
    return path


def make_rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = ["alpha", "beta", "gamma"][i % 3]
        color = ["red", "green", "blue"][i % 3]
        noise = " ".join(rng.choice(WORDS, 3))
        rows.append(
            {
                "state": f"{color} {noise}",
                "answers": {"topic": label, "level": i % 3, "flag": i % 2 == 0},
            }
        )
    return rows


def test_upstream_names_round_trip(checkpoint):
    header = engine.safetensors_header(checkpoint / "model.safetensors")
    assert "head.layers.0.self_attn.in_proj_weight" in header
    assert "scorer.0.weight" in header and "act_head.2.weight" in header
    mlx_names = set(sanitize_weights({k: 0 for k in header}))
    assert {engine.upstream_name(k) for k in mlx_names} == set(header)


def test_answer_targets_accept_common_spellings():
    choice, score, noul = QUESTIONS["topic"], QUESTIONS["level"], QUESTIONS["flag"]
    assert engine.answer_target(choice, "Beta") == [0.0, 1.0, 0.0]
    assert engine.answer_target(choice, {"alpha": 3, "gamma": 1}) == [0.75, 0.0, 0.25]
    assert engine.answer_target(score, "2") == [0.0, 0.0, 1.0]
    assert engine.answer_target(score, "mid") == [0.0, 1.0, 0.0]
    assert engine.answer_target(noul, "yes") == [0.0, 1.0]
    assert engine.answer_target(noul, "0") == [1.0, 0.0]
    assert engine.answer_target(noul, 0.8) == pytest.approx([0.2, 0.8])
    with pytest.raises(ValueError):
        engine.answer_target(choice, "delta")
    with pytest.raises(ValueError):
        engine.answer_target(score, 5)


def test_create_dataset_from_jsonl_and_csv(tmp_path):
    rows = make_rows(40)
    jsonl = "\n".join(json.dumps(r) for r in rows) + '\n{"state": "red", "answers": {"topic": "x"}}'
    meta = engine.create_dataset("demo", QUESTIONS, jsonl, "train.jsonl", workspace=tmp_path)
    assert meta["error_count"] == 1 and "unknown label" in meta["errors"][0]["error"]
    assert sum(meta["rows"].values()) == 40
    assert meta["rows"]["val"] >= 1 and meta["rows"]["test"] >= 1
    csv = "text,topic,flag\n" + "\n".join(f"red blue,{r['answers']['topic']},yes" for r in rows)
    meta2 = engine.create_dataset("csv", QUESTIONS, csv, "train.csv", workspace=tmp_path)
    questions, loaded, _ = engine.load_dataset(meta2["id"], tmp_path)
    assert set(loaded[0]["targets"]) == {"topic", "flag"}
    with pytest.raises(ValueError, match="already exists"):
        engine.create_dataset("csv", QUESTIONS, csv, "train.csv", workspace=tmp_path)


def test_training_logits_match_inference_model(checkpoint):
    hp = {**engine.HYPERPARAMETERS, "precision": "float32"}
    model, cfg = engine.load_training_model(checkpoint, hp)
    agent = laya_mlx.load(str(checkpoint), dtype="float32")
    items, internal = agent.prepare("red green", QUESTIONS)
    for item in items:
        item["target"] = [0.0] * len(item["markers"])
        item["weight"] = 1.0
    batch = engine.collate(items, agent.tok.pad_token_id)
    # LoRA B starts at zero, so the adapted model must equal the original exactly.
    ours = engine.decision_logits(model, batch, training=False)
    reference, _ = agent.model(
        **{k: batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask")},
        qtype=batch["qtype"],
    )
    reference = mx.where(batch["marker_mask"], reference, -1e4)
    np.testing.assert_allclose(np.asarray(ours), np.asarray(reference), atol=1e-4)


def test_lora_fusion_is_exact():
    import mlx.nn as nn

    mx.random.seed(0)
    base = nn.Linear(8, 6, bias=False)
    lora = engine.lora_class()(base, rank=4, alpha=8, dropout=0.0)
    lora.lora_b = mx.random.normal(lora.lora_b.shape)
    lora.eval()
    x = mx.random.normal((3, 8))
    np.testing.assert_allclose(
        np.asarray(lora(x)), np.asarray(lora.fused(mx.float32)(x)), rtol=1e-5, atol=1e-5
    )


def test_temperature_fit_recovers_overconfidence():
    rng = np.random.default_rng(0)
    triples = []
    for _ in range(400):
        z = rng.normal(size=4)
        p = np.exp(z) / np.exp(z).sum()
        target = np.eye(4)[rng.choice(4, p=p)]
        triples.append((0, z * 3.0, target))  # logits three times too sharp
    assert engine.fit_temperature(triples) == pytest.approx(3.0, rel=0.2)


def test_statistics_helpers():
    assert engine.mcnemar(0, 0) == 1.0
    assert engine.mcnemar(0, 10) < 0.01
    low, high = engine.wilson(80, 100)
    assert low < 0.8 < high


def test_end_to_end_training_writes_portable_checkpoint(checkpoint, tmp_path):
    workspace = tmp_path / "ws"
    rows = "\n".join(json.dumps(r) for r in make_rows(60))
    meta = engine.create_dataset("tiny", QUESTIONS, rows, "t.jsonl", workspace=workspace)
    analysis = engine.analyze_dataset(meta["id"], checkpoint, workspace)
    assert set(analysis["questions"]) == set(QUESTIONS)
    events = []
    spec = {
        "run_id": "tiny-run",
        "dataset": meta["id"],
        "base_model": f"path:{checkpoint}",
        "hyperparameters": {"epochs": 2, "batch_size": 4, "objective": "rlcd", "patience": 0},
    }
    engine.train(spec, lambda kind, **data: events.append((kind, data)), workspace)
    kinds = [k for k, _ in events]
    for kind in ("phase", "info", "step", "epoch", "calibration", "result"):
        assert kind in kinds
    out = workspace / "runs/tiny-run/model"
    assert set(engine.safetensors_header(out / "model.safetensors")) == set(
        engine.safetensors_header(checkpoint / "model.safetensors")
    )
    cfg = json.loads((out / "rl_agent_config.json").read_text())
    assert cfg["fine_tuned"]["method"] == "lora"
    assert all(0.5 <= t <= 5.0 for t in cfg["temperature"])
    assert json.loads((out / "questions.json").read_text()) == QUESTIONS
    result = laya_mlx.load(str(out)).predict("red", QUESTIONS)
    assert all(math.isfinite(a.get("confidence", 0)) for a in result["answers"].values())
    comparison = json.loads((workspace / "runs/tiny-run/comparison.json").read_text())
    assert comparison["base"]["overall"]["n"] == comparison["finetuned"]["overall"]["n"]
    assert "p_value" in comparison["paired"]["overall"]


@pytest.mark.parametrize("method", ["head", "full"])
def test_other_methods_train(checkpoint, tmp_path, method):
    workspace = tmp_path / "ws"
    rows = "\n".join(json.dumps(r) for r in make_rows(30))
    meta = engine.create_dataset("tiny", QUESTIONS, rows, "t.jsonl", workspace=workspace)
    spec = {
        "run_id": f"tiny-{method}",
        "dataset": meta["id"],
        "base_model": f"path:{checkpoint}",
        "baseline": False,
        "hyperparameters": {"method": method, "epochs": 1, "batch_size": 8, "lr": 1e-4},
    }
    summary = engine.fit(
        spec, {**engine.HYPERPARAMETERS, **spec["hyperparameters"]}, lambda *a, **k: None, workspace
    )
    assert summary["trainable_params"] > 0
    assert (workspace / f"runs/tiny-{method}/model/model.safetensors").exists()
