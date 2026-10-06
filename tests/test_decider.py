"""Decider: its prompt, answers and calibration; LoRA on PyTorch (PEFT) and MLX-LM; GGUF and
MLX exports; its NoulXP package.

The model is tests/tiny.py's random 2-layer Qwen3.5 (one Gated DeltaNet layer, one attention
layer), on the CPU. Decider's own prompt code is noulxp.native.decider, a rebuild of
decider-ai's checked token for token against it: the studio must build the same rows.
"""

import json
import math
import os
import random
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import tiny  # noqa: E402
from common import QUESTIONS, make_rows  # noqa: E402

from layastudio import decider, engine, kinds, noulxp_package  # noqa: E402
from layastudio.export import export  # noqa: E402


def importable(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


needs_peft = pytest.mark.skipif(not importable("peft"), reason="peft is not installed")
needs_mlx_lm = pytest.mark.skipif(not importable("mlx_lm"), reason="mlx-lm is not installed")
TOKENIZER = Path(
    os.environ.get(
        "DECIDER_TOKENIZER",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Mapika--decider-2b/snapshots/533964dae8be954c5b5e19fa4948e48408094c1e",
    )
)
# llama.cpp's converter knows Qwen's real tokenizer only, and is fetched into LAYASTUDIO_TOOLS.
GGUF_READY = (
    importable("llama_cpp")
    and (TOKENIZER / "tokenizer.json").is_file()
    and bool(os.environ.get("LAYASTUDIO_TOOLS"))
)
needs_gguf = pytest.mark.skipif(
    not GGUF_READY,
    reason="needs llama-cpp-python, Decider's tokenizer in the HF cache and LAYASTUDIO_TOOLS",
)

MORE = {
    **QUESTIONS,
    "described": {
        "type": "choice",
        "instructions": "Which colour?",
        "criteria": {"red": "warm", "green": None, "blue": {"tone": "cold"}},
    },
    "wide": {"type": "choice", "instructions": "Pick", "criteria": [f"o{i}" for i in range(14)]},
    "untold": {"type": "noul", "instructions": "", "criteria": {"true": "it is red"}},
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tiny.decider_checkpoint(tmp_path_factory.mktemp("decider") / "decider")


def answers_close(a, b, tol):
    worst = 0.0
    for qid, x in a.items():
        y = b[qid]
        if x["type"] == "noul":
            worst = max(worst, abs(x["noul"] - y["noul"]))
        else:
            for key, value in x["probabilities"].items():
                worst = max(worst, abs(value - y["probabilities"][key]))
    assert worst < tol, worst
    return worst


def test_the_checkpoint_is_a_decider_checkpoint(checkpoint):
    assert kinds.check(checkpoint) == kinds.DECIDER
    assert kinds.FAMILY[kinds.DECIDER] == "letter"
    assert kinds.PROFILE[kinds.DECIDER] == "causal-letters"
    with pytest.raises(ValueError, match="plain layout"):
        bad = checkpoint.parent / "chat"
        shutil.copytree(checkpoint, bad)
        (bad / "decider_config.json").write_text(json.dumps({"layout": "chat"}))
        decider.config(bad)


def test_the_prompt_is_deciders_own(checkpoint):
    """Every row, token for token, as noulxp.native.decider (decider-ai's prompt code) builds
    it: narrow and wide option lists, isolated score levels, JSON states, noul fallbacks."""
    native = pytest.importorskip("noulxp.native.decider")
    theirs, cfg = native.load_prompter(checkpoint)
    ours = decider.Prompter(decider.Tokens(checkpoint), decider.config(checkpoint))
    assert ours.label_ids == theirs.label_ids
    for state in ("red green blue", {"rows": list(range(9)), "note": "élan"}):
        plan, _ = theirs.plan(state, MORE, True)
        context = ours.context(state)
        for qid, _, kind, rows in plan:
            mine_kind, pieces = ours.rows(decider.render_question(MORE[qid]))
            assert mine_kind == kind
            built = [(context + ours.question(t, o), len(o)) for t, o in pieces]
            assert built == [(ids, n) for ids, n in rows], qid


def test_options_follow_the_studio_order():
    for qdef in MORE.values():
        if qdef["type"] != "score":
            assert len(decider.render_question(qdef)["options"]) == len(engine.option_labels(qdef))
    rq = decider.render_question(MORE["described"])
    assert rq["options"] == ["red: warm", "green", 'blue: {"tone": "cold"}']
    assert decider.render_question(MORE["untold"])["question"] == decider.NOUL_WITHOUT_INSTRUCTIONS


def test_training_rows(checkpoint):
    prompter = decider.Prompter(decider.Tokens(checkpoint), decider.config(checkpoint))
    rows = [{"state": "red", "targets": {"level": [0.0, 1.0, 0.0], "flag": [0.0, 1.0]}}]
    items, skipped = decider.encode_items(prompter, rows, QUESTIONS)
    assert skipped == 0 and len(items) == 4  # 3 isolated levels + 1 noul
    levels = [it for it in items if it["type"] == "score"]
    assert [it["target"] for it in levels] == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    assert all(it["n"] == 2 for it in items)
    wide = [{"state": "x", "targets": {"wide": [0.0] * 13 + [1.0]}}]
    hp = {**decider.HYPERPARAMETERS, "max_train_options": 5}
    items, _ = decider.encode_items(prompter, wide, MORE, random.Random(0), True, None, hp)
    (item,) = items
    assert item["n"] == 5 and sum(item["target"]) == pytest.approx(1.0)
    assert max(item["target"]) == 1.0  # the gold is always kept


def test_isolated_scores_assemble_as_decider_does():
    qdef = QUESTIONS["level"]
    rows = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]
    answer = decider.assemble(qdef, "iso", rows)
    fit = [0.1, 0.8, 0.4]
    assert answer["probabilities"]["1"] == pytest.approx(0.8 / sum(fit))
    assert answer["fit_mass"] == pytest.approx(sum(fit))
    assert answer["score"] == pytest.approx(sum(i * f / sum(fit) for i, f in enumerate(fit)))


def test_calibration_finds_the_temperature_that_made_the_data():
    rng = random.Random(0)
    answers = []
    for _ in range(400):
        z = [rng.gauss(0, 3) for _ in range(4)]
        p = decider.softmax(z, 1.8)
        gold = rng.choices(range(4), weights=p)[0]
        answers.append(("choice", "list", [z], [float(i == gold) for i in range(4)]))
    by_type, fitted = decider.fit_temperatures(answers, {"temperature": 1.0})
    assert fitted == ["choice"] and by_type["choice"] == pytest.approx(1.8, rel=0.15)
    assert by_type["noul"] == 1.0  # too few answers: the base's temperature


@needs_mlx_lm
def test_mlx_and_pytorch_read_the_same_letters(checkpoint):
    a = decider.Agent(checkpoint, backend="torch").predict("red green blue", MORE)["answers"]
    b = decider.Agent(checkpoint, backend="mlx").predict("red green blue", MORE)["answers"]
    answers_close(a, b, 1e-5)


def workspace_with_data(root, n=120):
    workspace = root / "ws"
    text = "\n".join(json.dumps(r) for r in make_rows(n))
    meta = engine.create_dataset("tiny", QUESTIONS, text, "t.jsonl", workspace=workspace)
    return workspace, meta


def train(fit, checkpoint, workspace, meta, run_id, **hp):
    spec = {"run_id": run_id, "dataset": meta["id"], "base_model": f"path:{checkpoint}"}
    events = []
    settings = {
        **engine.hyperparameters("decider"),
        "epochs": 3,
        "batch_size": 8,
        "lr": 3e-3,
        "precision": "float32",
        "grad_checkpoint": "off",
        **hp,
    }
    summary = fit(spec, settings, lambda kind, **d: events.append((kind, d)), workspace)
    return summary, workspace / "runs" / run_id / "model"


def check_written(base, out, summary):
    """Decider's own files, the base's tensor names and dtypes, the fitted temperatures."""
    assert kinds.check(out) == kinds.DECIDER
    names = {
        k: v["dtype"]
        for f in base.glob("*.safetensors")
        for k, v in engine.safetensors_header(f).items()
    }
    written = {
        k: v["dtype"]
        for f in out.glob("*.safetensors")
        for k, v in engine.safetensors_header(f).items()
    }
    assert written == names
    cfg = json.loads((out / "decider_config.json").read_text())
    assert cfg["layout"] == "plain" and cfg["isolated_levels"] is True
    assert cfg["temperature_by_type"] == summary["calibration"]["temperature_by_type"]
    assert cfg["fine_tuned"]["tool"] == "System One Studio"
    assert (out / "questions.json").is_file() and (out / "tokenizer.json").is_file()
    assert summary["best_epoch"] > 0


@needs_peft
def test_pytorch_lora_writes_a_decider_checkpoint(checkpoint, tmp_path, monkeypatch):
    from layastudio import decider_engine

    monkeypatch.setenv("LAYASTUDIO_DEVICE", "cpu")
    workspace, meta = workspace_with_data(tmp_path)
    summary, out = train(
        decider_engine.torch_fit, checkpoint, workspace, meta, "dt", loraplus_ratio=2.0
    )
    check_written(checkpoint, out, summary)
    assert summary["backend"] == "torch" and summary["trainable_params"] > 0
    answers = decider.Agent(out, backend="torch").predict("red low", QUESTIONS)["answers"]
    assert set(answers) == set(QUESTIONS)
    # The adapters were merged: the weights moved, the shapes did not.
    before = engine.safetensors_header(checkpoint / "model.safetensors")
    after = engine.safetensors_header(out / "model.safetensors")
    assert {k: v["shape"] for k, v in before.items()} == {k: v["shape"] for k, v in after.items()}


def test_qlora_off_cuda_is_refused_before_the_baseline(monkeypatch):
    """4-bit QLoRA needs CUDA on PyTorch: said before the baseline evaluates anything."""
    from layastudio import decider_engine

    monkeypatch.setenv("LAYASTUDIO_DEVICE", "cpu")
    hp = {**decider.HYPERPARAMETERS, "quantization": "4bit"}
    ran = []
    with pytest.raises(ValueError, match="4-bit QLoRA needs an NVIDIA GPU"):
        decider_engine.torch_fit({}, hp, lambda *a, **k: None, before_model=lambda: ran.append(1))
    assert ran == []


@needs_mlx_lm
def test_mlx_lora_writes_the_same_checkpoint_format(checkpoint, tmp_path):
    from layastudio import decider_mlx

    workspace, meta = workspace_with_data(tmp_path)
    summary, out = train(decider_mlx.fit, checkpoint, workspace, meta, "dm", dora=True)
    check_written(checkpoint, out, summary)
    assert summary["backend"] == "mlx"
    # Written by MLX, read by PyTorch and by MLX: the same answers.
    a = decider.Agent(out, backend="torch").predict("green mid", QUESTIONS)["answers"]
    b = decider.Agent(out, backend="mlx").predict("green mid", QUESTIONS)["answers"]
    answers_close(a, b, 1e-5)


# A question of more options than an epoch's rows carry (max_train_options, 10).
WIDE = {
    "wide": {"type": "choice", "instructions": "Which?", "criteria": [f"l{i}" for i in range(25)]}
}


def wide_rows(n=90):
    from layastudio import datasets

    text = "\n".join(
        json.dumps({"state": f"red {i}", "answers": {"wide": f"l{i % 25}"}}) for i in range(n)
    )
    questions, rows, _ = datasets.validate(WIDE, text, "w.jsonl", seed=13)
    return questions, rows


def test_updates_are_counted_as_an_epoch_encodes_its_rows(checkpoint):
    """Above max_train_options an epoch's rows carry 10 options: their batches, not those of
    rows with every option, are the updates the schedule and the progress count."""
    questions, rows = wide_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    hp = {**decider.HYPERPARAMETERS, "batch_size": 8, "batch_tokens": 256}
    prompter = decider.Prompter(decider.Tokens(checkpoint), decider.config(checkpoint))

    def batches(items):
        return len(
            decider.token_batches(items, hp["batch_size"], hp["batch_tokens"], random.Random(0))
        )

    probe, _ = decider.training_probe(prompter, train_rows, questions, hp)
    every, _ = decider.encode_items(prompter, train_rows, questions, hp=hp)
    rng = random.Random(hp["seed"])
    epochs = [
        batches(decider.encode_items(prompter, train_rows, questions, rng, True, None, hp)[0])
        for _ in range(3)
    ]
    assert all(abs(batches(probe) - e) <= max(1, e // 10) for e in epochs)
    assert batches(every) > 1.3 * max(epochs)  # every option: longer rows, more batches
    # Without shuffled options an epoch's rows carry every option, and so does the probe.
    plain = {**hp, "shuffle_options": False}
    assert batches(decider.training_probe(prompter, train_rows, questions, plain)[0]) == batches(
        every
    )


def step_events(fit, checkpoint, tmp_path, epochs=2):
    workspace = tmp_path / "ws"
    text = "\n".join(
        json.dumps({"state": f"red {i}", "answers": {"wide": f"l{i % 25}"}}) for i in range(90)
    )
    meta = engine.create_dataset("wide", WIDE, text, "w.jsonl", workspace=workspace)
    spec = {"run_id": "steps", "dataset": meta["id"], "base_model": f"path:{checkpoint}"}
    events = []
    hp = {
        **engine.hyperparameters("decider"),
        "epochs": epochs,
        "batch_size": 4,
        "batch_tokens": 256,
        "patience": 0,
        "precision": "float32",
        "grad_checkpoint": "off",
    }
    fit(spec, hp, lambda kind, **d: events.append({"type": kind, **d}), workspace)
    return events


def check_steps(events, epochs):
    """Step events all through training: the epoch of epochs, a fraction that only grows and
    reaches 1 at the last update, and the planned updates within an epoch's worth of it."""
    steps = [e for e in events if e["type"] == "step"]
    assert len(steps) >= 2 * epochs
    assert all(e["epochs"] == epochs and 0 < e["fraction"] <= 1 for e in steps)
    assert [e["fraction"] for e in steps] == sorted(e["fraction"] for e in steps)
    assert steps[-1]["fraction"] == 1 and steps[-1]["eta_s"] == 0
    assert {e["epoch"] for e in steps} == set(range(1, epochs + 1))
    planned = next(e for e in events if e["type"] == "info")["updates"]
    assert abs(planned - len(steps)) <= max(epochs, len(steps) // 10)


@needs_peft
def test_pytorch_training_reports_its_progress_in_steps(checkpoint, tmp_path, monkeypatch):
    from layastudio import decider_engine

    monkeypatch.setenv("LAYASTUDIO_DEVICE", "cpu")
    check_steps(step_events(decider_engine.torch_fit, checkpoint, tmp_path), 2)


@needs_mlx_lm
def test_mlx_training_reports_its_progress_in_steps(checkpoint, tmp_path):
    from layastudio import decider_mlx

    check_steps(step_events(decider_mlx.fit, checkpoint, tmp_path), 2)


@needs_mlx_lm
def test_the_mlx_export_is_measured(checkpoint, tmp_path):
    from layastudio import decider_mlx

    workspace, meta = workspace_with_data(tmp_path, 60)
    run_dir = workspace / "runs" / "dq"
    shutil.copytree(checkpoint, run_dir / "model")
    engine.write_json(run_dir / "run.json", {"id": "dq", "dataset": meta["id"]})
    report = decider_mlx.export("run:dq", workspace, precision="int8")
    assert report["verification"]["rows"] > 0
    assert report["verification"]["max_probability_difference"] < 0.05
    config = json.loads((run_dir / "exports/mlx-int8/config.json").read_text())
    assert config["quantization"]["bits"] == 8 and config["model_type"] == "qwen3_5"


def test_the_analysis_counts_rows_per_decision(checkpoint, tmp_path):
    workspace, meta = workspace_with_data(tmp_path, 30)
    report = decider.analyze(meta["id"], checkpoint, workspace)
    assert report["questions"]["level"]["rows_per_decision"] == 3
    assert report["questions"]["topic"]["readout"] == "letters"


# ----------------------------------------------------------------------------- GGUF


@pytest.fixture(scope="module")
def real_tokenizer_checkpoint(tmp_path_factory):
    if not GGUF_READY:
        pytest.skip("GGUF tooling")
    return tiny.decider_checkpoint(
        tmp_path_factory.mktemp("decider-gguf") / "decider", tokenizer_dir=TOKENIZER
    )


@needs_gguf
def test_gguf_conversion_reads_the_same_letters(real_tokenizer_checkpoint, tmp_path):
    from layastudio import gguf

    ck = real_tokenizer_checkpoint
    out = gguf.convert(ck, tmp_path / "tiny.gguf", "bf16")
    prompter = decider.Prompter(decider.Tokens(ck), decider.config(ck))
    context = prompter.context("The sky is a deep blue today.")
    rows = []
    for qdef in (QUESTIONS["topic"], QUESTIONS["level"], MORE["wide"]):
        _, pieces = prompter.rows(decider.render_question(qdef))
        rows += [(context + prompter.question(t, o), len(o)) for t, o in pieces]
    reader = gguf.Reader(out)
    try:
        got = [reader.logits(ids, prompter.label_ids[:n]) for ids, n in rows]
    finally:
        reader.close()
    want = decider.Agent(ck, backend="torch").row_logits(rows)
    result = gguf.compare(want, got, [{"n": n} for _, n in rows])
    assert result["same_answer"] == len(rows) and result["max_probability_difference"] < 5e-3


@needs_gguf
def test_a_decider_fine_tune_gets_a_noulxp_package_that_passes(real_tokenizer_checkpoint, tmp_path):
    """The run's GGUF (bf16 here, small), noulxp export decider, conformance from Decider's own
    GGUF readout, validate and check: the whole build on the CPU."""
    if noulxp_package.missing_tooling("decider"):
        pytest.skip(noulxp_package.missing_tooling("decider"))
    workspace, meta = workspace_with_data(tmp_path, 30)
    run_dir = workspace / "runs" / "dn"
    shutil.copytree(real_tokenizer_checkpoint, run_dir / "model")
    engine.write_json(run_dir / "model/questions.json", QUESTIONS)
    engine.write_json(
        run_dir / "run.json",
        {"id": "dn", "dataset": meta["id"], "base_model": "hub:Mapika/decider-2b"},
    )
    report = export("run:dn", "noulxp", workspace, test_rows=3, gguf="bf16")
    assert report["state"] == "passed" and report["profile"] == "causal-letters"
    check = report["check"]
    assert check["cases_passed"] == check["cases"] and check["max_abs_dp"] < 0.01
    manifest = json.loads((run_dir / "noulxp/noulxp.json").read_text())
    assert manifest["weights"]["format"] == "gguf"
    assert manifest["source"]["gguf"]["precision"] == "bf16"
    calibration = json.loads((run_dir / "noulxp/calibration.json").read_text())
    assert calibration["temperature"]["noul"] == pytest.approx(1.5)
    info = noulxp_package.passing_package(run_dir, run_dir / "model")
    assert info and "GGUF readout" in noulxp_package.card_line(info)

    # A publish dry run carries Decider's own files, the package (its GGUF) and a Decider card.
    from common import fake_systemone

    from layastudio import publish_systemone

    command, seen = fake_systemone(tmp_path)
    publish_systemone.cli_command, saved = (lambda: command), publish_systemone.cli_command
    try:
        publish_systemone.publish("run:dn", "me/decider-tiny", workspace, dry_run=True)
    finally:
        publish_systemone.cli_command = saved
    pushed = json.loads(seen.read_text())
    for name in ("decider_config.json", "config.json", "model.safetensors", "noulxp/model.gguf"):
        assert name in pushed["files"], name
    card = (run_dir / "model/README.md").read_text()
    assert "Mapika" in card and "base_model: Mapika/decider-2b" in card
    # The package is the run's own GGUF, converted from these weights: change them, and it
    # no longer counts.
    weights = run_dir / "model/model.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 1
    weights.write_bytes(bytes(data))
    assert noulxp_package.passing_package(run_dir, run_dir / "model") is None
    assert math.isfinite(report["seconds"])
