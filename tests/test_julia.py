"""Julia 1: its prompt, its checkpoint, its trainers on both backends and its NoulXP package.

Every model here is tests/tiny.py's random 2-layer Julia checkpoint, on the CPU (MLX tests run
where MLX is installed). Julia's own inference is noulxp.native.julia, a rebuild of
Supersonic Labs' code checked against their 100 parity cases: the studio must agree with it.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("laya")

import tiny  # noqa: E402
from common import QUESTIONS, make_rows  # noqa: E402

from layastudio import engine, julia, kinds, noulxp_package, torch_engine  # noqa: E402
from layastudio.export import export  # noqa: E402

try:
    import mlx.core  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed")

MORE = {
    **QUESTIONS,
    "described": {
        "type": "choice",
        "instructions": "Which colour is named first?",
        "criteria": {"red": "red", "green": None, "blue": "blue green"},
    },
    "flag2": {
        "type": "noul",
        "instructions": "Flagged?",
        "criteria": {"true": "yes red", "false": "no blue"},
    },
}
STATES = ["red green blue", "alpha beta " * 30, {"board": ["red", "blue"], "note": "mid"}]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tiny.julia_checkpoint(tmp_path_factory.mktemp("julia") / "julia")


def answers_close(a, b, tol):
    worst = 0.0
    for qid, x in a.items():
        y = b[qid]
        if x["type"] == "noul":
            worst = max(worst, abs(x["noul"] - y["noul"]))
        else:
            worst = max(
                worst,
                max(abs(x["probabilities"][k] - y["probabilities"][k]) for k in x["probabilities"]),
            )
    assert worst < tol, worst
    return worst


def test_the_checkpoint_is_a_julia_checkpoint(checkpoint):
    assert kinds.check(checkpoint) == kinds.JULIA
    assert kinds.FAMILY[kinds.JULIA] == "laya" and kinds.PROFILE[kinds.JULIA] == "encoder-markers"
    cfg = julia.config(checkpoint)
    assert (cfg["max_len"], cfg["head_max_len"], cfg["head_layers"]) == (128, 48, 2)


def test_options_follow_the_studio_order():
    for qdef in MORE.values():
        assert len(julia.option_texts(qdef)) == len(engine.option_labels(qdef))
    assert julia.option_texts(MORE["described"]) == ["red", "green", "blue green"]
    assert julia.option_texts(MORE["flag2"]) == ["no blue", "yes red"]
    assert julia.option_texts(QUESTIONS["flag"]) == ["false", "true"]
    assert julia.option_texts(QUESTIONS["level"]) == ["low", "mid", "high"]


def test_the_prompt_is_julias_own(checkpoint):
    """Token for token what noulxp.native.julia (Julia's data.py) builds, strict or not."""
    native = pytest.importorskip("noulxp.native.julia")
    tok = julia.tokenizer(checkpoint)
    theirs = native._Tokens(checkpoint / "tokenizer")
    for state in STATES:
        text = state if isinstance(state, str) else json.dumps(state)
        rows, meta = native._rows(text, MORE)
        for (qid, kind, keys, _), row in zip(meta, rows):
            assert keys == engine.option_labels(MORE[qid])
            expected = native._sequence(theirs, row, 128, 48)
            ids, markers, _ = julia.sequence(
                tok,
                text,
                kind,
                julia.instructions(MORE[qid]),
                julia.option_texts(MORE[qid]),
                128,
                48,
                strict=True,
            )
            assert (ids, markers) == (expected["ids"], expected["markers"])
    with pytest.raises(ValueError, match="too long"):
        julia.sequence(tok, "red " * 200, "choice", "Pick", ["red", "blue"], 128, 48, strict=True)
    ids, _, cut = julia.sequence(tok, "red " * 200, "choice", "Pick", ["red", "blue"], 128, 48)
    assert cut and len(ids) == 128  # the state is cut instead, as Julia's training does


def test_answers_match_julias_own_runtime(checkpoint):
    native = pytest.importorskip("noulxp.native.julia")
    ours = julia.Agent(checkpoint, backend="torch")
    theirs = native.Julia(checkpoint, threads=1)
    for state in ("red green blue", "mid low high"):
        # theirs rounds to 4 decimals
        answers_close(
            ours.predict(state, MORE)["answers"], theirs.system_one(state, MORE)["answers"], 1e-4
        )


@needs_mlx
def test_mlx_and_pytorch_give_the_same_answers(checkpoint):
    a = julia.Agent(checkpoint, backend="torch").predict("red green blue", MORE)["answers"]
    b = julia.Agent(checkpoint, backend="mlx").predict("red green blue", MORE)["answers"]
    answers_close(a, b, 1e-5)


def test_the_temperature_folds_exactly():
    rng = np.random.default_rng(0)
    w, b = rng.normal(size=(1, 8)), rng.normal(size=1)
    tensors = {"scorer.3.weight": torch.tensor(w), "scorer.3.bias": torch.tensor(b)}
    julia.fold_temperature(tensors, 1.7)
    h = rng.normal(size=(5, 8))
    folded = h @ tensors["scorer.3.weight"].numpy().T + tensors["scorer.3.bias"].numpy()
    np.testing.assert_allclose(folded, (h @ w.T + b) / 1.7)


def workspace_with_data(root, n=120):
    workspace = root / "ws"
    text = "\n".join(json.dumps(r) for r in make_rows(n))
    meta = engine.create_dataset("tiny", QUESTIONS, text, "t.jsonl", workspace=workspace)
    return workspace, meta


def train(fit, checkpoint, workspace, meta, run_id, **hp):
    spec = {"run_id": run_id, "dataset": meta["id"], "base_model": f"path:{checkpoint}"}
    events = []
    settings = {
        **engine.hyperparameters("julia"),
        "epochs": 3,
        "batch_size": 8,
        "head_lr": 2e-3,
        "lr": 2e-3,
        **hp,
    }
    summary = fit(spec, settings, lambda kind, **d: events.append((kind, d)), workspace)
    return summary, workspace / "runs" / run_id / "model", events


def check_written(base, out):
    """A Julia checkpoint: Julia's files, the base's tensor names, float32, loadable by
    Julia's own runtime; the policy names the new weights and the folded temperature."""
    assert kinds.check(out) == kinds.JULIA
    header = engine.safetensors_header(out / "model.safetensors")
    assert set(header) == set(engine.safetensors_header(base / "model.safetensors"))
    assert {v["dtype"] for v in header.values()} == {"F32"}
    policy = json.loads((out / "inference-policy.json").read_text())
    assert policy["calibration"]["folded_into"] == "scorer.3"
    import hashlib

    assert (
        policy["weights_sha256"]
        == hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()
    )
    assert json.loads((out / "julia_config.json").read_text())["weight_dtype"] == "float32"
    native = pytest.importorskip("noulxp.native.julia")
    native.Julia(out, threads=1)  # loads strictly


def test_pytorch_training_writes_a_julia_checkpoint(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setenv("LAYASTUDIO_DEVICE", "cpu")
    workspace, meta = workspace_with_data(tmp_path)
    summary, out, events = train(
        torch_engine.fit, checkpoint, workspace, meta, "jt", dora=True, loraplus_ratio=2.0
    )
    check_written(checkpoint, out)
    assert summary["kind"] == "julia" and summary["best_epoch"] > 0
    history = [h["val_loss"] for h in summary["history"]]
    first = next(d["val_loss"] for k, d in events if k == "epoch" and d["epoch"] == 0)
    assert min(history) < first  # it learned something
    # What Julia's runtime reads is the trained model with the temperature folded in.
    T = summary["calibration"]["folded_temperature"]
    agent = julia.Agent(out, backend="torch")
    native = pytest.importorskip("noulxp.native.julia").Julia(out, threads=1)
    answers_close(
        agent.predict("red low", QUESTIONS)["answers"],
        native.system_one("red low", QUESTIONS)["answers"],
        1e-4,
    )
    assert 0.5 <= T <= 5.0


def test_a_run_that_cannot_train_stops_before_the_baseline(checkpoint, tmp_path, monkeypatch):
    """The trainer encodes its rows before the base model is evaluated: a run none of whose
    decisions fit the model's budget fails before the baseline's GPU minutes, not after."""
    monkeypatch.setenv("LAYASTUDIO_BACKEND", "torch")
    monkeypatch.setenv("LAYASTUDIO_DEVICE", "cpu")
    workspace, meta = workspace_with_data(tmp_path, 30)
    spec = {"run_id": "jb", "dataset": meta["id"], "base_model": f"path:{checkpoint}"}
    events, baselines = [], []

    class Stop(Exception):
        pass

    def baseline(*args):
        baselines.append([kind for kind, _ in events])
        raise Stop

    monkeypatch.setattr(engine, "baseline", baseline)
    with pytest.raises(Stop):  # the rows are encoded first, then the baseline runs
        engine.train(spec, lambda kind, **d: events.append((kind, d)), workspace)
    assert baselines == [["phase"]] and events[0][1]["phase"] == "prepare"
    monkeypatch.setattr(julia, "encode_items", lambda *a, **k: ([], 99))  # nothing fits
    with pytest.raises(ValueError, match="No training decisions fit"):
        engine.train(spec, lambda kind, **d: None, workspace)
    assert len(baselines) == 1


def test_frozen_weights_are_written_as_the_base_has_them(checkpoint, tmp_path, monkeypatch):
    """bfloat16 training never rounds the float32 checkpoint: untouched tensors are the
    base's bit for bit, and adapted ones are the base plus the LoRA update."""
    from safetensors.torch import load_file

    monkeypatch.setenv("LAYASTUDIO_DEVICE", "cpu")
    workspace, meta = workspace_with_data(tmp_path, 40)
    _, out, _ = train(
        torch_engine.fit, checkpoint, workspace, meta, "jbf", precision="bfloat16", epochs=1
    )
    base = load_file(str(checkpoint / "model.safetensors"))
    tuned = load_file(str(out / "model.safetensors"))
    for frozen in ("encoder.embeddings.tok_embeddings.weight", "encoder.final_norm.weight"):
        assert torch.equal(base[frozen], tuned[frozen]), frozen
    adapted = "encoder.layers.1.mlp.Wo.weight"
    delta = (tuned[adapted] - base[adapted]).abs().max().item()
    assert 0 < delta < 1  # a LoRA update, on top of the float32 base


@needs_mlx
def test_mlx_training_writes_the_same_kind_of_checkpoint(checkpoint, tmp_path):
    workspace, meta = workspace_with_data(tmp_path)
    summary, out, _ = train(engine.fit, checkpoint, workspace, meta, "jm", rslora=True)
    check_written(checkpoint, out)
    assert summary["backend"] == "mlx" and summary["best_epoch"] > 0
    # Written by MLX, read by PyTorch: the same answers as MLX gives.
    a = julia.Agent(out, backend="mlx").predict("green mid", QUESTIONS)["answers"]
    b = julia.Agent(out, backend="torch").predict("green mid", QUESTIONS)["answers"]
    answers_close(a, b, 1e-5)


def test_the_dataset_analysis_counts_what_julia_would_refuse(checkpoint, tmp_path):
    workspace, meta = workspace_with_data(tmp_path, 30)
    report = julia.analyze(meta["id"], checkpoint, workspace)
    assert report["kind"] == "julia" and set(report["questions"]) == set(QUESTIONS)
    assert all(q["options"] in (2, 3) for q in report["questions"].values())


# ----------------------------------------------------------------------------- NoulXP

TOOLING = noulxp_package.missing_tooling("julia")


@pytest.mark.skipif(TOOLING is not None, reason=f"NoulXP tooling: {TOOLING}")
def test_a_julia_fine_tune_gets_a_noulxp_package_that_passes(checkpoint, tmp_path):
    """noulxp export julia, conformance from Julia's own runtime, validate and check: the
    whole build on the run's checkpoint, on the CPU."""
    workspace, meta = workspace_with_data(tmp_path, 30)
    run_dir = workspace / "runs" / "jn"
    run_dir.mkdir(parents=True)
    import shutil

    shutil.copytree(checkpoint, run_dir / "model")
    engine.write_json(run_dir / "model/questions.json", QUESTIONS)
    engine.write_json(
        run_dir / "run.json",
        {"id": "jn", "dataset": meta["id"], "base_model": "hub:SupersonicLabs/Julia-1"},
    )
    report = export("run:jn", "noulxp", workspace, test_rows=4)
    assert report["state"] == "passed" and report["kind"] == "julia"
    assert report["profile"] == "encoder-markers"
    check = report["check"]
    assert check["cases_passed"] == check["cases"] and check["max_abs_dp"] < 0.01
    manifest = json.loads((run_dir / "noulxp/noulxp.json").read_text())
    assert manifest["source"]["model"] == "SupersonicLabs/Julia-1"
    template = json.loads((run_dir / "noulxp/template.json").read_text())
    assert (template["budgets"]["total"], template["budgets"]["head"]) == (128, 48)
    info = noulxp_package.passing_package(run_dir, run_dir / "model")
    assert info and "Julia 1's own inference" in noulxp_package.card_line(info)

    # A publish dry run carries Julia's own files, the package and a Julia card.
    from common import fake_systemone

    from layastudio import publish_systemone

    command, seen = fake_systemone(tmp_path)
    publish_systemone.cli_command, saved = (lambda: command), publish_systemone.cli_command
    try:
        publish_systemone.publish("run:jn", "me/julia-tiny", workspace, dry_run=True)
    finally:
        publish_systemone.cli_command = saved
    pushed = json.loads(seen.read_text())
    assert "--dry-run" in pushed["args"]
    for name in (
        "julia_config.json",
        "inference-policy.json",
        "model.safetensors",
        "noulxp/model.onnx",
    ):
        assert name in pushed["files"], name
    card = (run_dir / "model/README.md").read_text()
    assert "Supersonic Labs" in card and "base_model: SupersonicLabs/Julia-1" in card
    assert "**NoulXP:** this version carries a NoulXP package" in card
