"""Headless runs: the checks every fine-tune starts with, shared with the server, and a run
from a config file with no UI. Nothing here needs MLX, PyTorch or a real model, except the
end-to-end runs, which use the tests' tiny random checkpoints.
"""

import json
import signal

import pytest
from common import QUESTIONS, make_rows
from test_server import call, studio  # noqa: F401 - studio is a fixture

from layastudio import cloud, datasets, engine, kinds

# A checkpoint folder of each kind, as far as the checks look: its files, not its weights.
FILES = {
    kinds.LAYA: kinds.REQUIRED[kinds.LAYA],
    kinds.JULIA: kinds.REQUIRED[kinds.JULIA],
    kinds.DECIDER: (*kinds.REQUIRED[kinds.DECIDER], "model.safetensors"),
}
WIDE = {
    "wide": {"type": "choice", "instructions": "Which?", "criteria": [f"l{i}" for i in range(25)]}
}


def fake_checkpoint(path, kind):
    for name in FILES[kind]:
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text("{}")
    return f"path:{path}"


def jsonl(rows):
    return "\n".join(json.dumps(r) for r in rows)


class Context:
    """A workspace with two datasets and a checkpoint of every kind, one with no licence."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.dataset = engine.create_dataset(
            "tiny", QUESTIONS, jsonl(make_rows(45)), "t.jsonl", workspace=workspace
        )["id"]
        wide = [{"state": f"red {i}", "answers": {"wide": f"l{i % 25}"}} for i in range(30)]
        self.wide = engine.create_dataset(
            "wide", WIDE, jsonl(wide), "w.jsonl", workspace=workspace
        )["id"]
        models = workspace.parent / "models"
        self.laya = fake_checkpoint(models / "laya", kinds.LAYA)
        self.julia = fake_checkpoint(models / "julia", kinds.JULIA)
        self.decider = fake_checkpoint(models / "decider", kinds.DECIDER)
        self.tev1 = fake_checkpoint(models / "tev1", kinds.DECIDER)
        self.empty = models / "empty"
        self.empty.mkdir()
        self.partial = models / "partial"
        self.partial.mkdir()
        (self.partial / "rl_agent_config.json").write_text("{}")
        # Imported from the registry as a model whose licence allows no derivatives.
        engine.write_json(
            workspace / "imports.json", [{"ref": self.tev1, "repo": "together-ai/tev1"}]
        )

    def spec(self, base="laya", **extra):
        return {"dataset": self.dataset, "base_model": getattr(self, base), **extra}


# Every way a fine-tune is refused. The same cases go to the studio's server (POST /api/jobs)
# and to prepare_run, which is the server's check: both refuse each one, and write nothing.
def hp(**values):
    return lambda c, base="laya": c.spec(base, hyperparameters=values)


BAD = {
    "no dataset": lambda c: {"base_model": c.laya},
    "unknown dataset": lambda c: {**c.spec(), "dataset": "no-such-dataset"},
    "dataset id that is a path": lambda c: {**c.spec(), "dataset": "../datasets"},
    "dataset files sent to the server": lambda c: {
        **c.spec(),
        "dataset": {"questions": QUESTIONS, "train": "t.jsonl"},
    },
    "no base model": lambda c: {"dataset": c.dataset},
    "unknown model reference": lambda c: {**c.spec(), "base_model": "ftp:x"},
    "model not downloaded": lambda c: {**c.spec(), "base_model": "hub:aac6fef/no-such-model"},
    "not a checkpoint": lambda c: {**c.spec(), "base_model": f"path:{c.empty}"},
    "incomplete checkpoint": lambda c: {**c.spec(), "base_model": f"path:{c.partial}"},
    "licence allows no derivatives": lambda c: c.spec("tev1"),
    "hyperparameters not an object": lambda c: {**c.spec(), "hyperparameters": [1, 2]},
    "dora not true or false": hp(dora="yes"),
    "loraplus ratio over 64": hp(loraplus_ratio=100),
    "epochs as text": hp(epochs="four"),
    "no epochs": hp(epochs=0),
    "half an epoch": hp(epochs=1.5),
    "infinite learning rate": hp(lr=float("inf")),
    "negative learning rate": hp(lr=-1e-4),
    "unknown method": hp(method="prompt"),
    "unknown objective": hp(objective="mse"),
    "unknown precision": hp(precision="int3"),
    "julia, unknown method": lambda c: hp(method="adapter")(c, "julia"),
    "decider full fine-tune": lambda c: hp(method="full")(c, "decider"),
    "decider rlcd objective": lambda c: hp(objective="rlcd")(c, "decider"),
    "decider 8-bit": lambda c: hp(quantization="8bit")(c, "decider"),
    "decider no batch tokens": lambda c: hp(batch_tokens=0)(c, "decider"),
    "julia with no question it answers": lambda c: {**c.spec("julia"), "dataset": c.wide},
    "name not text": lambda c: c.spec(name=7),
    "run id not an id": lambda c: c.spec(run_id="Not An Id"),
    "export the kind does not have": lambda c: c.spec(exports=[{"target": "gguf"}]),
    "export at an unknown precision": lambda c: c.spec(
        exports=[{"target": "onnx", "precision": "int2"}]
    ),
    "a GGUF choice for a Laya package": lambda c: c.spec(
        exports=[{"target": "noulxp", "gguf": "q8_0"}]
    ),
    "a decider package GGUF that does not exist": lambda c: c.spec(
        "decider", exports=[{"target": "noulxp", "gguf": "q4_k"}]
    ),
    "over the platform's bound": lambda c: c.spec(
        hyperparameters={"epochs": 9}, bounds={"epochs": {"min": 1, "max": 5}}
    ),
    "outside the platform's choices": lambda c: c.spec(
        hyperparameters={"lora_rank": 12}, bounds={"lora_rank": {"choices": [8, 16, 32]}}
    ),
}


@pytest.fixture
def context(studio, monkeypatch):  # noqa: F811
    url, server_studio = studio
    # Setup state is not what these cases are about (on a machine without a training stack
    # it never becomes ready, and the server answers 409 to every fine-tune).
    monkeypatch.setattr(type(server_studio.bootstrap), "ready", property(lambda self: True))
    return url, Context(server_studio.workspace)


def written(workspace):
    return sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*"))


@pytest.mark.parametrize("case", list(BAD))
def test_prepare_run_refuses_what_the_server_refuses(context, case):
    url, c = context
    spec = BAD[case](c)
    before = written(c.workspace)
    with pytest.raises(cloud.Refused) as refused:
        cloud.prepare_run(spec, c.workspace)
    assert written(c.workspace) == before
    status, body = call(url, "/api/jobs", {"kind": "train", **spec})
    assert (status, body["error"]) == (400, str(refused.value))  # the same rule, said the same
    assert written(c.workspace) == before


def test_a_run_is_prepared_as_the_server_starts_it(context):
    url, c = context
    spec = c.spec(
        name="  tiny run  ",
        hyperparameters={"epochs": 2.0, "lora_rank": 8, "head_lr": 1.0, "nonsense": 1},
        baseline=False,
    )
    run = cloud.prepare_run(spec, c.workspace, stamp="1006-120000")
    assert run["run_id"] == "tiny-run-1006-120000" and run["kind"] == "laya"
    # The job carries the overrides the trainer has, counts as whole numbers.
    assert run["job"] == {
        "run_id": "tiny-run-1006-120000",
        "dataset": c.dataset,
        "base_model": c.laya,
        "hyperparameters": {"epochs": 2, "lora_rank": 8, "head_lr": 1.0},
        "baseline": False,
    }
    assert run["ignored"] == ["nonsense"] and "nonsense" in run["warnings"][0]
    record = engine.read_json(c.workspace / "runs" / run["run_id"] / "run.json")
    assert record == run["run"]
    assert record["hyperparameters"] == {**engine.HYPERPARAMETERS, **run["job"]["hyperparameters"]}
    assert record["dataset_name"] == "tiny" and record["questions"] == list(QUESTIONS)
    with pytest.raises(cloud.Refused, match="exists already"):
        cloud.prepare_run(spec, c.workspace, stamp="1006-120000")
    # Decider keeps only its own keys; the default name is the dataset's and the method.
    decider = cloud.prepare_run(c.spec("decider", hyperparameters={"head_lr": 1.0}), c.workspace)
    assert decider["kind"] == "decider" and decider["job"]["hyperparameters"] == {}
    assert decider["name"] == "tiny · lora" and decider["ignored"] == ["head_lr"]


def test_a_kind_that_answers_some_questions_says_which_it_leaves_out(context):
    _, c = context
    rows = [
        {"state": f"red {i}", "answers": {"wide": f"l{i % 25}", "flag": i % 2 == 0}}
        for i in range(30)
    ]
    questions = {**WIDE, "flag": QUESTIONS["flag"]}
    mixed = engine.create_dataset("mixed", questions, jsonl(rows), "m.jsonl", workspace=c.workspace)
    run = cloud.prepare_run({**c.spec("julia", name="j"), "dataset": mixed["id"]}, c.workspace)
    assert run["warnings"] == ["Julia 1 leaves out wide: 25 options; Julia 1 answers 2 to 20"]
    laya = cloud.prepare_run({**c.spec(name="l"), "dataset": mixed["id"]}, c.workspace)
    assert laya["warnings"] == []


# ----------------------------------------------------------------------------- run files


def test_a_run_file_brings_its_dataset_split_as_the_studio_splits_it(context, tmp_path):
    url, c = context
    rows = make_rows(50)
    (tmp_path / "train.jsonl").write_text(jsonl(rows))
    (tmp_path / "questions.json").write_text(json.dumps(QUESTIONS))
    # The studio's own dataset of the same file and seed, made through its API.
    status, made = call(
        url,
        "/api/datasets",
        {
            "name": "train",
            "questions": QUESTIONS,
            "seed": 5,
            "train": {"name": "train.jsonl", "text": jsonl(rows)},
        },
    )
    assert status == 201, made
    checked = datasets.validate(QUESTIONS, jsonl(rows), "train.jsonl", seed=5)[2]
    given = {
        "questions": "questions.json",
        "train": "train.jsonl",
        "seed": 5,
        "expected": {"rows": checked["rows"], "sha256": checked["sha256"]},
    }
    run = cloud.prepare_run({"dataset": given, "base_model": c.laya}, c.workspace, base=tmp_path)
    assert run["dataset"]["id"] == made["id"]  # the same dataset: same rows, same split
    assert run["dataset"]["rows"] == made["rows"] == checked["rows"]
    wrong = {**given, "expected": {"rows": {**checked["rows"], "test": 0}}}
    with pytest.raises(cloud.Refused, match="does not match the one that was checked"):
        cloud.prepare_run({"dataset": wrong, "base_model": c.laya}, c.workspace, base=tmp_path)
    for bad, message in (
        ({**given, "train": "missing.jsonl"}, "Cannot read missing.jsonl"),
        ({**given, "seed": "5"}, "seed is a whole number"),
        ({k: v for k, v in given.items() if k != "train"}, "names its questions and its train"),
        ({**given, "questions": {"q": {"type": "rank"}}}, "Unknown question type"),
    ):
        with pytest.raises(cloud.Refused, match=message):
            cloud.prepare_run({"dataset": bad, "base_model": c.laya}, c.workspace, base=tmp_path)
    limits = {"max_bytes": 100}
    with pytest.raises(cloud.Refused, match="the limit is"):
        cloud.prepare_run(
            {"dataset": given, "base_model": c.laya, "limits": limits}, c.workspace, base=tmp_path
        )
    with pytest.raises(cloud.Refused, match="the limit is 49"):
        cloud.prepare_run(
            {"dataset": given, "base_model": c.laya, "limits": {"max_rows": 49}},
            c.workspace,
            base=tmp_path,
        )


# ----------------------------------------------------------------------------- jobs


def test_an_export_job_passes_the_gguf_choice_on(tmp_path, monkeypatch):
    """A Decider package's GGUF (bf16 or q8_0) reaches the export from a job's spec."""
    import layastudio.export

    monkeypatch.setattr(signal, "signal", lambda *a: None)  # run_job's SIGTERM handler
    seen = {}
    monkeypatch.setattr(layastudio.export, "export", lambda *a, **k: seen.update(k))
    job = tmp_path / "job"
    for gguf in ("q8_0", None):
        engine.write_json(
            job / "spec.json",
            {
                "kind": "export",
                "model": "run:x",
                "target": "noulxp",
                "precision": "float",
                **({"gguf": gguf} if gguf else {}),
                "workspace": str(tmp_path),
            },
        )
        assert engine.run_job(job) == 0
        assert seen["gguf"] == gguf and seen["precision"] == "float"
