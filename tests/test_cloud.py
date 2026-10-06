"""Headless runs: the checks every fine-tune starts with, shared with the server, and a run
from a config file with no UI. Nothing here needs MLX, PyTorch or a real model, except the
end-to-end runs, which use the tests' tiny random checkpoints.
"""

import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

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
        # Two questions, and only the one Julia 1 leaves out (25 options) labeled.
        rows = [{"state": f"red {i}", "answers": {"wide": f"l{i % 25}"}} for i in range(30)]
        self.wide_only = engine.create_dataset(
            "wide-only",
            {**WIDE, "flag": QUESTIONS["flag"]},
            jsonl(rows),
            "w.jsonl",
            workspace=workspace,
        )["id"]
        self.job = "export-1006-120000"
        (workspace / "jobs" / self.job).mkdir(parents=True)

    def spec(self, base="laya", **extra):
        return {"dataset": self.dataset, "base_model": getattr(self, base), **extra}


# Every way a fine-tune is refused. The same cases go to the studio's server (POST /api/jobs)
# and to prepare_run, which is the server's check: both refuse each one, and write nothing.
def hp(**values):
    return lambda c, base="laya": c.spec(base, hyperparameters=values)


BAD = {
    "no dataset": lambda c: {"base_model": c.laya},
    "a hub reference with an empty revision": lambda c: {
        **c.spec(),
        "base_model": "hub:aac6fef/laya-mlx@",
    },
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
    "a dropout of more than 1": hp(lora_dropout=1.5),
    "a head dropout of 1": lambda c: hp(head_dropout=1.0)(c, "julia"),
    "a warm-up of more than every update": lambda c: hp(warmup=1.5)(c, "decider"),
    "julia with no question it answers": lambda c: {**c.spec("julia"), "dataset": c.wide},
    "julia with no row on the question it answers": lambda c: {
        **c.spec("julia"),
        "dataset": c.wide_only,
    },
    "name not text": lambda c: c.spec(name=7),
}
# What only a run file brings (layastudio train --config): the studio's server takes the Train
# button's fields alone, so prepare_run refuses these and the server never sees them.
BAD_RUN_FILE = {
    "run id not an id": lambda c: c.spec(run_id="Not An Id"),
    "run id of a job that exists": lambda c: c.spec(run_id=c.job),
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
    "bounds for a key the trainer lacks": lambda c: c.spec(bounds={"nope": {"max": 1}}),
    "bounds that do not fit the value": lambda c: c.spec(bounds={"method": {"min": 1}}),
    "a change the bounds do not name": lambda c: c.spec(
        hyperparameters={"epochs": 2, "batch_size": 512}, bounds={"epochs": {"max": 5}}
    ),
}


@pytest.fixture
def context(studio, monkeypatch):  # noqa: F811
    url, server_studio = studio
    # Setup state is not what these cases are about (on a machine without a training stack
    # it never becomes ready, and the server answers 409 to every fine-tune).
    monkeypatch.setattr(type(server_studio.bootstrap), "ready", property(lambda self: True))
    c = Context(server_studio.workspace)
    c.studio = server_studio
    return url, c


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


@pytest.mark.parametrize("case", list(BAD_RUN_FILE))
def test_prepare_run_refuses_a_run_file_that_cannot_train(context, case):
    _, c = context
    before = written(c.workspace)
    with pytest.raises(cloud.Refused):
        cloud.prepare_run(BAD_RUN_FILE[case](c), c.workspace)
    assert written(c.workspace) == before


def test_the_server_takes_only_what_the_train_button_sends(context, monkeypatch):
    """A run's id, bounds, limits and exports are a run file's: the server starts the run it
    always started, under its own id, and records no exports it would never make."""
    url, c = context
    started = []
    monkeypatch.setattr(c.studio.jobs, "start", lambda *a: started.append(a) or a[2])
    spec = c.spec(
        name="plain",
        run_id=c.job,
        exports=[{"target": "noulxp"}],
        bounds={"epochs": {"max": 1}},
        hyperparameters={"epochs": 3},
        limits={"max_rows": 1},
    )
    status, body = call(url, "/api/jobs", {"kind": "train", **spec})
    assert status == 201, body
    [(kind, job, job_id, _)] = started
    assert kind == "train" and job_id == job["run_id"] != c.job
    assert job["run_id"].startswith("plain-") and job["hyperparameters"] == {"epochs": 3}
    assert "exports" not in engine.read_json(c.workspace / "runs" / job_id / "run.json")


EXPORTS = {
    "a target the kind does not have": {"target": "gguf"},
    "an unknown target": {"target": "tflite"},
    "an unknown precision": {"target": "onnx", "precision": "int2"},
    "test rows for an ONNX export": {"target": "onnx", "test_rows": 5},
    "test rows that are text": {"target": "noulxp", "test_rows": "5"},
    "a GGUF choice for a Laya package": {"target": "noulxp", "gguf": "q8_0"},
}


@pytest.mark.parametrize("case", list(EXPORTS))
def test_the_export_button_and_a_run_file_refuse_the_same_exports(context, case, monkeypatch):
    url, c = context
    run_id = cloud.prepare_run(c.spec(name="x"), c.workspace)["run_id"]
    (c.workspace / "runs" / run_id / "model").symlink_to(c.laya[5:], target_is_directory=True)
    monkeypatch.setattr(c.studio.jobs, "start", lambda *a: pytest.fail("a job was started"))
    item = EXPORTS[case]
    with pytest.raises(cloud.Refused) as refused:
        cloud.check_export(kinds.LAYA, item, f"run:{run_id}")
    status, body = call(url, "/api/jobs", {"kind": "export", "model": f"run:{run_id}", **item})
    assert (status, body["error"]) == (400, str(refused.value))


def test_exports_default_to_their_own_precision(context):
    _, c = context
    assert cloud.check_export(kinds.DECIDER, "gguf", "run:x")["precision"] == "bf16"
    assert cloud.check_export(kinds.DECIDER, "mlx", "run:x")["precision"] == "int4"
    assert cloud.check_export(kinds.LAYA, "onnx", "run:x")["precision"] == "float"
    # A run's ONNX and Core ML folders are its own, as its GGUF and MLX exports are.
    run = cloud.prepare_run(
        c.spec(name="o", exports=[{"target": "onnx", "precision": "int8"}]), c.workspace
    )
    assert run["exports"][0]["out_dir"] == f"runs/{run['run_id']}/exports/onnx-int8"


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
    left_out = [w for w in run["warnings"] if "leaves out" in w]
    assert left_out == ["Julia 1 leaves out wide: 25 options; Julia 1 answers 2 to 20"]
    laya = cloud.prepare_run({**c.spec(name="l"), "dataset": mixed["id"]}, c.workspace)
    assert not any("leaves out" in w for w in laya["warnings"])
    # These bases are folders with no published name: trained, and said so.
    assert any("no published name" in w for w in laya["warnings"])


def hub_cache(tmp_path, monkeypatch, repo, kind):
    """A Hugging Face cache holding repo at one commit, as a download by commit leaves it:
    snapshots/<commit> and no refs/main. Returns the commit."""
    import huggingface_hub.constants

    commit = hashlib.sha1(repo.encode()).hexdigest()
    folder = tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    folder.mkdir(parents=True)
    fake_checkpoint(folder, kind)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    return commit


def test_a_base_pinned_to_a_revision_is_found_licence_checked_and_named(
    context, tmp_path, monkeypatch
):
    from layastudio import noulxp_package

    _, c = context
    commit = hub_cache(tmp_path, monkeypatch, "Mapika/decider-2b", kinds.DECIDER)
    with pytest.raises(cloud.Refused, match="not downloaded yet"):  # main is not cached
        cloud.prepare_run(c.spec("decider", base_model="hub:Mapika/decider-2b"), c.workspace)
    ref = f"hub:Mapika/decider-2b@{commit}"
    run = cloud.prepare_run({**c.spec(name="pinned"), "base_model": ref}, c.workspace)
    assert run["kind"] == kinds.DECIDER and run["job"]["base_model"] == ref
    assert not any("no published name" in w for w in run["warnings"])
    # The package names the model and the revision it was trained from.
    source = noulxp_package.provenance(run["run"], run["run_id"], c.workspace)
    assert source["model"] == "Mapika/decider-2b" and source["revision"] == commit
    assert source["license"]
    # A pinned revision of a model whose licence allows no derivatives is refused all the same.
    tev1 = hub_cache(tmp_path, monkeypatch, "together-ai/tev1", kinds.DECIDER)
    with pytest.raises(cloud.Refused) as refused:
        cloud.prepare_run({**c.spec(), "base_model": f"hub:together-ai/tev1@{tev1}"}, c.workspace)
    with pytest.raises(cloud.Refused) as imported:
        cloud.prepare_run(c.spec("tev1"), c.workspace)
    assert str(refused.value) == str(imported.value)


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
        "expected": {"rows": checked["rows"], "digest": checked["sha256"]},
    }
    run = cloud.prepare_run({"dataset": given, "base_model": c.laya}, c.workspace, base=tmp_path)
    assert run["dataset"]["id"] == made["id"]  # the same dataset: same rows, same split
    assert run["dataset"]["rows"] == made["rows"] == checked["rows"]
    wrong = {**given, "expected": {"rows": {**checked["rows"], "test": 0}}}
    with pytest.raises(cloud.Refused, match="does not match the one that was checked"):
        cloud.prepare_run({"dataset": wrong, "base_model": c.laya}, c.workspace, base=tmp_path)
    for bad, message in (
        ({**given, "train": "missing.jsonl"}, "Cannot read missing.jsonl"),
        ({**given, "train": 5}, "The train file is a path"),
        ({**given, "train": {"name": "train.jsonl"}}, "The train file is a path"),
        ({**given, "train": {"path": "train.jsonl", "name": 5}}, "name is text"),
        ({**given, "test": ["test.jsonl"]}, "The test file is a path"),
        ({**given, "questions": 5}, "The questions are a file"),
        ({**given, "seed": "5"}, "seed is a whole number"),
        ({k: v for k, v in given.items() if k != "train"}, "names its questions and its train"),
        ({**given, "questions": {"q": {"type": "rank"}}}, "Unknown question type"),
        ({**given, "expected": [1]}, "expected is"),
        # The platform's file hash is the file's: it goes with the file, not with expected.
        ({**given, "expected": {"sha256": checked["sha256"]}}, "not 'sha256'"),
        ({**given, "train": {"path": "train.jsonl", "sha256": "0" * 64}}, "not the one uploaded"),
    ):
        with pytest.raises(cloud.Refused, match=message):
            cloud.prepare_run({"dataset": bad, "base_model": c.laya}, c.workspace, base=tmp_path)
    limits = {"max_bytes": 100}
    with pytest.raises(cloud.Refused, match="the limit is"):
        cloud.prepare_run(
            {"dataset": given, "base_model": c.laya, "limits": limits}, c.workspace, base=tmp_path
        )
    for limits, message in (
        ([1], "limits is"),
        ({"max_bytes": "25MB"}, "limits.max_bytes is a whole number"),
        ({"max_rows": -1}, "limits.max_rows is a whole number"),
    ):
        with pytest.raises(cloud.Refused, match=message):
            cloud.prepare_run(
                {"dataset": given, "base_model": c.laya, "limits": limits},
                c.workspace,
                base=tmp_path,
            )
    with pytest.raises(cloud.Refused, match="the limit is 49"):
        cloud.prepare_run(
            {"dataset": given, "base_model": c.laya, "limits": {"max_rows": 49}},
            c.workspace,
            base=tmp_path,
        )


def test_a_file_is_read_as_its_bytes_and_by_its_own_name(context, tmp_path):
    """A CSV whose quoted cells hold CRLF line breaks is the dataset the studio makes of the same
    upload (and the platform of the same bytes): read as text, the cells would lose their CR
    and the dataset would be another. A file saved without its name is read by the name given."""
    url, c = context
    lines = ["state,topic,level,flag"] + [
        f'"red {i}\r\nsecond line {i}",{["alpha", "beta", "gamma"][i % 3]},{i % 3},{i % 2 == 0}'
        for i in range(20)
    ]
    data = ("\r\n".join(lines) + "\r\n").encode()
    (tmp_path / "train.csv").write_bytes(data)
    (tmp_path / "upload.bin").write_bytes(data)
    text = data.decode()
    checked = datasets.validate(QUESTIONS, text, "train.csv", seed=13)[2]
    status, made = call(
        url,
        "/api/datasets",
        {"name": "crlf", "questions": QUESTIONS, "train": {"name": "train.csv", "text": text}},
    )
    assert status == 201, made
    assert made["sha256"] == checked["sha256"]
    sha256 = hashlib.sha256(data).hexdigest()
    for n, train in enumerate(
        ("train.csv", {"path": "upload.bin", "name": "train.csv", "sha256": sha256})
    ):  # the second as an agent may save it
        given = {
            "questions": QUESTIONS,
            "train": train,
            "name": "crlf",
            "expected": {"rows": checked["rows"], "digest": checked["sha256"]},
        }
        run = cloud.prepare_run(
            {"dataset": given, "base_model": c.laya, "name": f"run {n}"}, c.workspace, base=tmp_path
        )
        assert run["dataset"]["id"] == made["id"]
        rows = engine.load_dataset(made["id"], c.workspace)[1]
        assert rows[0]["state"].startswith("red ") and "\r\n" in rows[0]["state"]
    # Read by its saved name, the CSV would be taken for JSONL and refused.
    with pytest.raises(cloud.Refused, match="valid labeled rows"):
        cloud.prepare_run(
            {"dataset": {"questions": QUESTIONS, "train": "upload.bin"}, "base_model": c.laya},
            c.workspace,
            base=tmp_path,
        )


# ----------------------------------------------------------------------------- jobs


# llama.cpp's converter as far as its imports go: its own conversion/ package and gguf-py.
CONVERTER = """
import json, sys
import conversion, gguf
out = sys.argv[sys.argv.index("--outfile") + 1]
with open(out, "wb") as handle:
    handle.write(b"GGUF" + json.dumps(sys.path).encode())
"""


def test_the_converter_finds_its_own_modules_with_pythonsafepath(tmp_path, monkeypatch):
    """The trainer image sets PYTHONSAFEPATH=1, which keeps a script's own folder off the
    import path: the converter's conversion/ package comes from PYTHONPATH, and the current
    folder is not on its path."""
    from layastudio import gguf

    tool = tmp_path / "tool"
    for package in ("conversion", "gguf-py/gguf"):
        (tool / package).mkdir(parents=True)
        (tool / package / "__init__.py").write_text("")
    (tool / "convert_hf_to_gguf.py").write_text(CONVERTER)
    monkeypatch.setattr(gguf, "converter", lambda *args, **kwargs: tool)
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    (tmp_path / "here").mkdir()
    monkeypatch.chdir(tmp_path / "here")
    out = gguf.convert(tmp_path / "model", tmp_path / "out" / "model.gguf", "bf16")
    path = json.loads(out.read_bytes()[4:])
    assert path[:2] == [str(tool), str(tool / "gguf-py")]
    assert str(tmp_path / "here") not in path and os.getcwd() not in path


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


# ----------------------------------------------------------------------------- headless runs


def write_run(folder, base_model, **extra):
    """A run file with its own dataset files, as a cloud GPU gets one."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = jsonl(make_rows(45))
    (folder / "train.jsonl").write_text(rows)
    (folder / "questions.json").write_text(json.dumps(QUESTIONS))
    checked = datasets.validate(QUESTIONS, rows, "train.jsonl", seed=13)[2]
    config = {
        "name": "tiny",
        "base_model": base_model,
        "dataset": {
            "questions": "questions.json",
            "train": "train.jsonl",
            "seed": 13,
            "expected": {"rows": checked["rows"], "digest": checked["sha256"]},
        },
        "hyperparameters": {"epochs": 1, "batch_size": 4},
        "workspace": "ws",
        **extra,
    }
    (folder / "run.json").write_text(json.dumps(config))
    return folder / "run.json", checked


def test_a_run_file_that_cannot_train_ends_with_a_refusal(tmp_path):
    config, _ = write_run(tmp_path / "run", "hub:aac6fef/no-such-model")
    out = io.StringIO()
    assert cloud.run(config, stream=out) == 2
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [e["type"] for e in events] == ["refused", "finished"]
    assert "pinned to a commit" in events[0]["message"]  # a run from a file downloads its base
    # The last event says why, as the result file does.
    assert events[1]["exit_code"] == 2 and events[1]["error"]["stage"] == "prepare"
    assert events[1]["error"]["message"] == events[0]["message"]
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "refused" and result["error"]["stage"] == "prepare"
    assert result["exit_code"] == 2
    assert not (tmp_path / "run/ws/runs").exists()  # nothing was written for it
    for text in (
        "{not json",
        json.dumps({"workspace": 5}),
        json.dumps({"dataset": {"questions": 5, "train": 5}}),
        json.dumps({"keep_checkpoint": "yes"}),
        json.dumps({"template": 5}),
    ):
        config.write_text(text)  # refused, with a result file, whatever is wrong with it
        assert cloud.run(config, result=tmp_path / "r.json", stream=io.StringIO()) == 2
        assert json.loads((tmp_path / "r.json").read_text())["state"] == "refused"


# A job that only writes events, as the scripted plan for its kind says: what the headless
# runner does with a job's events and exit code, with no model and no training.
CHILD = """
import json, sys, time
from pathlib import Path
job = Path(sys.argv[1])
plan = json.loads(sys.argv[2])[json.loads((job / "spec.json").read_text())["kind"]]
with open(job / "events.jsonl", "ab", buffering=0) as out:
    for step in plan["steps"]:
        if isinstance(step, (int, float)):
            time.sleep(step)
        else:
            out.write(bytes.fromhex(step))
sys.exit(plan["exit"])
"""


def scripted(monkeypatch, **plan):
    """engine.start_job, replaced by CHILD with this plan. Returns the processes started."""
    started = []

    def start_job(job_dir, kind, log, env=None):
        command = [sys.executable, "-c", CHILD, str(job_dir), json.dumps(plan)]
        started.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        return started[-1]

    monkeypatch.setattr(engine, "start_job", start_job)
    return started


def line(type_, **data):
    return (json.dumps({"t": 0, "type": type_, **data}, ensure_ascii=False) + "\n").encode()


def scripted_run(tmp_path, **extra):
    base = fake_checkpoint(tmp_path / "laya", kinds.LAYA)
    return write_run(tmp_path / "run", base, **extra)[0]


TRAINED = {"steps": [line("result", accuracy=0.9).hex(), line("done").hex()], "exit": 0}


def test_a_run_whose_export_fails_ends_partial(tmp_path, monkeypatch):
    """Trained, but an export failed: exit 3 and state partial, never success, with the
    failed exports named in the result and in the last event."""
    failed = {"steps": [line("error", message="RuntimeError: no package").hex()], "exit": 1}
    scripted(monkeypatch, train=TRAINED, export=failed)
    config = scripted_run(tmp_path, exports=[{"target": "onnx"}])
    out = io.StringIO()
    assert cloud.run(config, stream=out) == 3
    last = json.loads(out.getvalue().splitlines()[-1])
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert last["type"] == "finished" and last["state"] == result["state"] == "partial"
    assert last["exit_code"] == result["exit_code"] == 3
    assert last["failed_exports"] == result["failed_exports"] == ["onnx"]
    assert (
        last["error"]
        == result["error"]
        == {"stage": "export:onnx", "message": "RuntimeError: no package"}
    )
    assert result["exports"][0]["state"] == "failed" and "outputs" in result


def test_a_run_says_what_it_keeps_and_warns_when_that_is_only_its_card(tmp_path, monkeypatch):
    """keep_checkpoint is off unless the run says so; the prepared event and the result repeat
    it, and a run with neither the checkpoint nor a NoulXP package to keep is told."""
    scripted(monkeypatch, train=TRAINED)
    out = io.StringIO()
    assert cloud.run(scripted_run(tmp_path), stream=out) == 0
    prepared = json.loads(out.getvalue().splitlines()[0])
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert prepared["keep_checkpoint"] is result["keep_checkpoint"] is False
    assert result["template"] is None
    assert any("keeps no checkpoint" in w for w in prepared["warnings"])
    assert [f["path"] for f in result["outputs"]["card"]] == [
        "card/README.md",
        "card/finetune.json",
    ]

    config = scripted_run(tmp_path / "kept", keep_checkpoint=True, template="laya.lora")
    assert cloud.run(config, stream=io.StringIO()) == 0
    result = json.loads((tmp_path / "kept/run/result.json").read_text())
    assert result["keep_checkpoint"] is True and result["template"] == "laya.lora"
    assert not any("keeps no checkpoint" in w for w in result["warnings"])


def test_a_card_that_cannot_be_written_is_a_warning(tmp_path, monkeypatch):
    """The run trained and its outputs stand: a card that fails is a warning, not a failure."""
    scripted(monkeypatch, train=TRAINED)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cloud, "write_card", broken)
    assert cloud.run(scripted_run(tmp_path, keep_checkpoint=True), stream=io.StringIO()) == 0
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "succeeded" and "card" not in result["outputs"]
    assert "No model card: OSError: disk full" in result["warnings"]


def test_a_card_that_fails_halfway_leaves_no_card(tmp_path, monkeypatch):
    """finetune.json fails after README.md is written: the run has no card, not a card with
    one file, and nothing of it is left for the manifest to list."""
    scripted(monkeypatch, train=TRAINED)
    write_json, there = engine.write_json, []

    def full_disk(path, value):
        if os.path.basename(path) == "finetune.json":
            there.extend(sorted(os.listdir(os.path.dirname(path))))
            raise OSError("disk full")
        return write_json(path, value)

    monkeypatch.setattr(engine, "write_json", full_disk)
    assert cloud.run(scripted_run(tmp_path, keep_checkpoint=True), stream=io.StringIO()) == 0
    assert there == ["README.md"]  # the README was written when the record failed
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "succeeded" and "card" not in result["outputs"]
    assert "No model card: OSError: disk full" in result["warnings"]
    run_dir = tmp_path / "run/ws/runs" / result["run_id"]
    assert not (run_dir / "card").exists() and not (run_dir / "card.tmp").exists()


def test_step_progress_reaches_the_end_whatever_the_epochs_hold():
    """Every trainer's step events: the share done grows with each update, epochs of
    different lengths included, and is 1 at the last."""
    epochs, shares = 3, []
    for epoch, updates in zip(range(1, epochs + 1), (7, 5, 6)):
        for done in range(1, updates + 1):
            fields = engine.step_progress(epoch, done, updates, epochs, elapsed=10.0)
            assert fields["epochs"] == epochs and fields["eta_s"] >= 0
            shares.append(fields["fraction"])
    assert shares == sorted(shares) and shares[-1] == 1 and 0 < shares[0] < 1 / epochs
    assert engine.step_progress(1, 1, 2, 1, elapsed=10.0)["eta_s"] == 10


def test_an_event_read_in_the_middle_of_a_character_arrives_whole(tmp_path, monkeypatch):
    whole = line("phase", phase="train", message="Training ✅ · epoch 1")
    cut = whole.index("✅".encode()) + 1  # inside the check mark's three bytes
    train = {"steps": [whole[:cut].hex(), 0.6, whole[cut:].hex(), line("done").hex()], "exit": 0}
    scripted(monkeypatch, train=train)
    out = io.StringIO()
    assert cloud.run(scripted_run(tmp_path), stream=out) == 0
    events = [json.loads(e) for e in out.getvalue().splitlines()]
    assert any(e.get("message") == "Training ✅ · epoch 1" for e in events)


def test_a_signal_during_the_checks_cancels_before_anything_runs(tmp_path, monkeypatch):
    started = scripted(monkeypatch, train=TRAINED)
    prepare = cloud.prepare_run

    def signalled(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)  # handled: the checks finish, nothing starts
        return prepare(*args, **kwargs)

    monkeypatch.setattr(cloud, "prepare_run", signalled)
    before = signal.getsignal(signal.SIGTERM)
    out = io.StringIO()
    assert cloud.run(scripted_run(tmp_path), stream=out) == 143
    assert started == []
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "cancelled" and result["error"]["stage"] == "prepare"
    assert not list((tmp_path / "run/ws/runs").iterdir())  # the record of a run that never ran
    assert json.loads(out.getvalue().splitlines()[-1])["state"] == "cancelled"
    assert signal.getsignal(signal.SIGTERM) == before  # the caller's handler is back


class Broken(io.StringIO):
    """stdout whose reader went away after a few lines."""

    def __init__(self, lines):
        super().__init__()
        self.left = lines

    def write(self, text):
        if self.left == 0:
            raise BrokenPipeError(32, "Broken pipe")
        self.left -= 1
        return super().write(text)


def test_a_job_is_never_left_running(tmp_path, monkeypatch):
    """Whatever ends the following of a job's events, the job is stopped and waited for."""
    train = {"steps": [line("phase", phase="train", message="Training").hex(), 60], "exit": 0}
    started = scripted(monkeypatch, train=train)
    config = scripted_run(tmp_path)
    with pytest.raises(BrokenPipeError):  # "prepared" and "stage" are written, then no more
        cloud.run(config, stream=Broken(2))
    [child] = started
    assert child.returncode is not None  # stopped (SIGTERM), long before its 60 s
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "failed" and result["error"]["stage"] == "run"


def tiny_base(kind, folder):
    """A tiny random base checkpoint of a kind, and the environment its run trains in: the
    CPU through PyTorch when this machine has the PyTorch stack, else MLX."""
    found = __import__("importlib").util.find_spec
    has_torch = all(found(m) for m in ("torch", "transformers", "laya"))
    if kind == "decider":
        if not all(found(m) for m in ("torch", "transformers", "peft")):
            pytest.skip("Decider trains through PyTorch and PEFT here")
        import tiny

        path = tiny.decider_checkpoint(folder / "decider")
        return f"path:{path}", {"LAYASTUDIO_BACKEND": "torch", "LAYASTUDIO_DEVICE": "cpu"}
    if kind == "julia":
        if not has_torch:
            pytest.skip("Julia's tiny checkpoint is built with PyTorch")
        import tiny

        path = tiny.julia_checkpoint(folder / "julia")
    else:
        pytest.importorskip("mlx.core", reason="Laya's tiny checkpoint is built with MLX")
        from test_noulxp import tiny_checkpoint

        path = tiny_checkpoint(folder / "laya")
    env = {"LAYASTUDIO_BACKEND": "torch", "LAYASTUDIO_DEVICE": "cpu"} if has_torch else {}
    return f"path:{path}", env


def headless(config, env, *args):
    """`python -m layastudio.cloud --config <file>`, as a cloud GPU's agent calls it."""
    return subprocess.Popen(
        [sys.executable, "-m", "layastudio.cloud", "--config", str(config), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **env},
        cwd=str(engine.PACKAGE.parent),
    )


@pytest.mark.parametrize("kind", ["julia", "laya", "decider"])
def test_a_run_file_trains_exports_and_says_what_it_made(tmp_path, kind):
    """End to end with no UI: baseline, training and evaluation, then a NoulXP package where
    this machine has the tooling, and the card; every event a JSON line, the training's
    progress in step events, the result file at the end."""
    from layastudio import noulxp_package

    ref, env = tiny_base(kind, tmp_path / "models")
    packaged = noulxp_package.missing_tooling(kind) is None
    exports = [{"target": "noulxp"}] if packaged else []
    # Without a package the run keeps its checkpoint, or its card would be all it keeps.
    config, checked = write_run(
        tmp_path / "run",
        ref,
        exports=exports,
        keep_checkpoint=not packaged,
        template=f"{kind}.lora",
    )
    process = headless(config, env)
    stdout, stderr = process.communicate(timeout=900)
    assert process.returncode == 0, stderr[-3000:]
    events = [json.loads(line) for line in stdout.splitlines()]  # every line is one JSON event
    assert events[0]["type"] == "prepared" and events[-1]["type"] == "finished"
    assert events[-1]["exit_code"] == 0 and "error" not in events[-1]
    assert events[0]["dataset"]["rows"] == checked["rows"]
    stages = [(e["stage"], e["state"]) for e in events if e["type"] == "stage"]
    expected = [("train", "running"), ("train", "done")]
    if packaged:
        expected += [("export:noulxp", "running"), ("export:noulxp", "done")]
    assert stages == expected
    trained = {e["type"] for e in events if e.get("stage") == "train" and "job" not in e}
    assert {"phase", "epoch", "step", "result", "done"} <= trained
    steps = [e for e in events if e["type"] == "step"]
    assert all(e["epochs"] == 1 and e["epoch"] == 1 and 0 < e["fraction"] <= 1 for e in steps)
    assert [e["fraction"] for e in steps] == sorted(e["fraction"] for e in steps)
    assert steps[-1]["fraction"] == 1 and steps[-1]["step"] == len(steps)
    assert events[0]["keep_checkpoint"] is (not packaged)
    assert not any("keeps no checkpoint" in w for w in events[0]["warnings"])

    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "succeeded" and result["kind"] == kind
    assert result["dataset"]["sha256"] == checked["sha256"]
    assert result["comparison"]["finetuned"]["overall"]["n"] > 0
    assert result["training"]["kind"] == kind and "records" not in result["eval"]
    run_dir = tmp_path / "run/ws/runs" / result["run_id"]
    assert engine.read_json(run_dir / "run.json")["kind"] == kind
    model = {f["path"]: f for f in result["outputs"]["model"]}
    weights = run_dir / "model/model.safetensors"
    assert model["model/model.safetensors"]["size"] == weights.stat().st_size
    assert (
        model["model/model.safetensors"]["sha256"]
        == hashlib.sha256(weights.read_bytes()).hexdigest()
    )
    if packaged:
        [package] = result["exports"]
        assert package["state"] == "done" and package["result"]["state"] == "passed"
        assert any(f["path"] == "noulxp/noulxp.json" for f in result["outputs"]["noulxp"])
    else:
        assert "noulxp" not in result["outputs"]
    assert result["keep_checkpoint"] is (not packaged) and result["template"] == f"{kind}.lora"
    check_card(tmp_path, result, run_dir, checked, packaged)


def check_card(tmp_path, result, run_dir, checked, packaged):
    """card/: the model card and the fine-tune's record, listed with the outputs; neither
    names a path on the machine that trained nor holds a row of the dataset."""
    card = {f["path"]: f for f in result["outputs"]["card"]}
    assert sorted(card) == ["card/README.md", "card/finetune.json"]
    for path, entry in card.items():
        data = (run_dir / path).read_bytes()
        assert entry["size"] == len(data) and entry["sha256"] == hashlib.sha256(data).hexdigest()
    readme = (run_dir / "card/README.md").read_text()
    record = json.loads((run_dir / "card/finetune.json").read_text())
    for text in (readme, json.dumps(record)):
        assert str(tmp_path) not in text and "/private/" not in text
        assert not any(row["state"] in text for row in make_rows(45))
    assert cloud.CARD_REPO in readme and "## Measured on the held-out test split" in readme
    assert ("carries a NoulXP package" in readme) is packaged
    assert record["kind"] == result["kind"] and record["template"] == result["template"]
    assert record["dataset"]["sha256"] == checked["sha256"]
    assert record["dataset"]["rows"] == checked["rows"]
    assert record["hyperparameters"] == result["hyperparameters"]
    assert record["metrics"]["finetuned"]["overall"]["n"] > 0
    assert "model" not in record["metrics"]["base"]
    assert record["training"]["best_epoch"] == result["training"]["best_epoch"]
    if packaged:
        assert record["noulxp"]["cases_passed"] == record["noulxp"]["cases"] > 0
    else:
        assert record["noulxp"] is None


def test_a_cancelled_run_stops_its_job_and_says_so(tmp_path):
    for kind in ("julia", "laya"):
        try:
            ref, env = tiny_base(kind, tmp_path / "models" / kind)
            break
        except pytest.skip.Exception:
            continue
    else:
        pytest.skip("no tiny checkpoint can be built here")
    config, _ = write_run(tmp_path / "run", ref, hyperparameters={"epochs": 50, "batch_size": 4})
    process = headless(config, env)
    for line in process.stdout:
        event = json.loads(line)
        if event["type"] == "stage" and event["state"] == "running":
            process.send_signal(signal.SIGTERM)
            break
    rest, _ = process.communicate(timeout=120)
    assert process.returncode == 143
    last = json.loads(rest.splitlines()[-1])
    assert last == {**last, "type": "finished", "state": "cancelled"}
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "cancelled" and result["stages"][0]["state"] == "cancelled"
    assert "outputs" not in result


# ----------------------------------------------------------------------------- base downloads

COMMIT = "0123456789abcdef0123456789abcdef01234567"


class FakeHub:
    """huggingface_hub.snapshot_download with no network. A lookup in the cache
    (local_files_only) is the real one; a download writes a checkpoint of the kind into the
    cache as a download by commit leaves it (snapshots/<commit>), or raises `fail`. Every
    call is kept. With `hold` (an Event), a download sets `downloading` and then waits for
    it (60 s at most), as a big file on a slow link does."""

    def __init__(self, real):
        self.real, self.fail, self.files, self.calls = real, None, None, []
        self.hold, self.downloading = None, threading.Event()

    def __call__(self, repo, revision=None, allow_patterns=None, cache_dir=None, **options):
        import huggingface_hub.constants

        local = options.get("local_files_only", False)
        self.calls.append(
            {
                "repo": repo,
                "revision": revision,
                "allow_patterns": allow_patterns,
                "cache_dir": cache_dir,
                "local": local,
            }
        )
        if local:
            return self.real(repo, revision=revision, cache_dir=cache_dir, **options)
        self.downloading.set()
        if self.hold is not None:
            self.hold.wait(60)
        if self.fail is not None:
            raise self.fail
        cache = Path(cache_dir or huggingface_hub.constants.HF_HUB_CACHE)
        folder = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
        folder.mkdir(parents=True, exist_ok=True)
        fake_checkpoint(folder, kinds.LAYA)
        for name in set(FILES[kinds.LAYA]) - set(self.files or FILES[kinds.LAYA]):
            (folder / name).unlink()
        return str(folder)

    @property
    def downloads(self):
        return [call for call in self.calls if not call["local"]]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """The fake hub, with the environment's cache in tmp_path/hf: nothing reaches the network
    or this machine's own cache."""
    import huggingface_hub
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hf"))
    fake = FakeHub(huggingface_hub.snapshot_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return fake


def recorded_envs(monkeypatch):
    """The env each job of a run is started with (scripted jobs, as TRAINED plans)."""
    scripted(monkeypatch, train=TRAINED)
    start, envs = engine.start_job, []

    def start_job(job_dir, kind, log, env=None):
        envs.append(env)
        return start(job_dir, kind, log, env)

    monkeypatch.setattr(engine, "start_job", start_job)
    return envs


def test_a_run_file_downloads_its_base_at_the_pinned_commit_into_its_cache(
    tmp_path, monkeypatch, hub
):
    """A GPU starts with an empty cache: the base model is downloaded at the commit the run
    pins, the files its kind needs only, into the run's cache_dir, after a "phase" event; the
    run's jobs read it there; a second run finds it there and asks the hub nothing."""
    envs = recorded_envs(monkeypatch)
    ref = f"hub:aac6fef/laya-mlx@{COMMIT}"
    config, _ = write_run(tmp_path / "run", ref, cache_dir="cache")
    out = io.StringIO()
    assert cloud.run(config, stream=out) == 0
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    assert events[0] == {**events[0], "type": "phase", "phase": "download"}
    assert "aac6fef/laya-mlx at 0123456789ab" in events[0]["message"]
    assert events[1]["type"] == "prepared" and events[1]["base_model"] == ref
    cache = (tmp_path / "run/cache").resolve()
    assert hub.downloads == [
        {
            "repo": "aac6fef/laya-mlx",
            "revision": COMMIT,
            "allow_patterns": list(kinds.DOWNLOAD[kinds.LAYA]),
            "cache_dir": cache,
            "local": False,
        }
    ]
    snapshot = cache / "models--aac6fef--laya-mlx" / "snapshots" / COMMIT
    assert (snapshot / "model.safetensors").is_file()
    assert envs and all(env["HF_HUB_CACHE"] == str(cache) for env in envs)
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "succeeded" and result["base_model"] == ref
    run_dir = tmp_path / "run/ws/runs" / result["run_id"]
    assert engine.read_json(run_dir / "run.json")["base_model"] == ref
    assert json.loads((run_dir / "card/finetune.json").read_text())["base_model"] == {
        "repo": "aac6fef/laya-mlx",
        "revision": COMMIT,
        "license": "apache-2.0",
    }

    hub.calls.clear()
    again, _ = write_run(tmp_path / "again", ref, cache_dir="../run/cache")
    assert cloud.run(again, stream=io.StringIO()) == 0
    assert hub.calls and not hub.downloads


def test_without_cache_dir_the_base_goes_to_the_environments_cache(tmp_path, monkeypatch, hub):
    """The pod agent points $HF_HOME into the job's folder: with no cache_dir, that cache is
    the run's, and the jobs, started with the run's own env, read it there."""
    envs = recorded_envs(monkeypatch)
    config, _ = write_run(tmp_path / "run", f"hub:aac6fef/laya-mlx@{COMMIT}")
    assert cloud.run(config, stream=io.StringIO()) == 0
    [download] = hub.downloads
    assert download["cache_dir"] is None
    assert (tmp_path / "hf/models--aac6fef--laya-mlx/snapshots" / COMMIT).is_dir()
    assert envs and not any("HF_HUB_CACHE" in env for env in envs)


@pytest.mark.parametrize(
    "revision", ["", "@main", "@v1.0", "@0123456", f"@{COMMIT.upper()}", f"@{COMMIT}0"]
)
def test_a_run_file_refuses_a_base_not_pinned_to_a_commit(tmp_path, hub, revision):
    """A branch or a tag can move between the platform's check and the GPU: a run from a file
    trains a commit or nothing, and downloads nothing for a run it refuses."""
    config, _ = write_run(tmp_path / "run", f"hub:aac6fef/laya-mlx{revision}")
    out = io.StringIO()
    assert cloud.run(config, stream=out) == 2
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [e["type"] for e in events] == ["refused", "finished"]
    assert "pinned to a commit, hub:aac6fef/laya-mlx@<commit>" in events[0]["message"]
    assert hub.calls == [] and not (tmp_path / "run/ws/runs").exists()


def test_a_run_file_checks_the_licence_before_it_downloads(tmp_path, hub):
    from layastudio import families

    config, _ = write_run(tmp_path / "run", f"hub:together-ai/tev1@{COMMIT}")
    out = io.StringIO()
    assert cloud.run(config, stream=out) == 2
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [e["type"] for e in events] == ["refused", "finished"]
    reason = families.trainer_status(families.find("together-ai/tev1"))["reason"]
    assert events[0]["message"] == reason and hub.calls == []


def test_a_base_that_cannot_be_downloaded_is_refused(tmp_path, hub):
    """Exit 2, as for any run that never trained (the agent credits it), saying why in one
    line; nothing is written for the run."""
    hub.fail = OSError("Connection reset by peer\nwhile reading the response")
    ref = f"aac6fef/laya-mlx@{COMMIT}"
    config, _ = write_run(tmp_path / "run", f"hub:{ref}")
    out = io.StringIO()
    assert cloud.run(config, stream=out) == 2
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [e["type"] for e in events] == ["phase", "refused", "finished"]
    assert events[1]["message"] == (
        f"{ref} could not be downloaded: OSError: Connection reset by peer"
    )
    assert len(hub.downloads) == 1 and not (tmp_path / "run/ws/runs").exists()
    # A download that leaves no complete checkpoint is refused too.
    hub.fail, hub.files = None, ("rl_agent_config.json",)
    config, _ = write_run(tmp_path / "partial", f"hub:{ref}")
    out = io.StringIO()
    assert cloud.run(config, stream=out) == 2
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    assert events[1]["message"].startswith("Not a complete Laya checkpoint")


def test_the_train_button_downloads_nothing(context, hub):
    """The studio's own runs are unchanged: a hub base is one this machine has downloaded, at
    any revision; prepare_run and the server only look in the cache."""
    url, c = context
    for ref in (f"hub:aac6fef/laya-mlx@{COMMIT}", "hub:aac6fef/laya-mlx"):
        spec = {**c.spec(), "base_model": ref}
        with pytest.raises(cloud.Refused, match="not downloaded yet"):
            cloud.prepare_run(spec, c.workspace)
        status, body = call(url, "/api/jobs", {"kind": "train", **spec})
        assert status == 400 and "not downloaded yet" in body["error"]
    assert hub.calls and not hub.downloads


def test_a_run_file_trains_a_hub_base_from_its_own_cache(tmp_path):
    """End to end, offline: the base model, pinned, is in the run's cache_dir; the run finds
    it there without the hub, and its train job (a child process, HF_HUB_OFFLINE=1, whose
    own $HF_HUB_CACHE points elsewhere) reads it from the same cache."""
    import shutil

    path, env = tiny_base("julia", tmp_path / "models")
    snapshot = tmp_path / "run/cache/models--SupersonicLabs--Julia-1/snapshots" / COMMIT
    shutil.copytree(path.removeprefix("path:"), snapshot)
    ref = f"hub:SupersonicLabs/Julia-1@{COMMIT}"
    config, _ = write_run(tmp_path / "run", ref, cache_dir="cache", keep_checkpoint=True)
    elsewhere = {"HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(tmp_path / "elsewhere")}
    process = headless(config, {**env, **elsewhere})
    stdout, stderr = process.communicate(timeout=900)
    assert process.returncode == 0, stderr[-3000:]
    events = [json.loads(line) for line in stdout.splitlines()]
    assert [e["type"] for e in events[:2]] == ["phase", "prepared"]
    assert events[1]["kind"] == "julia" and events[1]["base_model"] == ref
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "succeeded" and result["outputs"]["model"]
    run_dir = tmp_path / "run/ws/runs" / result["run_id"]
    record = json.loads((run_dir / "card/finetune.json").read_text())
    assert record["base_model"]["repo"] == "SupersonicLabs/Julia-1"
    assert record["base_model"]["revision"] == COMMIT


def test_a_cancel_during_the_base_download_ends_the_run_at_once(tmp_path, hub):
    """SIGTERM while the base model downloads (no job is running yet): the run stops waiting
    for the download at once, which goes on, left behind on its thread, and ends cancelled
    (143) with its result file and last event; nothing is written for the run."""
    hub.hold = threading.Event()
    ref = f"hub:aac6fef/laya-mlx@{COMMIT}"
    config, _ = write_run(tmp_path / "run", ref, cache_dir="cache")

    def cancel():
        if hub.downloading.wait(60):
            os.kill(os.getpid(), signal.SIGTERM)  # cloud.run's handler: Headless.cancel

    threading.Thread(target=cancel, daemon=True).start()
    before = signal.getsignal(signal.SIGTERM)
    out = io.StringIO()
    began = time.monotonic()
    try:
        code = cloud.run(config, stream=out)
        seconds = time.monotonic() - began
        downloading = [t for t in threading.enumerate() if t.name == cloud.DOWNLOAD_THREAD]
    finally:
        hub.hold.set()  # the download "finishes" behind the run's back
    assert code == 143 and seconds < 10  # not the 60 s the download is held for
    assert [t.is_alive() for t in downloading] == [True]  # left behind, never waited for
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [e["type"] for e in events] == ["phase", "finished"]
    message = "Cancelled while the base model was downloading"
    assert events[1] == {
        **events[1],
        "state": "cancelled",
        "exit_code": 143,
        "error": {"stage": "prepare", "message": message},
    }
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "cancelled" and result["exit_code"] == 143
    assert result["error"] == {"stage": "prepare", "message": message}
    assert len(hub.downloads) == 1 and not (tmp_path / "run/ws/runs").exists()
    assert signal.getsignal(signal.SIGTERM) == before  # the caller's handler is back


def test_a_cancel_during_the_checks_downloads_nothing(tmp_path, monkeypatch, hub):
    """SIGTERM while the dataset is checked, before the download: nothing is fetched for a run
    that is already over."""
    checks = cloud.dataset

    def signalled(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        return checks(*args, **kwargs)

    monkeypatch.setattr(cloud, "dataset", signalled)
    config, _ = write_run(tmp_path / "run", f"hub:aac6fef/laya-mlx@{COMMIT}")
    assert cloud.run(config, stream=io.StringIO()) == 143
    assert hub.downloads == [] and not (tmp_path / "run/ws/runs").exists()
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "cancelled" and result["error"]["stage"] == "prepare"


# `layastudio train` (cloud.main) with a hub whose download never ends, as a checkpoint's
# gigabytes on a slow link: on worker threads of a pool, as huggingface_hub downloads, which
# the interpreter's exit waits for. The cache lookup finds nothing.
ENDLESS_DOWNLOAD = """
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
import huggingface_hub

def snapshot_download(repo, revision=None, local_files_only=False, **options):
    if local_files_only:
        raise FileNotFoundError("not in the cache")
    print(json.dumps({"t": 0, "type": "test", "message": "downloading"}), flush=True)
    with ThreadPoolExecutor(2) as pool:
        pool.submit(time.sleep, 600).result()

huggingface_hub.snapshot_download = snapshot_download
from layastudio.cloud import main
sys.exit(main(sys.argv[1:]))
"""


def test_a_cancelled_download_ends_the_process_at_once(tmp_path):
    """As the pod agent stops a run: SIGTERM to `layastudio train` while its base model
    downloads. The process exits 143 within seconds, its result file written, instead of
    downloading on until the agent's SIGKILL 30 s later, with no result."""
    config, _ = write_run(tmp_path / "run", f"hub:aac6fef/laya-mlx@{COMMIT}", cache_dir="cache")
    process = subprocess.Popen(
        [sys.executable, "-c", ENDLESS_DOWNLOAD, "--config", str(config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(engine.PACKAGE.parent),
    )
    try:
        seen = []
        for line in process.stdout:
            seen.append(json.loads(line))
            if seen[-1]["type"] == "test":  # the download has started
                break
        assert [e["type"] for e in seen] == ["phase", "test"]
        began = time.monotonic()
        process.send_signal(signal.SIGTERM)
        rest, stderr = process.communicate(timeout=60)
        seconds = time.monotonic() - began
    finally:
        process.kill()
        process.wait()
    assert process.returncode == 143, stderr[-3000:]
    assert seconds < 15
    last = json.loads(rest.splitlines()[-1])
    assert last == {**last, "type": "finished", "state": "cancelled", "exit_code": 143}
    result = json.loads((tmp_path / "run/result.json").read_text())
    assert result["state"] == "cancelled" and result["error"]["stage"] == "prepare"


# ----------------------------------------------------------------------------- the manifest


def old_manifest(run_dir):
    """cloud.manifest as it was before it hashed each file once: every path by itself."""
    out = {}
    for group in cloud.OUTPUTS:
        folder = run_dir / group
        if not folder.is_dir():
            continue
        out[group] = [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": cloud._sha256(path),
            }
            for path in sorted(folder.rglob("*"))
            if path.is_file() and not path.name.endswith(".tmp")
        ]
    return out


def output_tree(run_dir):
    """A run's outputs as a Decider run leaves them: the GGUF hard-linked into the package,
    the weights into the checkpoint, nested folders, a .tmp file and an empty group."""
    for name, data in (
        ("noulxp/model.gguf", b"GGUF" + bytes(300)),
        ("noulxp/model.safetensors", b"weights" * 50),
        ("noulxp/conformance.jsonl", b"{}\n"),
        ("model/tokenizer/tokenizer.json", b"{}"),
        ("model/same-size-a.bin", b"a" * 64),
        ("model/same-size-b.bin", b"b" * 64),
        ("card/README.md", b"# card"),
        ("noulxp/upload.tmp", b"partial"),
    ):
        (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / name).write_bytes(data)
    (run_dir / "exports").mkdir()
    os.link(run_dir / "noulxp/model.gguf", run_dir / "exports/model-bf16.gguf")
    os.link(run_dir / "noulxp/model.safetensors", run_dir / "model/model.safetensors")
    return run_dir


@pytest.mark.parametrize("threads", [0, 1, 4])
def test_the_manifest_is_what_it_was_with_each_file_hashed_once(tmp_path, monkeypatch, threads):
    run_dir = output_tree(tmp_path / "run")
    (run_dir / "exports/gguf-bf16.json").write_text("{}")
    want = old_manifest(run_dir)
    hashed = []
    real = cloud._sha256
    monkeypatch.setattr(cloud, "_sha256", lambda path: hashed.append(path.name) or real(path))
    got = cloud.manifest(run_dir, threads=threads)
    assert got == want and list(got) == list(want)  # the same groups, in the same order
    assert sorted(hashed) == sorted(
        [
            "model.gguf",
            "model.safetensors",
            "conformance.jsonl",
            "tokenizer.json",
            "same-size-a.bin",
            "same-size-b.bin",
            "README.md",
            "gguf-bf16.json",
        ]
    )  # one read for each linked pair, two for two files of one size
    entries = {e["path"]: e for group in got.values() for e in group}
    assert entries["exports/model-bf16.gguf"]["sha256"] == entries["noulxp/model.gguf"]["sha256"]
    assert entries["model/same-size-a.bin"]["sha256"] != entries["model/same-size-b.bin"]["sha256"]
    assert "noulxp/upload.tmp" not in entries


def test_an_empty_output_group_is_listed_empty(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "exports").mkdir(parents=True)
    (run_dir / "card").mkdir()
    (run_dir / "card" / "x.tmp").write_text("partial")
    assert cloud.manifest(run_dir) == old_manifest(run_dir) == {"exports": [], "card": []}
