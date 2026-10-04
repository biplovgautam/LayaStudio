"""NoulXP packages: built from a tiny fine-tune, checked on the CPU, published with the run.

The model is the tiny random checkpoint the other tests use, in float16 like a real run's;
nothing is downloaded and no real model runs.
"""

import hashlib
import json
import shutil
import sys
import textwrap
from pathlib import Path

import pytest
from test_engine import QUESTIONS, WORDS, make_rows
from test_server import call, studio  # noqa: F401 - studio is a fixture

from layastudio import engine, families, noulxp_package, publish_systemone
from layastudio.export import export

TOOLING = noulxp_package.missing_tooling()
needs_tooling = pytest.mark.skipif(TOOLING is not None, reason=f"NoulXP tooling: {TOOLING}")
RUN = "tiny-run-1004-120000"
TEST_ROWS = 12


def tiny_checkpoint(path):
    """test_engine's random checkpoint, with float16 weights and real special-token ids."""
    import mlx.core as mx
    from laya_mlx.model import DecisionModel, EncoderConfig
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer, models, pre_tokenizers

    cfg = {
        "model_type": "modernbert",
        "vocab_size": 64,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 3,
        "num_attention_heads": 1,
        "local_attention": 16,
        "max_position_embeddings": 256,
        # transformers' ModernBERT defaults these to the real 50k vocabulary
        "pad_token_id": 0,
        "cls_token_id": 2,
        "bos_token_id": 2,
        "sep_token_id": 3,
        "eos_token_id": 3,
    }
    agent_cfg = {
        "encoder": "test/tiny",
        "head_layers": 1,
        "max_len": 96,
        "head_max_len": 40,
        "act_costs": {"escalate": 0.5},
        "temperature": [1.3, 1.1, 2.0],
        "temperature_by_options": {"choice:3-5": 0.8},
    }
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    (path / "encoder/config.json").write_text(json.dumps(cfg))
    (path / "rl_agent_config.json").write_text(json.dumps(agent_cfg))
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {t: i for i, t in enumerate(specials + WORDS + [":", "?", "level", "0", "1", "2"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer/tokenizer.json"))
    names, tokens = ("pad", "cls", "sep", "mask"), ("[PAD]", "[CLS]", "[SEP]", "[MASK]")
    (path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps({f"{n}_token": t for n, t in zip(names, tokens)})
    )
    mx.random.seed(3)
    model = DecisionModel(EncoderConfig.from_dict(cfg), agent_cfg)
    weights = {
        engine.upstream_name(k): v.astype(mx.float16) for k, v in tree_flatten(model.parameters())
    }
    mx.save_safetensors(str(path / "model.safetensors"), weights, metadata={"format": "pt"})
    return path


def make_workspace(root, base_model="hub:aac6fef/laya-mlx"):
    """A workspace with a dataset and a finished run whose checkpoint is the tiny model."""
    workspace = root / "ws"
    rows = make_rows(60)
    for row in rows[:20]:
        row["split"] = "test"
    rows[0]["state"] = {"board": ["red", "green"], "note": "élan"}  # a JSON state
    text = "\n".join(json.dumps(r) for r in rows)
    meta = engine.create_dataset("tiny", QUESTIONS, text, "t.jsonl", workspace=workspace)
    run_dir = workspace / "runs" / RUN
    tiny_checkpoint(run_dir / "model")
    engine.write_json(run_dir / "model/questions.json", QUESTIONS)
    engine.write_json(
        run_dir / "run.json",
        {
            "id": RUN,
            "name": "tiny run",
            "dataset": meta["id"],
            "base_model": base_model,
            "hyperparameters": {**engine.HYPERPARAMETERS},
            "created": engine.now(),
        },
    )
    engine.write_json(run_dir / "training.json", {"dataset_sha256": meta["sha256"]})
    return workspace


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One real build, shared: export, conformance, validate and check of the tiny run."""
    if TOOLING is not None:
        pytest.skip(f"NoulXP tooling: {TOOLING}")
    workspace = make_workspace(tmp_path_factory.mktemp("noulxp"))
    model = workspace / "runs" / RUN / "model"
    before = {p.relative_to(model): p.read_bytes() for p in model.rglob("*") if p.is_file()}
    events = []
    report = export(
        f"run:{RUN}",
        "noulxp",
        workspace,
        lambda kind, **data: events.append({"type": kind, **data}),
        test_rows=TEST_ROWS,
    )
    return workspace, report, events, before


def fake_systemone(tmp_path):
    """A stand-in for `systemone push` that writes down every file it was asked to upload."""
    seen = tmp_path / "pushed.json"
    script = tmp_path / "systemone.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, sys
            from pathlib import Path
            folder = Path(sys.argv[2])
            files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
            Path({str(seen)!r}).write_text(json.dumps({{"args": sys.argv[1:], "files": files}}))
            print("Published (pretend)")
            """
        )
    )
    return [sys.executable, str(script)], seen


# ----------------------------------------------------------------------------- families


def test_every_family_says_where_noulxp_stands():
    assert set(families.NOULXP) == set(families.FAMILIES)
    ready = [k for k, v in families.NOULXP.items() if v["status"] == "ready"]
    assert ready == ["laya"] and families.NOULXP["laya"]["profile"] == "encoder-markers"
    assert families.NOULXP["letter"]["profile"] == "causal-letters"
    assert families.NOULXP["crossenc"]["profile"] == "encoder-pairs"
    for key in ("head", "gliner", "embedder", "tiny"):
        assert families.NOULXP[key]["profile"] is None
        assert "profile first" in noulxp_package.refusal(key)
    for key in ("letter", "crossenc"):
        assert "arrives with the trainer" in noulxp_package.refusal(key)
    assert all("noulxp" in f for f in families.catalogue()["families"])


def test_a_family_without_noulxp_export_is_refused(tmp_path):
    workspace = make_workspace(tmp_path, base_model="hub:Mapika/decider-2b")
    with pytest.raises(ValueError, match="arrives with the trainer"):
        export(f"run:{RUN}", "noulxp", workspace)
    assert not (workspace / "runs" / RUN / noulxp_package.PACKAGE).exists()
    with pytest.raises(ValueError, match="fine-tuned runs"):
        export(f"path:{workspace / 'runs' / RUN / 'model'}", "noulxp", workspace)


def test_own_requests_follow_the_package_limits():
    pytest.importorskip("noulxp")
    questions = {
        **QUESTIONS,
        "many": {
            "type": "choice",
            "instructions": "Pick",
            "criteria": [f"o{i}" for i in range(25)],
        },
        "deep": {"type": "score", "instructions": "Rate", "criteria": [str(i) for i in range(12)]},
        "odd": {"type": "noul", "instructions": "Odd?", "criteria": {"maybe": "x"}},
        "topic2": {**QUESTIONS["topic"], "note": "not a System One field"},
    }
    rows = [{"id": f"r{i}", "state": "red", "split": "test"} for i in range(5)]
    rows += [{"id": "j", "state": {"a": "é", "b": [1, 2]}, "split": "test"}]
    rows += [{"id": "t", "state": "green", "split": "train"}]
    limits = {"max_options": 20, "max_levels": 10}
    requests, asked = noulxp_package.own_requests(questions, rows, limits, count=6)
    assert set(asked["left_out"]) == {"many", "deep", "odd"}
    assert "25 options" in asked["left_out"]["many"] and "12 levels" in asked["left_out"]["deep"]
    assert asked["asked"] == ["topic", "level", "flag", "topic2"]
    assert len(requests) == 6 and len({r["id"] for r in requests}) == 6
    assert all(noulxp_package.TEST_TAG in r["tags"] for r in requests)
    assert requests[-1]["request"]["state"] == json.dumps(
        {"a": "é", "b": [1, 2]}, ensure_ascii=False
    )
    assert "note" not in requests[0]["request"]["questions"]["topic2"]
    assert noulxp_package.own_requests(questions, rows, limits, count=0)[0] == []


# ----------------------------------------------------------------------------- build


@needs_tooling
def test_the_package_is_built_checked_and_kept_with_the_run(built):
    workspace, report, events, before = built
    run_dir = workspace / "runs" / RUN
    assert report["state"] == "passed", report
    check = report["check"]
    assert check["cases"] == 52 + TEST_ROWS and check["cases_passed"] == check["cases"]
    assert check["compatible"] and check["max_abs_dp"] < 0.01 and check["device"] == "cpu"
    assert report["conformance"]["test_rows"] == TEST_ROWS
    package = run_dir / noulxp_package.PACKAGE
    for name in ("noulxp.json", "model.onnx", "model.safetensors", "tokenizer.json"):
        assert (package / name).is_file(), name
    for name in ("template.json", "calibration.json", "conformance.jsonl", "check-cpu.json"):
        assert (package / name).is_file(), name
    assert not (run_dir / noulxp_package.BUILDING).exists()
    assert not (run_dir / noulxp_package.FAILED).exists()
    assert not list(run_dir.glob(".noulxp-*"))  # the scratch folder is gone

    manifest = json.loads((package / "noulxp.json").read_text())
    assert manifest["profile"] == "encoder-markers" and manifest["name"] == f"studio:{RUN}"
    assert manifest["source"]["model"] == "aac6fef/laya-mlx"
    assert manifest["source"]["fine_tune"]["run"] == RUN
    assert manifest["conformance"]["generated_by"]["runtime"] == "laya"
    assert manifest["conformance"]["generated_by"]["device"] == "cpu"
    assert str(workspace) not in (package / "noulxp.json").read_text()  # no local paths
    assert str(workspace) not in (package / "check-cpu.json").read_text()
    checked = json.loads((package / "check-cpu.json").read_text())["package"]
    assert checked["path"] == "noulxp"  # named as the version's folder it is published as

    # The checkpoint is as training left it: laya never touched its tokenizer config.
    model = run_dir / "model"
    after = {p.relative_to(model): p.read_bytes() for p in model.rglob("*") if p.is_file()}
    assert {name: after.get(name) for name in before} == before
    assert set(after) - set(before) <= {Path("README.md")}  # a publish writes the card

    phases = [e["phase"] for e in events if e["type"] == "phase"]
    assert phases == ["export", "conformance", "validate", "check"]
    result = next(e for e in events if e["type"] == "result")
    assert result["target"] == "noulxp" and result["cases_passed"] == check["cases"]
    assert any(e["type"] == "progress" for e in events)

    info = noulxp_package.passing_package(run_dir, model)
    assert info and info["test_rows"] == TEST_ROWS and info["cases"] == 52 + TEST_ROWS
    listed = noulxp_package.listing(run_dir)
    assert listed["state"] == "passed" and listed["model"] == f"run:{RUN}"
    line = noulxp_package.card_line(info)
    assert f"reproduced {info['cases']} of {info['cases']} cases" in line
    assert f"{TEST_ROWS} rows of this model's test split" in line


@needs_tooling
def test_publish_carries_the_package_and_cleans_up(built, tmp_path, monkeypatch):
    workspace, _, _, _ = built
    command, seen = fake_systemone(tmp_path)
    monkeypatch.setattr(publish_systemone, "cli_command", lambda: command)
    model = workspace / "runs" / RUN / "model"

    publish_systemone.publish(f"run:{RUN}", "me/tiny", workspace, dry_run=True)
    pushed = json.loads(seen.read_text())
    assert "--dry-run" in pushed["args"] and "model.safetensors" in pushed["files"]
    for name in ("noulxp.json", "model.onnx", "conformance.jsonl", "check-cpu.json"):
        assert f"noulxp/{name}" in pushed["files"]
    assert not (model / "noulxp").exists()  # only there while the upload runs
    card = (model / "README.md").read_text()
    assert "**NoulXP:** this version carries a NoulXP package" in card
    assert card.index("**NoulXP:**") < card.index("## Provenance")

    # Skipping leaves the package out, even one an interrupted upload left behind.
    (model / "noulxp").mkdir()
    (model / "noulxp/noulxp.json").write_text("{}")
    publish_systemone.publish(f"run:{RUN}", "me/tiny", workspace, dry_run=True, noulxp=False)
    pushed = json.loads(seen.read_text())
    assert not [f for f in pushed["files"] if f.startswith("noulxp/")]
    assert "published without a NoulXP package" in (model / "README.md").read_text()
    assert not (model / "noulxp").exists()


@needs_tooling
def test_a_failing_package_is_kept_apart_and_never_published(built, tmp_path, monkeypatch):
    source, _, _, _ = built
    workspace = tmp_path / "ws"
    shutil.copytree(source, workspace)
    run_dir = workspace / "runs" / RUN
    passing = source / "runs" / RUN / noulxp_package.PACKAGE

    # Skip the slow graph export: the files it wrote last time are the same files.
    def export_package(model_dir, out_dir, name, src, emit):
        out_dir.mkdir()
        for path in passing.iterdir():
            if path.name not in ("conformance.jsonl", "check-cpu.json"):
                shutil.copyfile(path, out_dir / path.name)

    # Record the conformance file for real, then falsify one expected answer, hash and all.
    real_record = noulxp_package.record_conformance

    def record_conformance(package_dir, checkpoint, requests, emit):
        real_record(package_dir, checkpoint, requests, emit)
        path = package_dir / "conformance.jsonl"
        cases = [json.loads(line) for line in path.read_text().splitlines()]
        expected = next(iter(cases[0]["expected"].values()))["probabilities"]
        for key in expected:  # every bit of the probability on the last option
            expected[key] = 0.0
        expected[key] = 1.0
        path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
        manifest = json.loads((package_dir / "noulxp.json").read_text())
        manifest["conformance"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (package_dir / "noulxp.json").write_text(json.dumps(manifest, indent=2))

    monkeypatch.setattr(noulxp_package, "export_package", export_package)
    monkeypatch.setattr(noulxp_package, "record_conformance", record_conformance)

    # A failed rebuild leaves the run's passing package alone.
    with pytest.raises(RuntimeError, match="never published"):
        export(f"run:{RUN}", "noulxp", workspace, test_rows=TEST_ROWS)
    failed = run_dir / noulxp_package.FAILED
    assert json.loads((failed / "check-cpu.json").read_text())["passed"] is False
    assert noulxp_package.passing_package(run_dir, run_dir / "model")
    listed = noulxp_package.listing(run_dir)
    assert listed["state"] == "passed" and "last_attempt" in listed

    # Without one, publishing builds it first, and stops when it fails: nothing goes up.
    shutil.rmtree(run_dir / noulxp_package.PACKAGE)
    command, seen = fake_systemone(tmp_path)
    monkeypatch.setattr(publish_systemone, "cli_command", lambda: command)
    with pytest.raises(RuntimeError, match="Nothing was uploaded"):
        publish_systemone.publish(f"run:{RUN}", "me/tiny", workspace, dry_run=True)
    assert not seen.exists()
    assert not (run_dir / "model" / "noulxp").exists()
    assert not (run_dir / noulxp_package.PACKAGE).exists()
    assert noulxp_package.listing(run_dir)["state"] == "failed"


def test_a_tampered_package_does_not_count_as_passing(tmp_path):
    """passing_package re-hashes every file: an edited package is no package."""
    run_dir = tmp_path / "run"
    package = run_dir / noulxp_package.PACKAGE
    model = run_dir / "model"
    package.mkdir(parents=True)
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    for name, data in (("model.onnx", b"graph"), ("conformance.jsonl", b"{}\n")):
        (package / name).write_bytes(data)
    shutil.copyfile(model / "model.safetensors", package / "model.safetensors")

    def entry(name):
        return {"path": name, "sha256": hashlib.sha256((package / name).read_bytes()).hexdigest()}

    manifest = {
        "standard": "noulxp/0.1",
        "profile": "encoder-markers",
        "weights": {**entry("model.onnx"), "data": [entry("model.safetensors")]},
        "conformance": entry("conformance.jsonl"),
    }
    (package / "noulxp.json").write_text(json.dumps(manifest))
    check = {
        "passed": True,
        "compatible": True,
        "cases": 1,
        "cases_passed": 1,
        "package": {
            "conformance_sha256": manifest["conformance"]["sha256"],
            "weights_sha256": manifest["weights"]["sha256"],
        },
    }
    (package / "check-cpu.json").write_text(json.dumps(check))
    assert noulxp_package.passing_package(run_dir, model)
    (model / "model.safetensors").write_bytes(b"other weights")  # not this checkpoint's package
    assert noulxp_package.passing_package(run_dir, model) is None
    (model / "model.safetensors").write_bytes(b"weights")
    (package / "model.onnx").write_bytes(b"edited")
    assert noulxp_package.passing_package(run_dir, model) is None
    (package / "model.onnx").write_bytes(b"graph")
    (package / "check-cpu.json").write_text(json.dumps({**check, "passed": False}))
    assert noulxp_package.passing_package(run_dir, model) is None


def test_the_card_says_when_a_version_has_no_package():
    line = noulxp_package.card_line(None)
    assert "without a NoulXP package" in line and "compatibility" in line
    card = publish_systemone.with_noulxp("# T\n\ntext\n\n## Provenance\n\n{}\n", line)
    assert card.index(line) < card.index("## Provenance")
    assert publish_systemone.with_noulxp("# T\n", line).endswith(line + "\n")


def test_the_cli_names_the_target(capsys):
    from layastudio.export import PRECISIONS, TARGETS, main

    assert "noulxp" in TARGETS and PRECISIONS["noulxp"] == ("float",)
    with pytest.raises(SystemExit):
        main(["run:x", "--target", "noulxp", "--help"])
    assert "--test-rows" in capsys.readouterr().out


def test_paths_in_the_package_stay_inside_it():
    assert noulxp_package._safe("model.onnx") and noulxp_package._safe("a/b.json")
    for bad in ("/etc/passwd", "../x", "a\\b", "c:x", ""):
        assert not noulxp_package._safe(bad)


@pytest.fixture
def no_tooling(monkeypatch):
    monkeypatch.setattr(noulxp_package, "missing_tooling", lambda: "noulxp is not installed")


def test_missing_tooling_is_explained(tmp_path, no_tooling):
    workspace = make_workspace(tmp_path)
    with pytest.raises(RuntimeError, match="uv sync --extra export"):
        export(f"run:{RUN}", "noulxp", workspace)
    with pytest.raises(RuntimeError, match="--skip-noulxp"):
        noulxp_package.for_publish(f"run:{RUN}", workspace)


def test_release_parsing():
    assert noulxp_package._release("0.4.0") == (0, 4)
    assert noulxp_package._release("0.4.12rc1") == (0, 4)
    assert noulxp_package._release("1.0") == (1, 0)
    assert noulxp_package._release("5") == (5, 0)


def test_runs_name_their_base_without_local_paths(tmp_path):
    workspace = tmp_path / "ws"
    imported = {"ref": "path:/somewhere/snapshot", "repo": "someone/laya-ft"}
    engine.write_json(workspace / "imports.json", [imported])
    engine.write_json(workspace / "runs/parent/run.json", {"base_model": "hub:aac6fef/laya-mlx"})
    assert noulxp_package.base_model("hub:aac6fef/laya-mlx", workspace) == "aac6fef/laya-mlx"
    assert noulxp_package.base_model("path:/somewhere/snapshot", workspace) == "someone/laya-ft"
    assert noulxp_package.base_model("path:/elsewhere", workspace) is None
    assert noulxp_package.base_model("run:parent", workspace) == "aac6fef/laya-mlx"
    assert noulxp_package.base_model("", workspace) is None


# ----------------------------------------------------------------------------- the API


def no_jobs(app, monkeypatch):
    """These tests start no job: a publish job would upload with this machine's login."""

    def refuse(*args, **kwargs):
        raise AssertionError(f"a job was started: {args[:1]}")

    monkeypatch.setattr(app.jobs, "start", refuse)


def test_the_api_explains_noulxp_before_starting_a_job(studio, tmp_path, monkeypatch):  # noqa: F811
    base, app = studio
    no_jobs(app, monkeypatch)
    workspace = make_workspace(tmp_path)  # the same workspace the studio serves
    assert app.workspace == workspace
    status, info = call(base, f"/api/runs/{RUN}/noulxp")
    # 0 by default: a test row in the conformance file is published with the package
    assert status == 200 and info["package"] is None and info["test_rows"] == 0
    status, catalogue = call(base, "/api/families")
    assert all(f["noulxp"]["status"] for f in catalogue["families"])

    model = workspace / "runs" / RUN / "model"
    status, body = call(
        base, "/api/jobs", {"kind": "export", "model": f"path:{model}", "target": "noulxp"}
    )
    assert status == 400 and "fine-tuned runs" in body["error"]
    status, body = call(
        base,
        "/api/jobs",
        {"kind": "export", "model": f"run:{RUN}", "target": "noulxp", "precision": "int8"},
    )
    assert status == 400 and "float" in body["error"]

    monkeypatch.setattr(noulxp_package, "missing_tooling", lambda: "noulxp is not installed")
    status, body = call(
        base, "/api/jobs", {"kind": "export", "model": f"run:{RUN}", "target": "noulxp"}
    )
    assert status == 409 and "uv sync --extra export" in body["error"]
    monkeypatch.setattr(app.account, "status", lambda: {"signed_in": True})
    status, body = call(base, "/api/jobs", {"kind": "publish", "model": f"run:{RUN}"})
    assert status == 409 and "publish without a NoulXP package" in body["error"]
    assert call(base, "/api/jobs")[1]["count"] == 0  # nothing was started


def test_the_api_refuses_noulxp_for_a_family_without_it(studio, tmp_path, monkeypatch):  # noqa: F811
    base, app = studio
    no_jobs(app, monkeypatch)
    make_workspace(tmp_path, base_model="hub:Mapika/decider-2b")
    status, body = call(
        base, "/api/jobs", {"kind": "export", "model": f"run:{RUN}", "target": "noulxp"}
    )
    assert status == 400 and "arrives with the trainer" in body["error"]
