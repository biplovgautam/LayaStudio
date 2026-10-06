"""The NoulXP steps' command lines and environments, and the merge of a conformance file
recorded beside the export: with stand-ins for the step processes, so they run anywhere.

The guards here keep what a package certifies where it is: the check always hashes every file
(validate may skip that only because the check does it), always runs on the CPU, at the build's
thread count (the recording runs at that count too, or one fewer beside the export), and never
with a serving option. Every step runs behind children.WATCH, with its own arguments after it."""

import hashlib
import io
import json
import os
import subprocess
import sys

import pytest

from systemone_studio import children, engine, gguf, noulxp_package

WATCHED = [sys.executable, "-c", children.RUN]  # what every step's argv starts with


def quiet(*_args, **_kwargs):
    pass


FORBIDDEN = ("--no-hashes", "--batch-rows", "--batch-cache", "--optimization", "--precision")


class FakePopen:
    """A step process that prints nothing and ends at once; check's writes its report."""

    seen = []

    def __init__(self, args, **kwargs):
        self.args, self.env = list(args), dict(kwargs.get("env") or {})
        self.stdin_given = kwargs.get("stdin")  # children.WATCH reads a pipe
        self.pid, self.returncode = 999999, 0
        self.stdout = io.StringIO("")
        FakePopen.seen.append(self)
        if "--report" in self.args:
            path = self.args[self.args.index("--report") + 1]
            with open(path, "w") as handle:
                json.dump({"passed": True, "compatible": True}, handle)

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


@pytest.fixture
def steps(monkeypatch):
    FakePopen.seen = []
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    for name in noulxp_package.THREAD_VARS:
        monkeypatch.delenv(name, raising=False)
    return FakePopen.seen


def test_the_check_runs_exactly_so(tmp_path, steps):
    package = tmp_path / "noulxp"
    package.mkdir()
    report = noulxp_package.check_package(package, lambda *a, **k: None, threads=6)
    assert report["passed"]
    [check] = steps
    assert check.args == [
        *WATCHED,
        "-m",
        "noulxp",
        "check",
        str(package),
        "--device",
        "cpu",
        "--report",
        str(package / noulxp_package.CHECK),
        "--threads",
        "6",
    ]
    assert not set(FORBIDDEN) & set(check.args)
    assert {name: check.env[name] for name in noulxp_package.THREAD_VARS} == {
        "OMP_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "6",
        "SYSTEMONE_STUDIO_THREADS": "6",
        "LAYASTUDIO_THREADS": "6",
        "NOULXP_THREADS": "6",
    }
    assert check.env["HF_HUB_OFFLINE"] == "1"
    assert check.env[children.PARENT] == str(os.getpid())
    assert check.stdin_given is subprocess.PIPE


def test_validate_leaves_the_hashes_to_the_check_and_only_to_it(tmp_path, steps):
    package = tmp_path / "noulxp"
    package.mkdir()
    emit = lambda *a, **k: None  # noqa: E731
    assert noulxp_package.validate_package(package, emit) == []
    noulxp_package.check_package(package, emit, threads=2)
    validate, check = steps
    assert validate.args[:3] == WATCHED
    assert validate.args[3:] == ["-m", "noulxp", "validate", str(package), "--no-hashes"]
    assert "--threads" not in validate.args and "OMP_NUM_THREADS" not in validate.env
    assert "--no-hashes" not in check.args


def test_the_recording_gets_its_threads_and_every_request(tmp_path, steps):
    noulxp_package.record_conformance(
        tmp_path / "pkg", tmp_path / "view", tmp_path / "r.jsonl", print, "decider", threads=6
    )
    [generate] = steps
    assert generate.args[:3] == WATCHED and generate.args[3:] == [
        "-m",
        "noulxp",
        "conformance",
        "generate",
        str(tmp_path / "pkg"),
        "--native",
        str(tmp_path / "view"),
        "--runtime",
        "decider",
        "--requests",
        str(tmp_path / "r.jsonl"),
        "--threads",
        "6",
    ]
    assert "--limit" not in generate.args and generate.env["OMP_NUM_THREADS"] == "6"


def test_a_cap_the_machine_sets_wins_and_is_what_is_reported(tmp_path, steps, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    noulxp_package.check_package(tmp_path, print, threads=6)
    [check] = steps
    assert check.env["OMP_NUM_THREADS"] == "2" and check.env["MKL_NUM_THREADS"] == "6"
    assert noulxp_package.thread_env(6)["OMP_NUM_THREADS"] == "2"
    # Without a count, a step's environment is as it always was.
    assert noulxp_package.thread_env(None) == {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": None,
        "OPENBLAS_NUM_THREADS": None,
        "SYSTEMONE_STUDIO_THREADS": None,
        "LAYASTUDIO_THREADS": None,
        "NOULXP_THREADS": None,
    }


def test_a_step_gets_its_own_count_whatever_the_job_was_given(tmp_path, steps, monkeypatch):
    """LAYASTUDIO_THREADS=8 set the job's budget; the export beside the recording still gets 1
    for its comparison (ORT_THREADS reads it)."""
    monkeypatch.setenv("LAYASTUDIO_THREADS", "8")
    noulxp_package.export_package(tmp_path, tmp_path / "out", "n", {}, print, "laya", threads=1)
    [export] = steps
    assert {name: export.env[name] for name in noulxp_package.THREAD_VARS} == dict.fromkeys(
        noulxp_package.THREAD_VARS, "1"
    )


def test_the_export_caps_its_threads_and_runs_its_exporter(tmp_path, steps):
    noulxp_package.export_package(tmp_path, tmp_path / "out", "n", {}, print, "julia", threads=1)
    [export] = steps
    assert export.args[:3] == WATCHED
    assert export.args[3] == "-c" and export.args[4] == noulxp_package.EXPORT
    assert export.args[4].startswith(noulxp_package.ORT_THREADS)
    assert export.env["LAYASTUDIO_THREADS"] == "1" and export.env["OMP_NUM_THREADS"] == "1"
    assert export.env["SYSTEMONE_STUDIO_THREADS"] == "1"
    assert export.env["NOULXP_THREADS"] == "1"


# ----------------------------------------------------------------------------- ORT_THREADS


@pytest.fixture
def ort_graph(tmp_path):
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    providers = pytest.importorskip("noulxp.providers")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    path = tmp_path / "one.onnx"
    onnx.save(model, str(path))
    original = providers.ort_session
    yield providers, path, original
    providers.ort_session = original


def intra(session):
    return session.get_session_options().intra_op_num_threads


def test_ort_threads_caps_a_session_opened_without_a_count(ort_graph, monkeypatch):
    providers, path, original = ort_graph
    monkeypatch.setenv("LAYASTUDIO_THREADS", "3")
    exec(noulxp_package.ORT_THREADS, {})
    assert providers.ort_session is not original
    assert providers.ort_session.__wrapped__ is original
    session, _ = providers.ort_session(path, provider="cpu")
    assert intra(session) == 3
    session, _ = providers.ort_session(path, provider="cpu", threads=5)  # asked for: kept
    assert intra(session) == 5
    session, _ = providers.ort_session(path, provider="cpu", optimization="disable")
    assert intra(session) == 3


def test_ort_threads_reads_the_new_name_before_the_old_one(ort_graph, monkeypatch):
    """SYSTEMONE_STUDIO_THREADS wins over LAYASTUDIO_THREADS, as everywhere (environment.py)."""
    providers, path, original = ort_graph
    monkeypatch.setenv("LAYASTUDIO_THREADS", "7")
    monkeypatch.setenv("SYSTEMONE_STUDIO_THREADS", "2")
    exec(noulxp_package.ORT_THREADS, {})
    session, _ = providers.ort_session(path, provider="cpu")
    assert intra(session) == 2


@pytest.mark.parametrize("value", [None, "", "0", "-1", "8.5", "abc"])
def test_ort_threads_does_nothing_without_a_positive_count(ort_graph, monkeypatch, value):
    providers, path, original = ort_graph
    if value is None:
        monkeypatch.delenv("LAYASTUDIO_THREADS", raising=False)
    else:
        monkeypatch.setenv("LAYASTUDIO_THREADS", value)
    exec(noulxp_package.ORT_THREADS, {})
    assert providers.ort_session is original


# The export step's own preamble, then the exporter's informative comparison as the noulxp
# installed (or on PYTHONPATH) runs it, on a pod whose quota binds: 8.5 CPUs of 10 visible.
# noulxp 0.4.1 reads that quota for any count it is not given; 0.4.0 has no such reading.
COMPARISON = noulxp_package.ORT_THREADS + (
    "import json, sys\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import torch\n"
    "import noulxp.providers as providers\n"
    "if hasattr(providers, 'cpu_quota'):\n"
    "    providers.cpu_quota = lambda *a, **k: 8.5\n"
    "if hasattr(providers, 'affinity'):\n"
    "    providers.affinity = lambda: 10\n"
    "seen = {'ort': [], 'torch': []}\n"
    "capped = providers.ort_session\n"
    "def spy(path, *args, **kwargs):\n"
    "    session, used = capped(path, *args, **kwargs)\n"
    "    seen['ort'].append(session.get_session_options().intra_op_num_threads)\n"
    "    return session, used\n"
    "providers.ort_session = spy\n"
    "class Positions(torch.nn.Module):\n"
    "    def forward(self, ids, mask, positions, markers, qtype):\n"
    "        seen['torch'].append(torch.get_num_threads())\n"
    "        return positions.float()\n"
    "from noulxp.export import onnx_graph\n"
    "feeds = {\n"
    "    'input_ids': np.array([[1, 2, 3, 4]], dtype=np.int64),\n"
    "    'attention_mask': np.ones((1, 4), dtype=np.int64),\n"
    "    'marker_positions': np.array([[0, 1]], dtype=np.int64),\n"
    "    'marker_mask': np.array([[True, True]]),\n"
    "    'question_type': np.array([0], dtype=np.int64),\n"
    "}\n"
    "result = onnx_graph.compare_with_torch(Path(sys.argv[1]), Positions(), [feeds])\n"
    "print(json.dumps({**seen, 'rows': result['rows']}))\n"
)


@pytest.fixture
def positions_graph(tmp_path):
    """option_logits = marker_positions as floats, over the five inputs of an encoder graph."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("torch")
    pytest.importorskip("noulxp.export.onnx_graph")
    from onnx import TensorProto, helper

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "tokens"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "tokens"]),
        helper.make_tensor_value_info("marker_positions", TensorProto.INT64, ["batch", "options"]),
        helper.make_tensor_value_info("marker_mask", TensorProto.BOOL, ["batch", "options"]),
        helper.make_tensor_value_info("question_type", TensorProto.INT64, ["batch"]),
    ]
    output = helper.make_tensor_value_info("option_logits", TensorProto.FLOAT, ["batch", "options"])
    node = helper.make_node("Cast", ["marker_positions"], ["option_logits"], to=TensorProto.FLOAT)
    graph = helper.make_graph([node], "positions", inputs, [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    path = tmp_path / "positions.onnx"
    onnx.save(model, str(path))
    return path


@pytest.mark.parametrize("threads", [1, 3])
def test_the_exporters_comparison_runs_at_the_steps_threads(positions_graph, monkeypatch, threads):
    """The export step's own count caps both halves of the exporter's informative comparison,
    onnxruntime and torch, even where a quota binds: with noulxp 0.4.0 (ORT_THREADS and
    OMP_NUM_THREADS) and with 0.4.1, which reads NOULXP_THREADS instead of the quota."""
    for name in noulxp_package.THREAD_VARS:
        monkeypatch.delenv(name, raising=False)
    done = subprocess.run(
        [sys.executable, "-c", COMPARISON, str(positions_graph)],
        env=noulxp_package._child_env(threads),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen["rows"] == 1
    assert seen["ort"] == [threads] and seen["torch"] == [threads]


# ----------------------------------------------------------------------------- the merge


def recorded(tmp_path, cases=2, extra=None):
    """A stub after `noulxp conformance generate` (its manifest and file) and an exported
    package without a conformance file."""
    stub, building = tmp_path / "stub", tmp_path / "noulxp"
    stub.mkdir(parents=True)
    building.mkdir(parents=True)
    lines = "".join(json.dumps({"id": f"c{i}", "request": {}}) + "\n" for i in range(cases))
    (stub / "conformance.jsonl").write_text(lines)
    summary = {
        "path": "conformance.jsonl",
        "sha256": hashlib.sha256(lines.encode()).hexdigest(),
        "cases": cases,
        "generated_by": {"runtime": "julia", "threads": 3, "seconds": 1.0},
    }
    (stub / "noulxp.json").write_text(json.dumps({"conformance": summary, **(extra or {})}))
    exported = {"standard": "noulxp/0.1", "name": "studio:é", "weights": {"path": "model.onnx"}}
    (building / "noulxp.json").write_text(json.dumps(exported))
    return stub, building, summary, exported


def test_the_merge_moves_the_file_and_adds_only_its_summary(tmp_path):
    stub, building, summary, exported = recorded(tmp_path)
    assert noulxp_package.merge_conformance(stub, building, 2) == summary
    manifest_text = (building / "noulxp.json").read_text(encoding="utf-8")
    # As `noulxp conformance generate` writes it: the exporter's keys, then "conformance".
    assert (
        manifest_text
        == json.dumps({**exported, "conformance": summary}, ensure_ascii=False, indent=2) + "\n"
    )
    assert (building / "conformance.jsonl").is_file()
    assert not (stub / "conformance.jsonl").exists()


@pytest.mark.parametrize(
    "spoil,match",
    [
        (lambda stub, building: (stub / "extra.txt").write_text("x"), "other files"),
        (lambda stub, building: (stub / "noulxp.json").write_text("[]"), "not what generate"),
        (
            lambda stub, building: (building / "conformance.jsonl").write_text("{}\n"),
            "already has",
        ),
        (
            lambda stub, building: (building / "noulxp.json").write_text(
                json.dumps({"conformance": {"path": "conformance.jsonl"}})
            ),
            "already has",
        ),
        (lambda stub, building: (building / "noulxp.json").unlink(), "has no noulxp.json"),
        (lambda stub, building: (building / "noulxp.json").write_text("[1]"), "has no"),
        (
            lambda stub, building: (stub / "conformance.jsonl").write_text("{}\n{}\n"),
            "SHA-256",
        ),
    ],
)
def test_the_merge_refuses_anything_unexpected(tmp_path, spoil, match):
    stub, building, _, _ = recorded(tmp_path)
    spoil(stub, building)
    with pytest.raises(RuntimeError, match=match):
        noulxp_package.merge_conformance(stub, building, 2)


def test_the_merge_refuses_another_name_count_or_key(tmp_path):
    stub, building, summary, _ = recorded(tmp_path, extra={"weights": {}})
    with pytest.raises(RuntimeError, match="not what generate"):
        noulxp_package.merge_conformance(stub, building, 2)
    stub, building, summary, _ = recorded(tmp_path / "b")
    with pytest.raises(RuntimeError, match="holds 2 cases, not 3"):
        noulxp_package.merge_conformance(stub, building, 3)
    stub, building, summary, _ = recorded(tmp_path / "c")
    (stub / "noulxp.json").write_text(
        json.dumps({"conformance": {**summary, "path": "../conformance.jsonl"}})
    )
    with pytest.raises(RuntimeError, match="named"):
        noulxp_package.merge_conformance(stub, building, 2)


# ----------------------------------------------------------------------------- the GGUF's reuse


def test_ensure_gguf_hashes_only_what_it_can_reuse(tmp_path, monkeypatch):
    """No GGUF yet: nothing hashed, the export reads the digests itself. An up-to-date one is
    reused. A stale one is converted again, with the digests computed for the decision."""
    run_dir = tmp_path / "runs" / "dv"
    model = run_dir / "model"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        noulxp_package, "locate", lambda ref, ws: ("dv", run_dir, model, {"id": "dv"})
    )
    hashed, exported = [], []
    real = noulxp_package._sha256
    monkeypatch.setattr(noulxp_package, "_sha256", lambda p: hashed.append(p) or real(p))
    target = run_dir / "exports" / "model-bf16.gguf"

    def export(model_ref, workspace, emit, precision, threads=None, source_sha256=None):
        exported.append({"threads": threads, "source_sha256": source_sha256})
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"GGUF converted")
        report = {
            "sha256": real(target),
            "source_sha256": {"model.safetensors": real(model / "model.safetensors")},
            "llama_cpp": gguf.LLAMA_CPP_COMMIT,
            "timings": {"convert_s": 1.0},
        }
        engine.write_json(run_dir / "exports/gguf-bf16.json", report)
        return report

    monkeypatch.setattr(gguf, "export", export)
    stats = {}
    noulxp_package.ensure_gguf("run:dv", tmp_path, quiet, "bf16", threads=3, stats=stats)
    assert hashed == [] and exported == [{"threads": 3, "source_sha256": None}]
    assert stats == {"cached": False}

    _, report = noulxp_package.ensure_gguf("run:dv", tmp_path, quiet, "bf16", stats=stats)
    assert len(exported) == 1 and stats == {"cached": True} and report["timings"]
    assert sorted(p.name for p in hashed) == ["model-bf16.gguf", "model.safetensors"]

    hashed.clear()
    (model / "model.safetensors").write_bytes(b"other weights")
    noulxp_package.ensure_gguf("run:dv", tmp_path, quiet, "bf16", stats=stats)
    assert [p.name for p in hashed] == ["model.safetensors"]  # the GGUF is not worth hashing
    assert exported[-1]["source_sha256"] == {"model.safetensors": real(model / "model.safetensors")}

    hashed.clear()
    engine.write_json(run_dir / "exports/gguf-bf16.json", {"llama_cpp": "another commit"})
    noulxp_package.ensure_gguf("run:dv", tmp_path, quiet, "bf16", stats=stats)
    assert hashed == [] and exported[-1]["source_sha256"] is None


# ----------------------------------------------------------------------------- the gate, the emit


def test_only_encoders_without_test_rows_record_beside_the_export(monkeypatch):
    gate = noulxp_package.overlap_conformance
    monkeypatch.delenv("LAYASTUDIO_PARALLEL_CONFORMANCE", raising=False)
    plenty = 64 * 2**30
    assert gate("laya", 0, 8, free=plenty) and gate("julia", 0, 3, free=plenty)
    assert not gate("decider", 0, 8, free=plenty)
    assert not gate("laya", 12, 8, free=plenty)
    assert not gate("laya", 0, 2, free=plenty)
    assert not gate("julia", 0, 8, free=4 * 2**30)
    monkeypatch.setenv("LAYASTUDIO_PARALLEL_CONFORMANCE", "0")
    assert not gate("laya", 0, 8, free=plenty)
    monkeypatch.setenv("LAYASTUDIO_PARALLEL_CONFORMANCE", "1")
    assert gate("laya", 0, 1) and not gate("decider", 0, 8) and not gate("julia", 4, 8)


def test_the_conformance_gate_says_what_decided_it(monkeypatch):
    """What the build records (steps.conformance.overlap_gate): the decision, the gate that
    made it and what that gate read, memory included (available_memory: page cache free)."""
    from systemone_studio import telemetry

    gate = noulxp_package.conformance_gate
    monkeypatch.delenv("LAYASTUDIO_PARALLEL_CONFORMANCE", raising=False)
    assert noulxp_package.CONFORMANCE_MEMORY == 12 * 2**30
    assert gate("decider", 0, 8) == {"overlap": False, "reason": "kind"}
    assert gate("laya", 5, 8) == {"overlap": False, "reason": "test_rows"}
    assert gate("julia", 0, 2) == {"overlap": False, "reason": "threads", "threads": 2}
    reading = {"free": 13 * 2**30, "source": "cgroup", "max": 51 * 2**30, "page_cache": 9}
    monkeypatch.setattr(telemetry, "available_memory", lambda: dict(reading))
    assert gate("laya", 0, 8) == {
        "overlap": True,
        "reason": "gates",
        "threads": 8,
        "free_bytes": 13 * 2**30,
        "needed_bytes": 12 * 2**30,
        "memory": reading,
    }
    reading["free"] = 12 * 2**30 - 1
    found = gate("julia", 0, 3)
    assert found["overlap"] is False and found["reason"] == "memory"
    assert found["free_bytes"] == 12 * 2**30 - 1
    reading["free"] = None  # unknown (macOS): one after the other
    assert gate("laya", 0, 8)["reason"] == "memory"
    assert gate("laya", 0, 8, free=40 * 2**30)["overlap"] is True  # given: never read
    for flag in ("0", "1"):
        monkeypatch.setenv("LAYASTUDIO_PARALLEL_CONFORMANCE", flag)
        assert gate("laya", 0, 1) == {
            "overlap": flag == "1",
            "reason": f"LAYASTUDIO_PARALLEL_CONFORMANCE={flag}",
        }


def test_progress_waits_for_the_conformance_phase():
    events = []

    def record(type_, /, **data):
        events.append((type_, data))

    emit = noulxp_package.Locked(record)
    emit("log", message="a", kind="not the event's type")
    emit.progress(done=1, total=2)
    emit.hold()
    emit.progress(done=10, total=52)
    emit("phase", phase="export")
    emit.release()
    emit.progress(done=20, total=52)
    emit.hold()
    emit.progress(done=5, total=52)
    emit.release(send=False)
    assert [(k, d.get("done") or d.get("phase") or d.get("message")) for k, d in events] == [
        ("log", "a"),
        ("progress", 1),
        ("phase", "export"),
        ("progress", 10),
        ("progress", 20),
    ]
    assert events[0][1]["kind"] == "not the event's type"
