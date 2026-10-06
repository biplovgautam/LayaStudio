"""A GGUF export's measurement: its float32 reference in a process of its own, the weights'
digests read while the converter runs, the readout's threads, and a sampled measurement.

The checkpoint is tests/tiny.py's random 2-layer Decider; the converter and llama.cpp are
stand-ins (a file with GGUF's magic, letter logits computed from the ids) unless the real
tooling is there (test_decider.py's needs_gguf). Nothing real runs.
"""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import tiny  # noqa: E402
from common import QUESTIONS, make_rows  # noqa: E402

from layastudio import engine, gguf, noulxp_package  # noqa: E402

RUN = "dv"
REPO = Path(__file__).resolve().parents[1]


def fake_logit(ids, i):
    return ((sum(ids) * 31 + i * 17) % 101) / 10.0


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tiny.decider_checkpoint(tmp_path_factory.mktemp("decider") / "decider")


def run_workspace(root, checkpoint, n=120):
    import shutil

    workspace = root / "ws"
    text = "\n".join(json.dumps(r) for r in make_rows(n))
    meta = engine.create_dataset("tiny", QUESTIONS, text, "t.jsonl", workspace=workspace)
    run_dir = workspace / "runs" / RUN
    shutil.copytree(checkpoint, run_dir / "model")
    engine.write_json(run_dir / "model/questions.json", QUESTIONS)
    engine.write_json(
        run_dir / "run.json",
        {"id": RUN, "dataset": meta["id"], "base_model": "hub:Mapika/decider-2b"},
    )
    return workspace, run_dir


class FakeReader:
    """llama.cpp's readout, stood in: logits from the ids, its thread count kept."""

    made = []
    delay = 0.0

    def __init__(self, path, threads=None):
        assert Path(path).read_bytes()[:4] == b"GGUF"
        self.threads = threads
        self.seen = []
        FakeReader.made.append(self)

    def logits(self, ids, letters):
        time.sleep(self.delay)
        self.seen.append(self.threads)
        return [fake_logit(ids, i) for i in range(len(letters))]

    def set_threads(self, threads):
        self.threads = threads

    def close(self):
        pass


CHILD = textwrap.dedent(
    """
    import json, os, sys, time
    sys.path.insert(0, {repo!r})
    from layastudio.gguf_reference import watch_parent
    watch_parent()
    mode = os.environ.get("FAKE_REFERENCE", "ok")
    model_dir, rows_path, out = sys.argv[1:4]
    rows = json.loads(open(rows_path).read())
    print("reference started", flush=True)
    if mode == "sleep":
        time.sleep(600)
    if mode == "slow":
        time.sleep(1.0)
    if mode == "fail":
        print("CUDA error: out of memory", flush=True)
        sys.exit(3)
    if mode == "noisy":
        sys.stdout.write("x" * (2 << 20) + "\\n")
        sys.stdout.flush()
    def fake(ids, i):
        return ((sum(ids) * 31 + i * 17) % 101) / 10.0
    logits = [[fake(ids, i) for i in range(n)] for ids, n in rows]
    if mode == "short":
        logits = logits[:-1]
    open(out, "w").write(json.dumps({{"device": "cpu", "logits": logits}}))
    """
)


@pytest.fixture
def stand_ins(tmp_path, monkeypatch):
    """A converter that writes a GGUF stand-in, FakeReader, and a stand-in reference process
    whose behaviour FAKE_REFERENCE picks. Records the order things happened in."""
    order = []
    script = tmp_path / "fake_reference.py"
    script.write_text(CHILD.format(repo=str(REPO)))
    monkeypatch.setattr(gguf, "REFERENCE", (str(script),))
    started = []
    real = gguf.Reference

    class Recorded(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            started.append(self)
            order.append("reference")

    monkeypatch.setattr(gguf, "Reference", Recorded)

    def convert(model_dir, out_file, precision="bf16", workspace=None, emit=None):
        order.append("convert")
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(b"GGUF" + bytes(64))
        return out_file

    monkeypatch.setattr(gguf, "convert", convert)
    FakeReader.made, FakeReader.delay = [], 0.0
    monkeypatch.setattr(gguf, "Reader", FakeReader)
    monkeypatch.setenv("LAYASTUDIO_DEVICE", "cpu")
    return {"order": order, "started": started, "convert": convert}


def gone(process, within=5.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.05)
    return False


# ----------------------------------------------------------------------------- the overlap


def test_the_reference_starts_before_the_converter(tmp_path, checkpoint, stand_ins, monkeypatch):
    workspace, run_dir = run_workspace(tmp_path, checkpoint)
    monkeypatch.setenv("LAYASTUDIO_PARALLEL_VERIFY", "1")
    events = []
    report = gguf.export(f"run:{RUN}", workspace, lambda k, **d: events.append((k, d)), threads=6)
    assert stand_ins["order"] == ["reference", "convert"]
    verification = report["verification"]
    # The stand-in reference computes what the stand-in readout does: every row agrees.
    assert verification["same_answer"] == verification["rows"] > 0
    assert verification["max_probability_difference"] == 0
    assert verification["reference"].endswith("PyTorch on cpu")
    assert report["timings"]["overlap"] is True and report["timings"]["threads"] == 6
    assert not list(run_dir.glob(".gguf-verify-*"))  # the scratch folder is gone
    assert all(p.poll() is not None for p in (r.process for r in stand_ins["started"]))
    written = json.loads((run_dir / "exports/gguf-bf16.json").read_text())
    assert written["sha256"] == report["sha256"] and "timings" in written
    result = next(d for k, d in events if k == "result")
    assert "timings" not in result and "source_sha256" not in result


@pytest.mark.skipif(os.name == "nt", reason="the sweep keeps every folder on Windows")
def test_an_export_removes_what_a_killed_one_left(tmp_path, checkpoint, stand_ins, monkeypatch):
    """A job killed outright leaves its .gguf-verify-<pid>-* folder; the run's next export
    removes it once that pid is gone, and leaves one whose job still runs."""
    workspace, run_dir = run_workspace(tmp_path, checkpoint)
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    stale = run_dir / f"{gguf.SCRATCH}{gone.pid}-x1"
    running = run_dir / f"{gguf.SCRATCH}{os.getppid()}-x2"
    for folder in (stale, running):
        folder.mkdir()
        (folder / "rows.json").write_text("[]")
    monkeypatch.setenv("LAYASTUDIO_PARALLEL_VERIFY", "1")
    report = gguf.export(f"run:{RUN}", workspace, threads=2)
    assert report["timings"]["overlap"] and report["verification"]["rows"] > 0
    assert not stale.exists() and running.is_dir()
    assert [p.name for p in run_dir.glob(f"{gguf.SCRATCH}*")] == [running.name]


def test_serial_and_overlapped_measurements_agree(tmp_path, checkpoint, stand_ins, monkeypatch):
    """The real reference (decider.Agent in float32, on the CPU here) in this process and in a
    process of its own: the same rows, the same numbers."""
    monkeypatch.setattr(gguf, "REFERENCE", ("-m", "layastudio.gguf_reference"))
    workspace, run_dir = run_workspace(tmp_path, checkpoint)
    monkeypatch.setenv("LAYASTUDIO_SERIAL_VERIFY", "1")
    serial = gguf.export(f"run:{RUN}", workspace, threads=4)
    monkeypatch.delenv("LAYASTUDIO_SERIAL_VERIFY")
    monkeypatch.setenv("LAYASTUDIO_PARALLEL_VERIFY", "1")
    overlapped = gguf.export(f"run:{RUN}", workspace, threads=4)
    assert serial["timings"]["overlap"] is False and overlapped["timings"]["overlap"] is True
    a, b = serial["verification"], overlapped["verification"]
    assert {k: a[k] for k in ("rows", "same_answer", "reference")} == {
        k: b[k] for k in ("rows", "same_answer", "reference")
    }
    assert abs(a["max_probability_difference"] - b["max_probability_difference"]) <= 1e-6
    keys = {"rows", "same_answer", "max_probability_difference", "reference", "ms_per_row_cpu"}
    assert set(a) == set(b) == keys | {"threads"}
    assert FakeReader.made[0].seen == [4] * a["rows"]  # the serial readout at N threads


def _importable(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


TOKENIZER = Path(
    os.environ.get(
        "DECIDER_TOKENIZER",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Mapika--decider-2b/snapshots/533964dae8be954c5b5e19fa4948e48408094c1e",
    )
)
GGUF_READY = (
    _importable("llama_cpp")
    and (TOKENIZER / "tokenizer.json").is_file()
    and bool(os.environ.get("LAYASTUDIO_TOOLS"))
)


@pytest.mark.skipif(not GGUF_READY, reason="needs llama-cpp-python, Decider's tokenizer, tools")
def test_the_real_overlap_measures_what_the_serial_one_does(tmp_path, monkeypatch):
    """The real converter, llama.cpp and float32 reference on the tiny checkpoint with Decider's
    real tokenizer: overlapped (forced, on the CPU) and serial agree."""
    ck = tiny.decider_checkpoint(tmp_path / "decider", tokenizer_dir=TOKENIZER)
    workspace, run_dir = run_workspace(tmp_path, ck, n=60)
    monkeypatch.setenv("LAYASTUDIO_DEVICE", "cpu")
    monkeypatch.setenv("LAYASTUDIO_SERIAL_VERIFY", "1")
    serial = gguf.export(f"run:{RUN}", workspace, threads=4)
    monkeypatch.delenv("LAYASTUDIO_SERIAL_VERIFY")
    monkeypatch.setenv("LAYASTUDIO_PARALLEL_VERIFY", "1")
    overlapped = gguf.export(f"run:{RUN}", workspace, threads=4)
    a, b = serial["verification"], overlapped["verification"]
    assert a["rows"] == b["rows"] > 0 and a["same_answer"] == b["same_answer"]
    assert abs(a["max_probability_difference"] - b["max_probability_difference"]) <= 1e-6
    assert serial["sha256"] == overlapped["sha256"]  # the same converter, the same file
    assert serial["source_sha256"] == overlapped["source_sha256"]
    assert overlapped["timings"]["overlap"] and not list(run_dir.glob(".gguf-verify-*"))


@pytest.mark.parametrize("raised", [RuntimeError("convert_hf_to_gguf.py failed"), "cancel"])
def test_a_converter_failure_ends_the_reference(
    tmp_path, checkpoint, stand_ins, monkeypatch, raised
):
    workspace, run_dir = run_workspace(tmp_path, checkpoint)
    monkeypatch.setenv("LAYASTUDIO_PARALLEL_VERIFY", "1")
    monkeypatch.setenv("FAKE_REFERENCE", "sleep")
    error = engine.Cancelled() if raised == "cancel" else raised

    def convert(*args, **kwargs):
        stand_ins["order"].append("convert")
        assert stand_ins["started"][0].alive()
        raise error

    monkeypatch.setattr(gguf, "convert", convert)
    started = time.monotonic()
    with pytest.raises(type(error)):
        gguf.export(f"run:{RUN}", workspace, threads=4)
    assert time.monotonic() - started < 20
    [reference] = stand_ins["started"]
    assert reference.process.poll() is not None  # ended and reaped
    assert not list(run_dir.glob(".gguf-verify-*"))
    assert not (run_dir / "exports/gguf-bf16.json").exists()


@pytest.mark.parametrize("mode,message", [("fail", "exit 3"), ("short", "rows of logits for")])
def test_a_reference_that_fails_fails_the_export(
    tmp_path, checkpoint, stand_ins, monkeypatch, mode, message
):
    workspace, run_dir = run_workspace(tmp_path, checkpoint)
    monkeypatch.setenv("LAYASTUDIO_PARALLEL_VERIFY", "1")
    monkeypatch.setenv("FAKE_REFERENCE", mode)
    events = []
    with pytest.raises(RuntimeError, match=message):
        gguf.export(f"run:{RUN}", workspace, lambda k, **d: events.append((k, d)), threads=4)
    assert not (run_dir / "exports/gguf-bf16.json").exists()  # measured again next time
    assert not list(run_dir.glob(".gguf-verify-*"))
    logged = [d.get("message") for k, d in events if k == "log"]
    assert "reference started" in logged  # its log's last lines come first
    if mode == "fail":
        assert "CUDA error: out of memory" in logged


def test_a_reference_that_prints_a_lot_does_not_hang(tmp_path, checkpoint, stand_ins, monkeypatch):
    workspace, _ = run_workspace(tmp_path, checkpoint)
    monkeypatch.setenv("LAYASTUDIO_PARALLEL_VERIFY", "1")
    monkeypatch.setenv("FAKE_REFERENCE", "noisy")
    report = gguf.export(f"run:{RUN}", workspace, threads=4)
    assert report["verification"]["same_answer"] == report["verification"]["rows"]


def test_the_readout_leaves_the_reference_two_threads_until_it_ends(
    tmp_path, checkpoint, stand_ins, monkeypatch
):
    workspace, _ = run_workspace(tmp_path, checkpoint)
    monkeypatch.setenv("LAYASTUDIO_PARALLEL_VERIFY", "1")
    monkeypatch.setenv("FAKE_REFERENCE", "slow")
    FakeReader.delay = 0.1
    report = gguf.export(f"run:{RUN}", workspace, threads=6)
    seen = FakeReader.made[0].seen
    assert seen[0] == 4 and seen[-1] == 6 and seen == sorted(seen)
    threads = report["timings"]["reader_threads"]
    assert threads["start"] == 4 and threads["raised_to"] == 6 and 0 < threads["at_row"]
    assert report["verification"]["threads"] == 4


def test_a_reference_outlives_no_job(tmp_path, checkpoint):
    """The job killed outright (SIGKILL: no handler runs): the reference sees its stdin close
    (and, on Linux, PR_SET_PDEATHSIG) and is gone within 5 s."""
    script = tmp_path / "fake_reference.py"
    script.write_text(CHILD.format(repo=str(REPO)))
    folder = tmp_path / "scratch"
    folder.mkdir()
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import sys, time
                sys.path.insert(0, {str(REPO)!r})
                from layastudio import gguf
                gguf.REFERENCE = ({str(script)!r},)
                ref = gguf.Reference({str(checkpoint)!r}, [{{"ids": [1, 2], "n": 2}}], {str(folder)!r})
                print(ref.process.pid, flush=True)
                time.sleep(600)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "FAKE_REFERENCE": "sleep"},
    )
    child = int(parent.stdout.readline())
    os.kill(parent.pid, signal.SIGKILL)
    parent.wait()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child, signal.SIGKILL)
        pytest.fail("the reference process outlived its job")


# ----------------------------------------------------------------------------- the weights' digests


def test_the_weights_are_hashed_while_the_converter_runs(tmp_path, checkpoint, stand_ins):
    workspace, run_dir = run_workspace(tmp_path, checkpoint)
    report = gguf.export(f"run:{RUN}", workspace, threads=2)
    model = run_dir / "model"
    assert report["source_sha256"] == {
        p.name: gguf._sha256(p) for p in sorted(model.glob("*.safetensors"))
    }
    assert report["timings"]["source_sha256_s"] is not None
    given = {"model.safetensors": "f" * 64}
    assert gguf.export(f"run:{RUN}", workspace, source_sha256=given)["source_sha256"] == given


def test_the_hasher_stops_and_raises(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(1 << 20) * 16)
    hasher = gguf.Hasher([big], chunk=1 << 10)
    hasher.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        hasher.result()
    with pytest.raises(FileNotFoundError):
        gguf.Hasher([tmp_path / "missing"]).result()
    import hashlib

    assert gguf.Hasher([big]).result() == {"big.bin": hashlib.sha256(big.read_bytes()).hexdigest()}


# ----------------------------------------------------------------------------- fewer rows


def test_spread_takes_every_type_at_an_even_stride():
    items = [{"type": "score", "key": (i, "q", 0)} for i in range(90)]
    items += [{"type": "choice", "key": (90 + i, "c", None)} for i in range(10)]
    picked = gguf.spread(items, 20)
    assert len(picked) == 20 and picked == gguf.spread(items, 20)
    kinds = [p["type"] for p in picked]
    assert kinds.count("choice") == 2 and kinds.count("score") == 18
    keys = [p["key"][0] for p in picked]
    assert keys == sorted(keys) and keys[0] < 10 and keys[-1] > 90
    assert gguf.spread(items[:5], 20) == items[:5] and gguf.spread(items, 0) == []
    three = [{"type": t} for t in ("a", "b", "c") for _ in range(5)]
    assert len(gguf.spread(three, 2)) == 2


def test_a_sampled_measurement_says_so(tmp_path, checkpoint, monkeypatch):
    workspace, _ = run_workspace(tmp_path, checkpoint, n=300)
    items, sample = gguf.verify_items(f"run:{RUN}", workspace, 20)
    assert 0 < len(items) <= 20 and sample["of_test_rows"] > 20
    assert 1 < sample["test_rows"] <= len(items)
    assert {it["type"] for it in items} == {"choice", "score", "noul"}
    every, _, _ = gguf.test_rows(f"run:{RUN}", workspace, limit=None)
    assert items[-1]["key"][0] > every[len(every) // 2]["key"][0]  # spans the split
    assert gguf.verify_items(f"run:{RUN}", workspace, 20)[0] == items
    full, none = gguf.verify_items(f"run:{RUN}", workspace, 60)
    assert none is None and full == gguf.test_rows(f"run:{RUN}", workspace, limit=60)[0]
    monkeypatch.setattr(gguf, "VERIFY_ROWS", 20)
    assert gguf.verify_rows("bf16") == 20
    assert gguf.verify_rows("q8_0") == gguf.verify_rows("f16") == 60


def test_the_measurement_never_decides_a_package(tmp_path):
    """passing_package and the card read the GGUF report's digests, never its verification."""
    run_dir = tmp_path / "run"
    package, model = run_dir / noulxp_package.PACKAGE, run_dir / "model"
    package.mkdir(parents=True)
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (package / "model.gguf").write_bytes(b"GGUF weights")
    (package / "conformance.jsonl").write_bytes(b"{}\n")
    sha = noulxp_package._sha256
    manifest = {
        "standard": "noulxp/0.1",
        "profile": "causal-letters",
        "weights": {"path": "model.gguf", "sha256": sha(package / "model.gguf"), "format": "gguf"},
        "conformance": {"path": "conformance.jsonl", "sha256": sha(package / "conformance.jsonl")},
        "source": {"gguf": {"precision": "bf16", "llama_cpp": gguf.LLAMA_CPP_COMMIT}},
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
    exports = run_dir / "exports"
    exports.mkdir()
    for verification in ({"rows": 60, "same_answer": 60}, {"rows": 20, "same_answer": 0}):
        report = {
            "sha256": sha(package / "model.gguf"),
            "source_sha256": {"model.safetensors": sha(model / "model.safetensors")},
            "verification": verification,
        }
        (exports / "gguf-bf16.json").write_text(json.dumps(report))
        info = noulxp_package.passing_package(run_dir, model)
        assert info and "same_answer" not in noulxp_package.card_line(info)
