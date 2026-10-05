import json
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from common import QUESTIONS, make_rows

try:  # the tiny Laya checkpoint is built with MLX, which only Apple silicon has
    from test_engine import checkpoint  # noqa: F401 - pytest fixture

    HAS_MLX = True
except (ImportError, pytest.skip.Exception):
    HAS_MLX = False
needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="the Laya fixture needs MLX")

from layastudio import engine, server  # noqa: E402
from layastudio.bootstrap import Bootstrap  # noqa: E402


@pytest.fixture
def studio(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    # Run setup inline, without touching the network: no download, no example datasets.
    setup = Bootstrap(tmp_path / "ws", download=False, fetch_examples=False)
    try:
        setup.run()
    except SystemExit:  # no training stack here (neither MLX nor PyTorch): setup stops, as it may
        pass
    server.Handler.studio = server.Studio(tmp_path / "ws", setup)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", server.Handler.studio
    server.Handler.studio.jobs.shutdown()
    srv.shutdown()


def call(base, path, body=None, method=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")


def test_page_and_state(studio):
    base, _ = studio
    with urllib.request.urlopen(base + "/") as response:
        page = response.read().decode()
        assert "System One Studio" in page
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    # Links out are fine; loading anything from outside this machine is not.
    loads = re.findall(r'(?:src|srcset)=["\']([^"\']+)|<link[^>]+href=["\']([^"\']+)', page)
    external = [u for pair in loads for u in pair if u.startswith(("http://", "https://", "//"))]
    assert external == [], external
    assert "@import" not in page and "fonts.googleapis" not in page
    status, state = call(base, "/api/state")
    assert status == 200 and state["datasets"] == []
    # The base models: Laya's own, and every other catalogue model the studio trains.
    from layastudio import families

    others = [m for m in families.trainable() if m["repo"] not in engine.BASE_MODELS]
    bases = [m for m in state["models"] if not m.get("demo")]
    assert len(bases) == len(engine.BASE_MODELS) + len(others)
    assert {m["kind"] for m in bases} == {"laya", "julia", "decider"}
    assert set(state["hyperparameters_by_kind"]) == {"laya", "julia", "decider"}
    assert state["system"]["setup"]["state"] in ("ready", "failed")
    assert state["system"]["chip"]


def test_rejects_cross_site_and_foreign_hosts(studio):
    base, _ = studio
    status, _ = call(base, "/api/state", headers={"Host": "evil.example"})
    assert status == 403
    status, _ = call(
        base,
        "/api/jobs",
        {"kind": "example", "name": "emotion"},
        headers={"Origin": "https://evil.example"},
    )
    assert status == 403
    request = urllib.request.Request(base + "/api/jobs", data=b"{}", method="POST")
    request.add_header("Content-Type", "text/plain")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 415


def test_jobs_list_and_unknown_job(studio):
    base, _ = studio
    status, body = call(base, "/api/jobs")
    assert status == 200 and body == {"jobs": [], "count": 0}
    status, body = call(base, "/api/jobs/no-such-job")
    assert status == 404 and "no-such-job" in body["error"]
    _, state = call(base, "/api/state")
    assert state["job_count"] == 0


@needs_mlx
def test_dataset_train_job_and_results(studio, checkpoint):  # noqa: F811
    base, _ = studio
    rows = "\n".join(json.dumps(r) for r in make_rows(45))
    status, meta = call(
        base,
        "/api/datasets",
        {"name": "tiny", "questions": QUESTIONS, "train": {"name": "t.jsonl", "text": rows}},
    )
    assert status == 201, meta
    status, report = call(
        base, f"/api/datasets/{meta['id']}/analyze", {"model": f"path:{checkpoint}"}
    )
    assert status == 200 and report["model"] == f"path:{checkpoint}"
    status, job = call(
        base,
        "/api/jobs",
        {
            "kind": "train",
            "dataset": meta["id"],
            "base_model": f"path:{checkpoint}",
            "name": "tiny run",
            "hyperparameters": {"epochs": 1, "batch_size": 4},
        },
    )
    assert status == 201, job
    status, busy = call(
        base,
        "/api/jobs",
        {"kind": "evaluate", "dataset": meta["id"], "model": f"path:{checkpoint}"},
    )
    assert status == 409
    deadline = time.time() + 120
    while time.time() < deadline:
        _, info = call(base, f"/api/jobs/{job['id']}")
        if info["job"]["state"] != "running":
            break
        time.sleep(0.5)
    assert info["job"]["state"] == "done", info["job"]
    _, jobs = call(base, "/api/jobs")
    assert jobs["count"] == 1 and jobs["jobs"][0]["id"] == job["id"]
    _, run = call(base, f"/api/runs/{job['id']}")
    assert run["comparison"]["finetuned"]["overall"]["n"] > 0
    assert "records" not in run["eval"]
    status, errors = call(base, f"/api/runs/{job['id']}/errors")
    assert status == 200 and "errors" in errors
    status, out = call(
        base,
        "/api/predict",
        {
            "models": [f"run:{job['id']}", f"path:{checkpoint}"],
            "questions": QUESTIONS,
            "state": "red",
        },
    )
    assert status == 200 and len(out["results"]) == 2
    _, state = call(base, "/api/state")
    assert state["finetuned"][0]["ref"] == f"run:{job['id']}"
    status, library = call(base, "/api/models")
    assert status == 200 and library["finetuned"][0]["ref"] == f"run:{job['id']}"
    assert library["finetuned"][0]["size_bytes"] > 0 and library["exports"] == []
    assert len(library["base"]) == len(state["models"])
    status, _ = call(base, f"/api/runs/{job['id']}", {}, method="DELETE")
    assert status == 200


@pytest.mark.parametrize("kind", ["julia", "decider"])
def test_other_kinds_train_through_the_api(studio, tmp_path, kind):
    """A Julia 1 or Decider base goes through the same job as Laya: baseline, training on this
    machine's backend, evaluation, comparison; its fine-tune gets its kind's exports."""
    pytest.importorskip("torch")
    if kind == "decider":
        import importlib.util

        from layastudio import runtime

        needed = "mlx_lm" if runtime.backend() == "mlx" else "peft"
        if importlib.util.find_spec(needed) is None:
            pytest.skip(f"{needed} is not installed")
    import tiny

    build = tiny.julia_checkpoint if kind == "julia" else tiny.decider_checkpoint
    model_dir = build(tmp_path / kind)
    base, _ = studio
    rows = "\n".join(json.dumps(r) for r in make_rows(45))
    status, meta = call(
        base,
        "/api/datasets",
        {"name": "tiny", "questions": QUESTIONS, "train": {"name": "t.jsonl", "text": rows}},
    )
    assert status == 201, meta
    status, report = call(
        base, f"/api/datasets/{meta['id']}/analyze", {"model": f"path:{model_dir}"}
    )
    assert status == 200 and report["kind"] == kind
    hp = {"epochs": 1, "batch_size": 4}
    if kind == "decider":
        hp.update(precision="float32", grad_checkpoint="off", head_lr=1.0)  # head_lr: not Decider's
    status, job = call(
        base,
        "/api/jobs",
        {
            "kind": "train",
            "dataset": meta["id"],
            "base_model": f"path:{model_dir}",
            "hyperparameters": hp,
        },
    )
    assert status == 201, job
    deadline = time.time() + 180
    while time.time() < deadline:
        _, info = call(base, f"/api/jobs/{job['id']}")
        if info["job"]["state"] != "running":
            break
        time.sleep(0.5)
    assert info["job"]["state"] == "done", info["job"]
    _, run = call(base, f"/api/runs/{job['id']}")
    assert run["run"]["kind"] == kind and "head_lr" not in (
        run["run"]["hyperparameters"] if kind == "decider" else {}
    )
    assert run["comparison"]["finetuned"]["overall"]["n"] > 0
    assert run["training"]["kind"] == kind
    status, out = call(
        base,
        "/api/predict",
        {
            "models": [f"run:{job['id']}", f"path:{model_dir}"],
            "questions": QUESTIONS,
            "state": "red",
        },
    )
    assert status == 200 and len(out["results"]) == 2
    _, library = call(base, "/api/models")
    tuned = library["finetuned"][0]
    assert tuned["kind"] == kind
    assert tuned["targets"] == (["noulxp"] if kind == "julia" else ["gguf", "mlx", "noulxp"])
