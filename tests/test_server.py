import json
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from test_engine import QUESTIONS, checkpoint, make_rows  # noqa: F401 - pytest fixture

from layastudio import engine, server
from layastudio.bootstrap import Bootstrap


@pytest.fixture
def studio(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    # Run setup inline, without touching the network: no download, no example datasets.
    setup = Bootstrap(tmp_path / "ws", download=False, fetch_examples=False)
    setup.run()
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
        assert "LayaStudio" in page
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    # Links out are fine; loading anything from outside this machine is not.
    loads = re.findall(r'(?:src|srcset)=["\']([^"\']+)|<link[^>]+href=["\']([^"\']+)', page)
    external = [u for pair in loads for u in pair if u.startswith(("http://", "https://", "//"))]
    assert external == [], external
    assert "@import" not in page and "fonts.googleapis" not in page
    status, state = call(base, "/api/state")
    assert status == 200 and state["datasets"] == []
    assert len([m for m in state["models"] if not m.get("demo")]) == len(engine.BASE_MODELS)
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
    status, _ = call(base, f"/api/runs/{job['id']}", {}, method="DELETE")
    assert status == 200
