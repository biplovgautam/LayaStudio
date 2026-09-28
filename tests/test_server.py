import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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


def test_jobs_list_and_unknown_job(studio):
    base, _ = studio
    status, body = call(base, "/api/jobs")
    assert status == 200 and body == {"jobs": [], "count": 0}
    status, body = call(base, "/api/jobs/no-such-job")
    assert status == 404 and "no-such-job" in body["error"]
    _, state = call(base, "/api/state")
    assert state["job_count"] == 0


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


class FakeEngine(BaseHTTPRequestHandler):
    """Just enough of the System One Engine's local HTTP contract to exercise the proxy."""

    seen = []

    def log_message(self, *args):
        pass

    def reply(self, status, body=None):
        data = b"" if body is None else json.dumps(body).encode()
        self.send_response(status)
        if body is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        FakeEngine.seen.append(("GET", self.path, None))
        if self.path == "/healthz":
            return self.reply(200, {"ok": True, "ready": 1})
        self.reply(404, {"detail": "Not Found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeEngine.seen.append(("POST", self.path, body))
        if self.path == "/v1/models" and body["model"] == "ns/name":
            return self.reply(202, {"model": "ns/name", "checkpoints": ["main"]})
        if self.path == "/v1/models":
            return self.reply(404, {"detail": f"No model {body['model']}"})
        self.reply(409, {"detail": "starting"})

    def do_DELETE(self):
        FakeEngine.seen.append(("DELETE", self.path, None))
        self.reply(204)


def test_engine_proxy(studio):
    base, running = studio
    with socket.socket() as probe:  # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    running.engine = server.EngineProxy(f"http://127.0.0.1:{free}")
    status, body = call(base, "/api/engine/healthz")
    assert status == 503 and "systemone run engine" in body["error"]
    assert body["engine_url"] == f"http://127.0.0.1:{free}"

    running.engine = server.EngineProxy(base)  # the studio itself: forwarding would loop
    status, body = call(base, "/api/engine/v1/models")
    assert status == 503 and "this studio" in body["error"]

    fake = ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    try:
        running.engine = server.EngineProxy(f"http://127.0.0.1:{fake.server_address[1]}/")
        assert call(base, "/api/engine/healthz") == (200, {"ok": True, "ready": 1})
        status, body = call(base, "/api/engine/v1/models", {"model": "ns/name"})
        assert (status, body) == (202, {"model": "ns/name", "checkpoints": ["main"]})
        assert ("POST", "/v1/models", {"model": "ns/name"}) in FakeEngine.seen
        status, body = call(base, "/api/engine/v1/models", {"model": "no/such"})
        assert status == 404 and body["detail"] == "No model no/such"
        question = {"q": {"type": "noul", "instructions": "Yes?"}}
        status, body = call(
            base,
            "/api/engine/v1/systemone",
            {"model": "ns/name", "state": "hi", "questions": question},
        )
        assert status == 409 and body["detail"] == "starting"
        status, _ = call(base, "/api/engine/v1/models/ns/name", method="DELETE")
        assert status == 204 and ("DELETE", "/v1/models/ns/name", None) in FakeEngine.seen
        request = urllib.request.Request(base + "/api/engine/v1/models", b"x", method="POST")
        request.add_header("Content-Type", "text/plain")  # JSON bodies only
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 415
    finally:
        fake.shutdown()
