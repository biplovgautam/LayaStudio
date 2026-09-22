#!/usr/bin/env python3
"""LayaStudio: fine-tune Laya typed-decision models on your own data, on your own Mac.

    layastudio                    # or: uv run layastudio

One command. The server answers immediately and finishes setting itself up in the
background: it detects this Mac, checks the MLX runtime, downloads a base checkpoint and
fetches the public example datasets, reporting every step on the page.

Frontend (HTML, CSS, JavaScript) and backend (JSON API) live in this one file and use only
the Python standard library, so there is nothing to build. Training, evaluation and
downloads run as child processes of layastudio.engine; this server schedules them, streams
their progress and serves the results.

Privacy: the server binds to 127.0.0.1, the page loads no external scripts, fonts or
analytics, and training/evaluation jobs run with HF_HUB_OFFLINE=1. Datasets, runs and
checkpoints stay in the workspace folder, which git ignores.
"""

import argparse
import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import engine
from .bootstrap import DEFAULT_MODEL, Bootstrap
from .examples import catalog

PACKAGE = Path(__file__).resolve().parent

MAX_BODY = 512 * 2**20
TERMINAL = {"done": "done", "error": "failed", "cancelled": "cancelled"}


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def shown_path(path, root=None):
    """Short, home-free paths: readable in the UI and safe in screenshots."""
    path = Path(path).resolve()
    for base in (root, Path.cwd()):
        if base is None:
            continue
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path).replace(str(Path.home()), "~", 1)


def strip_records(report):
    return {k: v for k, v in report.items() if k != "records"} if report else None


# ----------------------------------------------------------------------------- jobs


class Jobs:
    """One GPU job at a time, each a child process writing events.jsonl."""

    def __init__(self, workspace, before_start):
        self.workspace = workspace
        self.root = workspace / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.procs = {}
        self.lock = threading.Lock()
        self.before_start = before_start

    def active(self):
        with self.lock:
            return next((j for j, p in self.procs.items() if p.poll() is None), None)

    def start(self, kind, spec, job_id, title):
        with self.lock:
            running = next((j for j, p in self.procs.items() if p.poll() is None), None)
            if running:
                raise ApiError(HTTPStatus.CONFLICT, f"Job {running} is still running")
            path = self.root / engine.check_id(job_id)
            path.mkdir(parents=True)
            engine.write_json(
                path / "spec.json", {**spec, "kind": kind, "workspace": str(self.workspace)}
            )
            engine.write_json(
                path / "job.json",
                {"id": job_id, "kind": kind, "title": title, "created": engine.now()},
            )
            self.before_start()
            env = {**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1"}
            if kind in ("train", "evaluate"):
                env["HF_HUB_OFFLINE"] = "1"
            log = open(path / "output.log", "w")
            self.procs[job_id] = subprocess.Popen(
                [sys.executable, "-m", "layastudio.engine", "run", str(path)],
                cwd=str(PACKAGE.parent),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return job_id

    def events(self, job_id, since=0):
        path = self.root / engine.check_id(job_id) / "events.jsonl"
        if not path.exists():
            return [], 0
        lines = path.read_text().splitlines()
        events = []
        for line in lines[since:]:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a line still being written
        return events, since + len(events)

    def info(self, job_id):
        path = self.root / engine.check_id(job_id)
        job = engine.read_json(path / "job.json")
        if job is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"No job {job_id}")
        events, _ = self.events(job_id)
        last = next((e for e in reversed(events) if e["type"] in TERMINAL), None)
        proc = self.procs.get(job_id)
        if proc is not None and proc.poll() is None:
            state = "running"
        elif last:
            state = TERMINAL[last["type"]]
        elif proc is not None:
            state = "failed"
        else:
            state = "interrupted"
        job.update(state=state, events=len(events))
        if last and last["type"] == "error":
            job["error"] = last.get("message")
        elif state in ("failed", "interrupted"):
            log = path / "output.log"
            tail = log.read_text()[-2000:] if log.exists() else ""
            job["error"] = tail.strip().splitlines()[-1] if tail.strip() else "Process stopped"
        progress = [e for e in events if e["type"] in ("phase", "step", "progress")]
        job["last"] = progress[-1] if progress else None
        return job

    def list(self, limit=30):
        jobs = sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for path in jobs[:limit]:
            try:
                out.append(self.info(path.name))
            except (ApiError, ValueError, OSError):
                continue
        return out

    def cancel(self, job_id):
        proc = self.procs.get(job_id)
        if proc is None or proc.poll() is not None:
            raise ApiError(HTTPStatus.CONFLICT, "Job is not running")
        proc.terminate()

        def reap():
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                proc.kill()

        threading.Thread(target=reap, daemon=True).start()

    def shutdown(self):
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.terminate()


# ----------------------------------------------------------------------------- playground


class Playground:
    """Warm models for interactive predictions, all on one thread (MLX is used serially)."""

    def __init__(self, workspace, capacity=2):
        self.workspace = workspace
        self.capacity = capacity
        self.models = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self.pool.submit(engine.limit_mlx_cache)

    def _agent(self, ref):
        import laya_mlx

        if ref not in self.models:
            while len(self.models) >= self.capacity:
                self.models.pop(next(iter(self.models)))
            path = engine.resolve_model_ref(ref, self.workspace)
            self.models[ref] = laya_mlx.load(str(path))
        self.models[ref] = self.models.pop(ref)  # most recently used last
        return self.models[ref]

    def predict(self, refs, state, questions):
        def run():
            results = {}
            for ref in refs:
                agent = self._agent(ref)
                started = time.perf_counter()
                out = agent.predict(state, questions)
                results[ref] = {
                    "answers": out["answers"],
                    "usage": out["usage"],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            return results

        return self.pool.submit(run).result()

    def unload(self):
        def run():
            import mlx.core as mx

            self.models.clear()
            gc.collect()
            mx.clear_cache()

        self.pool.submit(run).result()


# ----------------------------------------------------------------------------- state


class Studio:
    def __init__(self, workspace, bootstrap=None):
        self.workspace = workspace
        for name in ("datasets", "runs", "evals", "jobs"):
            (workspace / name).mkdir(parents=True, exist_ok=True)
        self.playground = Playground(workspace)
        self.jobs = Jobs(workspace, self.playground.unload)
        self.bootstrap = bootstrap or Bootstrap(workspace, download=False, fetch_examples=False)

    # --- summaries

    def system(self):
        snapshot = self.bootstrap.snapshot()
        return {
            **snapshot["machine"],
            "workspace": shown_path(self.workspace, self.workspace.parent),
            "setup": {k: snapshot[k] for k in ("state", "ready", "steps", "seconds")},
        }

    def datasets(self):
        out = []
        for path in sorted((self.workspace / "datasets").iterdir(), reverse=True):
            meta = engine.read_json(path / "meta.json")
            if meta:
                questions = engine.read_json(path / "questions.json", {})
                out.append(
                    {
                        k: meta.get(k)
                        for k in ("id", "name", "created", "rows", "decisions", "error_count")
                    }
                    | {"questions": {q: v["type"] for q, v in questions.items()}}
                )
        return sorted(out, key=lambda d: d["created"] or "", reverse=True)

    def runs(self):
        out = []
        for path in (self.workspace / "runs").iterdir():
            run = engine.read_json(path / "run.json")
            if not run:
                continue
            comparison = engine.read_json(path / "comparison.json")
            try:
                job = self.jobs.info(run["id"])
                run["state"], run["error"] = job["state"], job.get("error")
            except (ApiError, ValueError):
                run["state"] = "unknown"
            if comparison:
                run["baseline_accuracy"] = comparison["base"]["overall"]["accuracy"]
                run["accuracy"] = comparison["finetuned"]["overall"]["accuracy"]
                run["p_value"] = comparison["paired"]["overall"]["p_value"]
            run["has_model"] = (path / "model/model.safetensors").exists()
            out.append(run)
        return sorted(out, key=lambda r: r["created"], reverse=True)

    def models(self):
        base = [
            {
                "ref": f"hub:{repo}",
                "repo": repo,
                "description": desc,
                "cached": engine.hub_cached(repo),
            }
            for repo, desc in engine.BASE_MODELS.items()
        ]
        tuned = [
            {
                "ref": f"run:{r['id']}",
                "name": r["name"],
                "base_model": r["base_model"],
                "dataset": r["dataset"],
                "accuracy": r.get("accuracy"),
                "path": shown_path(
                    self.workspace / "runs" / r["id"] / "model", self.workspace.parent
                ),
                "created": r["created"],
            }
            for r in self.runs()
            if r["has_model"] and r["state"] == "done"
        ]
        return base, tuned

    def overview(self):
        base, tuned = self.models()
        return {
            "system": self.system(),
            "active_job": self.jobs.active(),
            "datasets": self.datasets(),
            "runs": self.runs(),
            "models": base,
            "finetuned": tuned,
            "jobs": self.jobs.list(12),
            "examples": [
                {
                    "name": k,
                    "title": v["title"],
                    "description": v["description"],
                    "source": v["source"],
                }
                for k, v in self.examples.items()
            ],
            "hyperparameters": engine.HYPERPARAMETERS,
        }

    # --- datasets

    def dataset(self, dataset_id):
        try:
            questions, rows, meta = engine.load_dataset(dataset_id, self.workspace)
        except FileNotFoundError:
            raise ApiError(HTTPStatus.NOT_FOUND, f"No dataset {dataset_id}") from None
        sample = []
        for row in [r for r in rows if r["split"] == "train"][:30]:
            state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"])
            sample.append(
                {
                    "id": row["id"],
                    "state": state[:500],
                    "labels": {q: label_text(questions[q], t) for q, t in row["targets"].items()},
                }
            )
        evals = []
        for path in (self.workspace / "evals").glob(f"{dataset_id}--*.json"):
            report = engine.read_json(path)
            if report:
                evals.append(
                    {
                        "model": report["model"],
                        "created": report["created"],
                        "accuracy": report["overall"]["accuracy"],
                        "ece": report["overall"]["ece"],
                        "n": report["overall"]["n"],
                        "latency_ms": report["latency_ms"],
                    }
                )
        path = self.workspace / "datasets" / dataset_id
        return {
            "meta": meta,
            "questions": questions,
            "sample": sample,
            "analysis": engine.read_json(path / "analysis.json"),
            "evals": evals,
        }

    def create_dataset(self, body):
        train = body.get("train") or {}
        test = body.get("test") or {}
        if not train.get("text"):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Choose a training file")
        try:
            return engine.create_dataset(
                (body.get("name") or train.get("name") or "dataset").strip(),
                body.get("questions"),
                train["text"],
                train.get("name", "train.jsonl"),
                test.get("text"),
                test.get("name"),
                int(body.get("seed", 13)),
                workspace=self.workspace,
            )
        except (ValueError, json.JSONDecodeError) as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from None

    def analyze(self, dataset_id, ref):
        try:
            model_dir = engine.resolve_model_ref(ref, self.workspace)
        except FileNotFoundError as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from None
        report = engine.analyze_dataset(dataset_id, model_dir, self.workspace)
        report["model"] = ref
        engine.write_json(self.workspace / "datasets" / dataset_id / "analysis.json", report)
        return report

    # --- jobs

    @property
    def examples(self):
        try:
            return catalog()
        except Exception:  # noqa: BLE001 - offline just means no examples to offer
            return {}

    def start(self, body):
        kind = body.get("kind")
        if kind in ("train", "evaluate") and not self.bootstrap.ready:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "Setup is still running. Its progress is on the Datasets page.",
            )
        stamp = time.strftime("%m%d-%H%M%S")
        if kind == "train":
            questions, _, meta = engine.load_dataset(body["dataset"], self.workspace)
            engine.resolve_model_ref(body["base_model"], self.workspace)
            hp = {
                k: v
                for k, v in (body.get("hyperparameters") or {}).items()
                if k in engine.HYPERPARAMETERS
            }
            name = (body.get("name") or f"{meta['name']} · {hp.get('method', 'lora')}").strip()
            run_id = f"{engine.slugify(name, 'run')[:40]}-{stamp}"
            spec = {
                "run_id": run_id,
                "dataset": body["dataset"],
                "base_model": body["base_model"],
                "hyperparameters": hp,
                "baseline": bool(body.get("baseline", True)),
            }
            engine.write_json(
                self.workspace / "runs" / run_id / "run.json",
                {
                    "id": run_id,
                    "name": name,
                    "dataset": body["dataset"],
                    "dataset_name": meta["name"],
                    "base_model": body["base_model"],
                    "hyperparameters": {**engine.HYPERPARAMETERS, **hp},
                    "created": engine.now(),
                    "questions": list(questions),
                },
            )
            try:
                return self.jobs.start("train", spec, run_id, f"Fine-tune: {name}")
            except ApiError:
                shutil.rmtree(self.workspace / "runs" / run_id, ignore_errors=True)
                raise
        if kind == "evaluate":
            engine.load_dataset(body["dataset"], self.workspace)
            engine.resolve_model_ref(body["model"], self.workspace)
            return self.jobs.start(
                "evaluate",
                {"dataset": body["dataset"], "model": body["model"]},
                f"evaluate-{stamp}",
                f"Evaluate {body['model']}",
            )
        if kind == "download":
            repo = body.get("repo_id")
            if repo not in engine.BASE_MODELS:
                raise ApiError(HTTPStatus.BAD_REQUEST, "Unknown base model")
            return self.jobs.start(
                "download", {"repo_id": repo}, f"download-{stamp}", f"Download {repo}"
            )
        if kind == "example":
            if body.get("name") not in self.examples:
                raise ApiError(HTTPStatus.BAD_REQUEST, "Unknown example")
            return self.jobs.start(
                "example",
                {"name": body["name"]},
                f"example-{stamp}",
                f"Fetch example: {self.examples[body['name']]['title']}",
            )
        raise ApiError(HTTPStatus.BAD_REQUEST, f"Unknown job kind {kind!r}")

    # --- runs

    def run(self, run_id):
        path = self.workspace / "runs" / engine.check_id(run_id)
        run = engine.read_json(path / "run.json")
        if not run:
            raise ApiError(HTTPStatus.NOT_FOUND, f"No run {run_id}")
        job = self.jobs.info(run_id)
        base_eval = engine.read_json(
            engine.baseline_path(run["base_model"], run["dataset"], self.workspace)
        )
        return {
            "run": run,
            "job": job,
            "training": engine.read_json(path / "training.json"),
            "comparison": engine.read_json(path / "comparison.json"),
            "eval": strip_records(engine.read_json(path / "eval.json")),
            "base_eval": strip_records(base_eval),
            "model_path": (
                shown_path(path / "model", self.workspace.parent)
                if (path / "model").exists()
                else None
            ),
        }

    def errors(self, run_id, question=None, limit=60):
        path = self.workspace / "runs" / engine.check_id(run_id)
        run = engine.read_json(path / "run.json")
        tuned = engine.read_json(path / "eval.json")
        if not run or not tuned:
            raise ApiError(HTTPStatus.NOT_FOUND, "This run has no evaluation yet")
        base = (
            engine.read_json(
                engine.baseline_path(run["base_model"], run["dataset"], self.workspace)
            )
            or {}
        )
        questions, rows, _ = engine.load_dataset(run["dataset"], self.workspace)
        states = {r["id"]: r["state"] for r in rows}
        base_index = {(r["row"], r["qid"]): r for r in base.get("records", [])}
        out = []
        for rec in tuned["records"]:
            if question and rec["qid"] != question:
                continue
            gold = engine.argmax(rec["gold"])
            if engine.argmax(rec["p"]) == gold:
                continue
            qdef = questions[rec["qid"]]
            names = engine.option_names(qdef)
            state = states.get(rec["row"], "")
            state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            other = base_index.get((rec["row"], rec["qid"]))
            out.append(
                {
                    "row": rec["row"],
                    "question": rec["qid"],
                    "state": state[:800],
                    "gold": names[gold],
                    "predicted": names[engine.argmax(rec["p"])],
                    "confidence": max(rec["p"]),
                    "base_predicted": names[engine.argmax(other["p"])] if other else None,
                    "base_correct": bool(other and engine.argmax(other["p"]) == gold),
                }
            )
        out.sort(key=lambda e: -e["confidence"])
        return {"count": len(out), "errors": out[:limit]}

    def delete(self, kind, item_id):
        engine.check_id(item_id)
        if self.jobs.active() == item_id:
            raise ApiError(HTTPStatus.CONFLICT, "Cancel the running job first")
        if kind == "datasets":
            for path in (self.workspace / "evals").glob(f"{item_id}--*.json"):
                path.unlink()
        path = self.workspace / kind / item_id
        if not path.exists():
            raise ApiError(HTTPStatus.NOT_FOUND, "Not found")
        shutil.rmtree(path)
        if kind == "runs":
            shutil.rmtree(self.workspace / "jobs" / item_id, ignore_errors=True)
        return {"deleted": item_id}

    def predict(self, body):
        if self.jobs.active():
            raise ApiError(
                HTTPStatus.CONFLICT,
                "A job is running. The playground pauses so the "
                "job has the GPU and memory to itself.",
            )
        refs = body.get("models") or []
        if not refs or len(refs) > 4:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Choose between one and four models")
        try:
            questions = engine.validate_questions(body.get("questions"))
        except (ValueError, json.JSONDecodeError) as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from None
        state = body.get("state")
        if state in (None, ""):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Enter a state")
        try:
            return {"results": self.playground.predict(refs, state, questions)}
        except FileNotFoundError as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from None


def label_text(qdef, target):
    names = engine.option_names(qdef)
    if max(target) >= 0.999:
        return names[engine.argmax(target)]
    return ", ".join(f"{n} {p:.0%}" for n, p in zip(names, target) if p > 0)


# ----------------------------------------------------------------------------- http


class Handler(BaseHTTPRequestHandler):
    studio: Studio = None
    port = 8765
    server_version = "LayaStudio/1"

    def log_message(self, fmt, *args):
        if os.environ.get("LAYA_STUDIO_VERBOSE"):
            super().log_message(fmt, *args)

    def _allowed(self):
        """Refuse DNS-rebinding and cross-site requests: this API has no login."""
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
            return False
        return True

    def _send(self, status, body, content_type="application/json"):
        data = (
            body
            if isinstance(body, bytes)
            else json.dumps(engine.finite(body), ensure_ascii=False).encode()
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if content_type.startswith("text/html"):
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; frame-ancestors 'none'",
            )
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            raise ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Send application/json")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Upload is larger than 512 MB")
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid JSON body") from None

    def _dispatch(self, method):
        if not self._allowed():
            return self._send(HTTPStatus.FORBIDDEN, {"error": "Forbidden"})
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        studio = self.studio
        try:
            if method == "GET" and not parts:
                return self._send(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            if not parts or parts[0] != "api":
                raise ApiError(HTTPStatus.NOT_FOUND, "Not found")
            route = parts[1:]
            if method == "GET":
                if route == ["state"]:
                    return self._send(HTTPStatus.OK, studio.overview())
                if len(route) == 2 and route[0] == "datasets":
                    return self._send(HTTPStatus.OK, studio.dataset(engine.check_id(route[1])))
                if len(route) == 2 and route[0] == "jobs":
                    since = int(query.get("since", 0))
                    events, nxt = studio.jobs.events(route[1], since)
                    return self._send(
                        HTTPStatus.OK,
                        {"job": studio.jobs.info(route[1]), "events": events, "next": nxt},
                    )
                if len(route) == 2 and route[0] == "runs":
                    return self._send(HTTPStatus.OK, studio.run(route[1]))
                if len(route) == 3 and route[0] == "runs" and route[2] == "errors":
                    return self._send(
                        HTTPStatus.OK,
                        studio.errors(route[1], query.get("question"), int(query.get("limit", 60))),
                    )
            elif method == "POST":
                body = self._body()
                if route == ["datasets"]:
                    return self._send(HTTPStatus.CREATED, studio.create_dataset(body))
                if len(route) == 3 and route[0] == "datasets" and route[2] == "analyze":
                    return self._send(
                        HTTPStatus.OK, studio.analyze(engine.check_id(route[1]), body.get("model"))
                    )
                if route == ["jobs"]:
                    return self._send(HTTPStatus.CREATED, {"id": studio.start(body)})
                if len(route) == 3 and route[0] == "jobs" and route[2] == "cancel":
                    studio.jobs.cancel(engine.check_id(route[1]))
                    return self._send(HTTPStatus.OK, {"cancelled": route[1]})
                if route == ["predict"]:
                    return self._send(HTTPStatus.OK, studio.predict(body))
            elif method == "DELETE" and len(route) == 2 and route[0] in ("datasets", "runs"):
                self._body()
                return self._send(HTTPStatus.OK, studio.delete(route[0], route[1]))
            raise ApiError(HTTPStatus.NOT_FOUND, "Not found")
        except ApiError as error:
            self._send(error.status, {"error": str(error)})
        except (ValueError, KeyError, FileNotFoundError) as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": f"{type(error).__name__}: {error}"})
        except Exception as error:  # noqa: BLE001 - report instead of dropping the connection
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"}
            )

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")


def stop(*_):
    raise KeyboardInterrupt


def find_port(preferred, host="127.0.0.1", tries=20):
    """Use the next free port if the preferred one is taken, so a second copy still starts."""
    import socket

    for port in range(preferred, preferred + tries):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port between {preferred} and {preferred + tries}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="layastudio", description="Fine-tune Laya on your own data, on your own Mac"
    )
    parser.add_argument("--port", type=int, default=8765, help="Default 8765, or the next free")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=engine.WORKSPACE,
        help=f"Datasets, runs and checkpoints (default: {engine.WORKSPACE})",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base checkpoint to prepare")
    parser.add_argument("--no-download", action="store_true", help="Never download a model")
    parser.add_argument("--no-examples", action="store_true", help="Skip the example datasets")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args(argv)

    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LAYASTUDIO_HOME", str(workspace))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    engine.WORKSPACE = workspace

    port = find_port(args.port)
    url = f"http://127.0.0.1:{port}"
    bootstrap = Bootstrap(
        workspace,
        model=args.model,
        download=not args.no_download,
        fetch_examples=not args.no_examples,
    )
    Handler.studio = Studio(workspace, bootstrap)
    Handler.port = port
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(
        f"LayaStudio  {url}\n"
        f"Workspace   {workspace}\n"
        "Setting up in the background (machine check, model, examples) - the page shows "
        "progress.\nPress Ctrl+C to stop."
    )
    bootstrap.start()
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, [url]).start()
    signal.signal(signal.SIGTERM, stop)  # `kill` stops running jobs too, like Ctrl+C
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.studio.jobs.shutdown()
        server.server_close()


# ----------------------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LayaStudio</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%233b5bdb'/%3E%3Cpath d='M9 22V10h3v9.5h7V22z' fill='white'/%3E%3C/svg%3E">
<style>
:root{--bg:#f6f6f3;--panel:#fff;--ink:#1b1b19;--muted:#686862;--faint:#9a9a93;--line:#e3e3dd;--accent:#3b5bdb;--accent-soft:#e8ecfb;--on-accent:#fff;--good:#2b8a3e;--good-soft:#e6f4ea;--bad:#c92a2a;--bad-soft:#fbeaea;--warn:#a86a00;--warn-soft:#fdf3e1;--code:#f0f0ec;--base:#9a9a93;--ft:#3b5bdb;--shadow:0 1px 2px rgba(0,0,0,.05)}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--panel:#1b1b19;--ink:#ececea;--muted:#a3a39c;--faint:#77776f;--line:#2d2d2a;--accent:#8198f7;--accent-soft:#232a45;--on-accent:#0d1024;--good:#5bd27a;--good-soft:#16301f;--bad:#ff7a7a;--bad-soft:#3a1d1d;--warn:#f2b84b;--warn-soft:#352a14;--code:#232321;--base:#7d7d76;--ft:#8198f7;--shadow:none}}
*{box-sizing:border-box}
[hidden]{display:none!important}
html,body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",system-ui,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
code,pre,.mono{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:12.5px}
pre{background:var(--code);padding:12px;border-radius:8px;overflow:auto;margin:8px 0;max-width:100%}
header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:16px;padding:10px 20px;background:var(--panel);border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:10px;font-weight:650;font-size:15px}
.brand .mark{width:26px;height:26px;border-radius:7px;background:var(--accent);color:var(--on-accent);display:grid;place-items:center;font-weight:750}
.brand small{color:var(--muted);font-weight:450}
.spacer{flex:1}
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;background:var(--code);color:var(--muted);font-size:12px;white-space:nowrap}
.chip.live{background:var(--accent-soft);color:var(--accent);cursor:pointer}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor}
.live .dot{animation:pulse 1.2s infinite}@keyframes pulse{50%{opacity:.3}}
.layout{display:grid;grid-template-columns:190px minmax(0,1fr);min-height:calc(100vh - 51px)}
.mainwrap{min-width:0;display:flex;flex-direction:column}
#setup{padding:20px 28px 0}
.card.setup{border-color:var(--accent);margin-bottom:0}
nav{border-right:1px solid var(--line);padding:16px 10px;display:flex;flex-direction:column;gap:2px}
nav a{display:block;padding:7px 12px;border-radius:7px;color:var(--ink)}
nav a:hover{background:var(--code);text-decoration:none}
nav a.on{background:var(--accent-soft);color:var(--accent);font-weight:600}
nav .sep{height:1px;background:var(--line);margin:10px 4px}
nav .note{color:var(--faint);font-size:12px;padding:4px 12px}
main{padding:24px 28px 60px;max-width:1180px;width:100%;min-width:0}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:15px;margin:0 0 12px}
h3{font-size:13px;margin:16px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.lead{color:var(--muted);margin:0 0 20px;max-width:760px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:var(--shadow);margin-bottom:16px;min-width:0}
.grid{display:grid;gap:16px}.two{grid-template-columns:repeat(2,minmax(0,1fr))}.three{grid-template-columns:repeat(3,minmax(0,1fr))}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat .k{color:var(--muted);font-size:12px}.stat .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}
.stat .d{font-size:12px;margin-top:2px}
.up{color:var(--good)}.down{color:var(--bad)}.muted{color:var(--muted)}.faint{color:var(--faint)}
label{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px;font-weight:550}
input[type=text],input[type=number],select,textarea{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit}
textarea{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;min-height:120px;resize:vertical}
input:focus,select:focus,textarea:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
input[type=file]{font-size:12.5px;color:var(--muted)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 14px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--ink);font:inherit;font-weight:550;cursor:pointer}
.btn:hover{background:var(--code)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.btn.primary:hover{filter:brightness(1.08)}
.btn.danger{color:var(--bad)}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn.small{padding:3px 9px;font-size:12px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{text-align:left;font-size:12px;color:var(--muted);font-weight:550;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
.tablewrap{overflow-x:auto}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600;background:var(--code);color:var(--muted)}
.pill.running{background:var(--accent-soft);color:var(--accent)}.pill.done{background:var(--good-soft);color:var(--good)}
.pill.failed,.pill.interrupted{background:var(--bad-soft);color:var(--bad)}.pill.cancelled{background:var(--warn-soft);color:var(--warn)}
.notice{border-radius:9px;padding:10px 12px;margin:8px 0;font-size:13px}
.notice.warn{background:var(--warn-soft);color:var(--warn)}.notice.bad{background:var(--bad-soft);color:var(--bad)}.notice.good{background:var(--good-soft);color:var(--good)}.notice.info{background:var(--accent-soft);color:var(--accent)}
.bar{height:8px;background:var(--code);border-radius:4px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);border-radius:4px;transition:width .3s}
.hbar{display:grid;grid-template-columns:minmax(80px,180px) 1fr 52px;gap:8px;align-items:center;font-size:12.5px;margin:3px 0}
.hbar .t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hbar .n{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
.steps{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 14px}
.steps span{padding:3px 10px;border-radius:999px;font-size:12px;background:var(--code);color:var(--faint)}
.steps span.done{background:var(--good-soft);color:var(--good)}.steps span.now{background:var(--accent-soft);color:var(--accent);font-weight:600}
details{margin-top:8px}summary{cursor:pointer;color:var(--muted);font-size:13px}
.state{white-space:pre-wrap;word-break:break-word;max-height:7.5em;overflow:hidden}
.opt{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.choice{border:1px solid var(--line);border-radius:9px;padding:10px;cursor:pointer}
.choice.on{border-color:var(--accent);background:var(--accent-soft)}
.choice b{display:block;font-size:13px}.choice span{font-size:12px;color:var(--muted)}
.empty{color:var(--muted);padding:24px;text-align:center;border:1px dashed var(--line);border-radius:10px}
.toast{position:fixed;right:20px;bottom:20px;max-width:420px;background:var(--ink);color:var(--bg);padding:10px 14px;border-radius:9px;box-shadow:0 6px 20px rgba(0,0,0,.2);z-index:9;font-size:13px}
.cm td,.cm th{text-align:center;padding:4px 6px;font-size:12px;border:1px solid var(--line)}
.cm th.rowh{text-align:right}
svg text{fill:var(--muted);font-size:11px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--muted)}.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
@media (max-width:820px){.layout{grid-template-columns:minmax(0,1fr)}nav{flex-direction:row;overflow-x:auto;border-right:0;border-bottom:1px solid var(--line);padding:8px 12px}nav .sep,nav .note{display:none}main{padding:16px}.two,.three{grid-template-columns:1fr}header .chip.sys{display:none}}
</style>
</head>
<body>
<header>
  <div class="brand"><div class="mark">L</div>LayaStudio <small>fine-tune Laya on your Mac</small></div>
  <div class="spacer"></div>
  <span id="jobchip"></span>
  <span class="chip sys" id="syschip">…</span>
</header>
<div class="layout">
  <nav id="nav">
    <a href="#/datasets" data-v="datasets">Datasets</a>
    <a href="#/train" data-v="train">Fine-tune</a>
    <a href="#/runs" data-v="runs">Runs &amp; results</a>
    <a href="#/playground" data-v="playground">Playground</a>
    <a href="#/models" data-v="models">Models</a>
    <div class="sep"></div>
    <a href="#/guide" data-v="guide">How it works</a>
    <div class="note">Everything stays on this Mac.</div>
  </nav>
  <div class="mainwrap"><div id="setup" hidden></div><main id="main"></main></div>
</div>
<script>
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = (v, d = 1) => v == null ? "–" : (100 * v).toFixed(d) + "%";
const num = (v, d = 3) => v == null ? "–" : Number(v).toFixed(d);
const main = $("#main");
let OV = null, timers = [], ROUTE = 0;
const current = token => token === ROUTE;  // false once the user has navigated elsewhere

async function api(path, opts = {}) {
  const init = {method: opts.method || "GET", headers: {}};
  if (opts.body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.body); }
  const r = await fetch(path, init);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
function toast(msg, ms = 4000) {
  const t = document.createElement("div"); t.className = "toast"; t.textContent = msg;
  document.body.appendChild(t); setTimeout(() => t.remove(), ms);
}
// "?static=1" freezes the page after one render, for screenshots and headless capture.
const STATIC = new URLSearchParams(location.search).has("static");
function every(fn, ms) { if (STATIC) return null; const id = setInterval(fn, ms); timers.push(id); return id; }
function clearTimers() { timers.forEach(clearInterval); timers = []; }
function pill(state) { return `<span class="pill ${esc(state)}">${esc(state)}</span>`; }
const STEP_ICON = {pending: "○", running: "◐", done: "●", warning: "▲", failed: "✕", skipped: "–"};
function renderSetup(setup) {
  const host = $("#setup");
  if (!setup || setup.ready) { host.hidden = true; return; }
  host.hidden = false;
  const failed = setup.state === "failed";
  host.innerHTML = `<section class="card setup"><div class="row" style="justify-content:space-between">
    <h2 style="margin:0">${failed ? "Setup needs your attention" : "Setting up LayaStudio"}</h2>
    <span class="muted">${failed ? "" : "You can look around while this finishes · " + Math.round(setup.seconds) + "s"}</span></div>
    ${setup.steps.map(st => `<div class="hbar" style="grid-template-columns:18px minmax(120px,190px) 1fr">
      <span class="${st.state === "failed" ? "down" : st.state === "done" ? "up" : "muted"}">${STEP_ICON[st.state] || "○"}</span>
      <span class="t">${esc(st.title)}</span>
      <span class="muted">${esc(st.detail || (st.state === "running" ? "working…" : ""))}${st.progress != null ? ` <span class="bar" style="display:inline-block;width:120px;vertical-align:middle"><i style="width:${(100 * st.progress).toFixed(0)}%"></i></span>` : ""}</span>
    </div>`).join("")}</section>`;
}
function modelName(ref) {
  if (!ref) return "–";
  if (ref.startsWith("hub:")) return ref.slice(4).split("/").pop();
  if (ref.startsWith("run:")) { const r = (OV?.runs || []).find(x => "run:" + x.id === ref); return r ? r.name : ref.slice(4); }
  return ref;
}
function delta(a, b, lowerBetter = false, asPct = true) {
  if (a == null || b == null) return "";
  const d = b - a, good = lowerBetter ? d < 0 : d > 0;
  const txt = asPct ? (d >= 0 ? "+" : "") + (100 * d).toFixed(1) + " pts" : (d >= 0 ? "+" : "") + d.toFixed(3);
  return `<span class="${Math.abs(d) < 1e-9 ? "muted" : good ? "up" : "down"}">${txt}</span>`;
}
function readFile(input) { const f = input.files[0]; return f ? f.text().then(text => ({name: f.name, text})) : Promise.resolve(null); }

// ------------------------------------------------------------------ charts
function lineChart(series, {height = 180, width = 640, xLabel = "", yLabel = "", yMin = null, yMax = null} = {}) {
  const pts = series.flatMap(s => s.points);
  if (!pts.length) return `<div class="empty">No data yet</div>`;
  const W = width, H = height, L = 44, R = 12, T = 10, B = 26;
  let x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0]));
  let y0 = yMin ?? Math.min(...pts.map(p => p[1])), y1 = yMax ?? Math.max(...pts.map(p => p[1]));
  if (x1 === x0) x1 = x0 + 1; if (y1 === y0) { y1 += 0.5; y0 -= 0.5; }
  const pad = (y1 - y0) * 0.06; if (yMin == null) y0 -= pad; if (yMax == null) y1 += pad;
  const sx = x => L + (x - x0) / (x1 - x0) * (W - L - R), sy = y => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const y = y0 + (y1 - y0) * i / 4, py = sy(y);
    g += `<line x1="${L}" x2="${W - R}" y1="${py}" y2="${py}" stroke="var(--line)"/><text x="${L - 6}" y="${py + 4}" text-anchor="end">${Math.abs(y1 - y0) < 3 ? y.toFixed(2) : y.toFixed(1)}</text>`;
  }
  g += `<text x="${W - R}" y="${H - 6}" text-anchor="end">${esc(xLabel)}</text><text x="${L}" y="${H - 6}">${esc(x0.toFixed(x1 - x0 < 3 ? 2 : 0))}</text>`;
  for (const s of series) {
    if (!s.points.length) continue;
    const d = s.points.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" opacity="${s.opacity || 1}" stroke-linejoin="round"/>`;
    if (s.dots) for (const p of s.points) g += `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="3.5" fill="${s.color}"><title>${esc(s.name)}: ${p[1].toFixed(3)}</title></circle>`;
  }
  const legend = series.filter(s => s.name).map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("");
  return `<div class="legend">${legend}</div><svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${esc(yLabel)}">${g}</svg>`;
}
function hbars(counts, color = "var(--accent)") {
  const entries = Object.entries(counts); const max = Math.max(1, ...entries.map(e => e[1]));
  return entries.map(([k, v]) => `<div class="hbar"><span class="t" title="${esc(k)}">${esc(k)}</span><div class="bar"><i style="width:${100 * v / max}%;background:${color}"></i></div><span class="n">${v}</span></div>`).join("");
}

// ------------------------------------------------------------------ shell
async function refresh() {
  try { OV = await api("/api/state"); } catch (e) { return; }
  const s = OV.system;
  $("#syschip").textContent = s.ok ? `${s.chip} · ${s.memory_gb} GB · MLX ${s.mlx}` : (s.chip || "Setting up…");
  $("#syschip").title = s.ok ? `${s.cores} cores · ${s.usable_gpu_gb} GB usable by the GPU · laya-mlx ${s.laya_mlx} · ${s.os}` : (s.note || "");
  renderSetup(s.setup);
  const active = OV.jobs.find(j => j.state === "running");
  const chip = $("#jobchip");
  if (active) {
    const last = active.last || {};
    let p = "";
    if (last.type === "step") p = ` · ${Math.round(100 * last.step / last.updates)}%`;
    else if (last.type === "progress" && last.total) p = ` · ${Math.round(100 * last.done / last.total)}%`;
    chip.innerHTML = `<span class="chip live" title="Open job"><span class="dot"></span>${esc(active.title)}${p}</span>`;
    chip.onclick = () => location.hash = active.kind === "train" ? "#/runs/" + active.id : "#/jobs/" + active.id;
  } else chip.innerHTML = "";
  return OV;
}
const routes = {datasets: viewDatasets, dataset: viewDataset, train: viewTrain, runs: viewRuns, run: viewRun, playground: viewPlayground, models: viewModels, guide: viewGuide, jobs: viewJob};
async function route() {
  clearTimers();
  const token = ++ROUTE;
  const [path, qs] = location.hash.replace(/^#\/?/, "").split("?");
  const [a, b] = path.split("/");
  let view = a || "datasets", arg = b ? decodeURIComponent(b) : null;
  if (view === "datasets" && arg) view = "dataset";
  if (view === "runs" && arg) view = "run";
  $$("#nav a").forEach(n => n.classList.toggle("on", n.dataset.v === (a || "datasets")));
  await refresh();
  if (!current(token)) return;
  try { await (routes[view] || viewDatasets)(arg, new URLSearchParams(qs || ""), token); }
  catch (e) { main.innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }
  every(refresh, 3000);
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ datasets
const TEMPLATE = `{
  "intent": {
    "type": "choice",
    "instructions": "What does the user want?",
    "criteria": {
      "billing": "payments, invoices, refunds",
      "technical": "bugs, errors, outages",
      "account": "login, profile, settings",
      "other": "anything else"
    }
  },
  "urgency": {
    "type": "score",
    "instructions": "How urgent is this?",
    "criteria": ["can wait", "soon", "blocking right now"]
  },
  "escalate": {
    "type": "noul",
    "instructions": "Should a human take over this conversation?"
  }
}`;
async function viewDatasets() {
  const ds = OV.datasets;
  main.innerHTML = `
  <h1>Datasets</h1>
  <p class="lead">A dataset is your questions (the same <code>questions</code> object you pass to <code>agent.predict</code>) plus labeled examples of the right answers. Rows are split into train / validation / test so every result is measured on examples the model never trained on.</p>
  <div class="grid two">
    <section class="card">
      <h2>New dataset</h2>
      <label>Name</label><input type="text" id="dsname" placeholder="e.g. support-intents-v1">
      <label>Questions (JSON) <a href="#" id="qload" style="float:right">load file…</a></label>
      <textarea id="dsq" spellcheck="false" style="min-height:220px">${esc(TEMPLATE)}</textarea>
      <input type="file" id="qfile" accept=".json" hidden>
      <label>Labeled rows · JSONL, JSON or CSV</label><input type="file" id="dstrain" accept=".jsonl,.json,.csv,.tsv,.txt">
      <label>Separate test file (optional)</label><input type="file" id="dstest" accept=".jsonl,.json,.csv,.tsv,.txt">
      <div class="row" style="margin-top:14px"><button class="btn primary" id="dscreate">Create dataset</button><span class="muted" id="dsmsg"></span></div>
    </section>
    <section class="card">
      <h2>Row format</h2>
      <p class="muted" style="margin-top:0">One JSON object per line. <code>state</code> is the text (or a JSON object, or a chat as a list of messages); <code>answers</code> holds the correct answer for any subset of your questions.</p>
<pre>{"state": "I was charged twice this month", "answers": {"intent": "billing", "urgency": 1, "escalate": false}}
{"state": [{"role": "user", "content": "app crashes on login"}], "answers": {"intent": "technical"}}
{"state": "refund?", "answers": {"intent": {"billing": 0.7, "other": 0.3}}}</pre>
      <p class="muted">Answers: <b>choice</b> → a label · <b>score</b> → level number (0 = first) · <b>noul</b> → true/false or a probability. A <code>{label: probability}</code> object is a soft label (several annotators, or a teacher model). Add <code>"split": "test"</code> to pin rows to a split.</p>
      <p class="muted">CSV: a <code>state</code> (or <code>text</code>) column plus one column per question id.</p>
      <h3>Or try a public example</h3>
      ${OV.examples.map(x => `<div class="row" style="justify-content:space-between;margin:8px 0"><div><b>${esc(x.title)}</b><div class="muted" style="font-size:12.5px">${esc(x.description)}</div></div><button class="btn small" data-ex="${esc(x.name)}">Fetch</button></div>`).join("")}
    </section>
  </div>
  <section class="card"><h2>Your datasets</h2>
  ${ds.length ? `<div class="tablewrap"><table><tr><th>Name</th><th>Questions</th><th>Train</th><th>Val</th><th>Test</th><th>Created</th><th></th></tr>
  ${ds.map(d => `<tr><td><a href="#/datasets/${esc(d.id)}">${esc(d.name)}</a><div class="faint mono">${esc(d.id)}</div></td><td>${Object.entries(d.questions).map(([q, t]) => `<span class="pill">${esc(q)} · ${esc(t)}</span>`).join(" ")}</td><td>${d.rows.train}</td><td>${d.rows.val}</td><td>${d.rows.test}</td><td class="muted">${esc(d.created)}</td><td><a class="btn small" href="#/train?dataset=${esc(d.id)}">Fine-tune</a></td></tr>`).join("")}
  </table></div>` : `<div class="empty">No datasets yet. Create one above or fetch a public example.</div>`}
  </section>`;
  $("#qload").onclick = e => { e.preventDefault(); $("#qfile").click(); };
  $("#qfile").onchange = async () => { const f = await readFile($("#qfile")); if (f) $("#dsq").value = f.text; };
  $("#dscreate").onclick = async () => {
    const btn = $("#dscreate"), msg = $("#dsmsg");
    let questions;
    try { questions = JSON.parse($("#dsq").value); } catch (e) { msg.textContent = "Questions are not valid JSON: " + e.message; return; }
    const train = await readFile($("#dstrain")), test = await readFile($("#dstest"));
    if (!train) { msg.textContent = "Choose a file with labeled rows."; return; }
    btn.disabled = true; msg.textContent = "Parsing…";
    try {
      const meta = await api("/api/datasets", {method: "POST", body: {name: $("#dsname").value || train.name.replace(/\.[^.]+$/, ""), questions, train, test}});
      toast(`Created ${meta.name}: ${meta.rows.train} train · ${meta.rows.val} val · ${meta.rows.test} test` + (meta.error_count ? ` · ${meta.error_count} rows skipped` : ""));
      location.hash = "#/datasets/" + meta.id;
    } catch (e) { msg.textContent = e.message; btn.disabled = false; }
  };
  $$("[data-ex]").forEach(b => b.onclick = async () => {
    try { const r = await api("/api/jobs", {method: "POST", body: {kind: "example", name: b.dataset.ex}}); location.hash = "#/jobs/" + r.id; }
    catch (e) { toast(e.message); }
  });
}

async function viewDataset(id, _, token) {
  const d = await api("/api/datasets/" + encodeURIComponent(id));
  if (!current(token)) return;
  const m = d.meta, a = d.analysis;
  const cached = OV.models.filter(x => x.cached).concat(OV.finetuned);
  const qs = Object.entries(d.questions);
  main.innerHTML = `
  <div class="row" style="justify-content:space-between"><div><h1>${esc(m.name)}</h1><div class="faint mono">${esc(m.id)}</div></div>
  <div class="row"><a class="btn primary" href="#/train?dataset=${esc(m.id)}">Fine-tune on this</a><button class="btn danger" id="dsdel">Delete</button></div></div>
  <div class="stats" style="margin-top:16px">
    <div class="stat"><div class="k">Train rows</div><div class="v">${m.rows.train}</div><div class="d muted">${m.decisions.train} decisions</div></div>
    <div class="stat"><div class="k">Validation rows</div><div class="v">${m.rows.val}</div><div class="d muted">early stopping + calibration</div></div>
    <div class="stat"><div class="k">Test rows</div><div class="v">${m.rows.test}</div><div class="d muted">never trained on</div></div>
    <div class="stat"><div class="k">Skipped rows</div><div class="v ${m.error_count ? "down" : ""}">${m.error_count}</div><div class="d muted">parse / label errors</div></div>
  </div>
  ${m.error_count ? `<details class="card"><summary>${m.error_count} rows were skipped — see why</summary><table>${m.errors.map(e => `<tr><td class="mono faint">${esc(e.file)}:${e.line}</td><td>${esc(e.error)}</td></tr>`).join("")}</table></details>` : ""}
  <section class="card"><h2>Token budget check</h2>
    <p class="muted" style="margin-top:0">Laya reads at most 512 (English) or 1,024 (multilingual) tokens, and all option texts share a fixed budget. Anything past the window is cut silently — check before you train.</p>
    <div class="row"><select id="anmodel" style="max-width:340px">${cached.map(x => `<option value="${esc(x.ref)}" ${a && a.model === x.ref ? "selected" : ""}>${esc(x.repo || x.name)}</option>`).join("")}</select><button class="btn" id="anrun" ${cached.length ? "" : "disabled"}>Check</button>${cached.length ? "" : `<span class="muted">Download a base model first (Models).</span>`}</div>
    <div id="anout">${a ? renderAnalysis(a) : ""}</div>
  </section>
  <div class="grid two">
  ${qs.map(([q, def]) => {
    const c = m.labels[q];
    return `<section class="card"><h2>${esc(q)} <span class="pill">${esc(def.type)}</span></h2><div class="muted" style="margin:-6px 0 10px">${esc(def.instructions)}</div><h3>Training labels</h3>${hbars(c.train)}</section>`;
  }).join("")}
  </div>
  <section class="card"><h2>Model evaluations on the test split</h2>
  ${d.evals.length ? `<table><tr><th>Model</th><th>Accuracy</th><th>ECE</th><th>Decisions</th><th>p50 latency</th><th>When</th></tr>${d.evals.map(e => `<tr><td>${esc(modelName(e.model))}</td><td>${pct(e.accuracy)}</td><td>${num(e.ece)}</td><td>${e.n}</td><td>${num(e.latency_ms.p50, 1)} ms</td><td class="muted">${esc(e.created)}</td></tr>`).join("")}</table>` : `<div class="muted">None yet. Fine-tuning evaluates the base model first automatically.</div>`}
  <div class="row" style="margin-top:12px"><select id="evmodel" style="max-width:340px">${cached.map(x => `<option value="${esc(x.ref)}">${esc(x.repo || x.name)}</option>`).join("")}</select><button class="btn" id="evrun" ${cached.length ? "" : "disabled"}>Evaluate this model</button></div>
  </section>
  <section class="card"><h2>Sample training rows</h2><div class="tablewrap"><table><tr><th>State</th><th>Labels</th></tr>
  ${d.sample.map(r => `<tr><td><div class="state">${esc(r.state)}</div></td><td style="white-space:nowrap">${Object.entries(r.labels).map(([q, l]) => `<div><span class="faint">${esc(q)}:</span> ${esc(l)}</div>`).join("")}</td></tr>`).join("")}
  </table></div></section>`;
  $("#anrun").onclick = async () => {
    $("#anout").innerHTML = `<div class="muted">Tokenizing…</div>`;
    try { $("#anout").innerHTML = renderAnalysis(await api(`/api/datasets/${id}/analyze`, {method: "POST", body: {model: $("#anmodel").value}})); }
    catch (e) { $("#anout").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }
  };
  $("#evrun").onclick = async () => {
    try { const r = await api("/api/jobs", {method: "POST", body: {kind: "evaluate", dataset: id, model: $("#evmodel").value}}); location.hash = "#/jobs/" + r.id; }
    catch (e) { toast(e.message); }
  };
  $("#dsdel").onclick = async () => {
    if (!confirm(`Delete dataset "${m.name}" and its cached evaluations? Runs trained on it are kept.`)) return;
    try { await api("/api/datasets/" + id, {method: "DELETE", body: {}}); location.hash = "#/datasets"; } catch (e) { toast(e.message); }
  };
}
function renderAnalysis(a) {
  const rows = Object.entries(a.questions).map(([q, x]) => `<tr><td>${esc(q)}</td><td>${x.prefix_tokens}</td><td>${x.state_room}</td><td class="${x.truncated_rows ? "down" : ""}">${x.truncated_rows} / ${x.labeled_rows}</td><td>${x.options}</td><td class="${x.clipped_options.length ? "down" : ""}">${x.tokens_per_option}${x.clipped_options.length ? ` (${x.clipped_options.length} clipped)` : ""}</td></tr>`).join("");
  return `<p class="muted"><b>${esc(modelName(a.model))}</b> · window ${a.max_len} tokens · option budget ${a.head_max_len} · state length p50 ${a.state_tokens.p50}, p95 ${a.state_tokens.p95}, max ${a.state_tokens.max} tokens</p>
  <div class="tablewrap"><table><tr><th>Question</th><th>Question + options tokens</th><th>Room for state</th><th>Rows cut</th><th>Options</th><th>Tokens / option</th></tr>${rows}</table></div>
  ${a.warnings.length ? a.warnings.map(w => `<div class="notice warn">${esc(w)}</div>`).join("") : `<div class="notice good">Everything fits: no truncated states and no clipped options.</div>`}`;
}

// ------------------------------------------------------------------ train
const PRESETS = {
  balanced: {title: "Balanced", sub: "LoRA on every encoder layer + full decision head. Best accuracy per minute.", hp: {method: "lora", lora_layers: 0}},
  fast: {title: "Fast", sub: "LoRA on the top 8 encoder layers only. Roughly half the time.", hp: {method: "lora", lora_layers: 8}},
  head: {title: "Head only", sub: "Freeze the encoder, train the decision head. Fastest, smallest gains.", hp: {method: "head"}},
  full: {title: "Full top layers", sub: "Unfreeze the top 4 encoder layers fully. More memory, lower learning rate.", hp: {method: "full", full_layers: 4, lr: 2.5e-5}},
};
async function viewTrain(_, params) {
  if (!OV.datasets.length) { main.innerHTML = `<h1>Fine-tune</h1><div class="empty">Create a dataset first. <a href="#/datasets">Go to datasets</a></div>`; return; }
  // Defaults follow the machine the studio is running on, not the one it was written on.
  const tuned = OV.system.recommended || {};
  const H = {...OV.hyperparameters, ...(tuned.batch_size ? {batch_size: tuned.batch_size} : {})};
  const models = OV.models.concat(OV.finetuned.map(f => ({ref: f.ref, repo: f.name + " (fine-tuned)", description: "continue from " + modelName(f.base_model), cached: true})));
  main.innerHTML = `
  <h1>Fine-tune</h1>
  <p class="lead">Adapts Laya to your questions and labels. The run first scores the base model on your test split, trains with early stopping on the validation split, re-fits confidence calibration, then scores the fine-tuned model on the same test split.</p>
  <section class="card"><div class="grid two">
    <div><label>Dataset</label><select id="trds">${OV.datasets.map(d => `<option value="${esc(d.id)}" ${params.get("dataset") === d.id ? "selected" : ""}>${esc(d.name)} — ${d.rows.train} train rows</option>`).join("")}</select></div>
    <div><label>Base model</label><select id="trbase">${models.map(m => `<option value="${esc(m.ref)}" ${m.cached ? "" : "disabled"}>${esc(m.repo)} ${m.cached ? "" : "(download in Models)"}</option>`).join("")}</select><div class="muted" id="trbasedesc" style="font-size:12px;margin-top:4px"></div></div>
  </div>
  <label>Recipe</label><div class="opt" id="presets">${Object.entries(PRESETS).map(([k, p], i) => `<div class="choice ${i ? "" : "on"}" data-p="${k}"><b>${esc(p.title)}</b><span>${esc(p.sub)}</span></div>`).join("")}</div>
  <details><summary>Advanced settings</summary><div class="grid three" id="adv">
    ${field("epochs", "Epochs (max)", H.epochs)}${field("batch_size", "Batch size", H.batch_size)}${field("grad_accum", "Gradient accumulation", H.grad_accum)}
    ${field("lr", "Encoder / LoRA learning rate", H.lr)}${field("head_lr", "Head learning rate", H.head_lr)}${field("patience", "Early-stop patience (epochs)", H.patience)}
    ${field("lora_rank", "LoRA rank", H.lora_rank)}${field("lora_alpha", "LoRA alpha", H.lora_alpha)}${field("seed", "Seed", H.seed)}
    ${select("objective", "Objective", H.objective, [["proper", "proper — log + spherical + RPS scores"], ["rlcd", "rlcd — upstream policy gradient + CE"], ["ce", "ce — cross-entropy only"]])}
    ${select("class_weighting", "Class weighting", H.class_weighting, [["none", "none"], ["balanced", "balanced (rare labels count more)"]])}
    ${select("precision", "Frozen weights precision", H.precision, [["bfloat16", "bfloat16 (less memory)"], ["float32", "float32"]])}
    ${select("shuffle_options", "Shuffle choice options", String(H.shuffle_options), [["true", "yes — learn labels, not positions"], ["false", "no"]])}
  </div></details>
  <div class="grid two" style="margin-top:6px"><div><label>Run name (optional)</label><input type="text" id="trname" placeholder="auto"></div>
  <div><label>&nbsp;</label><label style="display:flex;gap:8px;align-items:center;color:var(--ink);font-weight:450"><input type="checkbox" id="trbl" checked> Evaluate the base model first (cached after the first run)</label></div></div>
  <div class="row" style="margin-top:16px"><button class="btn primary" id="trgo">Start fine-tuning</button><span class="muted" id="trmsg"></span></div>
  </section>
  <section class="card"><h2>What to expect on this Mac</h2><p class="muted" style="margin:0">${esc(OV.system.chip || "This Mac")} with ${OV.system.memory_gb || "?"} GB${tuned.note ? " — " + esc(tuned.note) : ""} On a 16 GB M4, the balanced recipe trains the 421M English model at about 7 decisions per second (roughly 10 minutes for 1,000 examples × 4 epochs) with a peak under 3 GB of GPU memory; the 322M multilingual model is lighter. Training pauses the playground so the job has the GPU to itself.</p></section>`;
  let preset = "balanced";
  $$("#presets .choice").forEach(c => c.onclick = () => { $$("#presets .choice").forEach(x => x.classList.remove("on")); c.classList.add("on"); preset = c.dataset.p;
    const lr = PRESETS[preset].hp.lr; $("#hp-lr").value = lr ?? H.lr; });
  const desc = () => { const m = models.find(x => x.ref === $("#trbase").value); $("#trbasedesc").textContent = m ? m.description : ""; };
  $("#trbase").onchange = desc; desc();
  $("#trgo").onclick = async () => {
    // Advanced fields hold every value (the recipe mirrors its learning rate into them);
    // the recipe then fixes the method and which layers adapt.
    const hp = {};
    $$("#adv [data-hp]").forEach(i => { let v = i.value; if (i.type === "number") v = Number(v); if (v === "true") v = true; if (v === "false") v = false; hp[i.dataset.hp] = v; });
    const {lr, ...fixed} = PRESETS[preset].hp; Object.assign(hp, fixed);
    $("#trgo").disabled = true; $("#trmsg").textContent = "Starting…";
    try {
      const r = await api("/api/jobs", {method: "POST", body: {kind: "train", dataset: $("#trds").value, base_model: $("#trbase").value, name: $("#trname").value, baseline: $("#trbl").checked, hyperparameters: hp}});
      location.hash = "#/runs/" + r.id;
    } catch (e) { $("#trmsg").textContent = e.message; $("#trgo").disabled = false; }
  };
}
function field(k, labelText, v) { return `<div><label>${esc(labelText)}</label><input type="number" step="any" data-hp="${k}" id="hp-${k}" value="${esc(v)}"></div>`; }
function select(k, labelText, v, opts) { return `<div><label>${esc(labelText)}</label><select data-hp="${k}" id="hp-${k}">${opts.map(([val, t]) => `<option value="${esc(val)}" ${String(val) === String(v) ? "selected" : ""}>${esc(t)}</option>`).join("")}</select></div>`; }

// ------------------------------------------------------------------ runs
async function viewRuns() {
  const runs = OV.runs;
  main.innerHTML = `<h1>Runs &amp; results</h1><p class="lead">Every fine-tuning run, with the base model and the fine-tuned model scored on the same held-out test rows.</p>
  <section class="card">${runs.length ? `<div class="tablewrap"><table><tr><th>Run</th><th>Dataset</th><th>Base</th><th>Status</th><th>Before</th><th>After</th><th>Change</th><th>Created</th></tr>
  ${runs.map(r => `<tr><td><a href="#/runs/${esc(r.id)}">${esc(r.name)}</a><div class="faint" style="font-size:12px">${esc(r.hyperparameters.method)} · ${esc(r.hyperparameters.objective)}</div></td><td>${esc(r.dataset_name)}</td><td>${esc(modelName(r.base_model))}</td><td>${pill(r.state)}</td><td>${pct(r.baseline_accuracy)}</td><td><b>${pct(r.accuracy)}</b></td><td>${delta(r.baseline_accuracy, r.accuracy)}${r.p_value != null ? `<div class="faint" style="font-size:11.5px">p = ${r.p_value < 0.001 ? "<0.001" : r.p_value.toFixed(3)}</div>` : ""}</td><td class="muted">${esc(r.created)}</td></tr>`).join("")}
  </table></div>` : `<div class="empty">No runs yet. <a href="#/train">Start one</a>.</div>`}</section>`;
}

const PHASES = [["baseline", "Baseline"], ["prepare", "Prepare"], ["train", "Train"], ["calibrate", "Calibrate"], ["save", "Save"], ["evaluate", "Evaluate"]];
async function viewRun(id, _, token) {
  let data = await api("/api/runs/" + encodeURIComponent(id));
  if (!current(token)) return;
  const events = [];
  let next = 0, rendered = false;
  const shell = () => {
    const r = data.run;
    main.innerHTML = `
    <div class="row" style="justify-content:space-between"><div><h1>${esc(r.name)}</h1><div class="muted">${esc(r.dataset_name)} · ${esc(modelName(r.base_model))} · ${esc(r.hyperparameters.method)}${r.hyperparameters.method === "lora" ? ` r${r.hyperparameters.lora_rank}${r.hyperparameters.lora_layers ? ", top " + r.hyperparameters.lora_layers + " layers" : ""}` : ""} · ${esc(r.hyperparameters.objective)}</div></div>
    <div class="row"><span id="rstate"></span><a class="btn" id="rplay" href="#/playground?run=${esc(r.id)}" hidden>Try in playground</a><button class="btn danger" id="rcancel" hidden>Cancel</button><button class="btn danger" id="rdel" hidden>Delete run</button></div></div>
    <section class="card" id="live" style="margin-top:16px"><div class="steps" id="rsteps"></div><div id="rprog"></div><div id="rchart" style="margin-top:12px"></div>
    <details><summary>Event log</summary><pre id="rlog" style="max-height:260px"></pre></details></section>
    <div id="results"></div>`;
    $("#rcancel").onclick = async () => { if (!confirm("Stop this run? The partial model is discarded.")) return; try { await api(`/api/jobs/${id}/cancel`, {method: "POST", body: {}}); } catch (e) { toast(e.message); } };
    $("#rdel").onclick = async () => { if (!confirm("Delete this run and its checkpoint?")) return; try { await api("/api/runs/" + id, {method: "DELETE", body: {}}); location.hash = "#/runs"; } catch (e) { toast(e.message); } };
  };
  const update = () => {
    const job = data.job;
    $("#rstate").innerHTML = pill(job.state);
    $("#rcancel").hidden = job.state !== "running"; $("#rdel").hidden = job.state === "running";
    $("#rplay").hidden = !(job.state === "done" && data.model_path);
    const phases = events.filter(e => e.type === "phase").map(e => e.phase);
    const cur = phases[phases.length - 1];
    const doneAll = job.state === "done";
    $("#rsteps").innerHTML = PHASES.map(([k, t]) => {
      const idx = PHASES.findIndex(p => p[0] === k), curIdx = PHASES.findIndex(p => p[0] === cur);
      const cls = doneAll || idx < curIdx ? "done" : k === cur && job.state === "running" ? "now" : "";
      return `<span class="${cls}">${t}</span>`;
    }).join("");
    const steps = events.filter(e => e.type === "step"), epochs = events.filter(e => e.type === "epoch");
    const info = events.find(e => e.type === "info"), last = events.filter(e => ["phase", "step", "progress"].includes(e.type)).pop();
    let prog = "";
    if (job.state === "running" && last) {
      const phaseMsg = events.filter(e => e.type === "phase").pop();
      let frac = null, detail = "";
      if (last.type === "step") { frac = last.step / last.updates; detail = `update ${last.step}/${last.updates} · epoch ${last.epoch} · ${last.decisions_per_s} decisions/s · ${last.peak_gb} GB peak · ~${fmtTime(last.eta_s)} left`; }
      else if (last.type === "progress") { frac = last.done / last.total; detail = `${last.done}/${last.total} rows${last.model ? " · " + modelName(last.model) : ""}`; }
      prog = `<div class="row" style="justify-content:space-between"><b>${esc(phaseMsg ? phaseMsg.message : "Starting")}</b><span class="muted">${esc(detail)}</span></div>${frac != null ? `<div class="bar" style="margin-top:8px"><i style="width:${(100 * frac).toFixed(1)}%"></i></div>` : ""}`;
    } else if (job.state === "failed" || job.state === "interrupted") prog = `<div class="notice bad"><b>Run ${esc(job.state)}.</b> ${esc(job.error || "")}</div>`;
    else if (job.state === "cancelled") prog = `<div class="notice warn">Run cancelled.</div>`;
    if (info) prog += `<div class="muted" style="font-size:12.5px;margin-top:10px">${(info.trainable_params / 1e6).toFixed(1)}M of ${(info.total_params / 1e6).toFixed(0)}M parameters trainable · ${info.train_decisions} train / ${info.val_decisions} validation decisions · ${info.updates} updates${info.gradient_checkpointing ? " · gradient checkpointing" : ""}${info.skipped_decisions ? ` · <span class="down">${info.skipped_decisions} skipped (too many options)</span>` : ""}</div>`;
    $("#rprog").innerHTML = prog;
    if (steps.length || epochs.length) {
      const per = info ? info.updates / (info.hyperparameters?.epochs || 1) : 1;
      let ema = 0; const smooth = steps.map((s, i) => { ema = 0.9 * ema + 0.1 * s.ce; return [s.step, ema / (1 - 0.9 ** (i + 1))]; });  // bias-corrected
      $("#rchart").innerHTML = `<div class="grid two"><div><h3>Loss (cross-entropy)</h3>${lineChart([
        {name: "train (raw)", points: steps.map(s => [s.step, s.ce]), color: "var(--base)", width: 1, opacity: .45},
        {name: "train (smoothed)", points: smooth, color: "var(--ft)"},
        {name: "validation", points: epochs.map(e => [e.epoch * per, e.val_loss]), color: "var(--warn)", dots: true}], {xLabel: "update", yMin: 0, width: 420, height: 220})}</div>
        <div><h3>Validation accuracy</h3>${lineChart([{name: "validation accuracy", points: epochs.map(e => [e.epoch, e.val_accuracy]), color: "var(--good)", dots: true}], {xLabel: "epoch", yMin: 0, yMax: 1, width: 420, height: 220})}</div></div>`;
    }
    $("#rlog").textContent = events.map(e => `${new Date(e.t * 1000).toLocaleTimeString()}  ${e.type.padEnd(11)} ${e.message || e.phase || ""} ${e.type === "step" ? `step ${e.step} loss ${num(e.loss)} ce ${num(e.ce)}` : ""}${e.type === "epoch" ? `epoch ${e.epoch} val_loss ${num(e.val_loss)} val_acc ${pct(e.val_accuracy)}` : ""}${e.type === "calibration" ? `ECE ${num(e.ece_uncalibrated)} → ${num(e.ece_calibrated)}` : ""}${e.type === "error" ? "\n" + (e.traceback || "") : ""}`).join("\n");
    if (job.state === "done" && !rendered && data.comparison) { rendered = true; renderResults(data, id); }
    if (job.state !== "running" && !rendered && data.eval) { rendered = true; renderResults(data, id); }
  };
  const poll = async () => {
    try {
      const r = await api(`/api/jobs/${encodeURIComponent(id)}?since=${next}`);
      if (!current(token)) return;
      events.push(...r.events); next = r.next;
      const was = data.job.state; data.job = r.job;
      if (was === "running" && r.job.state !== "running") data = await api("/api/runs/" + encodeURIComponent(id));
      update();
    } catch (e) { /* transient */ }
  };
  shell(); await poll();
  if (data.job.state === "running") every(poll, 1000);
}
function fmtTime(s) { if (s == null) return "–"; s = Math.round(s); return s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.round(s % 3600 / 60)}m` : s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`; }

function latencyStat(c, be, ev) {
  const lp = c?.latency_paired, refs = lp ? Object.keys(lp) : [];
  const [b, f] = refs.length === 2 ? [lp[refs[0]].p50, lp[refs[1]].p50] : [be?.latency_ms.p50, ev.latency_ms.p50];
  const note = refs.length === 2 ? `both timed interleaved on ${lp[refs[1]].rows} rows` : "separate passes; heat and load affect this";
  return `<div class="stat"><div class="k">Latency p50 / row</div><div class="v">${num(f, 1)} ms</div><div class="d">${b != null ? `<span class="muted">base ${num(b, 1)} ms</span>` : ""}<div class="faint">${note}</div></div></div>`;
}
function renderResults(data, id) {
  const c = data.comparison, ev = data.eval, be = data.base_eval, tr = data.training;
  if (!ev) return;
  const b = be ? be.overall : null, f = ev.overall;
  const qids = Object.keys(ev.questions);
  const paired = c ? c.paired.overall : null;
  let verdict = "";
  if (paired) {
    const sig = paired.p_value < 0.05;
    verdict = `<div class="notice ${sig ? (f.accuracy >= b.accuracy ? "good" : "bad") : "info"}">Fine-tuning fixed <b>${paired.fixed}</b> test decisions the base model got wrong and broke <b>${paired.broken}</b> it got right. ${sig ? `The difference is statistically significant (exact McNemar p ${paired.p_value < 0.001 ? "< 0.001" : "= " + paired.p_value.toFixed(3)}).` : `This is not statistically significant (p = ${paired.p_value.toFixed(3)}); add test rows or training data before relying on it.`}</div>`;
  }
  const stat = (k, bv, fv, fmt, lowerBetter, note) => `<div class="stat"><div class="k">${k}</div><div class="v">${fmt(fv)}</div><div class="d">${bv != null ? `<span class="muted">base ${fmt(bv)}</span> · ${delta(bv, fv, lowerBetter, fmt === pct)}` : ""}${note ? `<div class="faint">${note}</div>` : ""}</div></div>`;
  const cov = (rep) => rep ? rep.overall.coverage : [];
  const qrows = qids.map(q => {
    const x = ev.questions[q], y = be?.questions[q], p = c?.paired[q];
    return `<tr><td>${esc(q)}</td><td>${x.n}</td><td>${pct(y?.accuracy)}</td><td><b>${pct(x.accuracy)}</b> <span class="faint">[${pct(x.accuracy_ci95[0], 0)}–${pct(x.accuracy_ci95[1], 0)}]</span></td><td>${delta(y?.accuracy, x.accuracy)}</td><td>${num(y?.macro_f1)} → ${num(x.macro_f1)}</td><td>${num(y?.ece)} → ${num(x.ece)}</td><td>${x.mae != null ? num(y?.mae, 2) + " → " + num(x.mae, 2) : "–"}</td><td>${p ? (p.p_value < 0.001 ? "<0.001" : p.p_value.toFixed(3)) : "–"}</td></tr>`;
  }).join("");
  const thresholds = cov(data.eval ? ev : null).map((t, i) => { const bt = cov(be)[i]; return `<tr><td>≥ ${t.threshold.toFixed(2)}</td><td>${bt ? pct(bt.coverage, 0) : "–"}</td><td>${bt ? pct(bt.accuracy) : "–"}</td><td><b>${pct(t.coverage, 0)}</b></td><td><b>${pct(t.accuracy)}</b></td></tr>`; }).join("");
  $("#results").innerHTML = `
  <h2 style="margin-top:8px">Results on ${f.n} held-out test decisions</h2>
  <div class="stats">
    ${stat("Accuracy", b?.accuracy, f.accuracy, pct, false, `95% CI ${pct(f.accuracy_ci95[0], 0)}–${pct(f.accuracy_ci95[1], 0)}`)}
    ${stat("Calibration error (ECE)", b?.ece, f.ece, v => num(v), true, "lower = confidence you can trust")}
    ${stat("Log loss", b?.nll, f.nll, v => num(v), true)}
    ${stat("Brier score", b?.brier, f.brier, v => num(v), true)}
    ${latencyStat(c, be, ev)}
  </div>
  ${verdict}
  <section class="card"><h2>Per question</h2><div class="tablewrap"><table><tr><th>Question</th><th>Decisions</th><th>Before</th><th>After [95% CI]</th><th>Change</th><th>Macro F1</th><th>ECE</th><th>Score MAE</th><th>p</th></tr>${qrows}</table></div></section>
  <div class="grid two">
    <section class="card"><h2>Confidence gating</h2><p class="muted" style="margin-top:0">Answer automatically only when the top probability clears a threshold; send the rest to a person or a larger model. Coverage = share answered automatically.</p>
      ${lineChart([{name: "base", points: cov(be).map(t => [t.coverage, t.accuracy]), color: "var(--base)", dots: true}, {name: "fine-tuned", points: cov(ev).map(t => [t.coverage, t.accuracy]), color: "var(--ft)", dots: true}], {xLabel: "share answered automatically →", yMax: 1, width: 420, height: 240})}
      <div class="tablewrap"><table><tr><th>Threshold</th><th>Base coverage</th><th>Base accuracy</th><th>Tuned coverage</th><th>Tuned accuracy</th></tr>${thresholds}</table></div></section>
    <section class="card"><h2>Confusions</h2><select id="cmq" style="max-width:260px;margin-bottom:10px">${qids.map(q => `<option>${esc(q)}</option>`).join("")}</select><div id="cm"></div></section>
  </div>
  <section class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Mistakes the fine-tuned model still makes</h2><button class="btn small" id="errload">Show mistakes</button></div><div id="errs" style="margin-top:10px"></div></section>
  ${tr ? `<section class="card"><h2>Training</h2><div class="stats">
    <div class="stat"><div class="k">Trainable parameters</div><div class="v">${(tr.trainable_params / 1e6).toFixed(1)}M</div><div class="d muted">of ${(tr.total_params / 1e6).toFixed(0)}M</div></div>
    <div class="stat"><div class="k">Best epoch</div><div class="v">${tr.best_epoch}</div><div class="d muted">validation loss ${num(tr.best_val_loss)}</div></div>
    <div class="stat"><div class="k">Training time</div><div class="v">${fmtTime(tr.train_seconds)}</div><div class="d muted">${tr.updates} updates</div></div>
    <div class="stat"><div class="k">Peak memory</div><div class="v">${tr.peak_memory_gb} GB</div><div class="d muted">MLX active allocations</div></div>
    <div class="stat"><div class="k">Calibration (val ECE)</div><div class="v">${num(tr.calibration.ece_calibrated)}</div><div class="d muted">uncalibrated ${num(tr.calibration.ece_uncalibrated)}</div></div>
  </div><details><summary>Hyperparameters and temperatures</summary><pre>${esc(JSON.stringify({hyperparameters: tr.hyperparameters, temperature: tr.calibration.temperature, temperature_by_options: tr.calibration.temperature_by_options}, null, 2))}</pre></details></section>` : ""}
  ${data.model_path ? `<section class="card"><h2>Use the fine-tuned model</h2><p class="muted" style="margin-top:0">A standard Laya checkpoint: FP16 safetensors with the original PyTorch parameter names, your questions and the refitted calibration. Ask the <b>same questions</b> it was trained on.</p>
<pre># from the repository root
import json, laya_mlx as laya

agent = laya.load("${esc(data.model_path)}")
questions = json.load(open("${esc(data.model_path)}/questions.json"))
print(agent.predict("your text here", questions)["answers"])</pre></section>` : ""}`;
  const cm = () => {
    const q = $("#cmq").value, x = ev.questions[q];
    if (!x.confusion) { $("#cm").innerHTML = ""; return; }
    const L = x.labels, M = x.confusion;
    if (L.length > 12) {
      const pairs = []; M.forEach((row, i) => row.forEach((n, j) => { if (i !== j && n) pairs.push([n, L[i], L[j]]); }));
      pairs.sort((a, b) => b[0] - a[0]);
      $("#cm").innerHTML = pairs.length ? `<table><tr><th>Correct label</th><th>Predicted</th><th>Count</th></tr>${pairs.slice(0, 15).map(p => `<tr><td>${esc(p[1])}</td><td>${esc(p[2])}</td><td>${p[0]}</td></tr>`).join("")}</table>` : `<div class="notice good">No confusions.</div>`;
      return;
    }
    const max = Math.max(1, ...M.flat());
    $("#cm").innerHTML = `<div class="tablewrap"><table class="cm"><tr><th class="rowh">correct ↓ / predicted →</th>${L.map(l => `<th title="${esc(l)}">${esc(l.length > 10 ? l.slice(0, 9) + "…" : l)}</th>`).join("")}</tr>${M.map((row, i) => `<tr><th class="rowh">${esc(L[i])}</th>${row.map((n, j) => `<td style="background:color-mix(in srgb, ${i === j ? "var(--good)" : "var(--bad)"} ${n ? 12 + 60 * n / max : 0}%, transparent)">${n || ""}</td>`).join("")}</tr>`).join("")}</table></div>`;
  };
  $("#cmq").onchange = cm; cm();
  $("#errload").onclick = async () => {
    try {
      const r = await api(`/api/runs/${encodeURIComponent(id)}/errors?limit=40`);
      $("#errs").innerHTML = r.count ? `<p class="muted">${r.count} wrong decisions, most confident first. Confident mistakes often point to label noise or a missing option.</p><div class="tablewrap"><table><tr><th>State</th><th>Question</th><th>Correct</th><th>Fine-tuned</th><th>Base</th></tr>${r.errors.map(e => `<tr><td><div class="state">${esc(e.state)}</div></td><td>${esc(e.question)}</td><td>${esc(e.gold)}</td><td class="down">${esc(e.predicted)} <span class="faint">${pct(e.confidence, 0)}</span></td><td class="${e.base_correct ? "up" : "muted"}">${esc(e.base_predicted ?? "–")}</td></tr>`).join("")}</table></div>` : `<div class="notice good">No mistakes on the test split.</div>`;
    } catch (e) { $("#errs").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }
  };
}

// ------------------------------------------------------------------ jobs (non-training)
async function viewJob(id, _, token) {
  const events = []; let next = 0;
  main.innerHTML = `<h1 id="jt">Job</h1><section class="card"><div id="jstate"></div><div id="jprog" style="margin-top:10px"></div><pre id="jlog" style="max-height:300px"></pre></section>`;
  const poll = async () => {
    const r = await api(`/api/jobs/${encodeURIComponent(id)}?since=${next}`);
    if (!current(token)) return;
    events.push(...r.events); next = r.next;
    $("#jt").textContent = r.job.title; $("#jstate").innerHTML = pill(r.job.state) + (r.job.error ? `<div class="notice bad">${esc(r.job.error)}</div>` : "");
    const p = events.filter(e => e.type === "progress").pop();
    $("#jprog").innerHTML = r.job.state === "running" && p ? `<div class="bar"><i style="width:${100 * p.done / p.total}%"></i></div>` : "";
    $("#jlog").textContent = events.map(e => `${e.type.padEnd(9)} ${e.message || ""}${e.type === "progress" ? `${e.done}/${e.total}` : ""}${e.type === "result" ? JSON.stringify(e) : ""}`).join("\n");
    if (r.job.state !== "running") {
      clearTimers();
      const res = events.find(e => e.type === "result");
      if (res && res.dataset) { toast("Dataset ready"); location.hash = "#/datasets/" + res.dataset; }
      else if (r.job.kind === "download" && r.job.state === "done") { toast("Model downloaded"); location.hash = "#/models"; }
      else if (r.job.kind === "evaluate" && r.job.state === "done") { toast("Evaluation finished"); history.back(); }
    }
  };
  await poll(); every(poll, 1000);
}

// ------------------------------------------------------------------ playground
async function viewPlayground(_, params, token) {  // params: run, state, go
  const models = OV.models.filter(m => m.cached).map(m => ({ref: m.ref, name: m.repo})).concat(OV.finetuned.map(f => ({ref: f.ref, name: f.name + " (fine-tuned)"})));
  // "#/playground?run=<id>" compares a run against the model it started from.
  let preset = null;
  if (params.get("run")) {
    try {
      const r = await api("/api/runs/" + params.get("run"));
      if (!current(token)) return;
      const d = await api("/api/datasets/" + r.run.dataset);
      preset = {models: [r.run.base_model, "run:" + r.run.id], questions: d.questions,
                state: (d.sample[0] || {}).state || ""};
    } catch (e) { toast(e.message); }
  }
  const checked = m => preset ? preset.models.includes(m.ref) : false;
  main.innerHTML = `<h1>Playground</h1><p class="lead">Ask base and fine-tuned models the same questions side by side. Models stay loaded between requests, so later answers show real latency.</p>
  <section class="card"><label>Models (up to 4)</label><div class="row" id="pgm">${models.map((m, i) => `<label style="display:flex;gap:6px;align-items:center;margin:0;color:var(--ink);font-weight:450"><input type="checkbox" value="${esc(m.ref)}" ${preset ? (checked(m) ? "checked" : "") : (i === 0 ? "checked" : "")}>${esc(m.name)}</label>`).join("") || `<span class="muted">No models available. Download one in Models.</span>`}</div>
  <div class="grid two"><div><label>Questions <select id="pgqs" style="width:auto;display:inline-block;margin-left:8px;padding:2px 6px"><option value="">custom</option>${OV.datasets.map(d => `<option value="${esc(d.id)}">from ${esc(d.name)}</option>`).join("")}${OV.finetuned.map(f => `<option value="run:${esc(f.ref.slice(4))}">from run ${esc(f.name)}</option>`).join("")}</select></label><textarea id="pgq" spellcheck="false" style="min-height:240px">${esc(preset ? JSON.stringify(preset.questions, null, 2) : TEMPLATE)}</textarea></div>
  <div><label>State</label><textarea id="pgs" style="min-height:240px;font-family:inherit;font-size:14px" placeholder="Paste a message, ticket or JSON object">${esc(params.get("state") || (preset ? preset.state : "I was charged twice for my subscription and nobody answers my emails. Please fix this today."))}</textarea></div></div>
  <div class="row" style="margin-top:12px"><button class="btn primary" id="pggo">Predict</button><span class="muted" id="pgmsg"></span></div></section><div id="pgout" class="grid two"></div>`;
  $("#pgqs").onchange = async () => {
    const v = $("#pgqs").value; if (!v) return;
    try {
      const d = v.startsWith("run:") ? (await api("/api/runs/" + v.slice(4))).run.dataset : v;
      $("#pgq").value = JSON.stringify((await api("/api/datasets/" + d)).questions, null, 2);
    } catch (e) { toast(e.message); }
  };
  $("#pggo").onclick = async () => {
    const refs = $$("#pgm input:checked").map(i => i.value);
    let questions, state = $("#pgs").value;
    try { questions = JSON.parse($("#pgq").value); } catch (e) { $("#pgmsg").textContent = "Questions are not valid JSON"; return; }
    try { const t = state.trim(); if (t.startsWith("{") || t.startsWith("[")) state = JSON.parse(t); } catch (_) {}
    $("#pggo").disabled = true; $("#pgmsg").textContent = "Running (first use loads the model)…";
    try {
      const r = await api("/api/predict", {method: "POST", body: {models: refs, questions, state}});
      $("#pgmsg").textContent = "";
      $("#pgout").innerHTML = Object.entries(r.results).map(([ref, res]) => `<section class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">${esc(modelName(ref))}</h2><span class="chip">${res.latency_ms} ms · ${res.usage.input_tokens} tokens in · 0 out</span></div>
        ${Object.entries(res.answers).map(([q, a]) => answerCard(q, a, questions[q])).join("")}</section>`).join("");
    } catch (e) { $("#pgmsg").textContent = e.message; }
    $("#pggo").disabled = false;
  };
  if (params.get("go")) $("#pggo").click();
}
function answerCard(q, a, def) {
  let head = "", probs = a.probabilities || {};
  if (a.type === "choice") head = `<b>${esc(a.choice)}</b>`;
  else if (a.type === "score") { head = `<b>${num(a.score, 2)}</b> <span class="muted">expected level</span>`; probs = Object.fromEntries(Object.entries(a.probabilities).map(([k, v]) => [`${k}: ${a.legend[k]}`, v])); }
  else { head = `<b>${a.noul >= 0.5 ? "true" : "false"}</b> <span class="muted">P(true) = ${num(a.noul)}</span>`; probs = {"true": a.noul, "false": 1 - a.noul}; }
  return `<h3>${esc(q)} · ${esc(a.type)}</h3><div class="row" style="justify-content:space-between">${head}<span class="muted" style="font-size:12px">confidence ${num(a.confidence, 2)}</span></div>
  ${Object.entries(probs).map(([k, v]) => `<div class="hbar"><span class="t" title="${esc(k)}">${esc(k)}</span><div class="bar"><i style="width:${100 * v}%"></i></div><span class="n">${pct(v, 0)}</span></div>`).join("")}`;
}

// ------------------------------------------------------------------ models
async function viewModels() {
  main.innerHTML = `<h1>Models</h1><p class="lead">Base checkpoints come from Hugging Face (MLX conversions of the original Laya weights). Downloading is the only step that needs the internet.</p>
  <section class="card"><h2>Base models</h2><table><tr><th>Model</th><th>What it is</th><th>Status</th><th></th></tr>
  ${OV.models.map(m => `<tr><td class="mono">${esc(m.repo)}</td><td>${esc(m.description)}</td><td>${m.cached ? pill("done").replace(">done<", ">downloaded<") : `<span class="pill">not downloaded</span>`}</td><td>${m.cached ? "" : `<button class="btn small" data-dl="${esc(m.repo)}">Download</button>`}</td></tr>`).join("")}</table></section>
  <section class="card"><h2>Fine-tuned checkpoints</h2>
  ${OV.finetuned.length ? `<div class="tablewrap"><table><tr><th>Run</th><th>Base</th><th>Test accuracy</th><th>Location</th></tr>${OV.finetuned.map(f => `<tr><td><a href="#/runs/${esc(f.ref.slice(4))}">${esc(f.name)}</a></td><td>${esc(modelName(f.base_model))}</td><td>${pct(f.accuracy)}</td><td class="mono faint" style="word-break:break-all">${esc(f.path)}</td></tr>`).join("")}</table></div>` : `<div class="muted">None yet.</div>`}
  <h3>Format and portability</h3><p class="muted">Each checkpoint is <code>model.safetensors</code> (FP16, the original PyTorch parameter names), <code>rl_agent_config.json</code> (with refitted temperatures), <code>encoder/</code>, <code>tokenizer/</code>, <code>questions.json</code> and <code>laya_finetune.json</code> (provenance). It loads unchanged in <code>laya-mlx</code> on Apple silicon and in the upstream PyTorch <code>laya</code> package on Linux CPUs and NVIDIA GPUs. Dedicated exports (ONNX, Core ML, LiteRT, quantized) are the next step of this project.</p></section>`;
  $$("[data-dl]").forEach(b => b.onclick = async () => { try { const r = await api("/api/jobs", {method: "POST", body: {kind: "download", repo_id: b.dataset.dl}}); location.hash = "#/jobs/" + r.id; } catch (e) { toast(e.message); } });
}

// ------------------------------------------------------------------ guide
async function viewGuide() {
  main.innerHTML = `<h1>How it works</h1><p class="lead">A short tour of what Laya is, what fine-tuning changes, and what data you need.</p>
  <div class="grid two">
  <section class="card"><h2>1 · Laya scores options, it does not write text</h2><p>Each question becomes one input sequence:</p>
  <pre>[CLS] choice question: &lt;instructions&gt; [SEP]
[MASK] billing: payments… [MASK] technical: bugs… [SEP]
&lt;your state&gt; [SEP]</pre>
  <p>A bidirectional encoder (ModernBERT-large, 421M, or mmBERT-base, 322M) reads it once. A question-type embedding is added, a 2-layer decision transformer mixes the tokens, and a small scorer turns the hidden state at every <code>[MASK]</code> into one logit per option. Softmax with a calibrated temperature gives the probabilities. No tokens are generated.</p></section>
  <section class="card"><h2>2 · What fine-tuning changes</h2><p>The pretrained model knows language, but not your labels, your boundaries between them, or your domain vocabulary. Fine-tuning shows it thousands of your decisions and nudges the weights so the correct option's <code>[MASK]</code> scores higher.</p>
  <p><b>LoRA</b> (default) freezes the encoder and learns a low-rank update <code>W + (α/r)·A·B</code> for each of its attention and MLP matrices, while the decision head trains fully. That is ~33M of 428M parameters, so it fits comfortably in 16 GB. After training the updates are merged back, so the saved model is an ordinary Laya checkpoint with zero extra inference cost.</p></section>
  <section class="card"><h2>3 · The objective</h2><p>Laya was trained with RLCD: rewards from <i>strictly proper scoring rules</i> (log score, spherical score, and the ranked probability score for ordinal questions), which only reach their best value when the reported probabilities are honest.</p>
  <p><b>proper</b> (default) optimizes those same scores directly and deterministically. <b>rlcd</b> reproduces the upstream notebook: noisy Gaussian perturbations of the logits, a group-normalized policy gradient on that reward, plus cross-entropy. <b>ce</b> is plain cross-entropy (the log score alone).</p></section>
  <section class="card"><h2>4 · Calibration and honest measurement</h2><p>After training, temperatures are refitted per question type and option count on the validation split, so a 0.9 means right about nine times in ten. Early stopping keeps the epoch with the lowest validation loss.</p><p>The test split is never used for training or selection. Results show 95% confidence intervals and an exact McNemar test, so you can tell a real improvement from noise.</p></section>
  <section class="card"><h2>5 · The data you need</h2><ul style="padding-left:18px;margin:0">
  <li><b>Real inputs</b> in the form you will send in production (same cleaning, same fields, same language).</li>
  <li><b>Your final questions.</b> Instructions and option texts are part of the input; keep them identical after training.</li>
  <li><b>Enough examples per label:</b> ~30 is a start, 100+ is solid, more for labels that are easy to confuse.</li>
  <li><b>Honest test rows</b> that look like production traffic, ideally 200+ decisions.</li>
  <li><b>An escape hatch:</b> add an <code>other</code> option; the model always picks one of the options it is given.</li>
  <li><b>Soft labels</b> when annotators disagree, or when labels come from a larger teacher model (distillation).</li></ul></section>
  <section class="card"><h2>6 · Limits to keep in mind</h2><ul style="padding-left:18px;margin:0">
  <li>Inputs beyond 512 / 1,024 tokens are cut: check the token budget on each dataset.</li>
  <li>Many labels share one option budget, so long label lists get clipped; fine-tuning helps, shorter criteria help more.</li>
  <li>Fine-tuning specializes the model. Evaluate other questions you rely on before replacing a general checkpoint.</li>
  <li>Gate on confidence and route uncertain cases to a person or a larger model.</li></ul></section>
  </div>`;
}

route();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
