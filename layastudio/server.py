#!/usr/bin/env python3
"""System One Studio (formerly Laya Studio): fine-tune System One decision models on
your own data, on your own machine.

    layastudio                    # or: uv run layastudio
    layastudio train --config run.json    # one fine-tune with no UI (cloud.py)

One command. The server answers immediately and finishes setting itself up in the
background: it detects this machine, checks the training runtime (MLX on Apple silicon,
PyTorch elsewhere), downloads a base checkpoint and fetches the public example datasets,
reporting every step on the page.

Frontend (HTML, CSS, JavaScript) and backend (JSON API) live in this one file and use only
the Python standard library, so there is nothing to build. Training, evaluation and
downloads run as child processes of layastudio.engine; this server schedules them, streams
their progress and serves the results.

Privacy: the server binds to 127.0.0.1, the page loads no external scripts, fonts or
analytics, and training/evaluation jobs run with HF_HUB_OFFLINE=1. Datasets, runs and
checkpoints stay in the workspace folder, which git ignores.
"""

import argparse
import json
import os
import re
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

from . import cloud, engine, kinds, noulxp_package
from .account import Account
from .bootstrap import DEFAULT_MODEL, Bootstrap
from .examples import catalog

PACKAGE = Path(__file__).resolve().parent

MAX_BODY = 512 * 2**20
TERMINAL = {"done": "done", "error": "failed", "cancelled": "cancelled"}
# What the Train button sends, and all a fine-tune started here takes (cloud.prepare_run).
TRAIN_FIELDS = ("dataset", "base_model", "name", "hyperparameters", "baseline")


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def folders(path):
    """Sub-directories only: Finder leaves .DS_Store files in workspace folders."""
    try:
        return [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    except OSError:
        return []


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


def dir_size(path):
    """Bytes in a folder; the Hugging Face cache's symlinks count as the files they point to."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


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
            path = engine.write_job(self.workspace, job_id, kind, spec, title)
            self.before_start()
            log = open(path / "output.log", "w")
            self.procs[job_id] = engine.start_job(path, kind, log)
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

    def count(self):
        return len(folders(self.root))

    def list(self, limit=30):
        jobs = sorted(folders(self.root), key=lambda p: p.stat().st_mtime, reverse=True)
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
        from . import runtime

        if ref not in self.models:
            while len(self.models) >= self.capacity:
                self.models.pop(next(iter(self.models)))
            path = engine.resolve_model_ref(ref, self.workspace)
            self.models[ref] = runtime.load_agent(path)
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
            from . import runtime

            self.models.clear()
            runtime.clear_cache()

        self.pool.submit(run).result()


# ----------------------------------------------------------------------------- arena


class Arena:
    """Two models playing Snake side by side, live, for anyone watching the page.

    Moves run on the playground's single MLX thread, so the arena and the playground never
    touch the GPU at the same time, and a training job stops the arena first.
    """

    BOARD_CAP = 600

    def __init__(self, workspace, playground):
        self.workspace = workspace
        self.playground = playground
        self.lock = threading.Lock()
        self.sides = []
        self.running = False
        self.thread = None
        self.speed = 8
        self.error = None

    def _new_game(self, seed):
        from .laya_mlx_free import laya_mlx_module
        from .snake import HEIGHT, INITIAL_LENGTH, WIDTH

        SnakeGame = laya_mlx_module("snake.game").SnakeGame

        return SnakeGame(WIDTH, HEIGHT, seed=seed, initial_length=INITIAL_LENGTH)

    def _new_side(self, ref, seed):
        return {
            "ref": ref,
            "game": self._new_game(seed),
            "seed": seed,
            "games": 0,
            "moves": 0,
            "apples": 0,
            "legal": 0,
            "decisions": 0,
            "best_apples": 0,
            "best_moves": 0,
            "latency": [],
            "last_death": None,
        }

    def start(self, refs, speed=8):
        self.stop()
        seed = int(time.time()) % 10000
        with self.lock:
            self.sides = [self._new_side(ref, seed + i * 977) for i, ref in enumerate(refs)]
            self.running, self.error = True, None
        self.thread = threading.Thread(target=self._loop, name="arena", daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        thread, self.thread = self.thread, None
        if thread:
            thread.join(timeout=5)

    def _loop(self):
        interval = 1 / max(1, self.speed)
        while self.running:
            started = time.perf_counter()
            try:
                self.playground.pool.submit(self._step).result()
            except Exception as error:  # noqa: BLE001 - surfaced on the page
                with self.lock:
                    self.error, self.running = f"{type(error).__name__}: {error}", False
                return
            time.sleep(max(0, interval - (time.perf_counter() - started)))

    def _step(self):
        from .laya_mlx_free import laya_mlx_module
        from .snake import QUESTIONS, render

        DIRECTIONS = laya_mlx_module("snake.game").DIRECTIONS

        for side in self.sides:
            game = side["game"]
            if not game.alive or game.won or game.ticks >= self.BOARD_CAP:
                side["games"] += 1
                side["best_apples"] = max(side["best_apples"], game.score)
                side["best_moves"] = max(side["best_moves"], game.ticks)
                side["last_death"] = game.death_reason or ("finished" if game.won else "stalled")
                side["seed"] += 1
                side["game"] = game = self._new_game(side["seed"])
            agent = self.playground._agent(side["ref"])
            started = time.perf_counter()
            answer = agent.predict(render(game), QUESTIONS)["answers"]["move"]
            side["latency"].append((time.perf_counter() - started) * 1000)
            del side["latency"][:-200]
            choice = answer["choice"] if answer["choice"] in DIRECTIONS else "UP"
            legal = {m.direction: m for m in game.moves()}.get(choice)
            side["legal"] += bool(legal and legal.legal)
            side["decisions"] += 1
            side["moves"] += 1
            side["apples"] = side["apples"] + 1 if game.step(choice) else side["apples"]

    def snapshot(self):
        from .snake import HEIGHT, WIDTH

        with self.lock:
            sides = []
            for side in self.sides:
                game = side["game"]
                body = set(game.body)
                rows = [
                    "".join(
                        "H"
                        if (x, y) == game.head
                        else "F"
                        if (x, y) == game.food
                        else "o"
                        if (x, y) in body
                        else "."
                        for x in range(WIDTH)
                    )
                    for y in range(HEIGHT)
                ]
                latency = sorted(side["latency"])
                sides.append(
                    {
                        "ref": side["ref"],
                        "board": rows,
                        "alive": game.alive,
                        "ticks": game.ticks,
                        "score": game.score,
                        "length": len(game.body),
                        "games": side["games"],
                        "total_moves": side["moves"],
                        "total_apples": side["apples"],
                        "best_apples": side["best_apples"],
                        "best_moves": side["best_moves"],
                        "legal_rate": side["legal"] / max(1, side["decisions"]),
                        "ms": latency[len(latency) // 2] if latency else None,
                        "last_death": side["last_death"],
                    }
                )
            return {
                "running": self.running,
                "speed": self.speed,
                "sides": sides,
                "width": WIDTH,
                "height": HEIGHT,
                "error": self.error,
            }


# ----------------------------------------------------------------------------- state


class Studio:
    def __init__(self, workspace, bootstrap=None):
        self.workspace = workspace
        for name in ("datasets", "runs", "evals", "jobs"):
            (workspace / name).mkdir(parents=True, exist_ok=True)
        self.playground = Playground(workspace)
        self.arena = Arena(workspace, self.playground)
        self.jobs = Jobs(workspace, self.pause_gpu)
        self.bootstrap = bootstrap or Bootstrap(workspace, download=False, fetch_examples=False)
        self.account = Account()

    def exports(self):
        out = []
        for path in sorted((self.workspace / "exports").glob("*/export.json")):
            report = engine.read_json(path)
            if report:
                report["path"] = shown_path(path.parent, self.workspace.parent)
                out.append(report)
        # NoulXP packages live with their runs (runs/<id>/noulxp), which publish them.
        for path in sorted(folders(self.workspace / "runs")):
            entry = noulxp_package.listing(path, lambda p: shown_path(p, self.workspace.parent))
            if entry:
                out.append(entry)
        return out

    def noulxp(self, run_id):
        """What the publish dialog needs: the run's package, and whether one can be built."""
        run_dir = self.workspace / "runs" / run_id
        listing = noulxp_package.listing(run_dir, lambda p: shown_path(p, self.workspace.parent))
        kind = kinds.detect(run_dir / "model") or "laya"
        missing = noulxp_package.missing_tooling(kind)
        return {
            "package": listing if listing and listing.get("state") == "passed" else None,
            "last": listing,
            "kind": kind,
            "tooling": noulxp_package.tooling_message(missing, kind) if missing else None,
            "test_rows": noulxp_package.TEST_ROWS,
        }

    def pause_gpu(self):
        """Give a starting job the whole GPU: stop the arena, drop warm models."""
        self.arena.stop()
        self.playground.unload()

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
        for path in sorted(folders(self.workspace / "datasets"), reverse=True):
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
        for path in folders(self.workspace / "runs"):
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
        from . import families

        memory, accelerator = self._fit_inputs()
        fit = {m["repo"].lower(): m for m in families.trainable(memory, accelerator)}

        def assessed(repo, kind="laya"):
            known = fit.get(repo.lower()) or {}
            return {
                "kind": known.get("kind") or kind,
                "fit": known.get("fit"),
                "needed_gb": known.get("needed_gb"),
                "warnings": known.get("warnings") or [],
            }

        base = [
            {
                "ref": f"hub:{repo}",
                "repo": repo,
                "description": desc,
                "cached": engine.hub_cached(repo),
                **assessed(repo),
            }
            for repo, desc in engine.BASE_MODELS.items()
        ]
        # Every other catalogue model the studio trains (Julia 1, Decider), downloaded or not:
        # nothing is hidden, and one too big for this machine says so.
        listed = {repo.lower() for repo in engine.BASE_MODELS}
        for model in fit.values():
            if model["repo"].lower() in listed:
                continue
            size = model["params_b"]
            params = f"{size:.1f}B" if size >= 1 else f"{round(size * 1000)}M"
            base.append(
                {
                    "ref": f"hub:{model['repo']}",
                    "repo": model["repo"],
                    "description": f"{model['maker']} · {kinds.NAME[model['kind']]} · {params}"
                    + (f" · {model['note']}" if model.get("note") else ""),
                    "cached": engine.hub_cached(model["repo"]),
                    **assessed(model["repo"], model["kind"]),
                }
            )
        base += [
            {
                "ref": f"hub:{repo}",
                "repo": repo,
                "description": desc,
                "cached": engine.hub_cached(repo),
                "demo": True,
            }
            for repo, desc in engine.DEMO_MODELS.items()
        ]
        from . import families

        base += [
            {
                "ref": entry["ref"],
                "repo": f"{entry['repo']} (imported)",
                "description": f"Imported from {entry['source']}"
                + ("" if entry.get("trainable") else " · this family trains in a later version"),
                "cached": bool(entry.get("trainable")),
                "imported": True,
                "kind": entry.get("kind") or ("laya" if entry.get("trainable") else None),
                "warnings": [],
            }
            for entry in families.imports(self.workspace)
        ]
        tuned = [
            {
                "ref": f"run:{r['id']}",
                "name": r["name"],
                "kind": r.get("kind") or "laya",
                "base_model": r["base_model"],
                "dataset": r["dataset"],
                "dataset_name": r.get("dataset_name"),
                "method": (r.get("hyperparameters") or {}).get("method"),
                "accuracy": r.get("accuracy"),
                "baseline_accuracy": r.get("baseline_accuracy"),
                "p_value": r.get("p_value"),
                "path": shown_path(
                    self.workspace / "runs" / r["id"] / "model", self.workspace.parent
                ),
                "created": r["created"],
            }
            for r in self.runs()
            if r["has_model"] and r["state"] == "done"
        ]
        return base, tuned

    def library(self):
        """The Models page: your fine-tunes, the base checkpoints and exports, with sizes."""
        base, tuned = self.models()
        for model in base:
            model["size_bytes"] = None
            if model.get("cached"):
                try:
                    path = engine.resolve_model_ref(model["ref"], self.workspace)
                    model["size_bytes"] = dir_size(path)
                except (FileNotFoundError, ValueError, OSError):
                    pass
        from .export import KIND_TARGETS

        for model in tuned:
            model["size_bytes"] = dir_size(self.workspace / "runs" / model["ref"][4:] / "model")
            model["targets"] = list(KIND_TARGETS.get(model["kind"], ()))
        return {"base": base, "finetuned": tuned, "exports": self.exports()}

    def _fit_inputs(self):
        machine = getattr(self.bootstrap, "machine", None)
        memory = (machine.usable_gpu_gb or machine.memory_gb) if machine else None
        return memory, (machine.accelerator or machine.backend) if machine else None

    def families(self):
        from . import families

        memory, accelerator = self._fit_inputs()
        data = families.catalogue(memory, accelerator)
        imported = {e["repo"].lower() for e in families.imports(self.workspace)}
        for family in data["families"]:
            for model in family["models"]:
                model["downloaded"] = engine.hub_cached(model["repo"])
                model["imported"] = (model.get("registry") or "").lower() in imported
        data["machine"] = {"memory_gb": memory, "accelerator": accelerator}
        return data

    def registry(self, query):
        from . import families

        memory, accelerator = self._fit_inputs()
        return families.registry_search(query, memory, accelerator)

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
            "job_count": self.jobs.count(),
            "exports": self.exports(),
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
            "hyperparameters_by_kind": {k: engine.hyperparameters(k) for k in kinds.KINDS},
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
        kind = kinds.detect(model_dir)
        if kind == kinds.JULIA:
            from . import julia

            report = julia.analyze(dataset_id, model_dir, self.workspace)
        elif kind == kinds.DECIDER:
            from . import decider

            report = decider.analyze(dataset_id, model_dir, self.workspace)
        else:
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
        if kind == "publish" and not self.account.status().get("signed_in"):
            raise ApiError(
                HTTPStatus.UNAUTHORIZED,
                "Sign in to System One Models first (the button at the foot of the sidebar).",
            )
        stamp = time.strftime("%m%d-%H%M%S")
        if kind == "train":
            # The checks every fine-tune starts with, here and in the cloud: cloud.prepare_run.
            # Only what the Train button sends: a run's id, bounds, limits and exports are a
            # run file's (layastudio train --config), which the studio does not take here.
            spec = {k: body[k] for k in TRAIN_FIELDS if k in body}
            try:
                run = cloud.prepare_run(spec, self.workspace, stamp)
            except cloud.Refused as error:
                raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from None
            try:
                return self.jobs.start(
                    "train", run["job"], run["run_id"], f"Fine-tune: {run['name']}"
                )
            except Exception:
                shutil.rmtree(self.workspace / "runs" / run["run_id"], ignore_errors=True)
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
        if kind == "export":
            model_dir = engine.resolve_model_ref(body["model"], self.workspace)
            item = {k: body.get(k) for k in ("target", "precision", "gguf", "test_rows")}
            item["target"] = item["target"] or "onnx"
            if item["target"] == "noulxp":  # a run, of a family NoulXP has an exporter for
                self.noulxp_ready(body["model"], tooling=False)
            # The rules a run's exports follow too (cloud.check_export).
            try:
                spec = cloud.check_export(
                    kinds.detect(model_dir) or kinds.LAYA, item, body["model"]
                )
            except cloud.Unavailable as error:
                raise ApiError(HTTPStatus.CONFLICT, str(error)) from None
            except cloud.Refused as error:
                raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from None
            target, precision = spec["target"], spec["precision"]
            label = target.upper() + ("" if precision == "float" else f" ({precision})")
            if target == "noulxp":
                label = "a NoulXP package"
            return self.jobs.start(
                "export", spec, f"export-{stamp}", f"Export {modelname(body['model'])} to {label}"
            )
        if kind == "publish":
            engine.resolve_model_ref(body["model"], self.workspace)
            repo = (body.get("repo") or "").strip() or None
            if repo and repo.count("/") != 1:
                raise ApiError(HTTPStatus.BAD_REQUEST, "The repository is namespace/name")
            include = bool(body.get("noulxp", True))
            if include:
                run_id = noulxp_package.locate(body["model"], self.workspace)[0]
                if not self.noulxp(run_id)["package"]:  # it is built first: can it be?
                    self.noulxp_ready(body["model"], publishing=True)
            return self.jobs.start(
                "publish",
                {
                    "model": body["model"],
                    "repo": repo,
                    "private": bool(body.get("private")),
                    "noulxp": include,
                },
                f"publish-{stamp}",
                f"Publish {modelname(body['model'])} to System One",
            )
        if kind == "download":
            from . import families

            repo = body.get("repo_id")
            known = families.find(repo or "")
            if repo not in engine.BASE_MODELS and not known:
                raise ApiError(HTTPStatus.BAD_REQUEST, "Unknown model")
            if known and known.licence_kind == "closed":
                raise ApiError(HTTPStatus.BAD_REQUEST, f"{repo} publishes no weights")
            return self.jobs.start(
                "download",
                {"repo_id": known.repo if known else repo},
                f"download-{stamp}",
                f"Download {repo}",
            )
        if kind == "import":
            repo = (body.get("repo") or "").strip().lower()
            if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*", repo):
                raise ApiError(HTTPStatus.BAD_REQUEST, "The repository is namespace/name")
            return self.jobs.start(
                "import",
                {"repo": repo},
                f"import-{stamp}",
                f"Import {repo} from systemonemodels.tech",
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

    def noulxp_ready(self, ref, publishing=False, tooling=True):
        """Refuse up front what a NoulXP build would refuse: not a run, a family without a
        NoulXP exporter, or a machine without the tooling (unless the caller checks it)."""
        try:
            _, _, model_dir, run = noulxp_package.locate(ref, self.workspace)
        except ValueError as error:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from None
        family, entry = noulxp_package.support(run, self.workspace, model_dir)
        skip = " Or publish without a NoulXP package." if publishing else ""
        if entry["status"] != "ready":
            raise ApiError(HTTPStatus.BAD_REQUEST, noulxp_package.refusal(family) + skip)
        kind = kinds.detect(model_dir) or "laya"
        missing = noulxp_package.missing_tooling(kind) if tooling else None
        if missing:
            raise ApiError(
                HTTPStatus.CONFLICT, noulxp_package.tooling_message(missing, kind) + skip
            )

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

    def arena_start(self, body):
        if self.jobs.active():
            raise ApiError(HTTPStatus.CONFLICT, "A job is running; the arena pauses for it.")
        refs = body.get("models") or []
        if not 1 <= len(refs) <= 2:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Pick one or two models")
        for ref in refs:
            resolve = engine.resolve_model_ref(ref, self.workspace)  # fails loudly if missing
            del resolve
        self.arena.start(refs, speed=max(1, min(20, int(body.get("speed", 8)))))
        return self.arena.snapshot()

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


def modelname(ref):
    return ref.split(":", 1)[-1].split("/")[-1].partition("@")[0]


def label_text(qdef, target):
    names = engine.option_names(qdef)
    if max(target) >= 0.999:
        return names[engine.argmax(target)]
    return ", ".join(f"{n} {p:.0%}" for n, p in zip(names, target) if p > 0)


# ----------------------------------------------------------------------------- http


class Handler(BaseHTTPRequestHandler):
    studio: Studio = None
    port = 8765
    server_version = "SystemOneStudio/1"

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

    def _send_doc(self, name):
        """Screenshots and the logo, when the studio runs from a checkout."""
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,60}\.(png|svg)", name):
            raise ApiError(HTTPStatus.NOT_FOUND, "Not found")
        path = PACKAGE.parent / "docs" / name
        if not path.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "Not found")
        kind = "image/svg+xml" if name.endswith(".svg") else "image/png"
        self._send(HTTPStatus.OK, path.read_bytes(), kind)

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
            if method == "GET" and len(parts) == 2 and parts[0] == "docs":
                return self._send_doc(parts[1])
            if not parts or parts[0] != "api":
                raise ApiError(HTTPStatus.NOT_FOUND, "Not found")
            route = parts[1:]
            if method == "GET":
                if route == ["state"]:
                    return self._send(HTTPStatus.OK, studio.overview())
                if route == ["arena"]:
                    return self._send(HTTPStatus.OK, studio.arena.snapshot())
                if route == ["account"]:
                    return self._send(HTTPStatus.OK, studio.account.status())
                if route == ["families"]:
                    return self._send(HTTPStatus.OK, studio.families())
                if route == ["models"]:
                    return self._send(HTTPStatus.OK, studio.library())
                if route == ["jobs"]:
                    return self._send(
                        HTTPStatus.OK, {"jobs": studio.jobs.list(100), "count": studio.jobs.count()}
                    )
                if route == ["registry"]:
                    return self._send(HTTPStatus.OK, {"items": studio.registry(query.get("q"))})
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
                if len(route) == 3 and route[0] == "runs" and route[2] == "noulxp":
                    return self._send(HTTPStatus.OK, studio.noulxp(engine.check_id(route[1])))
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
                if route == ["account", "login"]:
                    return self._send(HTTPStatus.OK, studio.account.start())
                if route == ["account", "cancel"]:
                    return self._send(HTTPStatus.OK, studio.account.cancel())
                if route == ["account", "logout"]:
                    return self._send(HTTPStatus.OK, studio.account.logout())
                if route == ["arena", "start"]:
                    return self._send(HTTPStatus.OK, studio.arena_start(body))
                if route == ["arena", "stop"]:
                    studio.arena.stop()
                    return self._send(HTTPStatus.OK, studio.arena.snapshot())
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
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["train"]:  # a fine-tune with no UI: training, exports, a result file
        from .cloud import main as train

        return train(argv[1:])
    parser = argparse.ArgumentParser(
        prog="layastudio",
        description="Fine-tune Laya on your own data, on your own machine",
        epilog="layastudio train --config run.json runs one fine-tune with no UI "
        "(layastudio train --help)",
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
        f"System One Studio  {url}\n"
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
        Handler.studio.arena.stop()
        Handler.studio.jobs.shutdown()
        server.server_close()


# ----------------------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>System One Studio</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%235a58ca'/%3E%3Ctext x='32' y='43' text-anchor='middle' font-family='Helvetica,Arial' font-size='30' font-weight='700' letter-spacing='-1' fill='%23ffffff'%3Els%3C/text%3E%3C/svg%3E">
<style>
/* Tokens: the System One Models website's palette and scale (apps/web/app/globals.css).
   Periwinkle #d8dcff / #aeadf0 are surfaces and edges; text uses darker tones of the same
   hue in light mode. Every colour below is a token, so dark mode redefines variables only. */
:root{
  color-scheme:light;
  --bg:#ffffff;--bg-subtle:#f7f7fb;--bg-sunken:#f1f1f7;--surface:#ffffff;--surface-raised:#ffffff;--surface-inset:#f8f8fc;
  --border:#e5e5ee;--border-strong:#d3d3e0;--border-focus:#5a58ca;
  --text:#16161c;--text-secondary:#45454f;--text-muted:#6c6c7a;--text-faint:#6f6f7d;
  --accent:#5a58ca;--accent-hover:#3c3ab6;--accent-fg:#ffffff;--accent-text:#4744bd;
  --accent-subtle-bg:#d8dcff;--accent-subtle-border:#aeadf0;
  --success-text:#1f6b45;--success-bg:#eaf6ef;--success-border:#bfe0cd;
  --danger-text:#a52a37;--danger-bg:#fdeff0;--danger-border:#f2c7cc;
  --warning-text:#8a5314;--warning-bg:#fdf3e7;--warning-border:#f0d8b6;
  --shadow-sm:0 1px 2px rgb(22 22 40 / .06),0 1px 3px rgb(22 22 40 / .04);
  --shadow-md:0 4px 16px rgb(22 22 40 / .07);
  --shadow-lg:0 16px 48px rgb(22 22 40 / .1);
  --scrim:rgb(22 22 40 / .42);
  --chart-base:#a7a7ba;
  --font-sans:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-mono:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --radius-sm:5px;--radius:8px;--radius-lg:12px;
  --side:224px;
}
@media (prefers-color-scheme:dark){:root{
  color-scheme:dark;
  --bg:#0d0e11;--bg-subtle:#121318;--bg-sunken:#0a0b0d;--surface:#17181d;--surface-raised:#1c1d23;--surface-inset:#121317;
  --border:#292a32;--border-strong:#3a3b45;--border-focus:#aeadf0;
  --text:#f2f2f5;--text-secondary:#c9cad4;--text-muted:#9a9ba7;--text-faint:#8a8b95;
  --accent:#aeadf0;--accent-hover:#c5c4f7;--accent-fg:#15141f;--accent-text:#bfbef5;
  --accent-subtle-bg:#26253a;--accent-subtle-border:#423f63;
  --success-text:#8fd3a8;--success-bg:#17241c;--success-border:#2f4a38;
  --danger-text:#f0a8a8;--danger-bg:#261a1c;--danger-border:#543034;
  --warning-text:#edb989;--warning-bg:#261e15;--warning-border:#533f28;
  --shadow-sm:0 1px 2px rgb(0 0 0 / .4);--shadow-md:0 4px 18px rgb(0 0 0 / .45);--shadow-lg:0 18px 54px rgb(0 0 0 / .55);
  --scrim:rgb(0 0 0 / .6);
  --chart-base:#62636f;
}}
/* The names the views and charts already use, mapped onto the tokens. */
:root{--ink:var(--text);--muted:var(--text-muted);--faint:var(--text-faint);--line:var(--border);--panel:var(--surface);
  --code:var(--bg-sunken);--accent-soft:var(--accent-subtle-bg);--on-accent:var(--accent-fg);
  --good:var(--success-text);--good-soft:var(--success-bg);--bad:var(--danger-text);--bad-soft:var(--danger-bg);
  --warn:var(--warning-text);--warn-soft:var(--warning-bg);--base:var(--chart-base);--ft:var(--accent)}

/* ---- base */
*,*::before,*::after{box-sizing:border-box}
[hidden]{display:none!important}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--font-sans);font-size:.875rem;line-height:1.6;
  font-feature-settings:"cv05" 1,"ss01" 1;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
a{color:var(--accent-text);text-decoration:none}
a:hover{text-decoration:underline;text-underline-offset:3px}
button{font:inherit;color:inherit}
:is(a,button,input,select,textarea,summary,[tabindex]):focus-visible{outline:2px solid var(--border-focus);outline-offset:2px}
::selection{background:var(--accent);color:var(--accent-fg)}
h1,h2,h3,h4{font-weight:600;letter-spacing:-.022em;line-height:1.2;margin:0}
h1{font-size:clamp(1.75rem,3vw,2.25rem);margin:0 0 10px}
h2{font-size:1rem;letter-spacing:-.01em;margin:0 0 12px}
h3{font-family:var(--font-mono);font-size:.6875rem;font-weight:500;letter-spacing:.08em;text-transform:uppercase;color:var(--text-faint);margin:18px 0 8px}
code,pre,kbd,.mono{font-family:var(--font-mono);font-feature-settings:normal}
code{font-size:.85em;color:var(--accent-text)}
.mono{font-size:.75rem}
pre{font-size:.75rem;line-height:1.7;background:var(--surface-inset);border:1px solid var(--border);border-radius:var(--radius);
  padding:12px 14px;overflow:auto;margin:8px 0;max-width:100%;color:var(--text-secondary)}
pre code{font-size:inherit;color:inherit}
.up{color:var(--success-text)}.down{color:var(--danger-text)}.muted{color:var(--text-muted)}.faint{color:var(--text-faint)}
.small{font-size:.75rem}
.spacer{flex:1}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor;flex-shrink:0;display:inline-block}
.pulse,.jobchip .dot{animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{50%{opacity:.3}}

/* ---- shell: a sidebar on wide screens, a top bar with a menu below 820px */
.shell{display:grid;grid-template-columns:var(--side) minmax(0,1fr);min-height:100vh}
.side{border-right:1px solid var(--border);background:var(--bg-subtle);z-index:30}
.side-in{position:sticky;top:0;height:100vh;height:100dvh;display:flex;flex-direction:column;gap:16px;padding:18px 12px 14px;overflow-y:auto}
.side-top{display:flex;align-items:center;flex-wrap:wrap;gap:12px;padding:0 6px}
.brand{display:inline-flex;align-items:center;gap:9px;color:var(--text);font-size:.9375rem;font-weight:600;letter-spacing:-.02em;white-space:nowrap}
.brand:hover{text-decoration:none}
.brand-mark{width:26px;height:26px;border-radius:7px;background:var(--accent);color:var(--accent-fg);display:grid;place-items:center;
  font-size:.8125rem;font-weight:700;letter-spacing:-.04em;flex-shrink:0}
.brand-word b{color:var(--accent-text);font-weight:600}
.menu-toggle{display:none;align-items:center;gap:7px;min-height:34px;padding:6px 11px;border:1px solid var(--border-strong);
  border-radius:var(--radius);background:var(--surface);color:var(--text);font-size:.8125rem;font-weight:550;cursor:pointer}
.menu-toggle svg{color:var(--text-muted)}
#jobchip{flex-basis:100%;min-width:0}
#jobchip:empty{display:none}
.jobchip{display:flex;align-items:center;gap:8px;padding:7px 10px;border:1px solid var(--accent-subtle-border);border-radius:var(--radius);
  background:var(--accent-subtle-bg);color:var(--accent-text);font-size:.75rem;font-weight:500;min-width:0}
.jobchip:hover{text-decoration:none;border-color:var(--border-focus)}
.jobchip .jt{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.jobchip .jp{font-family:var(--font-mono);font-size:.6875rem;white-space:nowrap}
.side-nav{display:flex;flex-direction:column;gap:2px;flex:1}
.side-nav>a{display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:var(--radius-sm);font-size:.8125rem;font-weight:500;color:var(--text-secondary)}
.side-nav>a:hover{background:var(--bg-sunken);color:var(--text);text-decoration:none}
.side-nav>a svg{flex-shrink:0;color:var(--text-faint)}
.side-nav>a.on{background:var(--accent-subtle-bg);color:var(--accent-text)}
.side-nav>a.on svg{color:var(--accent-text)}
.count{margin-left:auto;font-family:var(--font-mono);font-size:.6875rem;color:var(--text-faint);font-variant-numeric:tabular-nums}
.count:empty{display:none}
.count.live{min-width:20px;padding:1px 6px;border-radius:999px;background:var(--accent);color:var(--accent-fg);font-weight:600;text-align:center}
.side-foot{margin-top:auto;display:grid;gap:10px;padding:12px 6px 0;border-top:1px solid var(--border)}
.machine{display:grid;gap:1px;font-family:var(--font-mono);font-size:.6875rem;line-height:1.55;color:var(--text-faint);min-width:0}
.machine span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.acct{display:flex;align-items:center;gap:8px;width:100%;min-height:32px;padding:5px 9px;border:1px solid var(--border);border-radius:var(--radius);
  background:var(--surface);color:var(--text-secondary);font-size:.8125rem;font-weight:500;cursor:pointer;text-align:left}
.acct:hover{border-color:var(--border-strong);color:var(--text)}
.acct .avatar{width:20px;height:20px;border-radius:50%;background:var(--accent);color:var(--accent-fg);display:grid;place-items:center;font-size:.6875rem;font-weight:700;flex-shrink:0}
.acct .who{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.side-links{display:flex;gap:12px;flex-wrap:wrap;font-size:.75rem}
.side-links a{color:var(--text-faint)}
.side-links a:hover{color:var(--accent-text)}
.mainwrap{min-width:0}
main{display:block;width:100%;max-width:1180px;margin:0 auto;padding:32px 36px 72px;min-width:0}
main:focus{outline:none}
#setup{max-width:1180px;margin:0 auto;padding:24px 36px 0}

/* ---- page structure */
.eyebrow{display:flex;align-items:center;gap:9px;font-family:var(--font-mono);font-size:.6875rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent-text);margin:0 0 14px}
.tiny-square{width:6px;height:6px;background:var(--accent);flex-shrink:0}
.lead{font-size:.9375rem;line-height:1.75;color:var(--text-muted);margin:0 0 24px;max-width:72ch}
.page-head{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap;padding-bottom:24px;margin-bottom:24px;border-bottom:1px solid var(--border)}
.page-head h1{margin:0 0 8px}
.page-head .lead{margin:0}
.head-actions{display:flex;gap:10px;flex-wrap:wrap}
.section{margin:0 0 32px}
.section-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px}
.section-head h2{margin:0;font-size:1.125rem}
.hint{font-size:.8125rem;line-height:1.7;color:var(--text-muted);margin:0 0 14px;max-width:80ch}
.text-link{display:inline-flex;align-items:center;gap:6px;font-size:.8125rem;font-weight:500;color:var(--text-secondary);white-space:nowrap}
.text-link:hover{color:var(--accent-text);text-decoration:none}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:18px 20px;margin-bottom:16px;min-width:0}
.card h2 .faint{font-weight:400;font-size:.8125rem}
.grid{display:grid;gap:16px}.two{grid-template-columns:repeat(2,minmax(0,1fr))}.three{grid-template-columns:repeat(3,minmax(0,1fr))}
.grid>.card,.grid>.panel{margin-bottom:0}
.grid.top{align-items:start}
.grid+.card,.grid+.panel,.grid+.section{margin-top:16px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}

/* panels: a header line and compact rows, like the website's dashboard */
.panel{border:1px solid var(--border);border-radius:var(--radius-lg);background:var(--surface);overflow:hidden;margin-bottom:16px;min-width:0}
.panel>header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:12px 18px;border-bottom:1px solid var(--border)}
.panel>header h2{margin:0;font-size:.9375rem}
.panel-empty{margin:0;padding:16px 18px;font-size:.8125rem;color:var(--text-muted)}
.panel-note{margin:0;padding:10px 18px;border-top:1px solid var(--border);font-size:.8125rem;color:var(--text-muted)}
.panel-note.warn{color:var(--warning-text);background:var(--warning-bg)}
.rows{list-style:none;margin:0;padding:0}
.rowi{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px 16px;align-items:center;padding:12px 18px}
.rowi+.rowi{border-top:1px solid var(--border)}
.rowi-main{display:grid;gap:2px;min-width:0}
.rowi-name{font-size:.875rem;font-weight:600;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
a.rowi-name:hover{color:var(--accent-text);text-decoration:none}
.rowi-sub{font-size:.75rem;color:var(--text-muted);overflow-wrap:anywhere}
.rowi-meta{display:flex;align-items:center;justify-content:flex-end;gap:10px;flex-wrap:wrap;font-size:.75rem;color:var(--text-muted);font-variant-numeric:tabular-nums}
.rowi-err{font-size:.75rem;color:var(--danger-text);overflow-wrap:anywhere}
.rowi .bar{margin-top:6px;max-width:320px}
.facts{margin:0;padding:4px 18px 8px}
.facts>div{display:grid;grid-template-columns:118px minmax(0,1fr);gap:12px;padding:8px 0;border-bottom:1px solid var(--border);font-size:.8125rem}
.facts>div:last-child{border-bottom:0}
.facts dt{font-family:var(--font-mono);font-size:.6875rem;letter-spacing:.08em;text-transform:uppercase;color:var(--text-faint);padding-top:2px}
.facts dd{margin:0;color:var(--text-secondary);overflow-wrap:anywhere}

/* stats, like .admin-stat */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(160px,100%),1fr));gap:12px;margin-bottom:24px}
.stat{display:grid;gap:4px;align-content:start;padding:16px 16px 14px;border:1px solid var(--border);border-radius:var(--radius-lg);background:var(--surface);min-width:0}
.stat .k{font-family:var(--font-mono);font-size:.6875rem;letter-spacing:.08em;text-transform:uppercase;color:var(--text-faint)}
.stat .v{font-size:1.6rem;font-weight:600;letter-spacing:-.02em;line-height:1.2;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.stat .d{font-size:.75rem;color:var(--text-muted);overflow-wrap:anywhere}
.stat .d a{color:var(--text-secondary)}.stat .d a:hover{color:var(--accent-text)}

/* ---- controls, like .button, .pill and the website's forms */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;min-height:36px;padding:7px 14px;border:1px solid var(--border-strong);
  border-radius:var(--radius);background:var(--surface);color:var(--text);font:inherit;font-size:.8125rem;font-weight:550;line-height:1.3;
  white-space:nowrap;cursor:pointer;transition:background .15s ease,border-color .15s ease}
.btn:hover{background:var(--bg-subtle);text-decoration:none}
.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-fg)}
.btn.primary:hover{background:var(--accent-hover);border-color:var(--accent-hover)}
.btn.danger{background:var(--danger-bg);border-color:var(--danger-border);color:var(--danger-text)}
.btn.danger:hover:not(:disabled){border-color:var(--danger-text)}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn.small{min-height:28px;padding:4px 10px;font-size:.75rem}
.pill{display:inline-flex;align-items:center;gap:5px;font-family:var(--font-mono);font-size:.6875rem;font-weight:500;line-height:1;padding:4px 7px;
  border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--bg-subtle);color:var(--text-secondary);white-space:nowrap;vertical-align:middle}
.pill.accent,.pill.running,.pill.downloading,.pill.loading{background:var(--accent-subtle-bg);border-color:var(--accent-subtle-border);color:var(--accent-text)}
.pill.done,.pill.ready{background:var(--success-bg);border-color:var(--success-border);color:var(--success-text)}
.pill.failed,.pill.interrupted,.pill.bad{background:var(--danger-bg);border-color:var(--danger-border);color:var(--danger-text)}
.pill.cancelled,.pill.warn{background:var(--warning-bg);border-color:var(--warning-border);color:var(--warning-text)}
.pill.running::before,.pill.downloading::before,.pill.loading::before{content:"";width:5px;height:5px;border-radius:50%;background:currentColor;animation:pulse 1.2s ease-in-out infinite}
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 8px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--bg-subtle);
  color:var(--text-muted);font-family:var(--font-mono);font-size:.6875rem;white-space:nowrap}
label{display:block;font-size:.8125rem;font-weight:500;color:var(--text-secondary);margin:12px 0 6px}
input:not([type]),input[type=text],input[type=number],input[type=search],input[type=url],select,textarea{width:100%;min-height:36px;padding:7px 11px;
  border:1px solid var(--border-strong);border-radius:var(--radius);background:var(--surface);color:var(--text);font:.875rem/1.4 var(--font-sans);transition:border-color .15s ease}
textarea{font:.8125rem/1.65 var(--font-mono);min-height:120px;resize:vertical}
input::placeholder,textarea::placeholder{color:var(--text-faint)}
input[type=checkbox],input[type=radio]{accent-color:var(--accent);margin:0}
input[type=file]{font-size:.8125rem;color:var(--text-muted);max-width:100%}
input[type=file]::file-selector-button{font:inherit;font-weight:550;margin-right:10px;padding:5px 10px;border:1px solid var(--border-strong);
  border-radius:var(--radius);background:var(--surface);color:var(--text);cursor:pointer}
.check{display:inline-flex;gap:8px;align-items:center;margin:0;color:var(--text);font-weight:450}
.checks{display:flex;flex-wrap:wrap;gap:8px 18px}
.segmented{display:inline-flex;flex-wrap:wrap;gap:2px;padding:3px;margin:0 0 16px;border:1px solid var(--border);border-radius:var(--radius);background:var(--bg-subtle);max-width:100%}
.segmented button{border:0;background:none;padding:6px 12px;border-radius:var(--radius-sm);font-size:.8125rem;font-weight:500;color:var(--text-secondary);cursor:pointer}
.segmented button:hover{color:var(--text)}
.segmented button[aria-selected=true]{background:var(--surface);color:var(--text);box-shadow:var(--shadow-sm)}

/* ---- tables, like .admin-table */
table{width:100%;border-collapse:collapse;font-size:.8125rem;font-variant-numeric:tabular-nums}
th{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border);font-family:var(--font-mono);font-size:.6875rem;font-weight:500;
  letter-spacing:.08em;text-transform:uppercase;color:var(--text-faint)}
td{padding:9px 10px;border-bottom:1px solid var(--border);vertical-align:top;color:var(--text-secondary)}
td b,td a{color:var(--text)}
td a:hover{color:var(--accent-text)}
tr:last-child td{border-bottom:0}
.tablewrap{overflow-x:auto;max-width:100%}
.tablewrap th{white-space:nowrap}
.tablewrap>table.wide{min-width:560px}
.boxed{border:1px solid var(--border);border-radius:var(--radius-lg);background:var(--surface)}
.boxed th{background:var(--bg-subtle);padding:10px 14px}
.boxed td{padding:10px 14px}

/* ---- feedback */
.notice{border:1px solid var(--border);border-radius:var(--radius);padding:10px 13px;margin:8px 0;font-size:.8125rem;line-height:1.6;background:var(--bg-subtle);color:var(--text-secondary)}
.notice.bad{border-color:var(--danger-border);background:var(--danger-bg);color:var(--danger-text)}
.notice.good{border-color:var(--success-border);background:var(--success-bg);color:var(--success-text)}
.notice.warn{border-color:var(--warning-border);background:var(--warning-bg);color:var(--warning-text)}
.notice.info{border-color:var(--accent-subtle-border);background:var(--accent-subtle-bg);color:var(--accent-text)}
.empty{border:1px dashed var(--border-strong);border-radius:var(--radius-lg);padding:28px 22px;text-align:center;color:var(--text-muted);background:var(--bg-subtle);font-size:.875rem;line-height:1.7}
.empty b{color:var(--text)}
.empty .row{justify-content:center;margin-top:14px}
.toast{position:fixed;right:20px;bottom:20px;max-width:min(420px,calc(100vw - 40px));background:var(--text);color:var(--bg);padding:10px 14px;
  border-radius:var(--radius);box-shadow:var(--shadow-md);z-index:60;font-size:.8125rem}
.bar{height:6px;background:var(--bg-sunken);border-radius:3px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);border-radius:3px;transition:width .3s}
.hbar{display:grid;grid-template-columns:minmax(80px,180px) minmax(0,1fr) 48px;gap:10px;align-items:center;font-size:.8125rem;margin:4px 0}
.hbar .t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hbar .n{text-align:right;color:var(--text-muted);font-variant-numeric:tabular-nums}
.steps{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 14px}
.steps span{padding:4px 8px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--bg-subtle);color:var(--text-faint);font-family:var(--font-mono);font-size:.6875rem}
.steps span.done{background:var(--success-bg);border-color:var(--success-border);color:var(--success-text)}
.steps span.now{background:var(--accent-subtle-bg);border-color:var(--accent-subtle-border);color:var(--accent-text)}
details{margin-top:10px}
summary{cursor:pointer;color:var(--text-muted);font-size:.8125rem}
summary:hover{color:var(--text)}
.adv-group{grid-column:1/-1;border-top:1px solid var(--border);padding-top:12px;font-size:.8125rem}
.state{white-space:pre-wrap;word-break:break-word;max-height:7.5em;overflow:hidden}
.opt{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(150px,100%),1fr));gap:8px}
.choice{border:1px solid var(--border);border-radius:var(--radius);padding:10px 12px;cursor:pointer;background:var(--surface)}
.choice:hover{border-color:var(--border-strong)}
.choice.on{border-color:var(--accent-subtle-border);background:var(--accent-subtle-bg)}
.choice.on b{color:var(--accent-text)}
.choice b{display:block;font-size:.8125rem;font-weight:600}.choice span{font-size:.75rem;color:var(--text-muted)}
.warnlist{margin:6px 0 0;padding-left:16px;font-size:.75rem;color:var(--warning-text)}
.cm td,.cm th{text-align:center;padding:4px 6px;font-size:.75rem;border:1px solid var(--border)}
.cm th{font-family:var(--font-sans);letter-spacing:0;text-transform:none;color:var(--text-muted)}
.cm th.rowh{text-align:right}
svg text{fill:var(--text-muted);font-family:var(--font-mono);font-size:10.5px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.75rem;color:var(--text-muted)}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}

/* ---- setup progress */
.panel.setup{border-color:var(--accent-subtle-border);margin-bottom:0}
.panel.setup.failed{border-color:var(--danger-border)}
.setup-steps{list-style:none;margin:0;padding:6px 0}
.setup-steps li{display:grid;grid-template-columns:8px minmax(110px,190px) minmax(0,1fr);gap:12px;align-items:baseline;padding:5px 18px;font-size:.8125rem}
.setup-steps .sd{color:var(--text-muted);min-width:0;overflow-wrap:anywhere}
.setup-steps .bar{display:inline-block;width:120px;vertical-align:middle}
.sdot{width:7px;height:7px;border-radius:50%;background:var(--border-strong);display:inline-block}
li.done .sdot,.sdot.ok{background:var(--success-text)}
li.running .sdot{background:var(--accent);animation:pulse 1.2s ease-in-out infinite}
li.warning .sdot{background:var(--warning-text)}
li.failed .sdot,.sdot.off{background:var(--danger-text)}

/* ---- home */
.steps4{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--border);border-radius:var(--radius-lg);background:var(--surface)}
.steps4 li{display:grid;gap:4px;align-content:start;padding:16px 18px}
.steps4 li+li{border-left:1px solid var(--border)}
.steps4 .n{font-family:var(--font-mono);font-size:.6875rem;letter-spacing:.08em;color:var(--accent-text)}
.steps4 b{font-size:.875rem;font-weight:600}
.steps4 span:last-child{font-size:.8125rem;line-height:1.65;color:var(--text-muted)}
footer.site{margin-top:40px;padding-top:28px;border-top:1px solid var(--border)}
footer.site .cols{display:grid;grid-template-columns:minmax(0,1.5fr) repeat(3,minmax(0,1fr));gap:28px}
footer.site .word{display:block;font-size:1.125rem;font-weight:600;letter-spacing:-.02em;margin-bottom:8px}
footer.site .word b{color:var(--accent-text);font-weight:600}
footer.site p{margin:0;font-size:.8125rem;line-height:1.7;color:var(--text-muted);max-width:36ch}
footer.site h4{margin:0 0 10px;font-family:var(--font-mono);font-size:.6875rem;font-weight:500;letter-spacing:.12em;text-transform:uppercase;color:var(--text-faint)}
footer.site ul{list-style:none;margin:0;padding:0;display:grid;gap:7px;font-size:.8125rem}
footer.site ul a{color:var(--text-secondary)}
footer.site ul a:hover{color:var(--accent-text);text-decoration:none}
footer.site .legal{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-top:28px;padding-top:16px;border-top:1px solid var(--border);color:var(--text-faint);font-size:.75rem}
footer.site .legal a{color:var(--text-muted)}

/* ---- modal: sign in */
.modal{position:fixed;inset:0;background:var(--scrim);display:grid;place-items:center;z-index:100;padding:16px}
.modal-card{background:var(--surface-raised);border:1px solid var(--border);border-radius:var(--radius-lg);max-width:440px;width:100%;padding:22px;box-shadow:var(--shadow-lg)}
.modal-card h2{margin:0 0 8px;font-size:1.125rem}
.modal-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:18px}
.signin-code{font-family:var(--font-mono);font-size:1.75rem;letter-spacing:.14em;font-weight:600;text-align:center;padding:14px;border-radius:var(--radius);
  background:var(--accent-subtle-bg);border:1px solid var(--accent-subtle-border);color:var(--accent-text);margin:14px 0 10px;user-select:all}
.signin-wait{display:flex;align-items:center;gap:8px;color:var(--text-muted);font-size:.8125rem;margin-top:10px}
.signin-wait i{width:7px;height:7px;border-radius:50%;background:var(--accent);animation:pulse 1.2s ease-in-out infinite}

/* ---- snake arena */
.arena{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr));gap:16px}
.board{display:grid;gap:2px;background:var(--bg-sunken);padding:8px;border-radius:var(--radius)}
.board i{aspect-ratio:1;border-radius:2px;background:var(--border);display:block}
.board i.o{background:var(--accent);opacity:.55}
.board i.H{background:var(--accent)}
.board i.F{background:var(--success-text)}
.board i.dead{background:var(--danger-text);opacity:.5}
.arena .num{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;font-variant-numeric:tabular-nums}
.arena .num b{display:block;font-size:1.125rem;font-weight:600}
.arena .num span{font-family:var(--font-mono);font-size:.6875rem;letter-spacing:.04em;color:var(--text-faint)}

/* ---- models: compact rows, like the website's .model-row */
.mrow{display:grid;grid-template-columns:34px minmax(0,1fr) auto;gap:4px 14px;align-items:center;padding:14px 18px}
.mrow+.mrow{border-top:1px solid var(--border)}
.tile{display:grid;place-items:center;width:34px;height:34px;border-radius:9px;border:1px solid var(--accent-subtle-border);background:var(--accent-subtle-bg);
  color:var(--accent-text);font-family:var(--font-mono);font-size:.875rem;font-weight:600}
.tile.base{border-color:var(--border);background:var(--bg-subtle);color:var(--text-muted)}
.mono-name{font-family:var(--font-mono);font-size:.8125rem;font-weight:500;color:var(--text)}
a.mono-name:hover{color:var(--accent-text)}
.rowi-name .pill{margin-left:6px}
.path{font-size:.6875rem;color:var(--text-faint);overflow-wrap:anywhere}
.mrow-score{display:grid;justify-items:end;gap:3px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;font-size:.8125rem}
.mrow-score b{font-size:1rem;font-weight:600;color:var(--text)}
.mrow-score span.faint{font-size:.75rem}
.mrow-actions{grid-column:2/-1;display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:6px}
.mrow-actions:empty{display:none}
.joined{display:inline-flex}
.joined select{width:auto;min-height:28px;padding:3px 8px;font-size:.75rem;border-radius:var(--radius) 0 0 var(--radius)}
.joined .btn{border-radius:0 var(--radius) var(--radius) 0;border-left:0}
.mrow-exp{grid-column:2/-1;margin:6px 0 0}
.mrow-exp>summary{font-size:.75rem}
.mrow-exports{grid-column:2/-1;list-style:none;margin:8px 0 0;padding:0;display:grid;gap:6px}
.mrow-exports li{display:flex;flex-wrap:wrap;align-items:center;gap:4px 12px;padding:8px 10px;border:1px solid var(--border);border-radius:var(--radius);
  background:var(--bg-subtle);font-size:.75rem;color:var(--text-muted);font-variant-numeric:tabular-nums}
.mrow-exports .path{flex-basis:100%}
details.family{margin:0 0 10px;border:1px solid var(--border);border-radius:var(--radius-lg);background:var(--surface)}
details.family>summary{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 18px;list-style:none;color:var(--text);font-size:.875rem}
details.family>summary::-webkit-details-marker{display:none}
details.family>summary::before{content:"";width:6px;height:6px;margin:0 4px 0 2px;border-right:1.5px solid var(--text-faint);border-bottom:1.5px solid var(--text-faint);transform:rotate(-45deg);transition:transform .15s ease}
details.family[open]>summary::before{transform:rotate(45deg)}
details.family>summary b{font-weight:600}
details.family .family-body{padding:0 18px 12px;border-top:1px solid var(--border)}

/* ---- responsive */
@media (max-width:1100px){.three{grid-template-columns:repeat(2,minmax(0,1fr))}
  .steps4{grid-template-columns:repeat(2,minmax(0,1fr))}.steps4 li:nth-child(3){border-left:0}.steps4 li:nth-child(n+3){border-top:1px solid var(--border)}
  footer.site .cols{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:819.98px){
  .shell{display:block}
  .side{position:sticky;top:0;border-right:0;border-bottom:1px solid var(--border);background:var(--bg)}
  .side-in{position:static;height:auto;gap:0;padding:10px 16px;overflow:visible}
  .side-top{flex-wrap:nowrap;padding:0;gap:8px}
  #jobchip{flex-basis:auto;margin-left:auto}
  #jobchip .jt{display:none}
  .menu-toggle{display:inline-flex;margin-left:auto}
  #jobchip:not(:empty)+.menu-toggle{margin-left:0}
  .side-nav{display:none}
  .side.open .side-nav{display:flex;position:absolute;left:0;right:0;top:100%;padding:10px 12px 14px;background:var(--bg);
    border-bottom:1px solid var(--border);box-shadow:var(--shadow-md);max-height:calc(100dvh - 58px);overflow-y:auto}
  .side-nav>a{padding:9px 10px}
  main{padding:20px 16px 56px}
  #setup{padding:16px 16px 0}
  .two,.three{grid-template-columns:minmax(0,1fr)}
  .page-head{padding-bottom:20px;margin-bottom:20px}
  .facts>div{grid-template-columns:96px minmax(0,1fr)}
  .setup-steps li{grid-template-columns:8px minmax(0,1fr)}
  .setup-steps .sd{grid-column:2}
  .rowi{grid-template-columns:minmax(0,1fr)}
  .rowi-meta{justify-content:flex-start}
  .mrow{grid-template-columns:34px minmax(0,1fr);padding:14px 16px}
  .mrow-score{grid-column:2;justify-items:start;text-align:left;white-space:normal;display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 8px}
  .toast{left:16px;right:16px;bottom:16px;max-width:none}
}
@media (max-width:560px){.steps4{grid-template-columns:minmax(0,1fr)}.steps4 li+li{border-left:0;border-top:1px solid var(--border)}
  footer.site .cols{grid-template-columns:minmax(0,1fr)}}
@media (prefers-reduced-motion:reduce){*,*::before{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div class="shell">
  <aside class="side" id="side"><div class="side-in">
    <div class="side-top">
      <a class="brand" href="#/home" aria-label="System One Studio home"><span class="brand-mark" aria-hidden="true">s1</span><span class="brand-word">system one <b>studio</b></span></a>
      <span id="jobchip"></span>
      <button class="menu-toggle" id="menutoggle" type="button" aria-expanded="false" aria-controls="sidenav">Menu</button>
    </div>
    <nav class="side-nav" id="sidenav" aria-label="Studio"></nav>
  </div></aside>
  <div class="mainwrap"><div id="setup" hidden></div><main id="main" tabindex="-1"></main></div>
</div>
<div class="modal" id="signin" hidden role="dialog" aria-modal="true" aria-labelledby="signin-title">
  <div class="modal-card">
    <h2 id="signin-title">Sign in to System One Models</h2>
    <p class="muted" id="signin-lede">Publishing puts a model on systemonemodels.tech under your name. Sign in once; the
      <code>systemone</code> command in your terminal is signed in too.</p>
    <div id="signin-body"></div>
    <div class="modal-actions">
      <button class="btn" id="signin-cancel" type="button">Cancel</button>
      <button class="btn primary" id="signin-go" type="button">Get a sign-in code</button>
    </div>
  </div>
</div>
<div class="modal" id="publish" hidden role="dialog" aria-modal="true" aria-labelledby="publish-title">
  <div class="modal-card">
    <h2 id="publish-title">Publish to System One Models</h2>
    <p class="muted" id="publish-lede"></p>
    <label for="publish-repo">Repository, as namespace/name</label>
    <input id="publish-repo" type="text" autocomplete="off" spellcheck="false">
    <label class="check" style="margin-top:14px"><input type="checkbox" id="publish-noulxp" checked> Include a NoulXP package</label>
    <div id="publish-noulxp-note" style="margin-top:8px"></div>
    <div class="modal-actions">
      <button class="btn" id="publish-cancel" type="button">Cancel</button>
      <button class="btn primary" id="publish-go" type="button">Publish</button>
    </div>
  </div>
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

// ---------------------------------------------------------------- System One Models account
let ACCOUNT = null;
async function refreshAccount() {
  try { ACCOUNT = await api("/api/account"); } catch (e) { return null; }
  const chip = $("#acctchip");
  if (!chip) return ACCOUNT;
  chip.hidden = !ACCOUNT.available;
  const name = ACCOUNT.username || "signed in";
  chip.innerHTML = ACCOUNT.signed_in
    ? `<span class="avatar" aria-hidden="true">${esc(name.charAt(0).toUpperCase())}</span><span class="who">${esc(name)}</span>`
    : `<span class="who">Sign in to System One Models</span>`;
  chip.title = ACCOUNT.signed_in ? `Signed in to ${ACCOUNT.site}. Click to sign out.` : "Sign in to System One Models to publish";
  return ACCOUNT;
}
function signIn(onDone) {
  const modal = $("#signin"), body = $("#signin-body"), go = $("#signin-go");
  let polling = null, done = false;
  const close = async (cancelled) => {
    modal.hidden = true; clearInterval(polling);
    if (cancelled) { try { await api("/api/account/cancel", {method: "POST", body: {}}); } catch (e) { /* already over */ } }
  };
  body.innerHTML = ""; go.hidden = false; go.disabled = false; modal.hidden = false; go.focus();
  $("#signin-cancel").onclick = () => close(true);
  go.onclick = async () => {
    go.disabled = true;
    // Opened now, while the click still counts, so the browser does not block it.
    const tab = window.open("about:blank", "_blank");
    let a;
    try { a = await api("/api/account/login", {method: "POST", body: {}}); }
    catch (e) { if (tab) tab.close(); body.innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; go.disabled = false; return; }
    if (a.error || !a.pending) { if (tab) tab.close(); body.innerHTML = `<div class="notice bad">${esc(a.error || "The registry did not answer. Try again.")}</div>`; go.disabled = false; return; }
    const p = a.pending;
    if (tab) tab.location = p.verification_uri_complete;
    go.hidden = true;
    body.innerHTML = `<p>Approve this machine on the page that just opened. It shows this code:</p>
      <div class="signin-code">${esc(p.user_code)}</div>
      <p class="muted">No tab? <a href="${esc(p.verification_uri_complete)}" target="_blank" rel="noreferrer">Open the approval page</a></p>
      <div class="signin-wait"><i></i> Waiting for approval…</div>`;
    polling = setInterval(async () => {
      const s = await refreshAccount();
      if (!s || done) return;
      if (s.signed_in) { done = true; close(false); toast(`Signed in as ${s.username}`); if (onDone) onDone(); }
      else if (s.error) { clearInterval(polling); body.innerHTML = `<div class="notice bad">${esc(s.error)}</div>`; go.hidden = false; go.disabled = false; }
    }, 2000);
  };
}
async function withAccount(action) {
  const a = await refreshAccount();
  if (!a) return toast("Could not reach the studio");
  if (!a.available) return toast(a.detail);
  if (!a.signed_in) return signIn(action);
  action();
}

async function api(path, opts = {}) {
  const init = {method: opts.method || "GET", headers: {}};
  if (opts.body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.body); }
  const r = await fetch(path, init);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { const e = new Error(data.error || r.statusText); e.status = r.status; throw e; }
  return data;
}
function toast(msg, ms = 4000) {
  const t = document.createElement("div"); t.className = "toast"; t.setAttribute("role", "status"); t.textContent = msg;
  document.body.appendChild(t); setTimeout(() => t.remove(), ms);
}
// "?static=1" freezes the page after one render, for screenshots and headless capture.
const STATIC = new URLSearchParams(location.search).has("static");
function every(fn, ms) { if (STATIC) return null; const id = setInterval(fn, ms); timers.push(id); return id; }
function clearTimers() { timers.forEach(clearInterval); timers = []; }
function pill(state, text) { return `<span class="pill ${esc(state)}">${esc(text || state)}</span>`; }

// Small line icons (24-unit grid, drawn for this page), inherited colour.
const ICONS = {
  home: '<path d="M3.5 10.5 12 3.5l8.5 7"/><path d="M5.5 9v11h13V9"/><path d="M10 20v-5.5h4V20"/>',
  arena: '<rect x="3.5" y="3.5" width="17" height="17" rx="2.5"/><path d="M7.5 16.5h4v-5h5v-4"/><circle cx="16.5" cy="16.5" r="1"/>',
  datasets: '<ellipse cx="12" cy="6" rx="7.5" ry="2.5"/><path d="M4.5 6v12c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5V6"/><path d="M4.5 12c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5"/>',
  train: '<path d="M4 6.5h9m4 0h3M4 12h3m4 0h9M4 17.5h11m4 0h1"/><circle cx="15" cy="6.5" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="17.5" r="2"/>',
  runs: '<path d="M4 4v16h16"/><path d="m7.5 14.5 3.5-4 3 2.5 5-6"/>',
  playground: '<path d="M9.5 3.5h5M10.5 3.5V9l-5.3 8.9a1.7 1.7 0 0 0 1.5 2.6h10.6a1.7 1.7 0 0 0 1.5-2.6L13.5 9V3.5"/><path d="M7.7 15h8.6"/>',
  models: '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9z"/><path d="m4 7.5 8 4.5 8-4.5M12 12v9"/>',
  guide: '<path d="M12 6.5C10.5 5 8 4.5 4 4.5v14c4 0 6.5.5 8 2 1.5-1.5 4-2 8-2v-14c-4 0-6.5.5-8 2z"/><path d="M12 6.5v14"/>',
  jobs: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/><path d="M15.5 8.5V6A1.5 1.5 0 0 0 14 4.5H6A1.5 1.5 0 0 0 4.5 6v8A1.5 1.5 0 0 0 6 15.5h2.5"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
};
const icon = (name, size = 15) => `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;

// Dates, ages and sizes people can read.
function when(iso) {
  if (!iso) return "–";
  const d = new Date(iso);
  if (isNaN(d)) return String(iso);
  const year = d.getFullYear() !== new Date().getFullYear() ? {year: "numeric"} : {};
  return d.toLocaleDateString(undefined, {month: "short", day: "numeric", ...year}) + ", " + d.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit"});
}
function ago(iso) {
  const t = new Date(iso).getTime();
  if (!iso || isNaN(t)) return "";
  const seconds = Math.max(0, (Date.now() - t) / 1000);
  for (const [size, name] of [[31536000, "year"], [2592000, "month"], [604800, "week"], [86400, "day"], [3600, "hour"], [60, "minute"]]) {
    const n = Math.floor(seconds / size);
    if (n >= 1) return `${n} ${name}${n === 1 ? "" : "s"} ago`;
  }
  return "just now";
}
function bytes(n) {
  if (n == null) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return (i && n < 10 ? n.toFixed(1) : Math.round(n)) + " " + units[i];
}

// This machine, in words, whatever it is: an Apple silicon Mac on MLX, or anything on PyTorch.
function machineName(s) {
  const chip = s.chip && s.chip !== "unknown" ? s.chip : "This machine";
  return chip + (s.memory_gb ? ` · ${s.memory_gb} GB` : "");
}
function runtimeName(s) { return s.backend === "mlx" ? "MLX" : "PyTorch"; }
function runtimeLine(s) {
  if (s.backend === "mlx") return [s.mlx ? "MLX " + s.mlx : "MLX", s.laya_mlx && "laya-mlx " + s.laya_mlx].filter(Boolean).join(" · ");
  const torch = s.torch && !String(s.torch).startsWith("broken") ? "PyTorch " + s.torch : "PyTorch";
  return [torch, s.laya && !String(s.laya).startsWith("broken") && "laya " + s.laya].filter(Boolean).join(" · ");
}
function trainsOn(s) {
  if (s.backend === "mlx") return `MLX${s.mlx ? " " + s.mlx : ""} on the Apple GPU`;
  const torch = s.torch && !String(s.torch).startsWith("broken") ? " " + s.torch : "";
  const where = !s.device || String(s.device).startsWith("unavailable") ? "" : s.device === "CPU" ? " on the CPU" : ` on ${s.device}`;
  return `PyTorch${torch}${where}`;
}
function acceleratorLine(s) {
  const gpus = (s.gpus || []).map(g => g.name + (g.memory_gb && s.backend !== "mlx" ? ` (${g.memory_gb} GB)` : "")).join(", ");
  if (s.backend === "mlx") return (gpus || "Apple GPU") + " through Metal";
  if (!s.device || String(s.device).startsWith("unavailable")) return gpus || "Not detected yet";
  if (s.device !== "CPU") return s.device + (gpus && !s.device.includes(gpus) ? ` · ${gpus}` : "");
  return gpus ? `${gpus}, not usable by PyTorch here: training runs on the CPU` : "None found: training runs on the CPU";
}
function machineTitle(s) {
  return [`${s.cores} cores`, s.os, runtimeLine(s), s.usable_gpu_gb ? `${s.usable_gpu_gb} GB usable for training` : ""].filter(Boolean).join(" · ");
}

function renderSetup(setup) {
  const host = $("#setup");
  if (!setup || setup.ready) { host.hidden = true; return; }
  host.hidden = false;
  const failed = setup.state === "failed";
  host.innerHTML = `<section class="panel setup${failed ? " failed" : ""}"><header>
    <h2>${failed ? "Setup needs your attention" : "Setting up System One Studio"}</h2>
    <span class="muted small">${failed ? "" : "You can look around while this finishes · " + Math.round(setup.seconds) + "s"}</span></header>
    <ul class="setup-steps">${setup.steps.map(st => `<li class="${esc(st.state)}"><span class="sdot" title="${esc(st.state)}"></span>
      <span>${esc(st.title)}</span>
      <span class="sd">${esc(st.detail || (st.state === "running" ? "working…" : ""))}${st.progress != null ? ` <span class="bar"><i style="width:${(100 * st.progress).toFixed(0)}%"></i></span>` : ""}</span>
    </li>`).join("")}</ul></section>`;
}
function modelName(ref) {
  if (!ref) return "–";
  if (ref.startsWith("hub:")) return ref.slice(4).split("/").pop().split("@")[0];
  if (ref.startsWith("run:")) { const r = (OV?.runs || []).find(x => "run:" + x.id === ref); return r ? r.name : ref.slice(4); }
  if (ref.startsWith("path:")) { const m = (OV?.models || []).find(x => x.ref === ref); return m ? m.repo.replace(" (imported)", "") : ref; }
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
const PAGES = [
  ["home", "Home"], ["arena", "Arena"], ["datasets", "Datasets"], ["train", "Train"], ["runs", "Runs"],
  ["playground", "Playground"], ["models", "Models"], ["guide", "Guide"], ["jobs", "Jobs"],
];
function buildNav() {
  $("#sidenav").innerHTML = PAGES.map(([key, label]) =>
    `<a href="#/${key}" data-v="${key}">${icon(key)}<span>${esc(label)}</span><span class="count" id="count-${key}"></span></a>`).join("") + `
    <div class="side-foot">
      <div class="machine" id="syschip"><span>Checking this machine…</span></div>
      <button class="acct" id="acctchip" type="button" hidden>Sign in</button>
      <div class="side-links"><a href="https://systemonemodels.tech" target="_blank" rel="noreferrer">systemonemodels.tech</a><a href="https://github.com/biplovgautam/LayaStudio" target="_blank" rel="noreferrer">GitHub</a></div>
    </div>`;
  const toggle = $("#menutoggle");
  toggle.innerHTML = icon("menu") + "<span>Menu</span>";
  toggle.onclick = e => { e.stopPropagation(); setMenu(!$("#side").classList.contains("open")); };
  document.addEventListener("click", e => { if (!e.target.closest("#side")) setMenu(false); });
  document.addEventListener("keydown", e => { if (e.key === "Escape") setMenu(false); });
}
function setMenu(open) {
  $("#side").classList.toggle("open", open);
  $("#menutoggle").setAttribute("aria-expanded", String(open));
}
function setCount(key, n, live = "") {
  const el = $("#count-" + key);
  if (!el) return;
  el.textContent = n ? String(n) : "";
  el.classList.toggle("live", !!live);
  el.title = live;
}
async function refresh() {
  try { OV = await api("/api/state"); } catch (e) { return null; }
  const s = OV.system;
  const sys = $("#syschip");
  sys.innerHTML = s.ok ? `<span>${esc(machineName(s))}</span><span>${esc(runtimeLine(s))}</span>`
    : `<span>${esc(s.chip && s.chip !== "unknown" ? machineName(s) : "Setting up…")}</span>`;
  sys.title = s.ok ? machineTitle(s) : (s.note || "");
  renderSetup(s.setup);
  const training = OV.runs.filter(r => r.state === "running").length;
  const running = OV.jobs.filter(j => j.state === "running").length;
  setCount("datasets", OV.datasets.length);
  setCount("runs", training || OV.runs.length, training ? `${training} training now` : "");
  setCount("models", OV.finetuned.length);
  setCount("jobs", running || OV.job_count, running ? `${running} running now` : "");
  const active = OV.jobs.find(j => j.state === "running");
  const chip = $("#jobchip");
  if (active) {
    const last = active.last || {};
    let p = "running";
    if (last.type === "step" && last.updates) p = `${Math.round(100 * (last.fraction ?? last.step / last.updates))}%`;
    else if (last.type === "progress" && last.total) p = `${Math.round(100 * last.done / last.total)}%`;
    const href = active.kind === "train" ? "#/runs/" + encodeURIComponent(active.id) : "#/jobs/" + encodeURIComponent(active.id);
    chip.innerHTML = `<a class="jobchip" href="${esc(href)}" title="${esc(active.title)} · open it"><span class="dot"></span><span class="jt">${esc(active.title)}</span><span class="jp">${esc(p)}</span></a>`;
  } else chip.innerHTML = "";
  return OV;
}
const routes = {home: viewHome, arena: viewArena, datasets: viewDatasets, dataset: viewDataset, train: viewTrain, runs: viewRuns, run: viewRun, playground: viewPlayground, models: viewModels, guide: viewGuide, jobs: viewJobs, job: viewJob};
async function route() {
  clearTimers();
  const token = ++ROUTE;
  const [path, qs] = location.hash.replace(/^#\/?/, "").split("?");
  const [a, b] = path.split("/");
  let view = a || "home", arg = b ? decodeURIComponent(b) : null;
  if (view === "datasets" && arg) view = "dataset";
  if (view === "runs" && arg) view = "run";
  if (view === "jobs" && arg) view = "job";
  const here = routes[a] ? a : "home";
  $$(".side-nav a[data-v]").forEach(n => {
    const on = n.dataset.v === here;
    n.classList.toggle("on", on);
    if (on) n.setAttribute("aria-current", "page"); else n.removeAttribute("aria-current");
  });
  setMenu(false);
  await refresh();
  if (!current(token)) return;
  if (!OV) {  // the server is restarting, or not up yet: keep trying, do not crash a view
    main.innerHTML = `<div class="empty">Connecting to the studio…<br>
      <span class="faint">If this persists, start it again with <code>systemone run studio</code>.</span></div>`;
    every(async () => { if (await refresh()) route(); }, 1500);
    return;
  }
  try { await (routes[view] || viewHome)(arg, new URLSearchParams(qs || ""), token); }
  catch (e) { if (current(token)) main.innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }
  every(refresh, 3000);
}
window.addEventListener("hashchange", route);


// ------------------------------------------------------------------ home
// Measured on an Apple M4 with 16 GB (MLX), balanced recipe, held-out test rows.
const SHIPPED = [
  ["Snake moves", "4 directions", "15.8%", "98.8%", "18 min"],
  ["Emotion", "6 labels", "47.5%", "88.2%", "12 min"],
  ["Prompt injection", "yes / no", "70.7%", "95.7%", "6 min"],
  ["Banking77", "77 intents", "34.2%", "64.2%", "18 min"],
];
function tile(label, value, detail) {
  return `<div class="stat"><div class="k">${esc(label)}</div><div class="v">${value}</div><div class="d">${detail}</div></div>`;
}

async function viewHome() {
  const s = OV.system;
  const measured = OV.runs.filter(r => r.state === "done" && r.accuracy != null && r.baseline_accuracy != null);
  const best = measured.slice().sort((a, b) => (b.accuracy - b.baseline_accuracy) - (a.accuracy - a.baseline_accuracy))[0];
  const training = OV.runs.filter(r => r.state === "running").length;
  const where = s.ok ? `, on this machine (${machineName(s)}, ${runtimeName(s)})` : ", on this machine";
  main.innerHTML = `
  <header class="page-head">
    <div>
      <p class="eyebrow"><span class="tiny-square"></span> System One Studio</p>
      <h1>Your decisions deserve your own model.</h1>
      <p class="lead">System One Studio fine-tunes open System One decision models on your own labeled data${esc(where)}. No cloud, no per-call bill, nothing leaves the machine, and every run proves whether it actually got better.</p>
    </div>
    <div class="head-actions">
      <a class="btn" href="#/arena">Watch it play</a>
      <a class="btn primary" href="#/datasets">Start fine-tuning</a>
    </div>
  </header>

  <div class="stats">
    ${tile("Datasets", OV.datasets.length, `<a href="#/datasets">${OV.datasets.length ? "labeled and split" : "add your first"}</a>`)}
    ${tile("Runs", OV.runs.length, `${measured.length} measured${training ? ` · ${training} training now` : ""}`)}
    ${tile("Fine-tuned models", OV.finetuned.length, `<a href="#/models">ready to use and export</a>`)}
    ${tile("Best gain", best ? `+${(100 * (best.accuracy - best.baseline_accuracy)).toFixed(1)} pts` : "–",
      best ? `<a href="#/runs/${esc(best.id)}">${esc(best.name)}</a>, ${pct(best.baseline_accuracy)} → ${pct(best.accuracy)}` : "no measured run yet")}
  </div>

  <div class="grid two top">
    <section class="panel">
      <header><h2>This machine</h2>${s.ok ? pill("done", "ready to train") : pill("warn", s.setup && s.setup.state === "failed" ? "needs attention" : "setting up")}</header>
      <dl class="facts">
        <div><dt>Processor</dt><dd>${esc(s.chip && s.chip !== "unknown" ? s.chip : "–")}${s.cores ? ` · ${s.cores} cores` : ""}</dd></div>
        <div><dt>Memory</dt><dd>${s.memory_gb ? `${s.memory_gb} GB` : "–"}${s.usable_gpu_gb ? ` · ${s.usable_gpu_gb} GB usable for training` : ""}</dd></div>
        <div><dt>Accelerator</dt><dd>${esc(acceleratorLine(s))}</dd></div>
        <div><dt>Trains with</dt><dd>${esc(runtimeLine(s))}</dd></div>
        <div><dt>System</dt><dd>${esc(s.os || "–")}${s.python ? ` · Python ${esc(s.python)}` : ""}</dd></div>
        <div><dt>Workspace</dt><dd><span class="mono">${esc(s.workspace)}/</span>${s.disk_free_gb != null ? ` · ${s.disk_free_gb} GB free` : ""}</dd></div>
      </dl>
      ${s.note ? `<p class="panel-note warn">${esc(s.note)}</p>` : ""}
    </section>
    <section class="panel">
      <header><h2>Recent runs</h2><a class="text-link" href="#/runs">All runs</a></header>
      ${OV.runs.length ? `<ul class="rows">${OV.runs.slice(0, 5).map(r => `<li class="rowi">
        <div class="rowi-main"><a class="rowi-name" href="#/runs/${esc(r.id)}">${esc(r.name)}</a>
          <span class="rowi-sub">${esc(r.dataset_name)} · ${esc(modelName(r.base_model))} · ${esc(ago(r.created))}</span></div>
        <div class="rowi-meta">${r.accuracy != null ? `<span>${pct(r.baseline_accuracy)} → <b>${pct(r.accuracy)}</b></span>` : ""}${pill(r.state)}</div>
      </li>`).join("")}</ul>` : `<p class="panel-empty">No runs yet. Pick a dataset and <a href="#/train">start one</a>: the base model is scored first, so you see the gain.</p>`}
    </section>
  </div>

  <section class="section" style="margin-top:32px">
    <div class="section-head"><h2>How a run works</h2><a class="text-link" href="#/guide">Read the guide</a></div>
    <ol class="steps4">
      <li><span class="n">01</span><b>Bring your decisions</b><span>JSONL or CSV, with the same questions you already ask.</span></li>
      <li><span class="n">02</span><b>Fine-tune here</b><span>LoRA plus the decision head: MLX on Apple silicon, PyTorch on NVIDIA, AMD, Intel or the CPU.</span></li>
      <li><span class="n">03</span><b>Prove it</b><span>Both models on the same untouched test rows, with a significance test.</span></li>
      <li><span class="n">04</span><b>Ship it</b><span>A standard checkpoint that loads on MLX and PyTorch, or an ONNX or Core ML export.</span></li>
    </ol>
  </section>

  <div class="grid two">
    <section class="section" style="margin:0">
      <div class="section-head"><h2>Measured, not promised</h2></div>
      <p class="hint">Base and fine-tuned model on the same held-out rows, balanced recipe, on an Apple M4 with 16 GB. Fine-tuning does not change inference speed: the adapters are merged into the weights.</p>
      <div class="tablewrap boxed"><table>
        <tr><th>Task</th><th>Answers</th><th>Before</th><th>After</th><th>Time</th></tr>
        ${SHIPPED.map(r => `<tr><td>${esc(r[0])}</td><td>${esc(r[1])}</td><td>${esc(r[2])}</td><td><b class="up">${esc(r[3])}</b></td><td>${esc(r[4])}</td></tr>`).join("")}
      </table></div>
    </section>
    <section class="section" style="margin:0">
      <div class="section-head"><h2>Does it really learn?</h2><a class="text-link" href="#/arena">Open the arena</a></div>
      <p class="hint">Snake, played unassisted: the model sees the board and four directions, its top answer is executed, and an illegal move ends the round. No hints, no safety layer.</p>
      <div class="tablewrap boxed"><table>
        <tr><th></th><th>Moves</th><th>Apples</th><th>Legal</th></tr>
        <tr><td>Base 322M</td><td>1.0</td><td>0.0</td><td>0%</td></tr>
        <tr><td><b>Fine-tuned</b></td><td><b>169</b></td><td><b>19.8</b></td><td><b>99.3%</b></td></tr>
        <tr><td>Planner (ceiling)</td><td>418</td><td>34.4</td><td>100%</td></tr>
      </table></div>
    </section>
  </div>

  <footer class="site">
    <div class="cols">
      <div>
        <span class="word">laya<b>studio</b></span>
        <p>Fine-tune System One decision models on your own data, on your own machine, and prove the result before you ship it.</p>
      </div>
      <div><h4>Project</h4><ul>
        <li><a href="https://github.com/biplovgautam/LayaStudio" target="_blank" rel="noreferrer">GitHub</a></li>
        <li><a href="https://github.com/biplovgautam/LayaStudio#readme" target="_blank" rel="noreferrer">Documentation</a></li>
        <li><a href="https://github.com/biplovgautam/LayaStudio/issues" target="_blank" rel="noreferrer">Issues</a></li>
        <li><a href="#/guide">How it works</a></li>
      </ul></div>
      <div><h4>System One Models</h4><ul>
        <li><a href="https://systemonemodels.tech" target="_blank" rel="noreferrer">systemonemodels.tech</a></li>
        <li><a href="https://systemonemodels.tech/system-one-models" target="_blank" rel="noreferrer">Every System One model</a></li>
        <li><a href="https://www.linkedin.com/company/system-one-models/" target="_blank" rel="noreferrer">LinkedIn</a></li>
        <li><a href="https://x.com/SystemoneModels" target="_blank" rel="noreferrer">X</a></li>
        <li><a href="https://huggingface.co/systemonemodels" target="_blank" rel="noreferrer">Hugging Face</a></li>
        <li><a href="https://www.instagram.com/systemonemodels.tech/" target="_blank" rel="noreferrer">Instagram</a></li>
        <li><a href="mailto:ceo@systemonemodels.tech">ceo@systemonemodels.tech</a></li>
      </ul></div>
      <div><h4>Built on</h4><ul>
        <li><a href="https://pypi.org/project/laya-mlx/" target="_blank" rel="noreferrer">laya-mlx</a></li>
        <li><a href="https://github.com/NandhaKishorM/laya" target="_blank" rel="noreferrer">Laya by Convai</a></li>
        <li><a href="https://github.com/ml-explore/mlx" target="_blank" rel="noreferrer">Apple MLX</a></li>
        <li><a href="https://pytorch.org" target="_blank" rel="noreferrer">PyTorch</a></li>
        <li><a href="https://huggingface.co/aac6fef" target="_blank" rel="noreferrer">Checkpoints</a></li>
      </ul></div>
    </div>
    <div class="legal">
      <span>Built by <a href="https://github.com/biplovgautam" target="_blank" rel="noreferrer">Biplov Gautam</a> · Apache-2.0, free to use, including commercially</span>
      <span>Your data, runs and checkpoints never leave this machine</span>
    </div>
  </footer>`;
}

// ------------------------------------------------------------------ snake arena
async function viewArena(_, params, token) {
  const models = OV.models.filter(m => m.cached).map(m => ({ref: m.ref, name: m.repo}))
    .concat(OV.finetuned.map(f => ({ref: f.ref, name: f.name})));
  const snakeRun = OV.finetuned.find(f => /snake/i.test(f.name))
    || models.find(m => /snake/i.test(m.name) && m.ref.startsWith("hub:"));
  const baseGuess = OV.models.find(m => m.cached && (!snakeRun || m.ref.includes("multilingual")));
  const pick = (id, chosen) => `<select id="${id}" style="max-width:280px">${models.map(m => `<option value="${esc(m.ref)}" ${chosen && chosen.ref === m.ref ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</select>`;
  main.innerHTML = `
  <h1>Snake arena</h1>
  <p class="lead">Two models play the same game side by side, live and unassisted: each one sees the board and four directions, its top answer is executed, and an illegal move ends the round. This is the difference fine-tuning makes, without a safety layer to hide behind.</p>
  <section class="card"><div class="row">
    ${pick("arenaA", baseGuess)} <span class="muted">vs</span> ${pick("arenaB", snakeRun)}
    <label style="margin:0;color:var(--ink);font-weight:450">speed <input type="number" id="arenaSpeed" value="8" min="1" max="20" style="width:70px"></label>
    <button class="btn primary" id="arenaGo">Start</button><button class="btn" id="arenaStop">Stop</button>
    <span class="muted" id="arenaMsg"></span>
  </div></section>
  <div class="arena" id="arenaBoards"></div>`;
  const draw = (state) => {
    if (state.error) $("#arenaMsg").textContent = state.error;
    $("#arenaBoards").innerHTML = (state.sides || []).map(side => `
      <section class="card">
        <div class="row" style="justify-content:space-between"><h2 style="margin:0">${esc(modelName(side.ref))}</h2>
          <span class="chip">${side.ms ? num(side.ms, 0) + " ms · " + (1000 / side.ms).toFixed(0) + " decisions/s" : "…"}</span></div>
        <div class="board" style="grid-template-columns:repeat(${state.width}, 1fr)">
          ${side.board.map(row => [...row].map(c => `<i class="${c === "." ? "" : c + (side.alive ? "" : " dead")}"></i>`).join("")).join("")}
        </div>
        <div class="num">
          <div><b>${side.ticks}</b><span>moves this round</span></div>
          <div><b>${side.score}</b><span>apples</span></div>
          <div><b>${pct(side.legal_rate, 0)}</b><span>legal moves</span></div>
          <div><b>${side.games}</b><span>rounds played</span></div>
          <div><b>${side.best_apples}</b><span>best apples</span></div>
        </div>
        ${side.alive ? "" : `<div class="notice bad" style="margin-bottom:0">Died: ${esc(side.last_death || "illegal move")}</div>`}
      </section>`).join("") || `<div class="empty">Pick two models and press start.</div>`;
  };
  const poll = async () => {
    try { const state = await api("/api/arena"); if (!current(token)) return; draw(state); } catch (e) { /* transient */ }
  };
  $("#arenaGo").onclick = async () => {
    $("#arenaMsg").textContent = "Loading models…";
    try {
      const state = await api("/api/arena/start", {method: "POST", body: {models: [$("#arenaA").value, $("#arenaB").value], speed: Number($("#arenaSpeed").value)}});
      $("#arenaMsg").textContent = ""; draw(state); every(poll, 300);
    } catch (e) { $("#arenaMsg").textContent = e.message; }
  };
  $("#arenaStop").onclick = async () => { try { draw(await api("/api/arena/stop", {method: "POST", body: {}})); clearTimers(); } catch (e) { toast(e.message); } };
  await poll();
  every(poll, 300);
}

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
  ${m.error_count ? `<details class="card"><summary>${m.error_count} rows were skipped — see why</summary><div class="tablewrap"><table>${m.errors.map(e => `<tr><td class="mono faint">${esc(e.file)}:${e.line}</td><td>${esc(e.error)}</td></tr>`).join("")}</table></div></details>` : ""}
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
  ${d.evals.length ? `<div class="tablewrap"><table><tr><th>Model</th><th>Accuracy</th><th>ECE</th><th>Decisions</th><th>p50 latency</th><th>When</th></tr>${d.evals.map(e => `<tr><td>${esc(modelName(e.model))}</td><td>${pct(e.accuracy)}</td><td>${num(e.ece)}</td><td>${e.n}</td><td>${num(e.latency_ms.p50, 1)} ms</td><td class="muted">${esc(when(e.created))}</td></tr>`).join("")}</table></div>` : `<div class="muted">None yet. Fine-tuning evaluates the base model first automatically.</div>`}
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
// Each checkpoint kind trains with its own recipes (kinds.py): Julia 1 with Laya's, plus a full
// fine-tune of its small encoder; Decider with LoRA on the attention and MLP projections.
const KIND_PRESETS = {
  laya: PRESETS,
  julia: {
    balanced: PRESETS.balanced,
    full: {title: "Full fine-tune", sub: "Every encoder layer and the head train (Julia 1 is 144M parameters). Lower learning rate.", hp: {method: "full", full_layers: 22, lr: 3e-5}},
    head: PRESETS.head,
  },
  decider: {
    lora: {title: "LoRA", sub: "Decider v11's own recipe shape: LoRA on the attention and MLP projections, a proper scoring rule on the option letters.", hp: {method: "lora"}},
    loraplus: {title: "LoRA+ ×4", sub: "The B matrices learn 4× faster: often better in few epochs.", hp: {method: "lora", loraplus_ratio: 4}},
    qlora: {title: "QLoRA (4-bit)", sub: "The frozen base in 4 bits: about half the memory, a little slower. NVIDIA GPUs and Apple silicon.", hp: {method: "lora", quantization: "4bit"}},
  },
};
const KIND_NAMES = {laya: "Laya", julia: "Julia 1", decider: "Decider"};
function advancedFields(kind, H) {
  const lora = `<div class="adv-group"><b>LoRA variants</b> <span class="muted">They combine freely, and all of them merge into the weights, so the exported model is the same size and speed.</span></div>
    ${select("dora", "DoRA", String(H.dora), [["false", "off"], ["true", "on — learn each row's magnitude and direction apart"]])}
    ${select("rslora", "Scaling", String(H.rslora), [["false", "alpha / rank (LoRA)"], ["true", "alpha / √rank (rsLoRA, for ranks above 16)"]])}
    ${field("loraplus_ratio", "LoRA+ ratio: B learns this many × faster (1 = off, 2–4 at the default rate)", H.loraplus_ratio)}
    <div id="lpwarn" style="grid-column:1/-1"></div>`;
  if (kind === "decider") return `
    ${field("epochs", "Epochs (max)", H.epochs)}${field("batch_size", "Rows per batch", H.batch_size)}${field("batch_tokens", "Tokens per batch", H.batch_tokens)}
    ${field("grad_accum", "Gradient accumulation", H.grad_accum)}${field("lr", "LoRA learning rate", H.lr)}${field("patience", "Early-stop patience (epochs)", H.patience)}
    ${field("lora_rank", "LoRA rank", H.lora_rank)}${field("lora_alpha", "LoRA alpha", H.lora_alpha)}${field("seed", "Seed", H.seed)}
    ${field("max_train_options", "Options per training row (gold kept; 0 = all)", H.max_train_options)}${field("max_state_tokens", "State tokens read in training", H.max_state_tokens)}${field("lora_dropout", "LoRA dropout", H.lora_dropout)}
    ${select("objective", "Objective", H.objective, [["proper", "proper — log + spherical scores (measured best)"], ["ce", "ce — cross-entropy on the letters (Decider's own)"]])}
    ${select("class_weighting", "Class weighting", H.class_weighting, [["none", "none"], ["balanced", "balanced (rare labels count more)"]])}
    ${select("precision", "Frozen weights precision", H.precision, [["bfloat16", "bfloat16 (less memory)"], ["float32", "float32"]])}
    ${select("quantization", "Frozen weights in 4 bits", H.quantization, [["none", "no (LoRA)"], ["4bit", "yes (QLoRA: NVIDIA or Apple silicon)"]])}
    ${select("shuffle_options", "Shuffle choice options", String(H.shuffle_options), [["true", "yes — learn labels, not positions"], ["false", "no"]])}
    ${lora}`;
  return `
    ${field("epochs", "Epochs (max)", H.epochs)}${field("batch_size", "Batch size", H.batch_size)}${field("grad_accum", "Gradient accumulation", H.grad_accum)}
    ${field("lr", "Encoder / LoRA learning rate", H.lr)}${field("head_lr", "Head learning rate", H.head_lr)}${field("patience", "Early-stop patience (epochs)", H.patience)}
    ${field("lora_rank", "LoRA rank", H.lora_rank)}${field("lora_alpha", "LoRA alpha", H.lora_alpha)}${field("seed", "Seed", H.seed)}
    ${select("objective", "Objective", H.objective, [["proper", "proper — log + spherical + RPS scores"], ["rlcd", "rlcd — upstream policy gradient + CE"], ["ce", "ce — cross-entropy only"]])}
    ${select("class_weighting", "Class weighting", H.class_weighting, [["none", "none"], ["balanced", "balanced (rare labels count more)"]])}
    ${select("precision", "Frozen weights precision", H.precision, [["bfloat16", "bfloat16 (less memory)"], ["float32", "float32"]])}
    ${select("shuffle_options", "Shuffle choice options", String(H.shuffle_options), [["true", "yes — learn labels, not positions"], ["false", "no"]])}
    ${lora}`;
}
async function viewTrain(_, params) {
  if (!OV.datasets.length) { main.innerHTML = `<h1>Fine-tune</h1><div class="empty">Create a dataset first. <a href="#/datasets">Go to datasets</a></div>`; return; }
  // Defaults follow the machine the studio is running on, not the one it was written on.
  const tuned = OV.system.recommended || {};
  const byKind = OV.hyperparameters_by_kind || {laya: OV.hyperparameters};
  const defaults = kind => ({...(byKind[kind] || OV.hyperparameters), ...(kind !== "decider" && tuned.batch_size ? {batch_size: tuned.batch_size} : {})});
  // Every base the studio trains, downloaded or not, too big or not: nothing is hidden.
  const models = OV.models.filter(m => m.kind).concat(OV.finetuned.map(f => ({ref: f.ref, repo: f.name + " (fine-tuned)", description: "continue from " + modelName(f.base_model), cached: true, kind: f.kind || "laya", warnings: []})));
  const wanted = params.get("base");
  const first = models.find(m => m.ref === wanted && m.cached) || models.find(m => m.cached) || models[0];
  const fitTag = m => m.fit === "too-big" ? " · too big here" : m.fit === "qlora" ? " · 4-bit QLoRA here" : "";
  main.innerHTML = `
  <h1>Fine-tune</h1>
  <p class="lead">Adapts a System One model (Laya, Julia 1 or Decider) to your questions and labels, in its own prompt format. The run first scores the base model on your test split, trains with early stopping on the validation split, re-fits confidence calibration, then scores the fine-tuned model on the same test split.</p>
  <section class="card"><div class="grid two">
    <div><label>Dataset</label><select id="trds">${OV.datasets.map(d => `<option value="${esc(d.id)}" ${params.get("dataset") === d.id ? "selected" : ""}>${esc(d.name)} — ${d.rows.train} train rows</option>`).join("")}</select></div>
    <div><label>Base model</label><select id="trbase">${models.map(m => `<option value="${esc(m.ref)}" ${m.cached ? "" : "disabled"} ${first && m.ref === first.ref ? "selected" : ""}>${esc(m.repo)}${m.kind ? " · " + esc(KIND_NAMES[m.kind] || m.kind) : ""}${esc(fitTag(m))}${m.cached ? "" : " (download in Models)"}</option>`).join("")}</select><div class="muted" id="trbasedesc" style="font-size:12px;margin-top:4px"></div><div id="trbasewarn"></div></div>
  </div>
  <label>Recipe</label><div class="opt" id="presets"></div>
  <details><summary>Advanced settings</summary><div class="grid three" id="adv"></div></details>
  <div class="grid two" style="margin-top:6px"><div><label>Run name (optional)</label><input type="text" id="trname" placeholder="auto"></div>
  <div><label>&nbsp;</label><label style="display:flex;gap:8px;align-items:center;color:var(--ink);font-weight:450"><input type="checkbox" id="trbl" checked> Evaluate the base model first (cached after the first run)</label></div></div>
  <div class="row" style="margin-top:16px"><button class="btn primary" id="trgo">Start fine-tuning</button><span class="muted" id="trmsg"></span></div>
  </section>
  <section class="card"><h2>What to expect on this machine</h2><p class="muted" style="margin:0">${esc(machineName(OV.system))}, training with ${esc(trainsOn(OV.system))}.${tuned.note ? " " + esc(tuned.note) : ""}${OV.system.note ? " " + esc(OV.system.note) : ""} For scale: on a 16 GB Apple M4 with MLX, the balanced recipe trains the 421M English Laya at about 7 decisions per second (roughly 10 minutes for 1,000 examples × 4 epochs) with a peak under 3 GB of GPU memory. Julia 1 (144M) is lighter still. Decider (1.9B) wants a GPU with about 10 GB for LoRA, or 4-bit QLoRA below that. Training pauses the playground and the arena so the job has the machine to itself.</p></section>`;
  let preset = null, kind = null, H = null;
  // Measured on Laya: LoRA+ with B at 8e-4 trained well; at 3.2e-3 it collapsed to chance.
  const loraWarnings = () => {
    const rate = Number($("#hp-lr").value), ratio = Number($("#hp-loraplus_ratio").value || 1);
    const rank = Number($("#hp-lora_rank").value), alpha = Number($("#hp-lora_alpha").value);
    const notes = [];
    if (ratio > 1 && rate * ratio > 1e-3) notes.push(`LoRA+ would train the B matrices at ${(rate * ratio).toExponential(1)}. On Laya, ratio 16 at the default 2e-4 collapsed to chance; ratio 4 (8e-4) trained best. Lower the ratio or the learning rate.`);
    if ($("#hp-rslora").value === "true" && rank <= 16) notes.push(`rsLoRA makes the update √${rank} = ${Math.sqrt(rank).toFixed(1)}× stronger at rank ${rank}. It is for higher ranks; at rank ${rank}, alpha ${Math.round(alpha / Math.sqrt(rank))} gives the same strength as plain LoRA.`);
    $("#lpwarn").innerHTML = notes.map(n => `<div class="notice warn">${esc(n)}</div>`).join("");
  };
  const choose = key => {
    preset = key;
    $$("#presets .choice").forEach(x => x.classList.toggle("on", x.dataset.p === key));
    const recipe = KIND_PRESETS[kind][key].hp;
    $("#hp-lr").value = recipe.lr ?? H.lr;
    if ($("#hp-loraplus_ratio")) $("#hp-loraplus_ratio").value = recipe.loraplus_ratio ?? H.loraplus_ratio;
    if ($("#hp-quantization")) $("#hp-quantization").value = recipe.quantization ?? H.quantization;
    loraWarnings();
  };
  const setKind = next => {
    if (next === kind) return;
    kind = next; H = defaults(kind);
    const presets = KIND_PRESETS[kind] || PRESETS;
    $("#presets").innerHTML = Object.entries(presets).map(([k, p]) => `<div class="choice" data-p="${k}"><b>${esc(p.title)}</b><span>${esc(p.sub)}</span></div>`).join("");
    $("#adv").innerHTML = advancedFields(kind, H);
    $$("#presets .choice").forEach(c => c.onclick = () => choose(c.dataset.p));
    ["lr", "loraplus_ratio", "rslora", "lora_rank", "lora_alpha"].forEach(k => { const el = $("#hp-" + k); if (el) el.oninput = el.onchange = loraWarnings; });
    choose(Object.keys(presets)[0]);
  };
  const desc = () => {
    const m = models.find(x => x.ref === $("#trbase").value);
    $("#trbasedesc").textContent = m ? m.description : "";
    $("#trbasewarn").innerHTML = m && m.warnings && m.warnings.length ? `<ul class="warnlist">${m.warnings.map(w => `<li>${esc(w)}</li>`).join("")}</ul>` : "";
    setKind((m && m.kind) || "laya");
    // A model that fits this machine only in 4 bits starts on the 4-bit recipe.
    if (m && m.fit === "qlora" && KIND_PRESETS[kind].qlora) choose("qlora");
  };
  $("#trbase").onchange = desc; desc();
  $("#trgo").onclick = async () => {
    // Advanced fields hold every value (the recipe mirrors its learning rate into them);
    // the recipe then fixes the method and which layers adapt.
    const hp = {};
    $$("#adv [data-hp]").forEach(i => { let v = i.value; if (i.type === "number") v = Number(v); if (v === "true") v = true; if (v === "false") v = false; hp[i.dataset.hp] = v; });
    const {lr, loraplus_ratio, quantization, ...fixed} = KIND_PRESETS[kind][preset].hp; Object.assign(hp, fixed);
    $("#trgo").disabled = true; $("#trmsg").textContent = "Starting…";
    try {
      const r = await api("/api/jobs", {method: "POST", body: {kind: "train", dataset: $("#trds").value, base_model: $("#trbase").value, name: $("#trname").value, baseline: $("#trbl").checked, hyperparameters: hp}});
      location.hash = "#/runs/" + r.id;
    } catch (e) { $("#trmsg").textContent = e.message; $("#trgo").disabled = false; }
  };
}
function loraTag(h) {
  if (h.method !== "lora") return "";
  const ratio = Number(h.loraplus_ratio || 1);
  const tags = [h.dora ? "DoRA" : "", h.rslora ? "rsLoRA" : "", ratio !== 1 ? `LoRA+ ×${ratio}` : ""].filter(Boolean);
  return tags.length ? esc(" + " + tags.join(" + ")) : "";
}
function field(k, labelText, v) { return `<div><label>${esc(labelText)}</label><input type="number" step="any" data-hp="${k}" id="hp-${k}" value="${esc(v)}"></div>`; }
function select(k, labelText, v, opts) { return `<div><label>${esc(labelText)}</label><select data-hp="${k}" id="hp-${k}">${opts.map(([val, t]) => `<option value="${esc(val)}" ${String(val) === String(v) ? "selected" : ""}>${esc(t)}</option>`).join("")}</select></div>`; }

// ------------------------------------------------------------------ runs
async function viewRuns() {
  const runs = OV.runs;
  main.innerHTML = `<h1>Runs &amp; results</h1><p class="lead">Every fine-tuning run, with the base model and the fine-tuned model scored on the same held-out test rows.</p>
  <section class="card">${runs.length ? `<div class="tablewrap"><table><tr><th>Run</th><th>Dataset</th><th>Base</th><th>Status</th><th>Before</th><th>After</th><th>Change</th><th>Created</th></tr>
  ${runs.map(r => `<tr><td><a href="#/runs/${esc(r.id)}">${esc(r.name)}</a><div class="faint" style="font-size:12px">${esc(r.hyperparameters.method)}${loraTag(r.hyperparameters)} · ${esc(r.hyperparameters.objective)}</div></td><td>${esc(r.dataset_name)}</td><td>${esc(modelName(r.base_model))}</td><td>${pill(r.state)}</td><td>${pct(r.baseline_accuracy)}</td><td><b>${pct(r.accuracy)}</b></td><td>${delta(r.baseline_accuracy, r.accuracy)}${r.p_value != null ? `<div class="faint" style="font-size:11.5px">p = ${r.p_value < 0.001 ? "<0.001" : r.p_value.toFixed(3)}</div>` : ""}</td><td class="muted">${esc(r.created)}</td></tr>`).join("")}
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
    <div class="row" style="justify-content:space-between"><div><h1>${esc(r.name)}</h1><div class="muted">${esc(r.dataset_name)} · ${esc(modelName(r.base_model))} · ${esc(r.hyperparameters.method)}${r.hyperparameters.method === "lora" ? ` r${r.hyperparameters.lora_rank}${r.hyperparameters.lora_layers ? ", top " + r.hyperparameters.lora_layers + " layers" : ""}${loraTag(r.hyperparameters)}` : ""} · ${esc(r.hyperparameters.objective)}</div></div>
    <div class="row"><span id="rstate"></span><a class="btn" id="rplay" href="#/playground?run=${esc(r.id)}" hidden>Try in playground</a><select id="rfmt" hidden style="width:auto" aria-label="Export format">${exportOptions()}</select><button class="btn" id="rexport" hidden>Export</button><button class="btn" id="rpublish" hidden title="Push this checkpoint and its measured numbers to systemonemodels.tech">Publish to System One</button><button class="btn danger" id="rcancel" hidden>Cancel</button><button class="btn danger" id="rdel" hidden>Delete run</button></div></div>
    <section class="card" id="live" style="margin-top:16px"><div class="steps" id="rsteps"></div><div id="rprog"></div><div id="rchart" style="margin-top:12px"></div>
    <details><summary>Event log</summary><pre id="rlog" style="max-height:260px"></pre></details></section>
    <div id="results"></div>`;
    $("#rcancel").onclick = async () => { if (!confirm("Stop this run? The partial model is discarded.")) return; try { await api(`/api/jobs/${id}/cancel`, {method: "POST", body: {}}); } catch (e) { toast(e.message); } };
    $("#rexport").onclick = () => startExport("run:" + id, $("#rfmt").value);
    $("#rpublish").onclick = () => startPublish("run:" + id);
    $("#rdel").onclick = async () => { if (!confirm("Delete this run and its checkpoint?")) return; try { await api("/api/runs/" + id, {method: "DELETE", body: {}}); location.hash = "#/runs"; } catch (e) { toast(e.message); } };
  };
  const update = () => {
    const job = data.job;
    $("#rstate").innerHTML = pill(job.state);
    $("#rcancel").hidden = job.state !== "running"; $("#rdel").hidden = job.state === "running";
    $("#rplay").hidden = !(job.state === "done" && data.model_path);
    $("#rexport").hidden = $("#rfmt").hidden = $("#rpublish").hidden = $("#rplay").hidden;
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
      if (last.type === "step") { frac = last.fraction ?? last.step / last.updates; detail = `update ${last.step}${last.step <= last.updates ? "/" + last.updates : ""} · epoch ${last.epoch} · ${last.decisions_per_s} decisions/s · ${last.peak_gb} GB peak · ~${fmtTime(last.eta_s)} left`; }
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
    <div class="stat"><div class="k">Peak memory</div><div class="v">${tr.peak_memory_gb} GB</div><div class="d muted">${tr.backend === "torch" ? "peak on " + esc(tr.device || "PyTorch") : "MLX active allocations"}</div></div>
    <div class="stat"><div class="k">Calibration (val ECE)</div><div class="v">${num(tr.calibration.ece_calibrated)}</div><div class="d muted">uncalibrated ${num(tr.calibration.ece_uncalibrated)}</div></div>
  </div><details><summary>Hyperparameters and temperatures</summary><pre>${esc(JSON.stringify({hyperparameters: tr.hyperparameters, temperature: tr.calibration.temperature, temperature_by_options: tr.calibration.temperature_by_options}, null, 2))}</pre></details></section>` : ""}
  ${data.model_path ? `<section class="card"><h2>Use the fine-tuned model</h2><p class="muted" style="margin-top:0">A standard Laya checkpoint: FP16 safetensors with the original PyTorch parameter names, your questions and the refitted calibration. It loads in <code>laya-mlx</code> on Apple silicon and in the PyTorch <code>laya</code> package everywhere else. Ask the <b>same questions</b> it was trained on.</p>
<pre># from the repository root
${OV.system.backend === "mlx" ? "import json, laya_mlx as laya" : "import json, laya"}

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

// ------------------------------------------------------------------ jobs
const JOB_KINDS = {train: "fine-tune", evaluate: "evaluate", export: "export", publish: "publish", download: "download", import: "import", example: "example"};
function jobHref(j) { return (j.kind === "train" ? "#/runs/" : "#/jobs/") + encodeURIComponent(j.id); }
function jobFraction(j) {
  const last = j.last || {};
  if (last.type === "step" && last.updates) return last.fraction ?? last.step / last.updates;
  if (last.type === "progress" && last.total) return last.done / last.total;
  return null;
}
function jobStarted(j) { return `started ${when(j.created)}${ago(j.created) ? ` (${ago(j.created)})` : ""}`; }
function jobRow(j) {
  const running = j.state === "running", f = running ? jobFraction(j) : null;
  const doing = running && j.last && j.last.message ? j.last.message : "";
  return `<li class="rowi">
    <div class="rowi-main">
      <a class="rowi-name" href="${esc(jobHref(j))}">${esc(j.title || j.id)}</a>
      <span class="rowi-sub"><span class="mono">${esc(j.id)}</span> · ${esc(jobStarted(j))}</span>
      ${doing ? `<span class="rowi-sub">${esc(doing)}</span>` : ""}
      ${f != null ? `<div class="bar"><i style="width:${(100 * f).toFixed(1)}%"></i></div>` : ""}
      ${j.error && !running ? `<span class="rowi-err">${esc(j.error)}</span>` : ""}
    </div>
    <div class="rowi-meta">${pill("", JOB_KINDS[j.kind] || j.kind)}${pill(j.state)}</div>
  </li>`;
}

async function viewJobs(_, __, token) {
  main.innerHTML = `<h1>Jobs</h1>
  <p class="lead">Everything the studio runs in the background: fine-tunes, evaluations, downloads, imports, exports and publishes. One job runs at a time, and while it runs the playground and the arena wait.</p>
  <div id="joblist"><div class="muted">Reading the jobs…</div></div>`;
  let shown = "";
  const load = async () => {
    const r = await api("/api/jobs");
    if (!current(token)) return;
    const html = r.jobs.length
      ? `<section class="panel"><header><h2>${r.count} ${r.count === 1 ? "job" : "jobs"}</h2><span class="muted small">${r.count > r.jobs.length ? `the newest ${r.jobs.length}` : "newest first"}</span></header>
         <ul class="rows">${r.jobs.map(jobRow).join("")}</ul></section>`
      : `<div class="empty"><b>No jobs yet.</b><br>Fetching an example dataset, downloading a model, fine-tuning, evaluating, exporting and publishing each run as a job, and every job shows up here with its progress and log.
         <div class="row"><a class="btn primary" href="#/datasets">Get a dataset</a><a class="btn" href="#/models">Download a model</a></div></div>`;
    if (html !== shown) { $("#joblist").innerHTML = html; shown = html; }
  };
  await load();
  every(() => load().catch(() => { /* transient */ }), 2000);
}

async function viewJob(id, _, token) {
  const path = `/api/jobs/${encodeURIComponent(id)}`;
  let first;
  try { first = await api(path + "?since=0"); }
  catch (e) {
    if (!current(token)) return;
    main.innerHTML = `<p class="eyebrow"><span class="tiny-square"></span> <a href="#/jobs">Jobs</a></p><h1>No such job</h1>
      <div class="empty">There is no job called <span class="mono">${esc(id)}</span>${e.status === 404 ? "" : ` (${esc(e.message)})`}. Deleting a run deletes its job too.
      <div class="row"><a class="btn primary" href="#/jobs">All jobs</a></div></div>`;
    return;
  }
  if (!current(token)) return;
  const events = [];
  let next = 0, poller = null;
  const watched = first.job.state === "running";  // only a job seen finishing moves the page on
  main.innerHTML = `<p class="eyebrow"><span class="tiny-square"></span> <a href="#/jobs">Jobs</a></p>
  <div class="row" style="justify-content:space-between;align-items:flex-start"><div style="min-width:0"><h1 id="jt"></h1><div class="muted small" id="jmeta"></div></div><div class="row" id="jact"></div></div>
  <section class="card" style="margin-top:18px"><div id="jstate"></div><div id="jprog" style="margin-top:10px"></div><div id="jres"></div><pre id="jlog" style="max-height:340px"></pre></section>`;
  let drawn = null;
  const draw = job => {
    $("#jt").textContent = job.title || job.id;
    $("#jmeta").innerHTML = `${esc(JOB_KINDS[job.kind] || job.kind)} · ${esc(jobStarted(job))} · <span class="mono">${esc(job.id)}</span>`;
    if (drawn !== job.state) { drawn = job.state; drawActions(job); }
    $("#jstate").innerHTML = pill(job.state) + (job.error ? `<div class="notice bad">${esc(job.error)}</div>` : "");
    const f = job.state === "running" ? jobFraction(job) : null;
    $("#jprog").innerHTML = f != null ? `<div class="bar"><i style="width:${(100 * f).toFixed(1)}%"></i></div>` : "";
    const nx = events.find(e => e.type === "result" && e.target === "noulxp");
    $("#jres").innerHTML = nx ? noulxpNotice(nx) : "";
    $("#jlog").textContent = events.map(e => `${e.type.padEnd(9)} ${e.message || ""}${e.type === "progress" ? `${e.done}/${e.total}` : ""}${e.type === "result" ? JSON.stringify(e) : ""}`).join("\n") || "No events yet.";
  };
  const drawActions = job => {
    $("#jact").innerHTML = (job.kind === "train" ? `<a class="btn" href="#/runs/${esc(encodeURIComponent(job.id))}">Open the run</a>` : "")
      + (job.state === "running" ? `<button class="btn danger" id="jcancel" type="button">Cancel</button>` : "");
    const cancel = $("#jcancel");
    if (cancel) cancel.onclick = async () => {
      if (!confirm("Stop this job?")) return;
      try { await api(path + "/cancel", {method: "POST", body: {}}); } catch (e) { toast(e.message); }
    };
  };
  const apply = r => {
    events.push(...r.events); next = r.next;
    draw(r.job);
    if (r.job.state === "running") return;
    if (poller) clearInterval(poller);
    if (!watched) return;
    const res = events.find(e => e.type === "result");
    if (res && res.dataset) { toast("Dataset ready"); location.hash = "#/datasets/" + res.dataset; }
    else if (r.job.kind === "download" && r.job.state === "done") { toast("Model downloaded"); location.hash = "#/models"; }
    else if (r.job.kind === "evaluate" && r.job.state === "done") { toast("Evaluation finished"); history.back(); }
  };
  apply(first);
  if (first.job.state === "running") poller = every(async () => {
    try { const r = await api(`${path}?since=${next}`); if (current(token)) apply(r); } catch (e) { /* transient */ }
  }, 1000);
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
  const wanted = (params.get("models") || "").split(",").filter(Boolean);
  const checked = m => preset ? preset.models.includes(m.ref) : wanted.includes(m.ref);
  main.innerHTML = `<h1>Playground</h1><p class="lead">Ask base and fine-tuned models the same questions side by side. Models stay loaded between requests, so later answers show real latency.</p>
  <section class="card"><label>Models (up to 4)</label><div class="row" id="pgm">${models.map((m, i) => `<label style="display:flex;gap:6px;align-items:center;margin:0;color:var(--ink);font-weight:450"><input type="checkbox" value="${esc(m.ref)}" ${preset || wanted.length ? (checked(m) ? "checked" : "") : (i === 0 ? "checked" : "")}>${esc(m.name)}</label>`).join("") || `<span class="muted">No models available. Download one in Models.</span>`}</div>
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
const EXPORT_FORMATS = [["onnx:float", "ONNX · float"], ["onnx:int8", "ONNX · int8"], ["onnx:int4", "ONNX · int4"], ["coreml:float", "Core ML · float"], ["coreml:int8", "Core ML · int8"], ["coreml:int4", "Core ML · int4"], ["gguf:bf16", "GGUF · bf16 (llama.cpp)"], ["gguf:q8_0", "GGUF · q8_0 (llama.cpp)"], ["mlx:int4", "MLX-LM · int4"], ["mlx:int8", "MLX-LM · int8"], ["noulxp:float", "NoulXP package · checked"]];
// The formats a model's kind exports to (server: export.KIND_TARGETS).
function exportOptions(targets) { return EXPORT_FORMATS.filter(([v]) => !targets || targets.includes(v.split(":")[0])).map(([v, t]) => `<option value="${v}">${t}</option>`).join(""); }
async function startExport(ref, format) {
  const [target, precision] = format.split(":");
  try { const job = await api("/api/jobs", {method: "POST", body: {kind: "export", model: ref, target, precision}}); location.hash = "#/jobs/" + job.id; }
  catch (e) { toast(e.message); }
}
// A fine-tune goes up with its NoulXP package unless the box is unticked: the run's package
// if it passed its check, else one built and checked first. A package that fails is never sent.
function startPublish(ref) {
  withAccount(async () => {
    const modal = $("#publish"), repo = $("#publish-repo"), box = $("#publish-noulxp");
    const note = $("#publish-noulxp-note"), go = $("#publish-go");
    let info = null;
    try { info = await api(`/api/runs/${encodeURIComponent(ref.replace(/^run:/, ""))}/noulxp`); } catch (e) { /* explained below */ }
    $("#publish-lede").textContent = `The model goes up to systemonemodels.tech as ${ACCOUNT.username}, with a card built from this run's measurements.`;
    repo.value = ""; repo.placeholder = `${ACCOUNT.username}/<this run's name>`;
    const explain = () => {
      const p = info && info.package;
      if (!box.checked) note.innerHTML = `<p class="hint">The checkpoint goes up alone. The registry will not check it for NoulXP compatibility, and its card says so.</p>`;
      else if (p) note.innerHTML = `<p class="hint">The run's package goes up as the version's <code>noulxp/</code> folder: ${p.cases_passed}/${p.cases} cases reproduced on the CPU, max |Δp| ${dpText(p.max_abs_dp)}. The registry checks it again before it shows the model as NoulXP compatible.${p.test_rows ? ` Its conformance file holds ${p.test_rows} rows of your test split, which are published with it.` : ""}</p>`;
      else if (info && info.tooling) note.innerHTML = `<div class="notice warn">This run has no NoulXP package, and this machine cannot build one: ${esc(info.tooling)}</div>`;
      else note.innerHTML = `<p class="hint">This run has no NoulXP package yet, so one is built and checked on the CPU first, which takes a few minutes. ${info && info.test_rows ? `Its conformance file holds up to ${info.test_rows} rows of your test split, which are published with it.` : `Its conformance file holds NoulXP's own requests only: none of your dataset goes up with it.`} If the package does not pass, nothing is uploaded.</p>`;
    };
    box.checked = true; box.onchange = explain; explain();
    modal.hidden = false; repo.focus();
    $("#publish-cancel").onclick = () => { modal.hidden = true; };
    go.onclick = async () => {
      go.disabled = true;
      try {
        const job = await api("/api/jobs", {method: "POST", body: {kind: "publish", model: ref, repo: repo.value.trim() || null, noulxp: box.checked}});
        modal.hidden = true; location.hash = "#/jobs/" + job.id;
      } catch (e) { note.innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }
      finally { go.disabled = false; }
    };
  });
}
function exportLabel(x) {
  return ({onnx: "ONNX", coreml: "Core ML", noulxp: "NoulXP", gguf: "GGUF", mlx: "MLX-LM"}[x.target] || String(x.target).toUpperCase()) + (x.precision && x.precision !== "float" ? " · " + x.precision : "");
}
function dpText(v) { return v == null ? "–" : Number(v).toExponential(1); }
function leftOut(x) {
  const qs = Object.entries(x.questions_left_out || {});
  return qs.length ? `Not in the conformance file: ${qs.map(([q, why]) => `${q} (${why})`).join("; ")}.` : "";
}
// A NoulXP export: the package the run keeps (it passed its check), or the attempt that failed.
function noulxpItem(x) {
  const ok = x.state === "passed", left = leftOut(x);
  return `<li>${pill(ok ? "done" : "bad", "NoulXP")}
    ${x.cases != null ? `<span>${x.cases_passed}/${x.cases} cases reproduced on the CPU</span>` : ""}
    ${x.max_abs_dp != null ? `<span>max |Δp| ${dpText(x.max_abs_dp)}</span>` : ""}
    ${ok ? `<span>${x.test_rows || 0} test rows in its conformance file</span><span>goes up when you publish</span>` : `<span class="down">${esc(x.error || "did not pass its check")}</span>`}
    ${x.last_attempt ? `<span class="faint">a later rebuild failed: ${esc(x.last_attempt.error || "")}</span>` : ""}
    ${left ? `<span class="faint">${esc(left)}</span>` : ""}
    <span>${esc(when(x.created))}</span>
    ${x.path ? `<span class="mono path">${esc(x.path)}</span>` : ""}</li>`;
}
function noulxpNotice(r) {
  const cases = r.cases != null ? ` ${r.cases_passed}/${r.cases} cases reproduced on the CPU, max |Δp| ${dpText(r.max_abs_dp)}.` : "";
  const left = leftOut(r);
  return r.state === "passed"
    ? `<div class="notice good"><b>The NoulXP package passed.</b>${esc(cases)}${r.test_rows ? ` ${r.test_rows} rows of your test split are in its conformance file, next to NoulXP's own requests.` : ""} It is kept with the run and goes up with it when you publish.${left ? " " + esc(left) : ""}</div>`
    : `<div class="notice bad"><b>The NoulXP package did not pass.</b>${esc(cases)} It is kept for reading and is never published.${left ? " " + esc(left) : ""}</div>`;
}
function exportItem(x) {
  if (x.target === "noulxp") return noulxpItem(x);
  const ms = x.ms_per_decision || x.ms_per_decision_cpu;
  return `<li>${pill("accent", exportLabel(x))}
    <span>${x.size_mb != null ? bytes(x.size_mb * 2 ** 20) : "–"}</span>
    ${ms ? `<span>${num(ms, 1)} ms per decision${x.ms_per_decision ? "" : " on the CPU"}</span>` : ""}
    ${x.test ? `<span>${pct(x.test.accuracy_exported)} on ${x.test.rows} test rows</span>` : ""}
    ${x.verification ? `<span>${x.verification.same_answer}/${x.verification.rows ?? x.verification.decisions} same answers</span>` : ""}
    <span>${esc(when(x.created))}</span>
    <span class="mono path">${esc(x.path)}</span></li>`;
}

async function viewModels(_, __, token) {
  const fitPill = m => ({
    "fits": pill("done", "fits"),
    "qlora": pill("warn", "4-bit QLoRA"),
    "too-big": pill("bad", "too big here"),
    "not-trainable": pill("", "no weights"),
    "unknown": pill("", "unknown"),
  }[m.fit] || "");
  const trainerPill = t => t === "ready" ? pill("done", "trains here") : t === "partial" ? pill("done", "some train here") : t === "next" ? pill("running", "trainer coming next") : pill("", "trainer planned");
  const noulxpPill = n => !n ? "" : n.status === "ready" ? pill("done", "NoulXP: " + n.profile) : n.status === "trainer" ? pill("", "NoulXP with its trainer") : pill("", "no NoulXP profile yet");
  const gb = n => n == null ? "–" : `${n} GB`;
  const params = b => b >= 1 ? `${b.toFixed(b >= 10 ? 0 : 1)}B` : b >= 0.001 ? `${Math.round(b * 1000)}M` : `${Math.round(b * 1e6)}K`;
  main.innerHTML = `<h1>Models</h1>
  <p class="lead">Your fine-tuned models, the base checkpoints they start from, and every System One model family with what this machine can do with it. A model that will not fit here says so, and what to do instead.</p>
  <section class="section">
    <div class="section-head"><h2>Your fine-tuned models</h2><a class="text-link" href="#/train">Fine-tune another</a></div>
    <div id="mine"><div class="muted">Reading the workspace…</div></div>
  </section>
  <section class="section">
    <div class="section-head"><h2>Base models</h2><span class="muted small">what a fine-tune starts from</span></div>
    <div id="bases"></div>
  </section>
  <section class="section">
    <div class="section-head"><h2>Import from systemonemodels.tech</h2></div>
    <p class="hint">Any model published on the registry, yours or anyone's: import it, then train from it here.</p>
    <div class="row"><input id="regq" type="search" placeholder="Search models, e.g. laya, jev, snake" style="flex:1;min-width:0"><button class="btn" id="regsearch" type="button">Search</button></div>
    <div id="regout" style="margin-top:12px"></div>
  </section>
  <section class="section">
    <div class="section-head"><h2>Every System One model family</h2></div>
    <div id="families"><div class="muted">Reading the catalogue…</div></div>
  </section>
  <section class="section">
    <div class="section-head"><h2>Format and portability</h2></div>
    <p class="hint">Every fine-tune keeps its maker's own files, so it loads where the base model loads. A <b>Laya</b> checkpoint is <code>model.safetensors</code> (FP16, the original PyTorch parameter names), <code>rl_agent_config.json</code> (with refitted temperatures), <code>encoder/</code>, <code>tokenizer/</code>, <code>questions.json</code> and <code>laya_finetune.json</code> (provenance); it loads in <code>laya-mlx</code> on Apple silicon and in the PyTorch <code>laya</code> package everywhere else, and exports to ONNX and Core ML in float, int8 or int4. A <b>Julia 1</b> checkpoint is Supersonic Labs' layout (<code>julia_config.json</code>, <code>inference-policy.json</code>, float32 weights with the fitted temperature folded into the scorer). A <b>Decider</b> checkpoint is Mapika's (merged bfloat16 safetensors, <code>decider_config.json</code> with the fitted temperatures, <code>decider/</code> code) and exports to GGUF through a pinned llama.cpp and to MLX-LM in int4 or int8. All three get a NoulXP package: the open standard's package with a conformance file of the fine-tune's own answers, checked on the CPU and kept with the run, which publishes it so systemonemodels.tech can check it for NoulXP compatibility.</p>
  </section>`;

  const bindDownloads = root => $$("[data-dl]", root).forEach(b => b.onclick = async () => { try { const r = await api("/api/jobs", {method: "POST", body: {kind: "download", repo_id: b.dataset.dl}}); location.hash = "#/jobs/" + r.id; } catch (e) { toast(e.message); } });
  const bindImports = root => $$("[data-import]", root).forEach(b => b.onclick = async () => { try { const r = await api("/api/jobs", {method: "POST", body: {kind: "import", repo: b.dataset.import}}); location.hash = "#/jobs/" + r.id; } catch (e) { toast(e.message); } });
  const warnings = ws => ws && ws.length ? `<ul class="warnlist">${ws.map(w => `<li>${esc(w)}</li>`).join("")}</ul>` : "";

  try {
    const lib = await api("/api/models");
    if (!current(token)) return;
    const byModel = {};
    for (const x of lib.exports) (byModel[x.model] = byModel[x.model] || []).push(x);
    const known = new Set([...lib.finetuned, ...lib.base].map(m => m.ref));
    const exportsOf = ref => {
      const xs = byModel[ref] || [];
      if (!xs.length) return "";
      const kinds = [...new Set(xs.map(exportLabel))].join(", ");
      return `<details class="mrow-exp"><summary>${xs.length} ${xs.length === 1 ? "export" : "exports"}: ${esc(kinds)}</summary><ul class="mrow-exports">${xs.map(exportItem).join("")}</ul></details>`;
    };
    const initial = name => esc((String(name).replace(/^.*\//, "").match(/[a-z0-9]/i) || ["?"])[0].toUpperCase());

    $("#mine").innerHTML = lib.finetuned.length ? `<section class="panel"><ul class="rows">${lib.finetuned.map(f => {
      const id = f.ref.slice(4);
      return `<li class="mrow">
        <span class="tile" aria-hidden="true">${initial(f.name)}</span>
        <div class="rowi-main">
          <a class="rowi-name" href="#/runs/${esc(id)}">${esc(f.name)}</a>
          <span class="rowi-sub">from ${esc(modelName(f.base_model))} · on ${esc(f.dataset_name || f.dataset)}${f.method ? " · " + esc(f.method) : ""} · ${bytes(f.size_bytes)} · ${esc(when(f.created))}${ago(f.created) ? ` (${esc(ago(f.created))})` : ""}</span>
          <span class="rowi-sub mono path">${esc(f.path)}</span>
        </div>
        <div class="mrow-score">${f.accuracy != null ? `<b>${pct(f.accuracy)}</b> ${delta(f.baseline_accuracy, f.accuracy)}<span class="faint">test accuracy${f.baseline_accuracy != null ? ", base " + pct(f.baseline_accuracy) : ""}</span>` : `<span class="faint">not measured</span>`}</div>
        <div class="mrow-actions">
          <a class="btn small" href="#/playground?run=${esc(encodeURIComponent(id))}">Try in playground</a>
          <span class="joined"><select aria-label="Export format" data-fmt="${esc(f.ref)}">${exportOptions(f.targets)}</select><button class="btn small" type="button" data-export="${esc(f.ref)}">Export</button></span>
          <button class="btn small" type="button" data-publish="${esc(f.ref)}" title="Push this checkpoint and its measured numbers to systemonemodels.tech">Publish to System One</button>
        </div>
        ${exportsOf(f.ref)}
      </li>`;
    }).join("")}</ul></section>`
      : `<div class="empty"><b>No fine-tuned models yet.</b><br>Fine-tune a base model on one of your datasets: the run measures it against the base model, and the result shows up here, ready to try, export or publish.
         <div class="row"><a class="btn primary" href="#/train">Fine-tune a model</a><a class="btn" href="#/datasets">Add a dataset</a></div></div>`;

    const orphans = lib.exports.filter(x => !known.has(x.model));
    if (orphans.length) $("#mine").innerHTML += `<section class="panel"><header><h2>Other exports</h2><span class="muted small">of models no longer in the workspace</span></header>
      <ul class="rows">${orphans.map(x => `<li class="mrow"><span class="tile base" aria-hidden="true">${initial(modelName(x.model))}</span><div class="rowi-main"><span class="rowi-name">${esc(modelName(x.model))}</span></div><div></div><ul class="mrow-exports">${exportItem(x)}</ul></li>`).join("")}</ul></section>`;

    $("#bases").innerHTML = `<section class="panel"><ul class="rows">${lib.base.map(m => {
      const repo = m.repo.replace(" (imported)", "");
      const tag = m.demo ? pill("accent", "demo fine-tune") : m.imported ? pill("", "imported") : "";
      const hub = m.ref.startsWith("hub:");
      return `<li class="mrow">
        <span class="tile base" aria-hidden="true">${initial(repo)}</span>
        <div class="rowi-main">
          <span class="rowi-name">${hub ? `<a class="mono-name" href="https://huggingface.co/${esc(repo)}" target="_blank" rel="noreferrer">${esc(repo)}</a>` : `<span class="mono-name">${esc(repo)}</span>`} ${tag}</span>
          <span class="rowi-sub">${esc(m.description)}</span>
          ${warnings(m.warnings)}
        </div>
        <div class="mrow-score">${m.cached ? `${pill("done", m.imported ? "ready" : "downloaded")}<span class="faint">${m.size_bytes ? bytes(m.size_bytes) + " on disk" : ""}</span>` : pill("", m.imported ? "not trainable yet" : "not downloaded")}${m.fit && m.fit !== "fits" ? fitPill(m) : ""}</div>
        <div class="mrow-actions">${m.cached
          ? `<a class="btn small" href="#/playground?models=${esc(encodeURIComponent(m.ref))}">Try in playground</a><a class="btn small" href="#/train?base=${esc(encodeURIComponent(m.ref))}">Fine-tune from it</a>`
          : !m.demo && !m.imported ? `<button class="btn small primary" type="button" data-dl="${esc(repo)}">Download</button>` : ""}</div>
        ${exportsOf(m.ref)}
      </li>`;
    }).join("")}</ul></section>`;

    $$("[data-export]").forEach(b => b.onclick = () => startExport(b.dataset.export, $(`select[data-fmt="${CSS.escape(b.dataset.export)}"]`).value));
    $$("[data-publish]").forEach(b => b.onclick = () => startPublish(b.dataset.publish));
    bindDownloads($("#bases"));
  } catch (e) { if (current(token)) $("#mine").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }

  try {
    const cat = await api("/api/families");
    if (!current(token)) return;
    const m = cat.machine || {};
    $("#families").innerHTML = `<p class="hint">This machine: ${m.memory_gb ? `${m.memory_gb} GB for training` : "memory unknown"}${m.accelerator ? " · " + esc(m.accelerator) : ""}. Estimates are for LoRA in bf16, and 4-bit QLoRA where that is the only way in.</p>` +
      cat.families.map((f, i) => {
        const fit = f.models.filter(x => x.fit === "fits" || x.fit === "qlora").length;
        return `<details class="family"${i === 0 ? " open" : ""}><summary><b>${esc(f.name)}</b><span class="muted small">${f.models.length} ${f.models.length === 1 ? "model" : "models"} · ${fit} fit here</span><span class="spacer"></span>${noulxpPill(f.noulxp)}${trainerPill(f.trainer)}</summary>
        <div class="family-body"><p class="hint" style="margin:12px 0">${esc(f.how)} <span class="faint">· ${esc(f.backends)}</span></p>
        ${f.noulxp ? `<p class="hint" style="margin:-4px 0 12px"><b>NoulXP</b>${f.noulxp.profile ? ` (${esc(f.noulxp.profile)} profile)` : ""}: ${esc(f.noulxp.note)}</p>` : ""}
        <div class="tablewrap"><table class="wide"><tr><th>Model</th><th>Maker</th><th>Size</th><th>Licence</th><th>Needs</th><th>Here</th><th></th></tr>
        ${f.models.map(x => `<tr><td><a class="mono" href="https://huggingface.co/${esc(x.repo)}" target="_blank" rel="noreferrer">${esc(x.repo)}</a>${x.note ? `<div class="faint small">${esc(x.note)}</div>` : ""}${warnings(x.warnings)}</td>
          <td>${esc(x.maker)}</td><td>${x.params_b ? params(x.params_b) : "–"}</td><td>${esc(x.licence)}</td>
          <td>${gb(x.needed_gb && x.needed_gb.lora)}${x.needed_gb && x.needed_gb.qlora ? `<div class="faint small">${gb(x.needed_gb.qlora)} QLoRA</div>` : ""}</td>
          <td>${fitPill(x)}${x.trains_here ? `<div style="margin-top:4px">${pill("done", "trains here")}</div>` : ""}</td>
          <td>${x.fit === "not-trainable" ? "" : x.downloaded ? pill("done", "downloaded") : `<button class="btn small" type="button" data-dl="${esc(x.repo)}">Download</button>`}</td></tr>`).join("")}
        </table></div></div></details>`;
      }).join("") +
      `<p class="hint">A model too big for this machine trains on a bigger GPU, or in the cloud studio at <a href="${esc(cat.cloud_studio)}" target="_blank" rel="noreferrer">${esc(cat.cloud_studio.replace("https://", ""))}</a> (coming).</p>`;
    bindDownloads($("#families"));
  } catch (e) { if (current(token)) $("#families").innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }

  const search = async () => {
    const out = $("#regout");
    out.innerHTML = `<div class="muted">Searching…</div>`;
    try {
      const r = await api("/api/registry?q=" + encodeURIComponent($("#regq").value.trim()));
      if (!current(token)) return;
      const all = out.dataset.all === "1" || r.items.length <= 8;
      const items = all ? r.items : r.items.slice(0, 6);
      out.innerHTML = r.items.length ? `<div class="tablewrap boxed"><table class="wide"><tr><th>Model</th><th>Family</th><th>Here</th><th></th></tr>${items.map(x => `<tr><td><span class="mono">${esc(x.repo)}</span><div class="faint small">${esc(x.maker)}${x.summary ? " · " + esc(x.summary.slice(0, 120)) : ""}</div>${warnings(x.warnings)}</td><td>${esc(x.family || "unknown")}</td><td>${fitPill(x)}</td><td>${x.availability === "hosted-api" ? pill("", "API only") : `<button class="btn small" type="button" data-import="${esc(x.repo)}">Import</button>`}</td></tr>`).join("")}</table></div>${all ? "" : `<div class="row" style="margin-top:10px"><button class="btn small" type="button" id="regall">Show all ${r.items.length}</button></div>`}` : `<div class="muted">Nothing found.</div>`;
      bindImports(out);
      const more = $("#regall");
      if (more) more.onclick = () => { out.dataset.all = "1"; search(); };
    } catch (e) { if (current(token)) out.innerHTML = `<div class="notice bad">${esc(e.message)}</div>`; }
  };
  $("#regsearch").onclick = () => { $("#regout").dataset.all = ""; search(); };
  $("#regq").onkeydown = e => { if (e.key === "Enter") { $("#regout").dataset.all = ""; search(); } };
  search();
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
  <p><b>LoRA</b> (default) freezes the encoder and learns a low-rank update <code>W + (α/r)·A·B</code> for each of its attention and MLP matrices, while the decision head trains fully. That is ~33M of 428M parameters, so it fits comfortably in 16 GB. After training the updates are merged back, so the saved model is an ordinary Laya checkpoint with zero extra inference cost.</p>
  <p><b>LoRA variants</b> (Advanced settings, combinable): <b>DoRA</b> also learns a magnitude for every output row and renormalises the adapted weight, which follows full fine-tuning more closely at a small cost in speed. <b>rsLoRA</b> scales the update by <code>α/√r</code> instead of <code>α/r</code>, so larger ranks keep learning instead of fading out; with the same α the update is √r times stronger, so lower α or the learning rate when you switch it on. <b>LoRA+</b> trains the B matrices faster than A by the ratio you set. The paper suggests 16, but on Laya at the default learning rate 16 collapsed to chance while 4 trained best, so start at 2–4. All of them merge into the weights, so the saved model is the same size and speed.</p></section>
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
  <section class="card"><h2>6 · Julia 1 and Decider train here too</h2><p><b>Julia 1</b> (Supersonic Labs, 144M) is Laya's network with its own prompt: an option is its description, up to 8,192 tokens and 20 options. It trains with the same recipes, and its fine-tunes keep Julia's own files, in float32, with the calibration temperature folded into the scorer.</p>
  <p><b>Decider</b> (Mapika, 0.8B to 4B) is a language model read by its answer letters: <code>Context: … Question: … (A) … (B) … Answer: (</code>. It trains with LoRA on its attention and MLP projections (or 4-bit QLoRA when memory is short), with cross-entropy on the letters, as Decider itself was made; a score question is one yes/no row per level. Its fine-tunes export to GGUF and MLX-LM, and every family gets a checked NoulXP package.</p></section>
  <section class="card"><h2>7 · Limits to keep in mind</h2><ul style="padding-left:18px;margin:0">
  <li>Inputs beyond 512 / 1,024 tokens are cut: check the token budget on each dataset.</li>
  <li>Many labels share one option budget, so long label lists get clipped; fine-tuning helps, shorter criteria help more.</li>
  <li>Fine-tuning specializes the model. Evaluate other questions you rely on before replacing a general checkpoint.</li>
  <li>Gate on confidence and route uncertain cases to a person or a larger model.</li></ul></section>
  </div>`;
}

buildNav();
$("#acctchip").onclick = async () => {
  const a = await refreshAccount();
  if (!a) return;
  if (!a.signed_in) return signIn(null);
  if (a.from_environment) return toast("Signed in through SYSTEMONE_TOKEN; unset it to sign out.");
  if (!confirm(`Sign out of ${a.site}? This also signs out the systemone command and revokes its token.`)) return;
  try { const r = await api("/api/account/logout", {method: "POST", body: {}}); await refreshAccount(); toast(r.message || "Signed out"); }
  catch (e) { toast(e.message); }
};
refreshAccount();
route();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
