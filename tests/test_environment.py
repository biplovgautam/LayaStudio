"""The studio's environment variables: SYSTEMONE_STUDIO_<KEY>, and LAYASTUDIO_<KEY>, the name
each had before the rename, which still counts; the new name wins whenever it is set
(systemone_studio/environment.py). What the studio sets for a child it sets under both names.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from systemone_studio import children, engine, environment, gguf, noulxp_package, runtime

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "systemone_studio"


def python(code, *args, env=None, cwd=None):
    """`python -c code` in a fresh interpreter that finds this checkout first."""
    env = {**os.environ, **(env or {})}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, *args, "-c", textwrap.dedent(code)] if code else [sys.executable, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd or ROOT,
        timeout=300,
    )


def last_json(result):
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture
def clean(monkeypatch):
    for key in environment.KEYS:
        for name in environment.names(key):
            monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_every_variable_has_both_names():
    assert environment.names("HOME") == ("SYSTEMONE_STUDIO_HOME", "LAYASTUDIO_HOME")
    assert environment.names("VERBOSE") == ("SYSTEMONE_STUDIO_VERBOSE", "LAYA_STUDIO_VERBOSE")
    read = set()
    for path in PACKAGE.glob("*.py"):
        text = path.read_text()
        read |= {k for k in environment.KEYS if f'"{k}"' in text or f"'{k}'" in text}
    # Every key is read somewhere, under environment.py's names.
    assert read >= set(environment.KEYS) - {"PARENT"}


def test_the_new_name_wins_and_the_old_one_still_counts(clean):
    env = {}
    assert environment.get("TOOLS", "default", env) == "default"
    assert environment.source("TOOLS", env) is None
    env["LAYASTUDIO_TOOLS"] = "/old"
    assert environment.get("TOOLS", environ=env) == "/old"
    assert environment.source("TOOLS", env) == "LAYASTUDIO_TOOLS"
    env["SYSTEMONE_STUDIO_TOOLS"] = "/new"
    assert environment.get("TOOLS", environ=env) == "/new"
    assert environment.source("TOOLS", env) == "SYSTEMONE_STUDIO_TOOLS"
    env["SYSTEMONE_STUDIO_TOOLS"] = ""  # set, even empty: it wins
    assert environment.get("TOOLS", environ=env) == ""
    assert environment.both("THREADS", 4) == {
        "SYSTEMONE_STUDIO_THREADS": "4",
        "LAYASTUDIO_THREADS": "4",
    }


def test_setdefault_sets_both_names_only_when_neither_is_set():
    env = {}
    assert environment.setdefault("HOME", "/w", env) == "/w"
    assert env == {"SYSTEMONE_STUDIO_HOME": "/w", "LAYASTUDIO_HOME": "/w"}
    env = {"LAYASTUDIO_HOME": "/old"}
    assert environment.setdefault("HOME", "/w", env) == "/old"
    assert env == {"LAYASTUDIO_HOME": "/old"}


def test_the_readers_take_the_new_name_first(clean, tmp_path):
    clean.setenv("LAYASTUDIO_HOME", str(tmp_path / "old"))
    assert engine.default_workspace() == (tmp_path / "old").resolve()
    clean.setenv("SYSTEMONE_STUDIO_HOME", str(tmp_path / "new"))
    assert engine.default_workspace() == (tmp_path / "new").resolve()

    clean.setenv("LAYASTUDIO_TOOLS", str(tmp_path / "old-tools"))
    assert gguf.tools_dir() == tmp_path / "old-tools"
    clean.setenv("SYSTEMONE_STUDIO_TOOLS", str(tmp_path / "new-tools"))
    assert gguf.tools_dir() == tmp_path / "new-tools"

    clean.setenv("LAYASTUDIO_BACKEND", "mlx")
    clean.setenv("SYSTEMONE_STUDIO_BACKEND", "torch")
    assert runtime.backend() == "torch"

    budget = runtime.cpu_budget(environ={"LAYASTUDIO_THREADS": "3"})
    assert (budget["threads"], budget["source"]) == (3, "env")
    budget = runtime.cpu_budget(
        environ={"LAYASTUDIO_THREADS": "3", "SYSTEMONE_STUDIO_THREADS": "5"}
    )
    assert budget["threads"] == 5


def test_a_gate_names_the_variable_that_decided(clean):
    clean.setenv("LAYASTUDIO_SERIAL_VERIFY", "1")
    assert gguf.verify_gate() == {"overlap": False, "reason": "LAYASTUDIO_SERIAL_VERIFY"}
    clean.setenv("SYSTEMONE_STUDIO_SERIAL_VERIFY", "0")  # the new name says no
    clean.setenv("SYSTEMONE_STUDIO_PARALLEL_VERIFY", "1")
    assert gguf.verify_gate() == {"overlap": True, "reason": "SYSTEMONE_STUDIO_PARALLEL_VERIFY"}

    clean.setenv("LAYASTUDIO_PARALLEL_CONFORMANCE", "1")
    assert noulxp_package.conformance_gate("laya", 0, 1)["reason"] == (
        "LAYASTUDIO_PARALLEL_CONFORMANCE=1"
    )
    clean.setenv("SYSTEMONE_STUDIO_PARALLEL_CONFORMANCE", "0")
    assert noulxp_package.conformance_gate("laya", 0, 8) == {
        "overlap": False,
        "reason": "SYSTEMONE_STUDIO_PARALLEL_CONFORMANCE=0",
    }


def test_a_child_gets_what_this_process_meant_under_both_names(clean):
    clean.setenv("SYSTEMONE_STUDIO_THREADS", "16")  # the job's own: a step gets its count
    clean.setenv("SYSTEMONE_STUDIO_PARENT", "1")
    env = noulxp_package._child_env(threads=2)
    assert env["SYSTEMONE_STUDIO_THREADS"] == env["LAYASTUDIO_THREADS"] == "2"
    assert env["SYSTEMONE_STUDIO_PARENT"] == env["LAYASTUDIO_PARENT"] == str(os.getpid())
    assert children.PARENT == "SYSTEMONE_STUDIO_PARENT"
    # The watch runs before anything is importable: it reads both names itself, new first.
    watch = children.WATCH
    assert "'SYSTEMONE_STUDIO_PARENT'" in watch and "'LAYASTUDIO_PARENT'" in watch
    assert watch.index("'SYSTEMONE_STUDIO_PARENT'") < watch.index("'LAYASTUDIO_PARENT'")


def test_an_empty_new_name_wins_over_the_old_one(tmp_path):
    """$SYSTEMONE_STUDIO_DEMO_MODELS="" turns the demo models off whatever the old name says."""
    found = last_json(
        python(
            """
            import json
            from systemone_studio import engine
            print(json.dumps(sorted(engine.DEMO_MODELS)))
            """,
            env={
                "SYSTEMONE_STUDIO_DEMO_MODELS": "",
                "LAYASTUDIO_DEMO_MODELS": "a/b",
                "SYSTEMONE_STUDIO_HOME": str(tmp_path),
            },
        )
    )
    assert found == []
    found = last_json(
        python(
            """
            import json
            from systemone_studio import engine
            print(json.dumps(sorted(engine.DEMO_MODELS)))
            """,
            env={"LAYASTUDIO_DEMO_MODELS": "a/b, c/d", "LAYASTUDIO_HOME": str(tmp_path)},
        )
    )
    assert found == ["a/b", "c/d"]


def test_a_run_gives_its_jobs_its_workspace_under_both_names(clean, tmp_path):
    """A job reads its workspace from the environment (engine.WORKSPACE): `systemone-studio
    train` gives it the run's under both names, so a new name the run itself inherited cannot
    send the job elsewhere."""
    from systemone_studio import cloud

    clean.setenv("SYSTEMONE_STUDIO_HOME", "/somewhere/else")
    headless = cloud.Headless(tmp_path, print, None)
    assert headless.env["SYSTEMONE_STUDIO_HOME"] == headless.env["LAYASTUDIO_HOME"] == str(tmp_path)
