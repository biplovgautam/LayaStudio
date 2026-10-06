"""A job's child processes end with it, even when it is killed outright, and its scratch folders
are removed by the next build once it is gone (children.py). Tiny scripts, nothing real."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from systemone_studio import children

REPO = Path(__file__).resolve().parents[1]
needs_posix = pytest.mark.skipif(os.name == "nt", reason="SIGKILL and os.kill(pid, 0)")


def watched(args, env=None):
    return subprocess.Popen(
        children.command(args),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=children.child_env(env),
    )


def finish(process, timeout=60):
    out = process.stdout.read()
    code = process.wait(timeout=timeout)
    children.close_stdin(process)
    return code, out


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def gone(pid, within=5.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


def first_pid(job, within=30):
    """The pid on the job's first line of output; fails (never hangs) when none comes."""
    import select

    ready, _, _ = select.select([job.stdout], [], [], within)
    line = job.stdout.readline() if ready else ""
    if not line.strip().isdigit():
        pytest.fail(f"no pid within {within} s: {line!r}")
    return int(line)


def dead_pid():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


# ----------------------------------------------------------------------------- what a child runs


def test_a_watched_child_runs_its_arguments_as_python_does(tmp_path):
    code, out = finish(watched(["-c", "import sys; print(sys.argv, __name__)", "a", "b"]))
    assert code == 0 and out.strip() == "['-c', 'a', 'b'] __main__"
    module = tmp_path / "shown.py"
    module.write_text("import sys\nprint(sys.argv[1:], __name__)\nraise SystemExit(7)\n")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    code, out = finish(watched(["-m", "shown", "--threads", "3"], env))
    assert code == 7 and out.strip() == "['--threads', '3'] __main__"
    code, out = finish(watched(["-x"]))
    assert code == 1 and "usage" in out


@needs_posix
def test_a_watched_child_ends_when_its_job_is_killed_outright(tmp_path):
    """SIGKILL: no handler runs in the job; its child sees its stdin close (and, on Linux,
    PR_SET_PDEATHSIG) and is gone within 5 s."""
    job = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import subprocess, sys, time
                sys.path.insert(0, {str(REPO)!r})
                from systemone_studio import children
                child = subprocess.Popen(
                    children.command(["-c", "import time; time.sleep(600)"]),
                    stdin=subprocess.PIPE,
                    env=children.child_env(),
                )
                print(child.pid, flush=True)
                time.sleep(600)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    child = int(job.stdout.readline())
    time.sleep(0.3)
    assert alive(child)
    os.kill(job.pid, signal.SIGKILL)
    job.wait()
    if not gone(child):
        os.kill(child, signal.SIGKILL)
        pytest.fail("the child outlived its job")


@needs_posix
def test_a_child_whose_job_is_gone_before_it_watches_ends_at_once():
    """The job starts its child and dies at once (os._exit: no handler, nothing closed by
    hand): the child's first read of its stdin is EOF, on every platform."""
    job = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import os, subprocess, sys
                sys.path.insert(0, {str(REPO)!r})
                from systemone_studio import children
                child = subprocess.Popen(
                    children.command(["-c", "import time; time.sleep(600)"]),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    env=children.child_env(),
                )
                print(child.pid, flush=True)
                os._exit(0)
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    child = int(job.stdout)
    if not gone(child, within=10):
        os.kill(child, signal.SIGKILL)
        pytest.fail("the child outlived its job")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG's race")
def test_on_linux_a_child_whose_job_is_already_gone_ends_at_once():
    """Its parent is not the job it names, and that job's pid is gone: the job died before
    PR_SET_PDEATHSIG was set, so no signal will come. (Here its stdin is still held open.)"""
    process = subprocess.Popen(
        children.command(["-c", "import time; time.sleep(600)"]),
        stdin=subprocess.PIPE,
        env={**os.environ, children.PARENT: str(dead_pid())},
    )
    try:
        assert process.wait(timeout=30) == 1
    finally:
        children.close_stdin(process)
        if process.poll() is None:
            process.kill()


# A launcher as a Windows virtual environment's python.exe is one (CPython's venv launcher,
# uv's trampoline): it starts the interpreter as its own child, with its own stdin, and waits.
LAUNCHER = "import subprocess, sys; sys.exit(subprocess.call([sys.executable, *sys.argv[1:]]))"
SHOW_PARENT = "import os; print(os.getppid() != int(os.environ[{parent!r}]), 'ran')".format(
    parent=children.PARENT
)


def launched(args, env=None):
    """watched(), through LAUNCHER: the interpreter is the job's grandchild."""
    return subprocess.Popen(
        [sys.executable, "-c", LAUNCHER, *children.command(args)[1:]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=children.child_env(env),
    )


def test_a_watched_child_runs_behind_a_launcher():
    """Its parent is the launcher, not the job it names: it runs all the same (it used to exit
    at once, so every NoulXP step failed in a Windows virtual environment)."""
    code, out = finish(launched(["-c", SHOW_PARENT]))
    assert code == 0 and out.strip() == "True ran"
    code, out = finish(launched(["-c", "raise SystemExit(5)"]))
    assert code == 5


@needs_posix
def test_a_child_behind_a_launcher_ends_when_its_job_is_killed_outright():
    job = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import subprocess, sys, time
                sys.path.insert(0, {str(REPO)!r})
                from systemone_studio import children
                code = "import os, time; print(os.getpid(), flush=True); time.sleep(600)"
                child = subprocess.Popen(
                    [sys.executable, "-c", {LAUNCHER!r}, *children.command(["-c", code])[1:]],
                    stdin=subprocess.PIPE,
                    env=children.child_env(),
                )
                time.sleep(600)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        child = first_pid(job)  # the interpreter behind the launcher
        time.sleep(0.3)
        assert alive(child)
    finally:
        os.kill(job.pid, signal.SIGKILL)
        job.wait()
    if not gone(child):
        os.kill(child, signal.SIGKILL)
        pytest.fail("the child outlived its job")


# ----------------------------------------------------------------------------- the GGUF converter

# A stand-in for llama.cpp's convert_hf_to_gguf.py: it writes its pid where FAKE_CONVERTER_PID
# says, then a GGUF's magic to --outfile; FAKE_CONVERTER=sleep sleeps instead, =fail fails.
FAKE_CONVERTER = """\
import os, sys, time
from pathlib import Path

Path(os.environ["FAKE_CONVERTER_PID"]).write_text(str(os.getpid()))
mode = os.environ.get("FAKE_CONVERTER", "")
if mode == "sleep":
    time.sleep(600)
if mode == "fail":
    print("loading the model", flush=True)
    print("KeyError: no such tensor", file=sys.stderr)
    sys.exit(2)
first = sys.path[0] == str(Path(__file__).resolve().parent)
print("argv", sys.argv[1:], __name__, Path(__file__).name, first)
print("written", file=sys.stderr)
Path(sys.argv[sys.argv.index("--outfile") + 1]).write_bytes(b"GGUF" + bytes(8))
"""


def fake_tools(root):
    """LAYASTUDIO_TOOLS holding llama.cpp's converter at the pinned commit as converter() finds
    a verified one, FAKE_CONVERTER in place of the real script."""
    from systemone_studio import gguf

    tool = root / f"llama.cpp-{gguf.LLAMA_CPP_COMMIT[:12]}"
    for name in gguf.PINNED:
        (tool / name).parent.mkdir(parents=True, exist_ok=True)
        (tool / name).write_text("")
    (tool / "convert_hf_to_gguf.py").write_text(FAKE_CONVERTER)
    (tool / ".verified").write_text(gguf.LLAMA_CPP_COMMIT + "\n")
    return tool


def converter_pid(path, within=30):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        text = path.read_text().strip() if path.is_file() else ""
        if text.isdigit():
            return int(text)
        time.sleep(0.05)
    pytest.fail(f"the converter did not start within {within} s")


def test_the_converter_runs_as_python_runs_a_script(tmp_path, monkeypatch):
    """Behind children.WATCH, the converter still sees its own argv, __name__ and __file__, and
    its folder first on its path as `python <script>` puts it there; its output reaches the
    job's log, and a failure names the last line it wrote to stderr."""
    from systemone_studio import gguf

    tools = tmp_path / "tools"
    fake_tools(tools)
    monkeypatch.setenv("LAYASTUDIO_TOOLS", str(tools))
    monkeypatch.setenv("FAKE_CONVERTER_PID", str(tmp_path / "pid"))
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    monkeypatch.delenv("FAKE_CONVERTER", raising=False)
    events = []
    out = gguf.convert(
        tmp_path / "model",
        tmp_path / "out" / "m.gguf",
        "bf16",
        tmp_path / "w",
        lambda kind, **data: events.append((kind, data)),
    )
    assert out.read_bytes()[:4] == b"GGUF"
    logged = [data["message"] for kind, data in events if kind == "log"]
    args = [str(tmp_path / "model"), "--outfile", str(out), "--outtype", "bf16", "--no-mtp"]
    assert f"argv {args} __main__ convert_hf_to_gguf.py True" in logged
    assert "written" in logged
    monkeypatch.setenv("FAKE_CONVERTER", "fail")
    with pytest.raises(RuntimeError, match="convert_hf_to_gguf.py failed: KeyError: no such"):
        gguf.convert(tmp_path / "model", tmp_path / "out" / "n.gguf", "bf16", tmp_path / "w")


def converter_job(tmp_path, on_term=False):
    """A job running gguf.convert with a converter that sleeps, and the converter's pid. With
    on_term, SIGTERM raises in the job as engine.run_job's handler makes it (a cancel)."""
    tools = tmp_path / "tools"
    fake_tools(tools)
    pid_file = tmp_path / "pid"
    handler = (
        "import signal\n"
        "def on_term(*_):\n"
        "    raise SystemExit(143)\n"
        "signal.signal(signal.SIGTERM, on_term)\n"
        if on_term
        else ""
    )
    job = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import sys\nsys.path.insert(0, {str(REPO)!r})\n{handler}"
            "from systemone_studio import gguf\n"
            f"gguf.convert({str(tmp_path / 'model')!r}, {str(tmp_path / 'm.gguf')!r}, 'bf16')\n",
        ],
        env={
            **os.environ,
            "LAYASTUDIO_TOOLS": str(tools),
            "FAKE_CONVERTER_PID": str(pid_file),
            "FAKE_CONVERTER": "sleep",
        },
    )
    try:
        child = converter_pid(pid_file)
    except BaseException:
        job.kill()
        job.wait()
        raise
    return job, child


@needs_posix
def test_a_converter_ends_when_its_job_is_killed_outright(tmp_path):
    """SIGKILL during the GGUF conversion: no handler runs in the job, and the converter (half
    a minute or more on a 2B Decider, writing the run's GGUF) is gone within 5 s all the same."""
    job, child = converter_job(tmp_path)
    time.sleep(0.3)
    assert alive(child)
    os.kill(job.pid, signal.SIGKILL)
    job.wait()
    if not gone(child):
        os.kill(child, signal.SIGKILL)
        pytest.fail("the converter outlived its job")


@needs_posix
def test_a_cancelled_conversion_ends_its_converter(tmp_path):
    """SIGTERM raising in the job during the conversion (a cancel): the converter is ended
    before the job's exception goes on."""
    job, child = converter_job(tmp_path, on_term=True)
    time.sleep(0.3)
    os.kill(job.pid, signal.SIGTERM)
    assert job.wait(timeout=30) == 143
    if not gone(child):
        os.kill(child, signal.SIGKILL)
        pytest.fail("the converter outlived its cancelled job")


# ----------------------------------------------------------------------------- scratch folders


@needs_posix
def test_the_sweep_removes_only_the_folders_of_jobs_that_are_gone(tmp_path):
    prefix = ".noulxp-"
    with children.scratch(tmp_path, prefix) as mine:
        assert Path(mine).name.startswith(f"{prefix}{os.getpid()}-")
        stale = tmp_path / f"{prefix}{dead_pid()}-abc"
        running = tmp_path / f"{prefix}{os.getppid()}-def"  # pytest's parent: running
        older = tmp_path / f"{prefix}k2x_9q"  # an older release's: no pid
        for folder in (stale, running, older):
            (folder / "inner").mkdir(parents=True)
        (tmp_path / f"{prefix}file").write_text("not a folder")
        (tmp_path / "noulxp-report.json").write_text("{}")
        (tmp_path / ".gguf-verify-1-x").mkdir()
        removed = children.sweep(tmp_path, prefix)
        assert sorted(removed) == sorted([stale.name, older.name])
        assert Path(mine).is_dir() and running.is_dir()
        assert (tmp_path / f"{prefix}file").is_file() and (tmp_path / ".gguf-verify-1-x").is_dir()
    assert children.sweep(tmp_path / "none", prefix) == []
