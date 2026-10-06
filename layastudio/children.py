"""A job's child processes and scratch folders, when the job is killed outright.

A job ends its child processes itself on any exception, a cancel included (SIGTERM, then
SIGKILL: noulxp_package._end, gguf.Reference.end), and its scratch folders go with the
`with` or `finally` that made them. A job killed outright (SIGKILL: the local server 15 s
after asking it to stop, a cloud run 30 s after) runs none of that, so:

- each child process watches for it itself (WATCH): it exits when its stdin, a pipe the job
  holds and never writes to, closes, which happens when the job dies (a job already gone
  before the child starts reading included: the first read is EOF), and on Linux also on
  PR_SET_PDEATHSIG. A NoulXP step runs as `python -c RUN <its own arguments>` (command());
  gguf_reference.py calls watch_parent(). The interpreter need not be the job's own child:
  in a Windows virtual environment sys.executable is a launcher that starts the base
  interpreter as its child and waits for it, and the stdin pipe reaches it all the same;
- a scratch folder in a run's folder carries the pid of the job that made it (scratch()), and
  the next build or export of that run removes the ones whose job is gone (sweep()).
"""

import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

PARENT = "LAYASTUDIO_PARENT"  # the pid of the job a child process belongs to

# The watch, as source: it runs before a child's own code, so it needs nothing importable.
WATCH = f"""\
def _watch_parent():
    import ctypes, os, sys, threading
    parent = os.environ.get({PARENT!r}, "")
    if sys.platform.startswith("linux"):
        try:  # PR_SET_PDEATHSIG (1): SIGKILL (9) when the job dies
            ctypes.CDLL(None, use_errno=True).prctl(1, 9)
        except (OSError, AttributeError):
            pass
        # A job that died before prctl sent no signal: this process has another parent now,
        # and the job's pid is gone. Linux only, and only on both: elsewhere the interpreter
        # can be its job's grandchild (a Windows virtual environment's python.exe is a
        # launcher that starts the base interpreter as its own child), and the stdin watch
        # below already ends a child whose job is gone.
        if parent.isdigit() and os.getppid() != int(parent):
            try:
                os.kill(int(parent), 0)
            except ProcessLookupError:
                os._exit(1)  # the job ended before this process was watching
            except OSError:
                pass  # someone else's process: running
    try:
        stdin = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        return

    def wait_for_eof():
        # The file descriptor, not sys.stdin: a daemon thread blocked in a buffered read holds
        # its lock, and the interpreter cannot shut down past that.
        try:
            while os.read(stdin, 4096):
                pass
        except OSError:
            pass
        os._exit(1)

    threading.Thread(target=wait_for_eof, name="parent-watch", daemon=True).start()


_watch_parent()
del _watch_parent
"""
# A step's own arguments after the watch, run as Python runs them: "-m <module> args..." as
# `python -m` does (sys.argv[0] its file), "-c <code> args..." as `python -c` does.
RUN = (
    WATCH
    + """
import sys

_args = sys.argv[1:]
if _args[:1] == ["-m"] and len(_args) > 1:
    import runpy

    sys.argv = _args[1:]
    runpy.run_module(_args[1], run_name="__main__", alter_sys=True)
elif _args[:1] == ["-c"] and len(_args) > 1:
    sys.argv = ["-c", *_args[2:]]
    _code = _args[1]
    del _args
    exec(compile(_code, "<string>", "exec"), globals())  # __main__'s, as `python -c` runs it
else:
    sys.exit("usage: python -c RUN (-m module | -c code) [arguments]")
"""
)


def watch_parent():
    """WATCH, in this process (gguf_reference.py's)."""
    exec(compile(WATCH, "<watch>", "exec"), {"__name__": __name__})


def command(args):
    """The argv of a child that ends with this job: `python -c RUN` and its own arguments,
    "-m <module> ..." or "-c <code> ...". Started with child_env() and stdin=subprocess.PIPE."""
    return [sys.executable, "-c", RUN, *args]


def child_env(env=None):
    """An environment that names this process as the child's job."""
    return {**(os.environ if env is None else env), PARENT: str(os.getpid())}


def close_stdin(process):
    """The watched pipe, closed once its process has ended (a stand-in may have none)."""
    handle = getattr(process, "stdin", None)
    if handle is not None:
        with contextlib.suppress(OSError):
            handle.close()


# ----------------------------------------------------------------------------- scratch folders


def scratch(folder, prefix):
    """A TemporaryDirectory in `folder` named prefix<pid>-<random>, so that sweep() can tell
    whether the job that made it is still running."""
    return tempfile.TemporaryDirectory(prefix=f"{prefix}{os.getpid()}-", dir=folder)


def _alive(pid):
    if os.name == "nt":
        return True  # os.kill would end it: such a folder is left where it is
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # someone else's process: running
        return True
    return True


def sweep(folder, prefix):
    """Removes the prefix* folders in `folder` whose job is gone: named for a pid that no
    process has now, or by an older release (no pid). A folder of this process or of one
    still running stays. Returns the names removed; never raises."""
    removed = []
    try:
        found = sorted(Path(folder).glob(f"{prefix}*"))
    except OSError:
        return removed
    for path in found:
        try:
            if path.is_symlink() or not path.is_dir():
                continue
        except OSError:
            continue
        pid = path.name[len(prefix) :].split("-", 1)[0]
        if pid.isdigit() and (int(pid) == os.getpid() or _alive(int(pid))):
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path.name)
    return removed
