"""What a NoulXP build's steps cost here: seconds, CPU, throttling, memory, threads.

Read-only and best-effort. Every reading is a file the kernel already keeps (cgroup cpu.stat,
memory.peak, /proc/<pid>/status, /proc/cpuinfo) or getrusage; a reading that is not there or
cannot be read is null, and none can raise into the step it measures. Nothing here writes a
cgroup file, starts or reaps a step's process, or changes a step's arguments or environment.

The numbers go to runs/<id>/noulxp-report.json ("steps", "machine") and a GGUF export's
exports/gguf-<precision>.json ("timings"), never into the package or a result event: they
explain a measurement, they certify nothing.

cgroup counters are the container's (container_wide): every process in it counts, the job's
own steps and anything beside them.
"""

import contextlib
import os
import sys
import threading
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None

from .runtime import CGROUP, PROC_CGROUP, _cgroups, _inside, _read, affinity

# What a reading may fail with: a missing or unreadable file, a malformed value.
READ_ERRORS = (OSError, ValueError, IndexError, KeyError)
CPU_STAT = ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")
RUSAGE = ("utime", "stime", "nvcsw", "nivcsw")
SAMPLE_EVERY = 0.5
_CHILDREN = []  # what each step process's sampler saw, in the order they ended


# ----------------------------------------------------------------------------- readings


def cgroup_dir(root=CGROUP, proc=PROC_CGROUP):
    """This process's cgroup v2 folder (from /proc/self/cgroup), else the root."""
    try:
        return _inside(root, _cgroups(proc).get("", "/"))
    except READ_ERRORS:
        return Path(root)


def _keyed(text):
    values = {}
    for line in (text or "").splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                values[fields[0]] = int(fields[1])
            except ValueError:
                continue
    return values


def cpu_stat(root=CGROUP, proc=PROC_CGROUP):
    """cgroup v2 cpu.stat: usage_usec, nr_periods, nr_throttled, throttled_usec; or None."""
    try:
        values = _keyed(_read(cgroup_dir(root, proc) / "cpu.stat"))
    except READ_ERRORS:
        return None
    found = {key: values[key] for key in CPU_STAT if key in values}
    return found or None


def _number(text):
    value = (text or "").strip()
    if not value:
        return None
    if value == "max":
        return "max"
    try:
        return int(value)
    except ValueError:
        return None


def memory(root=CGROUP, proc=PROC_CGROUP):
    """The container's memory now: memory.current, memory.peak (its lifetime high-water mark,
    page cache included), memory.max, and memory.events' oom_kill; each null when absent."""
    try:
        folder = cgroup_dir(root, proc)
        events = _keyed(_read(folder / "memory.events"))
        return {
            "current": _number(_read(folder / "memory.current")),
            "peak": _number(_read(folder / "memory.peak")),
            "max": _number(_read(folder / "memory.max")),
            "oom_kill": events.get("oom_kill"),
        }
    except READ_ERRORS:
        return {"current": None, "peak": None, "max": None, "oom_kill": None}


def free_memory(root=CGROUP, proc=PROC_CGROUP, meminfo="/proc/meminfo"):
    """Bytes this job can still use: the cgroup's memory.max minus memory.current, else the
    kernel's MemAvailable; None where neither is known (macOS)."""
    try:
        mem = memory(root, proc)
        if isinstance(mem["max"], int) and isinstance(mem["current"], int):
            return max(0, mem["max"] - mem["current"])
        for line in (_read(meminfo) or "").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except READ_ERRORS:
        pass
    return None


def cpu_model(cpuinfo="/proc/cpuinfo"):
    """(model name, flags) from /proc/cpuinfo; on a Mac, its brand string and no flags."""
    model = flags = None
    text = _read(cpuinfo)
    if text is None and sys.platform == "darwin":
        import subprocess

        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return (out.stdout.strip() or None), None
        except (OSError, subprocess.SubprocessError):
            return None, None
    for line in (text or "").splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "model name" and model is None:
            model = value.strip()
        elif key in ("flags", "Features") and flags is None:
            flags = value.strip()
        if model and flags:
            break
    return model, flags


def llama_system_info(timeout=60):
    """llama.cpp's own description of this CPU (its SIMD level), from a short process of its
    own: the job never imports llama_cpp for it. None on any failure."""
    import subprocess

    code = (
        "import llama_cpp\nprint(llama_cpp.llama_print_system_info().decode('utf-8', 'replace'))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout.strip()
    return text if out.returncode == 0 and text else None


def machine(kind=None, root=CGROUP, proc=PROC_CGROUP, cpuinfo="/proc/cpuinfo"):
    """Once per build: the CPUs, their quota and the memory limit this build ran under."""
    info = {"cpu_count": os.cpu_count(), "affinity": None, "cpu_max": None, "memory_max": None}
    try:
        info["affinity"] = affinity()
        folder = cgroup_dir(root, proc)
        info["cpu_max"] = (_read(folder / "cpu.max") or "").strip() or None
        info["memory_max"] = _number(_read(folder / "memory.max"))
        info["cpu_model"], info["cpu_flags"] = cpu_model(cpuinfo)
    except READ_ERRORS:
        pass
    if kind == "decider":
        info["llama_cpp_system_info"] = llama_system_info()
    return info


def _usage(who):
    if resource is None:
        return None
    try:
        r = resource.getrusage(who)
    except (OSError, ValueError):
        return None
    return {
        "utime": r.ru_utime,
        "stime": r.ru_stime,
        "nvcsw": r.ru_nvcsw,
        "nivcsw": r.ru_nivcsw,
        "maxrss": r.ru_maxrss,
    }


def maxrss_kb():
    """This process's peak resident memory in KiB (getrusage reports bytes on macOS)."""
    usage = _usage(resource.RUSAGE_SELF) if resource is not None else None
    if usage is None:
        return None
    return int(usage["maxrss"] / 1024) if sys.platform == "darwin" else int(usage["maxrss"])


# ----------------------------------------------------------------------------- step processes


class Sampler:
    """A step process's peak threads and peak resident memory (VmHWM), read from
    /proc/<pid>/status about twice a second by a daemon thread. Linux only; elsewhere it reads
    nothing. stop() ends it (an Event, then a bounded join) and returns what it saw."""

    def __init__(self, pid, every=SAMPLE_EVERY, proc="/proc"):
        self.path = Path(proc) / str(pid) / "status"
        self.every = every
        self.threads = self.vm_hwm_kb = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = None
        if sys.platform.startswith("linux") and self.path.parent.is_dir():
            self._thread = threading.Thread(target=self._run, name="step-sampler", daemon=True)
            self._thread.start()

    def _read(self):
        try:
            text = self.path.read_text()
        except (OSError, ValueError):
            return
        for line in text.splitlines():
            key, _, value = line.partition(":")
            fields = value.split()
            try:
                if key == "Threads":
                    self.threads = max(self.threads or 0, int(fields[0]))
                elif key == "VmHWM":
                    self.vm_hwm_kb = max(self.vm_hwm_kb or 0, int(fields[0]))
            except READ_ERRORS:
                continue
        self.samples += 1

    def _run(self):
        while not self._stop.is_set():
            self._read()
            self._stop.wait(self.every)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        seen = {"peak_threads": self.threads, "vm_hwm_kb": self.vm_hwm_kb, "samples": self.samples}
        _CHILDREN.append(seen)
        return seen


# ----------------------------------------------------------------------------- steps


def _delta(after, before, keys):
    if not isinstance(after, dict) or not isinstance(before, dict):
        return None
    out = {}
    for key in keys:
        a, b = after.get(key), before.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[key] = round(a - b, 3) if isinstance(a, float) or isinstance(b, float) else a - b
    return out or None


class Steps:
    """report["steps"][name] for each step of a build, timed where it is called:

        with steps("check", threads=8, device="cpu"):
            check_package(...)

    The step's own exception (engine.Cancelled included) goes on unchanged; what was measured
    is recorded on the way out either way, and recording never raises."""

    def __init__(self, report, root=CGROUP, proc=PROC_CGROUP, emit=None):
        self.steps = report.setdefault("steps", {})
        self.root, self.proc = root, proc
        self.emit = emit

    def _snapshot(self):
        return {
            "t": time.perf_counter(),
            "self": _usage(resource.RUSAGE_SELF) if resource is not None else None,
            "children": _usage(resource.RUSAGE_CHILDREN) if resource is not None else None,
            "cgroup": cpu_stat(self.root, self.proc),
            "oom_kill": memory(self.root, self.proc).get("oom_kill"),
            "sampled": len(_CHILDREN),
        }

    @contextlib.contextmanager
    def __call__(self, name, **settings):
        record = {key: value for key, value in settings.items() if value is not None}
        before = None
        try:
            before = self._snapshot()
        except READ_ERRORS:
            pass
        failed = True
        try:
            yield record
            failed = False
        finally:
            self._record(name, record, before, failed)

    def _record(self, name, record, before, failed):
        try:
            after = self._snapshot()
        except READ_ERRORS:
            after = None
        try:
            if before and after:
                seconds = round(after["t"] - before["t"], 2)
                record["seconds"] = seconds
                usage = {
                    "self": _delta(after["self"], before["self"], RUSAGE),
                    "children": _delta(after["children"], before["children"], RUSAGE),
                }
                record["rusage"] = usage
                cpu = sum(
                    (part or {}).get(key, 0.0)
                    for part in usage.values()
                    for key in ("utime", "stime")
                )
                record["cpu_s"] = round(cpu, 2)
                record["parallelism"] = round(cpu / seconds, 2) if seconds > 0 else None
                record["cgroup_cpu"] = _delta(after["cgroup"], before["cgroup"], CPU_STAT)
                if record["cgroup_cpu"] is not None:
                    record["cgroup_cpu"]["container_wide"] = True
                if isinstance(after["oom_kill"], int) and isinstance(before["oom_kill"], int):
                    record["oom_kills"] = after["oom_kill"] - before["oom_kill"]
                seen = _CHILDREN[before["sampled"] :]
                threads = [c["peak_threads"] for c in seen if c.get("peak_threads")]
                hwm = [c["vm_hwm_kb"] for c in seen if c.get("vm_hwm_kb")]
                record["child_peak_threads"] = max(threads) if threads else None
                record["child_vm_hwm_kb"] = max(hwm) if hwm else None
            if failed:
                record["failed"] = True
        except READ_ERRORS:
            pass
        self.steps[name] = record
        if self.emit is not None and "seconds" in record:
            try:
                self.emit(
                    "log",
                    message=f"NoulXP step {name}: {record['seconds']:.1f} s"
                    + (f", {record['threads']} threads" if record.get("threads") else "")
                    + (f", {record['device']}" if record.get("device") else ""),
                    step=name,
                    seconds=record["seconds"],
                    threads=record.get("threads"),
                    device=record.get("device"),
                )
            except READ_ERRORS:  # a closed log (BrokenPipeError) never ends a step
                pass
