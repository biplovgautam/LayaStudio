"""The build's telemetry (telemetry.py): readings from fake cgroup and /proc trees in tmp_path,
steps that record what they cost and never change what they measure. Nothing real runs."""

import json
import sys
import time

import pytest

from layastudio import engine, telemetry


def cgroup2(root, quota_by_level, proc_path="/"):
    """A cgroup v2 tree: {relative folder: cpu.max text}; /proc/self/cgroup names proc_path."""
    root.mkdir(parents=True, exist_ok=True)
    for folder, text in quota_by_level.items():
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "cpu.max").write_text(text)
    proc = root.parent / "proc-self-cgroup"
    proc.write_text(f"0::{proc_path}\n")
    return root, proc


def test_cgroup_readings_present_absent_and_malformed(tmp_path):
    root, proc = cgroup2(tmp_path / "cg", {".": "max 100000"})
    assert telemetry.cpu_stat(root, proc) is None
    assert telemetry.memory(root, proc) == {
        "current": None,
        "peak": None,
        "max": None,
        "oom_kill": None,
    }
    (root / "cpu.stat").write_text(
        "usage_usec 1000\nuser_usec 600\nnr_periods 10\nnr_throttled 2\nthrottled_usec 50\n"
    )
    (root / "memory.current").write_text("2048\n")
    (root / "memory.peak").write_text("4096\n")
    (root / "memory.max").write_text("max\n")
    (root / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 1\n")
    assert telemetry.cpu_stat(root, proc) == {
        "usage_usec": 1000,
        "nr_periods": 10,
        "nr_throttled": 2,
        "throttled_usec": 50,
    }
    assert telemetry.memory(root, proc) == {
        "current": 2048,
        "peak": 4096,
        "max": "max",
        "oom_kill": 1,
    }
    (root / "cpu.stat").write_text("usage_usec lots\nnr_throttled\n\x00")
    assert telemetry.cpu_stat(root, proc) is None
    (root / "memory.max").write_text("8192\n")
    assert telemetry.free_memory(root, proc) == 8192 - 2048
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 100 kB\nMemAvailable: 64 kB\n")
    (root / "memory.max").write_text("max\n")
    assert telemetry.free_memory(root, proc, meminfo) == 64 * 1024
    assert telemetry.free_memory(tmp_path / "x", tmp_path / "y", tmp_path / "z") is None


def test_free_memory_counts_the_page_cache_as_reclaimable(tmp_path):
    """The A40 pod's limit (51.2 GB) with 30 GB charged, 12 GB of it page cache (1 GB of that
    shmem): 32.2 GB free, not the 21.2 GB that memory.max minus memory.current says."""
    GB = 10**9
    root, proc = cgroup2(tmp_path / "cg", {".": "max 100000"})
    (root / "memory.max").write_text(f"{int(51.2 * GB)}\n")
    (root / "memory.current").write_text(f"{30 * GB}\n")
    (root / "memory.stat").write_text(
        f"anon {17 * GB}\nfile {12 * GB}\nkernel {GB}\nshmem {GB}\n"
        f"active_file {7 * GB}\ninactive_file {5 * GB}\n"
    )
    found = telemetry.available_memory(root, proc)
    assert found == {
        "free": int(51.2 * GB) - 19 * GB,
        "source": "cgroup",
        "max": int(51.2 * GB),
        "current": 30 * GB,
        "page_cache": 11 * GB,
    }
    assert telemetry.free_memory(root, proc) == found["free"]
    (root / "memory.stat").write_text(f"anon {GB}\nfile {40 * GB}\nshmem 0\n")  # stale, racy
    assert telemetry.available_memory(root, proc)["free"] == int(51.2 * GB)  # at most the limit
    (root / "memory.stat").write_text("anon 1\n")  # no file line: memory.current, whole
    assert telemetry.free_memory(root, proc) == int(51.2 * GB) - 30 * GB
    (root / "memory.stat").unlink()
    assert telemetry.available_memory(root, proc)["page_cache"] is None
    assert telemetry.free_memory(root, proc) == int(51.2 * GB) - 30 * GB
    (root / "memory.current").write_text(f"{60 * GB}\n")  # over the limit
    assert telemetry.free_memory(root, proc) == 0
    assert telemetry.available_memory(tmp_path / "x", tmp_path / "y", tmp_path / "z") == {
        "free": None,
        "source": None,
    }


def test_the_machine_block_reads_cpuinfo(tmp_path):
    root, proc = cgroup2(tmp_path / "cg", {".": "850000 100000"})
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz\n"
        "flags\t\t: fpu avx2 avx512f avx512_vnni\n\nprocessor\t: 1\n"
    )
    info = telemetry.machine("laya", root, proc, cpuinfo)
    assert info["cpu_max"] == "850000 100000" and info["cpu_model"].startswith("Intel")
    assert "avx512_vnni" in info["cpu_flags"] and "llama_cpp_system_info" not in info
    empty = telemetry.machine("julia", tmp_path / "nothing", tmp_path / "no-proc", tmp_path / "no")
    assert empty["cpu_max"] is None and empty["cpu_flags"] is None  # a Mac names its CPU


def steps_report(tmp_path):
    root, proc = cgroup2(tmp_path / "cg", {".": "max 100000"})
    (root / "cpu.stat").write_text("usage_usec 1000\nnr_periods 1\nnr_throttled 0\n")
    report = {}
    return report, telemetry.Steps(report, root=root, proc=proc), root


def test_a_step_is_timed_and_its_cgroup_counters_are_deltas(tmp_path):
    report, steps, root = steps_report(tmp_path)
    events = []
    steps.emit = lambda kind, **data: events.append((kind, data))
    with steps("check", threads=8, device="cpu") as record:
        (root / "cpu.stat").write_text("usage_usec 4000\nnr_periods 5\nnr_throttled 2\n")
        record["extra"] = 1
    step = report["steps"]["check"]
    assert step["threads"] == 8 and step["device"] == "cpu" and step["extra"] == 1
    assert step["seconds"] >= 0 and "failed" not in step
    assert step["cgroup_cpu"] == {
        "usage_usec": 3000,
        "nr_periods": 4,
        "nr_throttled": 2,
        "container_wide": True,
    }
    assert events and events[0][1]["step"] == "check" and events[0][1]["threads"] == 8
    path = tmp_path / "report.json"
    engine.write_json(path, report)  # JSON primitives only
    assert json.loads(path.read_text()) == report


@pytest.mark.parametrize("error", [RuntimeError("boom"), engine.Cancelled(), KeyboardInterrupt()])
def test_a_step_that_raises_raises_unchanged_and_is_still_recorded(tmp_path, error):
    report, steps, _ = steps_report(tmp_path)
    with pytest.raises(type(error)) as raised:
        with steps("export", threads=4):
            raise error
    assert raised.value is error
    assert report["steps"]["export"]["failed"] is True
    assert "seconds" in report["steps"]["export"]


def test_a_reading_that_fails_records_less_and_raises_nothing(tmp_path, monkeypatch):
    report, steps, _ = steps_report(tmp_path)

    def broken(*_args, **_kwargs):
        raise OSError("gone")

    monkeypatch.setattr(telemetry, "cpu_stat", broken)
    with steps("validate"):
        pass
    assert report["steps"]["validate"] == {}  # nothing measured, nothing raised

    def closed(*_args, **_kwargs):
        raise BrokenPipeError("the log is gone")

    monkeypatch.undo()
    steps.emit = closed
    with steps("check"):
        pass
    assert "seconds" in report["steps"]["check"]


def test_the_sampler_reads_proc_status(tmp_path, monkeypatch):
    status = tmp_path / "proc" / "123"
    status.mkdir(parents=True)
    (status / "status").write_text("Name:\tpython\nVmHWM:\t  2048 kB\nThreads:\t9\n")
    monkeypatch.setattr(sys, "platform", "linux")
    sampler = telemetry.Sampler(123, every=0.01, proc=tmp_path / "proc")
    deadline = time.monotonic() + 5
    while sampler.samples == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    (status / "status").write_text("VmHWM:\t4096 kB\nThreads:\t5\n")
    time.sleep(0.05)
    seen = sampler.stop()
    assert seen["peak_threads"] == 9 and seen["vm_hwm_kb"] == 4096
    # Not on Linux, or no such process: it reads nothing.
    monkeypatch.setattr(sys, "platform", "darwin")
    assert telemetry.Sampler(123, proc=tmp_path / "proc").stop()["samples"] == 0
