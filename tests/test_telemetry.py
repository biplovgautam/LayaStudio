"""The build's telemetry (telemetry.py): readings from fake cgroup and /proc trees in tmp_path,
steps that record what they cost and never change what they measure. Nothing real runs."""

import json
import sys
import time

import pytest

from systemone_studio import engine, telemetry

GB = 10**9
# What cgroup v1 shows for "no limit": the largest page-aligned 63-bit number, with 4 KiB pages
# and with 64 KiB pages.
V1_NO_LIMIT = (9223372036854771712, 9223372036854710272)


def cgroup2(root, quota_by_level, proc_path="/"):
    """A cgroup v2 tree: {relative folder: cpu.max text}; /proc/self/cgroup names proc_path."""
    root.mkdir(parents=True, exist_ok=True)
    for folder, text in quota_by_level.items():
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "cpu.max").write_text(text)
    proc = root.parent / "proc-self-cgroup"
    proc.write_text(f"0::{proc_path}\n")
    return root, proc


def cgroup1(root, levels, proc_path="/docker/7f3a9c"):
    """A cgroup v1 memory hierarchy, {folder under root/memory: {file: text}}, and the
    /proc/self/cgroup of a Docker container on a v1 host: no v2 line, and a memory cgroup named
    by a path the container's own mount does not show (its root is the container's cgroup)."""
    for folder, files in levels.items():
        (root / "memory" / folder).mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (root / "memory" / folder / name).write_text(text)
    proc = root.parent / "proc-self-cgroup"
    proc.write_text(
        f"12:memory:{proc_path}\n4:cpu,cpuacct:{proc_path}\n1:name=systemd:{proc_path}\n"
    )
    return root, proc


def v1_files(limit, usage, total_inactive, inactive=0):
    """One v1 cgroup's limit, usage and memory.stat, which has the cgroup's own lines (its
    inactive_file leaves its children out) and the total_ ones, as the kernel writes both."""
    return {
        "memory.limit_in_bytes": f"{limit}\n",
        "memory.usage_in_bytes": f"{usage}\n",
        "memory.stat": (
            f"cache {total_inactive}\nrss {usage - total_inactive}\nshmem 0\n"
            f"inactive_anon 0\nactive_anon {usage - total_inactive}\n"
            f"inactive_file {inactive}\nactive_file 0\nhierarchical_memory_limit {limit}\n"
            f"total_cache {total_inactive}\ntotal_rss {usage - total_inactive}\n"
            f"total_inactive_file {total_inactive}\ntotal_active_file 0\n"
        ),
    }


def host_meminfo(tmp_path, available):
    """A /proc/meminfo whose MemAvailable is `available` bytes: the host's, in a container too."""
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:       {503 * GB // 1024} kB\n"
        f"MemFree:        {9 * GB // 1024} kB\n"
        f"MemAvailable:   {available // 1024} kB\n"
    )
    return path


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
    assert telemetry.free_memory(root, proc, tmp_path / "no-meminfo") == 8192 - 2048
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 100 kB\nMemAvailable: 64 kB\n")
    (root / "memory.max").write_text("max\n")
    assert telemetry.free_memory(root, proc, meminfo) == 64 * 1024
    assert telemetry.free_memory(tmp_path / "x", tmp_path / "y", tmp_path / "z") is None


def test_under_cgroup_v1_free_memory_is_the_pods_not_the_hosts(tmp_path):
    """A RunPod A40 pod (cgroup v1): the host's MemAvailable says 480 GB, the pod's limit is
    46.6 GB. With 30 GB charged, 6 GB of it inactive page cache, 22.6 GB is free. The line read
    is total_inactive_file (the pod's child cgroups too, as memory.usage_in_bytes counts them),
    not the pod's own inactive_file."""
    limit = int(46.6 * GB)
    root, proc = cgroup1(tmp_path / "cg", {".": v1_files(limit, 30 * GB, 6 * GB, inactive=GB)})
    host = host_meminfo(tmp_path, 480 * GB)
    found = telemetry.available_memory(root, proc, host)
    assert found == {
        "free": limit - 24 * GB,
        "source": "cgroup1",
        "host_free": 480 * GB,
        "cgroup": "cgroup1",
        "max": limit,
        "current": 30 * GB,
        "inactive_file": 6 * GB,
        "container_free": limit - 24 * GB,
    }
    assert telemetry.free_memory(root, proc, host) == limit - 24 * GB
    # A hybrid host's /proc/self/cgroup has a v2 line too: no memory.max under it, so v1.
    proc.write_text("12:memory:/docker/7f3a9c\n0::/docker/7f3a9c\n")
    assert telemetry.available_memory(root, proc, host) == found


def test_under_cgroup_v2_free_memory_is_the_containers_not_the_hosts(tmp_path):
    """A 51.2 GB limit with 30 GB charged: 12 GB of it page cache, 5 GB of that inactive (1 GB
    shmem). 26.2 GB is free: the active page cache and shmem count as used, as under v1."""
    limit = int(51.2 * GB)
    root, proc = cgroup2(tmp_path / "cg", {".": "max 100000"})
    (root / "memory.max").write_text(f"{limit}\n")
    (root / "memory.current").write_text(f"{30 * GB}\n")
    (root / "memory.stat").write_text(
        f"anon {17 * GB}\nfile {12 * GB}\nkernel {GB}\nshmem {GB}\n"
        f"active_file {7 * GB}\ninactive_file {5 * GB}\n"
    )
    host = host_meminfo(tmp_path, 480 * GB)
    assert telemetry.available_memory(root, proc, host) == {
        "free": limit - 25 * GB,
        "source": "cgroup2",
        "host_free": 480 * GB,
        "cgroup": "cgroup2",
        "max": limit,
        "current": 30 * GB,
        "inactive_file": 5 * GB,
        "container_free": limit - 25 * GB,
    }


# Where each version keeps one cgroup's limit, usage and inactive page cache, and the
# /proc/self/cgroup line naming it: v2 at the root, v1 under memory/ with a path it cannot see.
VERSIONS = {
    "cgroup2": ("", "memory.max", "memory.current", "inactive_file", "0::/\n"),
    "cgroup1": (
        "memory",
        "memory.limit_in_bytes",
        "memory.usage_in_bytes",
        "total_inactive_file",
        "12:memory:/docker/7f3a9c\n",
    ),
}


def one_cgroup(tmp_path, version, limit, usage):
    """tmp_path/cg with one cgroup of `version` limited to `limit`, `usage` charged: its folder,
    the /proc/self/cgroup naming it, and its usage file and memory.stat line's names."""
    mount, limit_file, usage_file, key, line = VERSIONS[version]
    folder = tmp_path / "cg" / mount
    folder.mkdir(parents=True, exist_ok=True)
    (folder / limit_file).write_text(f"{limit}\n")
    (folder / usage_file).write_text(f"{usage}\n")
    proc = tmp_path / "proc-self-cgroup"
    proc.write_text(line)
    return folder, proc, usage_file, key


@pytest.mark.parametrize("version", ["cgroup2", "cgroup1"])
def test_both_versions_count_the_inactive_page_cache_alike(tmp_path, version):
    """The limit less the usage, the usage less its inactive page cache: at most the usage
    (memory.stat is read apart from it), the usage whole without that line, never below 0."""
    folder, proc, usage_file, key = one_cgroup(tmp_path, version, 50 * GB, 30 * GB)
    host = host_meminfo(tmp_path, 480 * GB)

    def read():
        return telemetry.available_memory(tmp_path / "cg", proc, host)

    (folder / "memory.stat").write_text(f"{key} {6 * GB}\n")
    assert read()["free"] == 26 * GB and read()["source"] == version
    assert read()["inactive_file"] == 6 * GB and read()["container_free"] == 26 * GB
    (folder / "memory.stat").write_text(f"{key} {40 * GB}\n")  # stale: at most the usage
    assert read()["free"] == 50 * GB and read()["inactive_file"] == 30 * GB
    (folder / "memory.stat").write_text(f"anon {GB}\nactive_file {20 * GB}\n")  # no such line
    assert read()["free"] == 20 * GB and read()["inactive_file"] is None
    if version == "cgroup1":  # the cgroup's own line, its children left out: not read
        (folder / "memory.stat").write_text(f"inactive_file {6 * GB}\n")
        assert read()["free"] == 20 * GB
    (folder / "memory.stat").unlink()
    assert read()["free"] == 20 * GB and read()["inactive_file"] is None
    (folder / usage_file).write_text(f"{60 * GB}\n")  # over the limit
    assert read()["free"] == 0 and read()["source"] == version


@pytest.mark.parametrize(
    "version, limit",
    [
        ("cgroup2", "max"),
        ("cgroup2", str(2**60)),
        ("cgroup1", str(V1_NO_LIMIT[0])),
        ("cgroup1", str(V1_NO_LIMIT[1])),
        ("cgroup1", "-1"),
        ("cgroup1", "lots"),
    ],
)
def test_no_limit_or_an_absurd_one_leaves_the_hosts_memory(tmp_path, version, limit):
    """No limit ("max"), cgroup v1's "no limit", anything of an exbibyte or more, or a value
    that is not one: the host's MemAvailable alone."""
    folder, proc, _, key = one_cgroup(tmp_path, version, limit, 300 * GB)
    (folder / "memory.stat").write_text(f"{key} {100 * GB}\n")
    host = host_meminfo(tmp_path, 480 * GB)
    found = telemetry.available_memory(tmp_path / "cg", proc, host)
    assert found == {"free": 480 * GB, "source": "meminfo", "host_free": 480 * GB}


def test_a_limit_above_what_the_host_has_left_leaves_the_hosts_memory(tmp_path):
    """A 500 GB limit on a host with 20 GB available: 20 GB is free (source "meminfo"), the
    container's reading kept beside it; without /proc/meminfo, the container's alone."""
    root, proc = cgroup1(tmp_path / "cg", {".": v1_files(500 * GB, 30 * GB, 6 * GB)})
    found = telemetry.available_memory(root, proc, host_meminfo(tmp_path, 20 * GB))
    assert found == {
        "free": 20 * GB,
        "source": "meminfo",
        "host_free": 20 * GB,
        "cgroup": "cgroup1",
        "max": 500 * GB,
        "current": 30 * GB,
        "inactive_file": 6 * GB,
        "container_free": 476 * GB,
    }
    alone = telemetry.available_memory(root, proc, tmp_path / "no-meminfo")
    assert alone["free"] == 476 * GB and alone["source"] == "cgroup1"
    assert "host_free" not in alone


def test_the_level_that_leaves_least_is_what_is_free(tmp_path):
    """Seen whole (the host's tree, or a process in a sub-cgroup): every folder from the
    process's own cgroup up to the root counts, the root's "no limit" does not, and the one
    that leaves the least is what is free, under either version."""
    host = host_meminfo(tmp_path, 480 * GB)
    levels = {
        ".": v1_files(V1_NO_LIMIT[0], 300 * GB, 100 * GB),
        "docker": v1_files(V1_NO_LIMIT[0], 200 * GB, 50 * GB),
        "docker/7f3a9c": v1_files(64 * GB, 40 * GB, 10 * GB),
    }
    root, proc = cgroup1(tmp_path / "v1", levels)
    found = telemetry.available_memory(root, proc, host)
    assert found["source"] == "cgroup1" and found["max"] == 64 * GB
    assert found["free"] == 34 * GB
    # v2: the container's own folder allows 64 GB; the pod above it, 40 GB with 35 GB charged.
    root, proc = cgroup2(tmp_path / "v2", {"pod/ctr": "max 100000"}, "/pod/ctr")
    for folder, limit, usage in (("pod", 40 * GB, 35 * GB), ("pod/ctr", 64 * GB, 30 * GB)):
        (root / folder / "memory.max").write_text(f"{limit}\n")
        (root / folder / "memory.current").write_text(f"{usage}\n")
        (root / folder / "memory.stat").write_text(f"inactive_file {GB}\n")
    found = telemetry.available_memory(root, proc, host)
    assert found["source"] == "cgroup2" and found["max"] == 40 * GB
    assert found["free"] == 6 * GB and found["current"] == 35 * GB
    # A process in a sub-cgroup of its container (systemd's init.scope): no limit there, the
    # container's root has it.
    (root / "pod" / "memory.max").write_text("max\n")
    (root / "pod/ctr" / "memory.max").write_text("max\n")
    (root / "memory.max").write_text(f"{48 * GB}\n")
    (root / "memory.current").write_text(f"{36 * GB}\n")
    found = telemetry.available_memory(root, proc, host)
    assert found["max"] == 48 * GB and found["free"] == 12 * GB


def test_without_a_limit_the_hosts_memory_and_without_either_none(tmp_path):
    host = host_meminfo(tmp_path, 480 * GB)
    nothing, no_proc = tmp_path / "no-cgroup", tmp_path / "no-proc"
    assert telemetry.available_memory(nothing, no_proc, host) == {
        "free": 480 * GB,
        "source": "meminfo",
        "host_free": 480 * GB,
    }
    assert telemetry.free_memory(nothing, no_proc, host) == 480 * GB
    unknown = {"free": None, "source": None}  # macOS
    assert telemetry.available_memory(nothing, no_proc, tmp_path / "no-meminfo") == unknown
    host.write_text("MemTotal: 1 kB\nMemAvailable: lots kB\n")
    assert telemetry.available_memory(nothing, no_proc, host) == unknown
    host.write_text("MemAvailable:\n")
    assert telemetry.available_memory(nothing, no_proc, host) == unknown
    assert telemetry.free_memory(nothing, no_proc, host) is None


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
