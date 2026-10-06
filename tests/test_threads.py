"""CPU threads from a container's quota (runtime.cpu_budget): fake cgroup and /proc trees in
tmp_path, so they run anywhere."""

import os

import pytest

from systemone_studio import runtime

PERIOD = 100000


def cgroup2(root, quota_by_level, proc_path="/"):
    """A cgroup v2 tree: {relative folder: cpu.max text}; /proc/self/cgroup names proc_path."""
    root.mkdir(parents=True, exist_ok=True)
    for folder, text in quota_by_level.items():
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "cpu.max").write_text(text)
    proc = root.parent / "proc-self-cgroup"
    proc.write_text(f"0::{proc_path}\n")
    return root, proc


def budget(root, proc, environ=None, cpus=64, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(runtime, "affinity", lambda: cpus)
    return runtime.cpu_budget(root=root, proc=proc, environ=environ or {})


# ----------------------------------------------------------------------------- the quota


def test_a_v2_quota_is_floored(tmp_path, monkeypatch):
    root, proc = cgroup2(tmp_path / "cg", {".": "850000 100000\n"})
    found = budget(root, proc, monkeypatch=monkeypatch)
    assert found == {"threads": 8, "quota": 8.5, "source": "cgroup2", "affinity": 64}


def test_the_tightest_level_of_a_nested_v2_tree_counts(tmp_path, monkeypatch):
    root, proc = cgroup2(
        tmp_path / "cg",
        {".": "max 100000", "pod": "400000 100000", "pod/job": "max 100000"},
        proc_path="/pod/job",
    )
    assert budget(root, proc, monkeypatch=monkeypatch)["threads"] == 4
    # A tighter child wins over its parent.
    (root / "pod/job/cpu.max").write_text("250000 100000")
    assert budget(root, proc, monkeypatch=monkeypatch)["threads"] == 2


def test_max_means_no_quota(tmp_path, monkeypatch):
    root, proc = cgroup2(tmp_path / "cg", {".": "max 100000"})
    monkeypatch.setattr(runtime, "_unquoted", lambda: 7)
    found = budget(root, proc, monkeypatch=monkeypatch)
    assert found == {"threads": 7, "quota": None, "source": "fallback", "affinity": 64}


def test_a_cgroup_path_outside_the_namespace_reads_the_root(tmp_path, monkeypatch):
    root, proc = cgroup2(tmp_path / "cg", {".": "300000 100000"}, proc_path="/../../elsewhere")
    assert budget(root, proc, monkeypatch=monkeypatch)["threads"] == 3


def test_cgroup_v1_quota_and_unlimited(tmp_path, monkeypatch):
    root = tmp_path / "cg"
    (root / "cpu,cpuacct").mkdir(parents=True)
    (root / "cpu,cpuacct/cpu.cfs_quota_us").write_text("-1\n")
    (root / "cpu,cpuacct/cpu.cfs_period_us").write_text(f"{PERIOD}\n")
    proc = tmp_path / "proc"
    proc.write_text("4:cpu,cpuacct:/\n")
    monkeypatch.setattr(runtime, "_unquoted", lambda: 5)
    assert budget(root, proc, monkeypatch=monkeypatch)["source"] == "fallback"
    (root / "cpu,cpuacct/cpu.cfs_quota_us").write_text("600000\n")
    found = budget(root, proc, monkeypatch=monkeypatch)
    assert found == {"threads": 6, "quota": 6.0, "source": "cgroup1", "affinity": 64}


@pytest.mark.parametrize("junk", ["", "lots 100000", "850000 zero", "\x00\x01", "-5 100000"])
def test_junk_is_no_quota(tmp_path, monkeypatch, junk):
    root, proc = cgroup2(tmp_path / "cg", {".": junk})
    monkeypatch.setattr(runtime, "_unquoted", lambda: 3)
    assert budget(root, proc, monkeypatch=monkeypatch)["source"] == "fallback"


@pytest.mark.skipif(os.name == "nt" or os.getuid() == 0, reason="needs file permissions")
def test_an_unreadable_file_is_no_quota(tmp_path, monkeypatch):
    root, proc = cgroup2(tmp_path / "cg", {".": "400000 100000"})
    (root / "cpu.max").chmod(0)
    monkeypatch.setattr(runtime, "_unquoted", lambda: 3)
    try:
        assert budget(root, proc, monkeypatch=monkeypatch)["source"] == "fallback"
    finally:
        (root / "cpu.max").chmod(0o644)


def test_no_cgroup_files_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_unquoted", lambda: 4)
    found = budget(tmp_path / "none", tmp_path / "none-proc", monkeypatch=monkeypatch)
    assert found["threads"] == 4 and found["source"] == "fallback"


def test_a_quota_under_one_cpu_gives_one_thread(tmp_path, monkeypatch):
    root, proc = cgroup2(tmp_path / "cg", {".": "50000 100000"})
    assert budget(root, proc, monkeypatch=monkeypatch)["threads"] == 1


def test_the_quota_is_capped_by_affinity_and_by_32(tmp_path, monkeypatch):
    root, proc = cgroup2(tmp_path / "cg", {".": "6400000 100000"})
    assert budget(root, proc, cpus=10, monkeypatch=monkeypatch)["threads"] == 10
    assert budget(root, proc, cpus=128, monkeypatch=monkeypatch)["threads"] == 32


def test_the_environment_overrides_and_bad_values_are_ignored(tmp_path, monkeypatch, capsys):
    root, proc = cgroup2(tmp_path / "cg", {".": "400000 100000"})
    found = budget(root, proc, {"LAYASTUDIO_THREADS": "12"}, monkeypatch=monkeypatch)
    assert found == {"threads": 12, "quota": None, "source": "env", "affinity": 64}
    runtime._warned.clear()
    for bad in ("8.5", "0", "-2", "abc"):
        found = budget(root, proc, {"LAYASTUDIO_THREADS": bad}, monkeypatch=monkeypatch)
        assert found["threads"] == 4 and found["source"] == "cgroup2"
    assert capsys.readouterr().err.count("LAYASTUDIO_THREADS") == 4  # once per value


def test_without_a_quota_a_mac_or_linux_uses_at_most_8():
    assert 1 <= runtime._unquoted() <= runtime.UNQUOTED_THREADS
    assert 1 <= runtime.cpu_threads() <= runtime.MAX_THREADS
