"""LayaStudio is System One Studio now, and its package `systemone_studio`; the old name keeps
working: `import layastudio` is systemone_studio, every `layastudio.<module>` is
systemone_studio's own module (one DeprecationWarning for all of them), and `python -m
layastudio[.<module>]` and the `layastudio` command run the same code (systemone_studio/_alias.py).

Each old-name import runs in a fresh interpreter: the alias installs a finder for the process.
"""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "systemone_studio"
MODULES = sorted(
    p.stem for p in PACKAGE.glob("*.py") if p.stem not in ("__init__", "__main__", "_alias")
)
# What other code imports under the old name: the trainer image (studio-image/: build_check,
# smoke), worker-studio and the platform's API tests, the research harness and run scripts.
CONSUMED = (
    "cloud",
    "datasets",
    "decider",
    "engine",
    "examples",
    "export",
    "gguf",
    "kinds",
    "laya_mlx_free",
    "noulxp_package",
    "publish_systemone",
    "runtime",
)


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


# ----------------------------------------------------------------------------- imports


def test_every_module_under_the_old_name_is_the_new_module(tmp_path):
    script = tmp_path / "old_names.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import importlib, json, sys, warnings

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                import layastudio
                import layastudio.engine
                from layastudio import cloud, gguf
                from layastudio.runtime import torch_device
                for name in {MODULES!r}:
                    importlib.import_module("layastudio." + name)
                import layastudio  # again: no second warning

            import systemone_studio

            deprecations = [
                (w.filename, w.lineno, str(w.message))
                for w in caught
                if issubclass(w.category, DeprecationWarning) and "layastudio" in str(w.message)
            ]
            print(json.dumps({{
                "same_package": layastudio is systemone_studio,
                "different": [
                    name for name in {MODULES!r}
                    if sys.modules["layastudio." + name] is not sys.modules["systemone_studio." + name]
                    or getattr(layastudio, name) is not sys.modules["systemone_studio." + name]
                ],
                "from_import": cloud is systemone_studio.cloud and gguf is systemone_studio.gguf,
                "function": torch_device is systemone_studio.runtime.torch_device,
                "own_specs": [
                    name for name in {MODULES!r}
                    if sys.modules["systemone_studio." + name].__spec__.name
                    != "systemone_studio." + name
                ],
                "version": layastudio.__version__,
                "repository": layastudio.REPOSITORY == systemone_studio.REPOSITORY,
                "deprecations": deprecations,
            }}))
            """
        )
    )
    found = last_json(python(None, str(script)))
    assert found["same_package"] and found["from_import"] and found["function"]
    assert found["different"] == []
    assert found["own_specs"] == []  # the alias leaves each module its own __spec__
    assert found["version"] and found["repository"]
    # One warning, on the importer's own line (where `import layastudio` is), not importlib's.
    [(filename, lineno, message)] = found["deprecations"]
    assert Path(filename) == script and lineno == 6
    assert "systemone_studio" in message


@pytest.mark.parametrize("name", MODULES)
def test_each_old_name_finds_the_new_file(name):
    found = last_json(
        python(
            f"""
            import importlib.util, json
            import layastudio
            spec = importlib.util.find_spec("layastudio.{name}")
            print(json.dumps(spec.origin))
            """
        )
    )
    assert Path(found) == PACKAGE / f"{name}.py"


def test_the_modules_other_code_imports_are_all_there():
    assert set(CONSUMED) <= set(MODULES)


def test_a_patch_through_the_old_name_is_seen_by_the_new_module():
    found = last_json(
        python(
            """
            import json
            from unittest import mock
            from systemone_studio import engine
            with mock.patch("layastudio.engine.WORKSPACE", "patched"):
                inside = engine.WORKSPACE
            print(json.dumps([inside, engine.WORKSPACE != "patched"]))
            """,
            "-W",
            "ignore::DeprecationWarning",
        )
    )
    assert found == ["patched", True]


def test_a_stand_in_put_in_under_the_old_name_is_the_one_imported():
    """As the platform's tests import cloud.py without MLX: a stub for
    layastudio.laya_mlx_free in sys.modules, then layastudio.cloud."""
    found = last_json(
        python(
            """
            import importlib, json, sys, types
            from unittest import mock
            stub = types.ModuleType("layastudio.laya_mlx_free")
            stub.laya_mlx_module = lambda name: mock.MagicMock()
            sys.modules["layastudio.laya_mlx_free"] = stub
            cloud = importlib.import_module("layastudio.cloud")
            import systemone_studio.engine
            print(json.dumps([
                sys.modules["systemone_studio.laya_mlx_free"] is stub,
                cloud is sys.modules["systemone_studio.cloud"],
                systemone_studio.engine.laya_mlx_module is stub.laya_mlx_module,
                callable(cloud.hyperparameters),
            ]))
            """,
            "-W",
            "ignore::DeprecationWarning",
        )
    )
    assert found == [True, True, True, True]


def test_a_module_dropped_from_sys_modules_is_imported_afresh_under_either_name():
    found = last_json(
        python(
            """
            import importlib, json, sys
            import layastudio.kinds
            first = sys.modules["systemone_studio.kinds"]
            del sys.modules["systemone_studio.kinds"]
            fresh = importlib.import_module("systemone_studio.kinds")
            del sys.modules["layastudio.kinds"]
            again = importlib.import_module("layastudio.kinds")
            print(json.dumps([fresh is not first, again is fresh]))
            """,
            "-W",
            "ignore::DeprecationWarning",
        )
    )
    assert found == [True, True]


def test_an_old_name_that_never_existed_is_not_found():
    result = python("import layastudio.nothing_here", "-W", "ignore::DeprecationWarning")
    assert result.returncode != 0
    assert "No module named 'layastudio.nothing_here'" in result.stderr


# ----------------------------------------------------------------------------- commands


def test_python_m_layastudio_runs_the_studio(tmp_path):
    new = python(None, "-m", "systemone_studio", "--help", cwd=tmp_path)
    old = python(None, "-m", "layastudio", "--help", cwd=tmp_path)
    assert new.returncode == 0 and old.returncode == 0, old.stderr
    assert old.stdout == new.stdout
    assert new.stdout.startswith("usage: systemone-studio") and "System One Studio" in new.stdout
    train = python(None, "-m", "layastudio", "train", "--help", cwd=tmp_path)
    assert train.returncode == 0 and train.stdout.startswith("usage: systemone-studio train")


@pytest.mark.parametrize(
    "module, code, expected",
    [
        ("engine", 1, "usage: python -m systemone_studio.engine run <job_dir>"),
        ("gguf_reference", 1, "usage: python -m systemone_studio.gguf_reference"),
        ("cloud", 0, "usage: systemone-studio train"),
    ],
)
def test_python_m_layastudio_module_runs_that_module(tmp_path, module, code, expected):
    args = ["--help"] if module == "cloud" else []
    result = python(None, "-m", f"layastudio.{module}", *args, cwd=tmp_path)
    assert result.returncode == code, result.stderr
    assert expected in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_both_commands_are_declared_with_one_target():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["name"] == "systemone-studio"
    scripts = project["project"]["scripts"]
    assert scripts["systemone-studio"] == scripts["layastudio"] == "systemone_studio.server:main"
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "systemone_studio",
        "layastudio",
    ]


def installed_commands():
    try:
        dist = importlib.metadata.distribution("systemone-studio")
    except importlib.metadata.PackageNotFoundError:
        return None
    scripts = {e.name: e.value for e in dist.entry_points if e.group == "console_scripts"}
    folder = Path(sys.executable).parent
    paths = {name: shutil.which(name, path=str(folder)) for name in scripts}
    return scripts, paths


@pytest.mark.skipif(installed_commands() is None, reason="systemone-studio is not installed")
def test_the_installed_layastudio_command_is_the_studio(tmp_path):
    scripts, paths = installed_commands()
    assert scripts == {
        "systemone-studio": "systemone_studio.server:main",
        "layastudio": "systemone_studio.server:main",
    }
    if not all(paths.values()):
        pytest.skip(f"the commands are not beside this interpreter: {paths}")
    run = {
        name: subprocess.run(
            [path, "--help"], capture_output=True, text=True, cwd=tmp_path, timeout=120
        )
        for name, path in paths.items()
    }
    assert all(r.returncode == 0 for r in run.values()), run
    assert run["layastudio"].stdout == run["systemone-studio"].stdout
    assert run["layastudio"].stdout.startswith("usage: systemone-studio")
