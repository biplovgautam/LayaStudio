"""The studio starts on machines without MLX (Linux, Windows, Intel Macs).

laya-mlx installs MLX only on Apple silicon, and its package __init__ imports its MLX agent, so
the studio must not import laya_mlx's plain-Python parts the usual way there.
"""

import subprocess
import sys

from systemone_studio.laya_mlx_free import laya_mlx_module

NO_MLX = "import sys; sys.modules['mlx'] = None; sys.modules['mlx.core'] = None; sys.modules['mlx.nn'] = None; "


def run(code):
    return subprocess.run(
        [sys.executable, "-c", NO_MLX + code], capture_output=True, text=True, timeout=120
    )


def test_laya_mlx_itself_cannot_be_imported_without_mlx():
    """The failure the studio works around: if this ever passes, laya-mlx fixed it upstream."""
    out = run("import laya_mlx.common")
    assert out.returncode != 0 and "mlx" in out.stderr


def test_the_studio_imports_without_mlx():
    out = run(
        "import systemone_studio.engine, systemone_studio.export, systemone_studio.snake, systemone_studio.server; print('ok')"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_the_plain_python_parts_load_from_their_files_without_mlx():
    code = (
        "from systemone_studio.laya_mlx_free import laya_mlx_module as m; "
        "c = m('common'); t = m('tokenizer'); g = m('snake.game'); "
        "print(sorted(c.QTYPES)[:1], callable(c.build_sequence), t.Tokenizer.__name__, g.SnakeGame.__name__)"
    )
    out = run(code)
    assert out.returncode == 0, out.stderr
    assert "Tokenizer SnakeGame" in out.stdout


def test_with_mlx_the_normal_import_is_used():
    import importlib

    try:
        normal = importlib.import_module("laya_mlx.common")
    except ImportError:  # this machine has no MLX: covered by the tests above
        return
    assert laya_mlx_module("common") is normal
