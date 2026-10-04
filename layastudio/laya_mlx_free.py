"""laya_mlx's plain-Python parts on machines without MLX.

laya-mlx installs MLX only on Apple silicon, but its package __init__ imports its MLX agent,
so `import laya_mlx.common` fails on Linux, Windows and Intel Macs. The parts the studio needs
everywhere (common.py, tokenizer.py, snake/game.py) use only the standard library, numpy and
tokenizers, so where the normal import fails they are loaded straight from their files.
On Apple silicon the normal import is used.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def laya_mlx_module(name: str) -> ModuleType:
    """`laya_mlx.<name>` ("common", "tokenizer", "snake.game"), with or without MLX."""
    try:
        return importlib.import_module(f"laya_mlx.{name}")
    except ImportError:
        pass
    alias = "_layastudio_laya_mlx_" + name.replace(".", "_")
    if alias in sys.modules:
        return sys.modules[alias]
    found = importlib.util.find_spec("laya_mlx")  # locates the package without running it
    if found is None or not found.submodule_search_locations:
        raise ImportError("laya-mlx is not installed: pip install laya-mlx")
    path = Path(next(iter(found.submodule_search_locations)), *name.split(".")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"laya_mlx.{name} is not in this laya-mlx ({path})")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[alias]
        raise
    return module
