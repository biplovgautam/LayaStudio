"""`layastudio`, this package's name before the rename, kept as an alias of its modules.

`import layastudio` (layastudio/__init__.py, in the same distribution) warns once
(DeprecationWarning), puts this package in sys.modules under the old name and installs
Finder first on sys.meta_path, so that:

- `import layastudio` gives systemone_studio itself;
- `layastudio.<name>` is systemone_studio.<name>: the same module object, imported once, so
  patching one patches the other (monkeypatch, mock.patch("layastudio.engine.WORKSPACE"));
- `python -m layastudio` and `python -m layastudio.<name>` run systemone_studio's code
  (runpy reads it through Loader.get_code);
- a stand-in put into sys.modules under the old name before its module was ever imported
  (sys.modules["layastudio.laya_mlx_free"] = stub, as tests do) is what the new name imports
  too, so code written against the old names keeps its stubs.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

OLD = "layastudio"
NEW = "systemone_studio"


class Loader(importlib.abc.Loader):
    """Gives a module name the object another name already has (or imports)."""

    def __init__(self, target, target_spec=None):
        self.target = target  # the name whose module object this name gets
        self.target_spec = target_spec  # its spec, for get_code and get_source
        self._kept = None

    def create_module(self, spec):
        module = sys.modules.get(self.target)
        if module is None:
            module = importlib.import_module(self.target)
        # The import system sets __spec__ (and an unset __loader__) to this alias's before
        # exec_module: kept, and put back there, so the module keeps its own.
        self._kept = (getattr(module, "__spec__", None), getattr(module, "__loader__", None))
        return module

    def exec_module(self, module):
        spec, loader = self._kept
        for name, value in (("__spec__", spec), ("__loader__", loader)):
            try:
                setattr(module, name, value)
            except AttributeError:
                pass

    def _target_loader(self):
        loader = getattr(self.target_spec, "loader", None)
        if loader is None:
            raise ImportError(f"no loader for {self.target}", name=self.target)
        return loader

    def get_code(self, fullname):
        """The target's code, for runpy (`python -m layastudio.<name>`)."""
        return self._target_loader().get_code(self.target)

    def get_source(self, fullname):
        return self._target_loader().get_source(self.target)

    def is_package(self, fullname):
        return getattr(self.target_spec, "submodule_search_locations", None) is not None


class Finder(importlib.abc.MetaPathFinder):
    """layastudio.<name> as systemone_studio.<name>, and the other way round for a stand-in
    put into sys.modules under the old name."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(OLD + "."):
            new = NEW + fullname[len(OLD) :]
            found = importlib.util.find_spec(new)
            if found is None:
                return None
            spec = importlib.machinery.ModuleSpec(
                fullname,
                Loader(new, found),
                origin=found.origin,
                is_package=found.submodule_search_locations is not None,
            )
            spec.has_location = found.has_location  # runpy's __file__
            return spec
        if fullname.startswith(NEW + "."):
            old = OLD + fullname[len(NEW) :]
            stand_in = sys.modules.get(old)
            # Not one of this package's own modules seen under the old name (those were
            # imported under the new one, and a fresh import of it is wanted).
            if stand_in is not None and getattr(stand_in, "__name__", None) != fullname:
                return importlib.machinery.ModuleSpec(fullname, Loader(old))
        return None


def install():
    """Finder first on sys.meta_path, once; and this package under the old name."""
    if not any(isinstance(finder, Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, Finder())
    sys.modules[OLD] = importlib.import_module(NEW)
