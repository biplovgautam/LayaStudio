"""The studio's environment variables, under their new names and their old ones.

Each variable is SYSTEMONE_STUDIO_<KEY>. The name it had before the rename,
LAYASTUDIO_<KEY> (LAYA_STUDIO_VERBOSE for VERBOSE), keeps working: the new name wins whenever
it is set (even to an empty value), and the old one is read only when the new one is not.
A variable the studio sets for a child process is set under both names (both()), so the
child reads what this process meant whichever name its inherited environment carried.
README.md, "Environment variables", lists them.
"""

import os

PREFIX = "SYSTEMONE_STUDIO_"
OLD_PREFIX = "LAYASTUDIO_"
# The one old name with another spelling than OLD_PREFIX + key.
OLD_NAMES = {"VERBOSE": "LAYA_STUDIO_VERBOSE"}
# Every variable the studio reads.
KEYS = (
    "HOME",
    "TOOLS",
    "BACKEND",
    "DEVICE",
    "THREADS",
    "DEMO_MODELS",
    "EXAMPLES_URL",
    "SERIAL_VERIFY",
    "PARALLEL_VERIFY",
    "PARALLEL_CONFORMANCE",
    "PARENT",
    "VERBOSE",
)


def names(key):
    """(new, old): ("SYSTEMONE_STUDIO_HOME", "LAYASTUDIO_HOME") for "HOME"."""
    return PREFIX + key, OLD_NAMES.get(key, OLD_PREFIX + key)


def source(key, environ=None):
    """The name in effect: the new one when it is set, else the old one when that is, else
    None."""
    environ = os.environ if environ is None else environ
    for name in names(key):
        if name in environ:
            return name
    return None


def get(key, default=None, environ=None):
    """The variable's value under the name in effect (source()), else default."""
    environ = os.environ if environ is None else environ
    name = source(key, environ)
    return default if name is None else environ[name]


def both(key, value):
    """The variable under both names, for a child process's environment."""
    return dict.fromkeys(names(key), str(value))


def setdefault(key, value, environ=None):
    """Both names set to value when neither is set; the value in effect."""
    environ = os.environ if environ is None else environ
    if source(key, environ) is None:
        environ.update(both(key, value))
    return get(key, environ=environ)
