import os
import sys
from pathlib import Path

# Run the tests against the checkout, not an installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from systemone_studio import environment  # noqa: E402

# The tests set the studio's variables under their old names (LAYASTUDIO_*), and a new name
# wins over its old one (systemone_studio/environment.py): one given to the whole run under its
# new name (SYSTEMONE_STUDIO_TOOLS=...) is moved to its old name here, so every test sees it and
# can still set its own.
for _key in environment.KEYS:
    _new, _old = environment.names(_key)
    if _new in os.environ:
        os.environ[_old] = os.environ.pop(_new)
