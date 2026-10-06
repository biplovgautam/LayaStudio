"""Deprecated: LayaStudio is now System One Studio, and this package is `systemone_studio`.

`import layastudio`, every `layastudio.<module>` and `python -m layastudio[.<module>]` keep
working: they are systemone_studio's own modules under the old name (systemone_studio/_alias.py),
and the first import says so once, with a DeprecationWarning. Import systemone_studio instead.
"""

import warnings

from systemone_studio import _alias

warnings.warn(
    "layastudio is now systemone_studio (System One Studio): import systemone_studio instead; "
    "the old name keeps working as an alias",
    DeprecationWarning,
    stacklevel=2,
)
_alias.install()
