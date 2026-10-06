"""System One Studio (formerly LayaStudio): fine-tune System One decision models on your own
data, on your own machine.

This package was `layastudio` before the rename. That name still imports, as a deprecated
alias of these modules (the `layastudio` package beside this one), and the `layastudio`
command still starts the studio.
"""

__version__ = "0.1.0"
# The studio's source repository, the one place the code names it (the page's links and the
# model cards).
REPOSITORY = "https://github.com/biplovgautam/LayaStudio"
__all__ = ["REPOSITORY", "__version__"]
