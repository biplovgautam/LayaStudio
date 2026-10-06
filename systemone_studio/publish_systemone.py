"""Publish a fine-tuned run to systemonemodels.tech, the System One model registry.

    pip install systemonemodels
    systemone login                                         # once, in the browser
    python -m systemone_studio.publish_systemone run:<id>   # or the Publish button in the studio

What goes up is the checkpoint System One Studio wrote - the safetensors, tokenizer,
calibration and questions - plus a model card built from the run's own measured
numbers, so the page on the registry shows what the studio recorded. The
registry's CLI reads the run's eval.json for accuracy, calibration and latency,
and records the base model with a link back to Hugging Face.

The version also carries the run's NoulXP package as its `noulxp/` folder, which the
registry checks to show the model as NoulXP compatible: the package the run keeps when it
passed its check, or one built and checked first (noulxp_package.py). When none passes,
nothing is uploaded; --skip-noulxp publishes the checkpoint without one.

Nothing here handles a token: the `systemone` CLI keeps its own login.
"""

import argparse
import os
import shutil
import subprocess
import sys

from . import noulxp_package
from .engine import WORKSPACE, check_id, read_json, resolve_model_ref
from .publish import build_card

USE_IT_MARK = "## Use it"
PROVENANCE_MARK = "\n## Provenance"
QUESTIONS_MARK = "Ask it **these** questions"
SYSTEMONE_USE = """## Use it

```bash
pip install systemonemodels
systemone pull {repo}
```

```python
import json
import laya_mlx as laya  # pip install laya-mlx, on Apple silicon
from systemone import snapshot_download

path = snapshot_download("{repo}")
agent = laya.load(str(path))
questions = json.loads((path / "questions.json").read_text())
print(agent.predict("your text here", questions)["answers"])
```

"""


SYSTEMONE_USE_JULIA = """## Use it

```bash
pip install systemonemodels "noulxp[onnx]"
systemone pull {repo}
noulxp run <the pulled folder>/noulxp --request request.json
```

Julia's own runtime (from SupersonicLabs/Julia-1) reads the pulled folder as it is.

"""
SYSTEMONE_USE_DECIDER = """## Use it

```bash
pip install systemonemodels
systemone pull {repo}
```

```python
from pathlib import Path
from systemone import snapshot_download

path = snapshot_download("{repo}")      # Decider's own files and its decider/ code
import sys; sys.path.insert(0, str(path))
from decider.infer import Decider
print(Decider(str(path)).system_one("your text here", questions)["answers"])
```

"""


def registry_card(card, repo, kind="laya"):
    """The Hugging Face card, with its "Use it" section pointed at the registry."""
    start = card.find(USE_IT_MARK)
    end = card.find(QUESTIONS_MARK)
    if start == -1 or end == -1 or end < start:
        return card
    use = {"julia": SYSTEMONE_USE_JULIA, "decider": SYSTEMONE_USE_DECIDER}.get(kind, SYSTEMONE_USE)
    return card[:start] + use.format(repo=repo) + card[end:]


def with_noulxp(card, line):
    """The card with its line on NoulXP, just before the provenance."""
    at = card.find(PROVENANCE_MARK)
    if at == -1:
        return card.rstrip("\n") + "\n\n" + line + "\n"
    return card[:at] + "\n" + line + "\n" + card[at:]


def cli_command():
    """The registry's CLI, wherever it is installed."""
    if shutil.which("systemone"):
        return ["systemone"]
    try:
        import systemone  # noqa: F401
    except ImportError:
        return None
    return [sys.executable, "-m", "systemone.cli"]


def publish(
    run_ref,
    repo=None,
    workspace=WORKSPACE,
    emit=None,
    private=False,
    dry_run=False,
    noulxp=True,
):
    """Push one run's checkpoint to the registry. Returns the CLI's exit code.

    With noulxp (the default) the version carries the run's NoulXP package, built and
    checked first if the run has none that passed; when none passes, nothing is uploaded."""
    emit = emit or (lambda kind, **data: None)
    run_id = check_id(run_ref.split(":", 1)[-1])
    run_dir = workspace / "runs" / run_id
    model_dir = resolve_model_ref(f"run:{run_id}", workspace)
    run = read_json(run_dir / "run.json") or {"name": run_id, "base_model": "unknown"}

    command = cli_command()
    if command is None:
        raise RuntimeError(
            "The registry's CLI is not installed. Run `pip install systemonemodels`, "
            "then `systemone login`, and try again."
        )

    from . import kinds
    from .families import find, licence_allows

    kind = kinds.detect(model_dir) or "laya"
    base = noulxp_package.base_model(run.get("base_model"), workspace)
    known = find(base) if base else None
    if known and not licence_allows(known.licence_kind, "publish"):
        raise RuntimeError(
            f"{known.repo} is licensed {known.licence}: the studio does not publish its fine-tunes."
        )

    # Before the card: a run whose package cannot pass publishes nothing, card included.
    package = noulxp_package.for_publish(f"run:{run_id}", workspace, emit) if noulxp else None

    # A card written for the Hub names a Hub repository in its examples; on
    # the registry the examples pull from the registry.
    target = repo or run_id
    card = registry_card(
        build_card(
            run,
            read_json(run_dir / "training.json"),
            read_json(run_dir / "comparison.json"),
            target,
            read_json(model_dir / "questions.json") or {},
            kind,
        ),
        target,
        kind,
    )
    card = with_noulxp(card, noulxp_package.card_line(package))
    (model_dir / "README.md").write_text(card)
    emit("phase", phase="card", message=f"Model card written from {run_id}'s measurements")

    args = [*command, "push", str(model_dir), "--yes"]
    if repo:
        args += ["--repo", repo]
    if private:
        args.append("--private")
    if dry_run:
        args.append("--dry-run")
    with noulxp_package.staged(run_dir, model_dir, package is not None):
        if package:
            emit("log", message="noulxp/ holds the run's NoulXP package, next to the checkpoint")
        emit("phase", phase="push", message="Uploading to systemonemodels.tech")
        code, last = _push(args, emit)
    if code != 0:
        hint = ""
        if "Not signed in" in last or "systemone login" in last:
            hint = " Sign in first: `systemone login` (opens the registry in your browser)."
        raise RuntimeError(f"systemone push failed: {last}.{hint}")
    emit("phase", phase="done", message=last or "Published")
    return code


def _push(args, emit):
    """`systemone push`, its output into the log. (exit code, last line printed)"""
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "COLUMNS": "100"},
    )
    assert process.stdout is not None
    last = ""
    try:
        for line in process.stdout:
            line = line.rstrip()
            if line:
                last = line
                emit("log", message=line)
        return process.wait(), last
    except BaseException:  # a cancelled job stops its upload too
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run:<id> (see Runs & results in the studio)")
    parser.add_argument(
        "--repo", help="namespace/name on the registry; default: your username and the run's name"
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Write the card and show the plan; upload nothing"
    )
    parser.add_argument(
        "--skip-noulxp",
        action="store_true",
        help="Publish without a NoulXP package (default: the run's package, built and checked "
        "first if it has none that passed)",
    )
    args = parser.parse_args(argv)
    try:
        publish(
            args.run,
            args.repo,
            private=args.private,
            dry_run=args.dry_run,
            emit=_print,
            noulxp=not args.skip_noulxp,
        )
    except Exception as error:  # noqa: BLE001 - a CLI should explain itself
        sys.exit(f"{type(error).__name__}: {error}")


def _print(kind, **data):
    message = data.get("message")
    if message:
        print(message)


if __name__ == "__main__":
    main()
