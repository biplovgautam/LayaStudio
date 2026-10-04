"""Questions and rows the tests share, with no MLX import: they run on Linux and Windows too."""

import sys
import textwrap

import numpy as np

QUESTIONS = {
    "topic": {"type": "choice", "instructions": "Choose", "criteria": ["alpha", "beta", "gamma"]},
    "level": {"type": "score", "instructions": "Level", "criteria": ["low", "mid", "high"]},
    "flag": {"type": "noul", "instructions": "Is this flagged?"},
}
WORDS = ["alpha", "beta", "gamma", "low", "mid", "high", "red", "green", "blue", "yes", "no"]


def make_rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = ["alpha", "beta", "gamma"][i % 3]
        color = ["red", "green", "blue"][i % 3]
        noise = " ".join(rng.choice(WORDS, 3))
        rows.append(
            {
                "state": f"{color} {noise}",
                "answers": {"topic": label, "level": i % 3, "flag": i % 2 == 0},
            }
        )
    return rows


def fake_systemone(tmp_path):
    """A stand-in for `systemone push` that writes down every file it was asked to upload."""
    seen = tmp_path / "pushed.json"
    script = tmp_path / "systemone.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, sys
            from pathlib import Path
            folder = Path(sys.argv[2])
            files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
            Path({str(seen)!r}).write_text(json.dumps({{"args": sys.argv[1:], "files": files}}))
            print("Published (pretend)")
            """
        )
    )
    return [sys.executable, str(script)], seen
