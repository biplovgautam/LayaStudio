"""Headless runs: the checks every fine-tune starts with, shared with the server, and a job
from a config file with no UI. Nothing here needs MLX, PyTorch or a real model, except the
end-to-end runs, which use the tests' tiny random checkpoints on whatever this machine has.
"""

import signal

from layastudio import engine


def test_an_export_job_passes_the_gguf_choice_on(tmp_path, monkeypatch):
    """A Decider package's GGUF (bf16 or q8_0) reaches the export from a job's spec."""
    import layastudio.export

    monkeypatch.setattr(signal, "signal", lambda *a: None)  # run_job's SIGTERM handler
    seen = {}
    monkeypatch.setattr(layastudio.export, "export", lambda *a, **k: seen.update(k))
    job = tmp_path / "job"
    for gguf in ("q8_0", None):
        engine.write_json(
            job / "spec.json",
            {
                "kind": "export",
                "model": "run:x",
                "target": "noulxp",
                "precision": "float",
                **({"gguf": gguf} if gguf else {}),
                "workspace": str(tmp_path),
            },
        )
        assert engine.run_job(job) == 0
        assert seen["gguf"] == gguf and seen["precision"] == "float"
