"""Publish a fine-tuned run to Hugging Face, model card and all.

    hf auth login                                     # once, with your own token
    python -m layastudio.publish run:<id> --repo <you>/laya-snake-mlx

The upload is the checkpoint LayaStudio already wrote - FP16 safetensors with the original
PyTorch parameter names, the tokenizer, the refitted calibration and the questions the
model was trained for - plus a card built from that run's measured numbers, so the claims
on the Hub are the ones the studio actually recorded.

Nothing here reads a token from arguments or the workspace: it uses the login the
huggingface_hub CLI stored for you.
"""

import argparse
import json
import sys

from .engine import WORKSPACE, check_id, read_json, resolve_model_ref

CARD = """---
license: apache-2.0
library_name: laya-mlx
pipeline_tag: text-classification
tags:
- laya
- typed-decisions
- mlx
- apple-silicon
- lora
- layastudio
base_model: {base_repo}
---

# {title}

A [Laya](https://github.com/NandhaKishorM/laya) typed-decision model, fine-tuned with
[LayaStudio](https://github.com/biplovgautam/LayaStudio) on an Apple silicon Mac. It
answers the questions below in a single forward pass, with calibrated probabilities and
**zero generated tokens**.

Base model: `{base_repo}` · method: {method}, {objective} objective · trained in
{minutes} minutes on {chip}.

## Measured on the held-out test split

{metrics}

{extra}

Test rows were never trained on. Accuracy intervals are Wilson intervals; the paired test
is an exact McNemar test between the base and the fine-tuned model on the same rows.

## Use it

```bash
pip install laya-mlx            # Apple silicon
```

```python
import json, laya_mlx as laya
from huggingface_hub import hf_hub_download

agent = laya.load("{repo}")
questions = json.load(open(hf_hub_download("{repo}", "questions.json")))
print(agent.predict("your text here", questions)["answers"])
```

Ask it **these** questions: the instructions and option texts are part of the model's
input, so changing them changes the task it was tuned for.

```json
{questions}
```

The same folder also loads in the upstream PyTorch `laya` package on Linux and NVIDIA, and
LayaStudio can export it to ONNX.

## Provenance

```json
{provenance}
```

## License and attribution

Apache-2.0. Laya and its pretrained weights are by
[Convai Innovations](https://github.com/NandhaKishorM/laya); this checkpoint is a
fine-tune of `{base_repo}` and carries the same licence. Fine-tuned and published with
[LayaStudio](https://github.com/biplovgautam/LayaStudio).
"""


def percent(value):
    return "–" if value is None else f"{100 * value:.1f}%"


def build_card(run, training, comparison, repo, questions):
    base_repo = run["base_model"].split(":", 1)[-1]
    rows, extra = [], ""
    if comparison:
        base, tuned = comparison["base"]["overall"], comparison["finetuned"]["overall"]
        paired = comparison["paired"]["overall"]
        rows = [
            "| Metric | Base model | This model |",
            "|---|---|---|",
            f"| Accuracy | {percent(base.get('accuracy'))} | **{percent(tuned.get('accuracy'))}** "
            f"[{percent(tuned['accuracy_ci95'][0])}–{percent(tuned['accuracy_ci95'][1])}] |",
            f"| Calibration error (ECE) | {base.get('ece', 0):.3f} | **{tuned.get('ece', 0):.3f}** |",
            f"| Log loss | {base.get('nll', 0):.3f} | **{tuned.get('nll', 0):.3f}** |",
            f"| Brier score | {base.get('brier', 0):.3f} | **{tuned.get('brier', 0):.3f}** |",
            f"| Decisions scored | {base.get('n')} | {tuned.get('n')} |",
        ]
        p = paired["p_value"]
        extra = (
            f"Fine-tuning fixed **{paired['fixed']}** test decisions the base model got wrong "
            f"and broke **{paired['broken']}** it got right "
            f"(exact McNemar p {'< 0.001' if p < 0.001 else f'= {p:.3f}'})."
        )
    calibration = (training or {}).get("calibration", {})
    provenance = {
        "base_model": base_repo,
        "hyperparameters": (training or {}).get("hyperparameters"),
        "train_decisions": (training or {}).get("train_decisions"),
        "best_epoch": (training or {}).get("best_epoch"),
        "temperature": calibration.get("temperature"),
        "temperature_by_options": calibration.get("temperature_by_options"),
        "dataset_sha256": (training or {}).get("dataset_sha256"),
        "trained_on": (training or {}).get("created"),
    }
    seconds = (training or {}).get("train_seconds") or 0
    return CARD.format(
        title=run.get("name", "Laya fine-tune"),
        base_repo=base_repo,
        repo=repo,
        method=(training or {}).get("hyperparameters", {}).get("method", "lora"),
        objective=(training or {}).get("hyperparameters", {}).get("objective", "proper"),
        minutes=f"{seconds / 60:.0f}",
        chip=(training or {}).get("chip", "an Apple silicon Mac"),
        metrics="\n".join(rows) or "_No comparison was recorded for this run._",
        extra=extra,
        questions=json.dumps(questions, indent=2, ensure_ascii=False),
        provenance=json.dumps(provenance, indent=2),
    )


def publish(run_ref, repo, workspace=WORKSPACE, private=False, dry_run=False):
    from huggingface_hub import HfApi

    run_id = check_id(run_ref.split(":", 1)[-1])
    run_dir = workspace / "runs" / run_id
    model_dir = resolve_model_ref(f"run:{run_id}", workspace)
    run = read_json(run_dir / "run.json") or {"name": run_id, "base_model": "unknown"}
    card = build_card(
        run,
        read_json(run_dir / "training.json"),
        read_json(run_dir / "comparison.json"),
        repo,
        read_json(model_dir / "questions.json") or {},
    )
    (model_dir / "README.md").write_text(card)
    files = sorted(p.name for p in model_dir.iterdir())
    size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 2**20
    print(f"{run_id} -> https://huggingface.co/{repo}\n  {len(files)} entries, {size:.0f} MB")
    if dry_run:
        print("  dry run: nothing uploaded. The model card is written to the checkpoint.")
        return {"repo": repo, "uploaded": False, "card": str(model_dir / "README.md")}

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        folder_path=str(model_dir),
        commit_message=f"LayaStudio: {run.get('name', run_id)}",
    )
    print(f"  published: https://huggingface.co/{repo}")
    return {"repo": repo, "uploaded": True, "url": f"https://huggingface.co/{repo}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run:<id> (see Runs & results in the studio)")
    parser.add_argument("--repo", required=True, help="Target repository, e.g. you/laya-snake-mlx")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write the card, upload nothing")
    args = parser.parse_args(argv)
    try:
        publish(args.run, args.repo, private=args.private, dry_run=args.dry_run)
    except Exception as error:  # noqa: BLE001 - a CLI should explain itself
        name = type(error).__name__
        if "Token" in name or "401" in str(error):
            sys.exit("Log in first with your own token:  hf auth login")
        sys.exit(f"{name}: {error}")


if __name__ == "__main__":
    main()
