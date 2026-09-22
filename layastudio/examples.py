"""Public example datasets, fetched from their source URLs so this repository ships no data.

Git is a poor place for datasets, so nothing is vendored: at startup (or on request) the
files are downloaded from Hugging Face and turned into ordinary workspace datasets, in the
same questions + rows format you would upload yourself. Point $LAYASTUDIO_EXAMPLES_URL at a
JSON manifest of the same shape to publish your own catalog.
"""

import json
import os
import random
import urllib.request

from .engine import WORKSPACE, create_dataset

EXAMPLES = {
    "emotion": {
        "title": "Emotion (6 labels)",
        "description": "Short English posts labeled with one of six emotions. Zero-shot Laya "
        "is weak here, which makes fine-tuning gains easy to see.",
        "source": "dair-ai/emotion",
        "files": {
            "train": "split/train-00000-of-00001.parquet",
            "test": "split/test-00000-of-00001.parquet",
        },
        "counts": {"train": 1200, "test": 600},
        "questions": {
            "emotion": {
                "type": "choice",
                "instructions": "Which emotion does the writer express?",
                "criteria": {
                    "sadness": "sad, hopeless, lonely, hurt",
                    "joy": "happy, content, excited, proud",
                    "love": "affection, tenderness, longing for someone",
                    "anger": "angry, irritated, resentful, offended",
                    "fear": "afraid, anxious, nervous, worried",
                    "surprise": "surprised, amazed, shocked, curious",
                },
            }
        },
        "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "question": "emotion",
    },
    "prompt-injections": {
        "title": "Prompt injection (yes/no)",
        "description": "English and German prompts, some trying to override an assistant's "
        "instructions. A guardrail-style noul question.",
        "source": "deepset/prompt-injections",
        "files": {
            "train": "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
            "test": "data/test-00000-of-00001-701d16158af87368.parquet",
        },
        "counts": {"train": 546, "test": 116},
        "questions": {
            "injection": {
                "type": "noul",
                "instructions": "Does this text try to make an AI assistant ignore, override or "
                "reveal its instructions?",
            }
        },
        "labels": [False, True],
        "question": "injection",
    },
    "snake": {
        "title": "Snake moves (generated on this Mac)",
        "description": "Boards from a real Snake game, labeled by a planner that never dies. "
        "No download: the games are played locally. Fine-tune it and the model plays "
        "unassisted - the demo that tells you whether fine-tuning really worked.",
        "source": "generated with laya_mlx.snake",
        "generator": "snake",
    },
    "banking77": {
        "title": "Banking77 (77 intents)",
        "description": "Customer-support questions across 77 fine-grained banking intents. "
        "Stresses the option token budget; zero-shot Laya scores poorly.",
        "source": "mteb/banking77",
        "files": {
            "train": "data/train-00000-of-00001.parquet",
            "test": "data/test-00000-of-00001.parquet",
        },
        "counts": {"train": 1001, "test": 770},
        "questions": None,  # built from the label names
        "question": "intent",
    },
}


def catalog():
    """The example catalog: built in, or a JSON manifest named by $LAYASTUDIO_EXAMPLES_URL."""
    url = os.environ.get("LAYASTUDIO_EXAMPLES_URL")
    if not url:
        return EXAMPLES
    with urllib.request.urlopen(url, timeout=20) as response:
        remote = json.loads(response.read())
    if not isinstance(remote, dict) or not remote:
        raise ValueError(f"{url} is not a JSON object of examples")
    return remote


def read_split(spec, key):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(spec["source"], spec["files"][key], repo_type="dataset")
    return pq.read_table(path).to_pylist()


def sample_split(spec, key, rng):
    """Class-balanced sample of the configured size (round-robin across labels)."""
    rows, count = read_split(spec, key), spec["counts"][key]
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    for group in by_label.values():
        rng.shuffle(group)
    chosen, i = [], 0
    while len(chosen) < min(count, len(rows)):
        for group in by_label.values():
            if i < len(group) and len(chosen) < count:
                chosen.append(group[i])
        i += 1
    rng.shuffle(chosen)
    return chosen


def fetch_example(name, emit, workspace=WORKSPACE):
    spec = catalog()[name]
    if spec.get("generator") == "snake":
        from .snake import build_dataset

        return build_dataset(workspace=workspace, emit=emit)
    rng = random.Random(7)
    emit("phase", phase="download", message=f"Fetching {spec['source']} (public dataset)")
    splits = {key: sample_split(spec, key, rng) for key in ("train", "test")}
    questions, qid = spec["questions"], spec["question"]
    if name == "banking77":
        names = sorted({r["label_text"] for rows in splits.values() for r in rows})
        questions = {
            qid: {
                "type": "choice",
                "instructions": "Which banking support intent does the customer have?",
                "criteria": names,
            }
        }

    def answer(row):
        if name == "banking77":
            return row["label_text"]
        return spec["labels"][row["label"]]

    def lines(rows):
        return "\n".join(
            json.dumps({"state": r["text"], "answers": {qid: answer(r)}}, ensure_ascii=False)
            for r in rows
        )

    emit("phase", phase="dataset", message="Creating the dataset")
    return create_dataset(
        spec["title"],
        questions,
        lines(splits["train"]),
        f"{name}-train.jsonl",
        lines(splits["test"]),
        f"{name}-test.jsonl",
        workspace=workspace,
        example=name,
    )
