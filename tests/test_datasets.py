"""datasets.py: the one reader, checker and splitter, with no ML library.

Nothing here needs MLX, PyTorch or a model: the same checks run on a Mac, on a Linux server
that checks uploads before any GPU is rented, and on the GPU itself.
"""

import json
import subprocess
import sys

import pytest
from common import QUESTIONS, make_rows

from systemone_studio import datasets, engine


def jsonl(rows):
    return "\n".join(json.dumps(r) for r in rows)


def test_nothing_heavy_is_imported():
    code = (
        "import sys, systemone_studio.datasets\n"
        "heavy = {'torch', 'mlx', 'numpy', 'transformers', 'laya_mlx', 'tokenizers'}\n"
        "print(sorted(heavy & {m.split('.')[0] for m in sys.modules}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_question_types_are_laya_mlxs():
    assert datasets.QUESTION_TYPES == tuple(engine.QTYPES)


def test_the_studio_saves_exactly_what_the_validator_reports(tmp_path):
    """Split parity: engine.create_dataset (the server's datasets) is validate, saved."""
    rows = make_rows(57)
    for row in rows[:4]:
        row["split"] = "val"
    text = jsonl(rows)
    questions, checked, report = datasets.validate(QUESTIONS, text, "t.jsonl", seed=7)
    meta = engine.create_dataset("parity", QUESTIONS, text, "t.jsonl", seed=7, workspace=tmp_path)
    _, saved, _ = engine.load_dataset(meta["id"], tmp_path)
    assert saved == checked and questions == QUESTIONS
    assert meta["sha256"] == report["sha256"] and meta["id"].endswith(report["sha256"][:8])
    assert {k: meta[k] for k in report} == report
    assert meta["rows"] == datasets.split_counts(saved)
    assert sum(meta["rows"].values()) == 57 and meta["rows"]["val"] >= 4


def test_the_split_is_deterministic_and_seeded():
    text = jsonl(make_rows(120))
    first = datasets.validate(QUESTIONS, text, "t.jsonl", seed=13)
    again = datasets.validate(QUESTIONS, text, "t.jsonl", seed=13)
    other = datasets.validate(QUESTIONS, text, "t.jsonl", seed=14)
    assert first == again
    assert first[2]["sha256"] != other[2]["sha256"]
    # Pinned: a change here changes every split, and a checked upload would no longer match the
    # rows its GPU trains on.
    assert first[2]["rows"] == {"train": 96, "val": 11, "test": 13}
    # Stratified by the first question's label: every split sees every label.
    for split in ("train", "val", "test"):
        assert all(n > 0 for n in first[2]["labels"]["topic"][split].values())


def test_a_test_file_is_the_test_split():
    rows = make_rows(40)
    questions, checked, report = datasets.validate(
        QUESTIONS, jsonl(rows), "t.jsonl", jsonl(make_rows(15, seed=1)), "test.jsonl"
    )
    assert report["rows"]["test"] == 15 and report["rows"]["train"] + report["rows"]["val"] == 40
    pinned = make_rows(30)
    for row in pinned[:6]:
        row["split"] = "test"
    _, _, report = datasets.validate(QUESTIONS, jsonl(pinned), "t.jsonl")
    assert report["rows"]["test"] == 6  # pinned test rows: no free row is moved to test


@pytest.mark.parametrize(
    "name, text",
    [
        ("t.json", json.dumps({"rows": make_rows(12)})),
        ("t.json", json.dumps(make_rows(12))),
        ("t.jsonl", "﻿" + jsonl(make_rows(12))),
        (
            "t.csv",
            "text,topic,flag\n"
            + "\n".join(f"red {i},{['alpha', 'beta'][i % 2]},yes" for i in range(12)),
        ),
        (
            "t.tsv",
            "state\ttopic\tlevel\n" + "\n".join(f"blue {i}\tgamma\t{i % 3}" for i in range(12)),
        ),
    ],
)
def test_every_file_format_reads(name, text):
    _, rows, report = datasets.validate(QUESTIONS, text, name)
    assert len(rows) == 12 and report["error_count"] == 0


def test_bad_rows_are_reported_and_left_out():
    rows = make_rows(20)
    text = (
        jsonl(rows)
        + "\n"
        + "\n".join(
            [
                '{"state": "red", "answers": {"topic": "delta"}}',
                '{"state": "", "answers": {"topic": "alpha"}}',
                '{"body": "red", "answers": {"topic": "alpha"}}',
                '{"state": "red", "answers": {"nope": 1}}',
                '{"state": "red", "answers": {"topic": "alpha"}, "split": "later"}',
                '{"state": "red", "answers": {"level": 9}}',
                '{"state": "red", "answers": {"flag": 1.5}}',
                "not json",
            ]
        )
    )
    _, kept, report = datasets.validate(QUESTIONS, text, "t.jsonl")
    assert len(kept) == 20 and report["error_count"] == 8
    errors = " | ".join(e["error"] for e in report["errors"])
    for expected in (
        "unknown label",
        "state is empty",
        "missing state",
        "unknown question",
        "split must be",
        "outside 0..2",
        "between 0 and 1",
        "invalid JSON",
    ):
        assert expected in errors
    many = jsonl(rows) + "\n" + "\n".join("{}" for _ in range(150))
    report = datasets.validate(QUESTIONS, many, "t.jsonl")[2]
    assert report["error_count"] == 150 and len(report["errors"]) == 100


def test_soft_labels_are_kept_as_distributions():
    rows = make_rows(12)
    rows[0]["answers"] = {"topic": {"alpha": 3, "gamma": 1}, "flag": 0.25}
    _, kept, _ = datasets.validate(QUESTIONS, jsonl(rows), "t.jsonl")
    first = next(r for r in kept if r["id"] == "0")
    assert first["targets"]["topic"] == [0.75, 0.0, 0.25]
    assert first["targets"]["flag"] == [0.75, 0.25]


@pytest.mark.parametrize(
    "questions, text, kwargs, message",
    [
        ({}, jsonl(make_rows(12)), {}, "non-empty JSON object"),
        ({"q": {"type": "rank", "instructions": "?"}}, "", {}, "Unknown question type"),
        ({"bad id!": QUESTIONS["topic"]}, "", {}, "must use letters"),
        (QUESTIONS, jsonl(make_rows(9)), {}, "at least 10 valid labeled rows, found 9"),
        (QUESTIONS, "nope\n" * 12, {}, "First error: line 1: invalid JSON"),
        (QUESTIONS, jsonl(make_rows(40)), {"max_bytes": 1000}, "the limit is"),
        (QUESTIONS, jsonl(make_rows(40)), {"max_rows": 39}, "the limit is 39"),
        (
            QUESTIONS,
            jsonl([{**r, "split": "test"} for r in make_rows(12)]),
            {},
            "Not enough rows to create train",
        ),
    ],
)
def test_what_cannot_train_is_refused(questions, text, kwargs, message):
    with pytest.raises(ValueError, match=message):
        datasets.validate(questions, text, "t.jsonl", **kwargs)


def test_the_limits_count_bad_rows_and_the_test_file():
    rows = jsonl(make_rows(12))
    text = rows + "\n" + "\n".join("{}" for _ in range(5))
    datasets.validate(QUESTIONS, text, "t.jsonl", max_rows=17)
    with pytest.raises(ValueError, match="17 rows; the limit is 16"):
        datasets.validate(QUESTIONS, text, "t.jsonl", max_rows=16)
    size = len(rows.encode())
    datasets.validate(QUESTIONS, rows, "t.jsonl", rows, "u.jsonl", max_bytes=2 * size)
    with pytest.raises(ValueError, match="limit"):
        datasets.validate(QUESTIONS, rows, "t.jsonl", rows, "u.jsonl", max_bytes=2 * size - 1)


def test_the_report_says_which_kinds_answer_the_questions():
    wide = {
        "type": "choice",
        "instructions": "Which?",
        "criteria": [f"label{i}" for i in range(25)],
    }
    deep = {"type": "score", "instructions": "How much?", "criteria": [str(i) for i in range(12)]}
    lone = {"type": "choice", "instructions": "Only one", "criteria": ["alpha"]}
    blank = {"type": "choice", "instructions": "", "criteria": ["alpha", "beta"]}
    report = datasets.validate(QUESTIONS, jsonl(make_rows(12)), "t.jsonl")[2]
    assert report["kinds"] == ["laya", "julia", "decider"]
    assert report["questions"] == {
        "topic": {"type": "choice", "options": 3},
        "level": {"type": "score", "options": 3},
        "flag": {"type": "noul", "options": 2},
    }
    assert report["max_options"] == 3 and report["size_bytes"] > 0
    problems = datasets.fits({"wide": wide, "deep": deep, "lone": lone, "blank": blank})
    assert problems["laya"] == []
    assert problems["julia"] == [
        "wide: 25 options; Julia 1 answers 2 to 20",
        "lone: 1 options; Julia 1 answers 2 to 20",
    ]
    assert [p.split(":")[0] for p in problems["decider"]] == ["deep", "lone", "blank"]
    huge = {**wide, "criteria": [f"label{i}" for i in range(256)]}
    assert datasets.fits({"huge": huge})["decider"] == ["huge: choice criteria: 2..255 options"]


def test_decisions_count_what_each_kind_can_learn_from():
    """A kind that leaves a question out has only the other questions' decisions: the platform
    and prepare_run count them the same way, and refuse a template with none."""
    wide = {"type": "choice", "instructions": "Which?", "criteria": [f"l{i}" for i in range(25)]}
    questions = {"wide": wide, "flag": QUESTIONS["flag"]}
    rows = [
        {
            "state": f"red {i}",
            "answers": {"wide": f"l{i % 25}", **({"flag": True} if i < 3 else {})},
        }
        for i in range(30)
    ]
    questions, rows, report = datasets.validate(questions, jsonl(rows), "t.jsonl")
    assert datasets.left_out(questions)["julia"] == {"wide": "25 options; Julia 1 answers 2 to 20"}
    every = datasets.decisions(rows, questions)
    assert every == report["decisions"] and sum(every.values()) == 33
    julia = datasets.decisions(rows, questions, "julia")
    assert sum(julia.values()) == 3 and julia == {
        s: sum("flag" in r["targets"] for r in rows if r["split"] == s) for s in julia
    }
    assert datasets.decisions(rows, questions, "laya") == every
