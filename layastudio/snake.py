"""Snake as a fine-tuning task: teach a 400M local model to actually play.

The well-known Laya Snake demo hands the model a state in which a classical planner has
already labeled every option ("Safe. Best route to food."), so the model only has to read
the labels. This module does the harder, honest thing:

  * the model sees the **raw board** and four plain directions, with no safety hints;
  * training rows are labeled by a planner teacher that follows a Hamiltonian cycle, with
    20% exploration so the data covers boards a fumbling player would actually reach;
  * the benchmark runs the model **unassisted** - top-1 move, no safety shield - so an
    illegal move ends the game, exactly as it would for any other decision engine.

    python -m layastudio.snake dataset          # generate and register the dataset
    python -m layastudio.snake bench --model run:<id> --games 10

The board and the rules come from laya_mlx.snake, the same game the demo uses.
"""

import argparse
import json
import random
import time

from laya_mlx.snake.game import DIRECTIONS, SnakeGame

from .engine import WORKSPACE, create_dataset, resolve_model_ref

WIDTH, HEIGHT, INITIAL_LENGTH = 12, 8, 4
QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": (
            "Snake board. Pick the next move: stay inside the board, do not hit the snake, "
            "and take the shortest safe path to the food."
        ),
        "criteria": {
            "UP": "one cell up",
            "DOWN": "one cell down",
            "LEFT": "one cell left",
            "RIGHT": "one cell right",
        },
    }
}
TITLE = "Snake moves (generated on this Mac)"


def render(game):
    """The board as text: the whole state the model gets, with no move hints."""
    head, body, food = game.head, set(game.body), game.food
    rows = []
    for y in range(game.height):
        row = []
        for x in range(game.width):
            cell = (x, y)
            row.append("H" if cell == head else "F" if cell == food else "o" if cell in body else ".")
        rows.append("".join(row))
    return (
        f"Snake {game.width}x{game.height}. Head {head}. Food {food}. Length {len(game.body)}.\n"
        "Board, top row first: . empty, H head, o body, F food.\n" + "\n".join(rows)
    )


def teacher(game):
    """The planner's move: the safe option that advances furthest along the cycle."""
    moves = [m for m in game.moves() if m.safe]
    if not moves:
        moves = [m for m in game.moves() if m.legal]
    if not moves:
        return None
    return max(moves, key=lambda m: (m.eats, m.advance)).direction


def generate(rows=2600, seed=11, explore=0.2, max_ticks=400):
    """Play teacher games, recording (board, best move) pairs.

    `explore` sometimes takes a different legal move, so the model also sees the messy
    boards it will create for itself once it plays on its own.
    """
    rng = random.Random(seed)
    data, game_index = [], 0
    while len(data) < rows:
        game = SnakeGame(WIDTH, HEIGHT, seed=seed + game_index, initial_length=INITIAL_LENGTH)
        game_index += 1
        for _ in range(max_ticks):
            if not game.alive or game.won:
                break
            best = teacher(game)
            if best is None:
                break
            data.append({"state": render(game), "answers": {"move": best}})
            legal = [m.direction for m in game.moves() if m.legal]
            step = rng.choice(legal) if legal and rng.random() < explore else best
            game.step(step)
            if len(data) >= rows:
                break
    rng.shuffle(data)
    return data


def build_dataset(workspace=WORKSPACE, rows=2600, test_rows=600, seed=11, emit=None):
    """Create the Snake dataset in the workspace (train rows and unseen test rows)."""
    if emit:
        emit("phase", phase="generate", message=f"Playing teacher games for {rows} boards")
    train = generate(rows, seed=seed)
    test = generate(test_rows, seed=seed + 5000)
    lines = lambda part: "\n".join(json.dumps(r, ensure_ascii=False) for r in part)  # noqa: E731
    return create_dataset(
        TITLE,
        QUESTIONS,
        lines(train),
        "snake-train.jsonl",
        lines(test),
        "snake-test.jsonl",
        workspace=workspace,
        example="snake",
    )


def play(agent, games=10, seed=101, max_ticks=500, on_game=None):
    """Play unassisted: the model's top choice is executed, mistakes end the game."""
    results = []
    for index in range(games):
        game = SnakeGame(WIDTH, HEIGHT, seed=seed + index, initial_length=INITIAL_LENGTH)
        safe_moves, latencies = 0, []
        while game.alive and not game.won and game.ticks < max_ticks:
            options = {m.direction: m for m in game.moves()}
            started = time.perf_counter()
            answer = agent.predict(render(game), QUESTIONS)["answers"]["move"]
            latencies.append((time.perf_counter() - started) * 1000)
            choice = answer["choice"] if answer["choice"] in DIRECTIONS else "UP"
            safe_moves += bool(options.get(choice) and options[choice].safe)
            game.step(choice)
        decisions = max(1, len(latencies))
        results.append({
            "game": index,
            "moves": game.ticks,
            "score": game.score,
            "length": len(game.body),
            "won": game.won,
            "death_reason": game.death_reason,
            "safe_rate": safe_moves / decisions,
            "ms_per_decision": sorted(latencies)[len(latencies) // 2] if latencies else None,
        })
        if on_game:
            on_game(results[-1])
    return results


def summarize(results):
    moves = sorted(r["moves"] for r in results)
    latency = [r["ms_per_decision"] for r in results if r["ms_per_decision"]]
    return {
        "games": len(results),
        "moves_mean": sum(moves) / len(moves),
        "moves_median": moves[len(moves) // 2],
        "moves_best": moves[-1],
        "score_mean": sum(r["score"] for r in results) / len(results),
        "score_best": max(r["score"] for r in results),
        "safe_rate": sum(r["safe_rate"] for r in results) / len(results),
        "deaths": sum(not r["won"] and r["moves"] < 500 for r in results),
        "ms_per_decision": sorted(latency)[len(latency) // 2] if latency else None,
        "decisions_per_s": round(1000 / (sorted(latency)[len(latency) // 2]), 1) if latency else None,
    }


def benchmark(model_ref, games=10, seed=101, workspace=WORKSPACE, max_ticks=500):
    import laya_mlx

    agent = laya_mlx.load(str(resolve_model_ref(model_ref, workspace)))
    for _ in range(3):  # warm the Metal kernels before timing
        agent.predict(render(SnakeGame(WIDTH, HEIGHT, seed=1)), QUESTIONS)
    results = play(agent, games=games, seed=seed, max_ticks=max_ticks)
    return {"model": model_ref, **summarize(results), "games_detail": results}


def teacher_benchmark(games=10, seed=101, max_ticks=500):
    """The planner's own score: the ceiling the model is being taught to approach."""
    results = []
    for index in range(games):
        game = SnakeGame(WIDTH, HEIGHT, seed=seed + index, initial_length=INITIAL_LENGTH)
        while game.alive and not game.won and game.ticks < max_ticks:
            game.step(teacher(game) or "UP")
        results.append({"game": index, "moves": game.ticks, "score": game.score,
                        "length": len(game.body), "won": game.won,
                        "death_reason": game.death_reason, "safe_rate": 1.0,
                        "ms_per_decision": None})
    return {"model": "planner (teacher)", **summarize(results), "games_detail": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("dataset", help="Generate the Snake dataset in the workspace")
    make.add_argument("--rows", type=int, default=2600)
    make.add_argument("--test-rows", type=int, default=600)
    make.add_argument("--seed", type=int, default=11)
    bench = sub.add_parser("bench", help="Play unassisted games with a model")
    bench.add_argument("--model", required=True, help="hub:<repo>, run:<id> or path:<dir>")
    bench.add_argument("--games", type=int, default=10)
    bench.add_argument("--seed", type=int, default=101)
    bench.add_argument("--output", help="Write the report as JSON")
    sub.add_parser("teacher", help="Score the planner teacher itself")
    args = parser.parse_args(argv)

    if args.command == "dataset":
        meta = build_dataset(rows=args.rows, test_rows=args.test_rows, seed=args.seed)
        print(json.dumps({k: meta[k] for k in ("id", "name", "rows", "decisions")}, indent=2))
        return
    report = (
        teacher_benchmark(games=args.games, seed=args.seed)
        if args.command == "teacher"
        else benchmark(args.model, games=args.games, seed=args.seed)
    )
    print(json.dumps({k: v for k, v in report.items() if k != "games_detail"}, indent=2))
    if getattr(args, "output", None):
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
