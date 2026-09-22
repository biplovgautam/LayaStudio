"""Regenerate the screenshots in this folder from a running studio.

    uv run python finetune/server.py --no-browser &
    uv run python finetune/docs/screenshots.py --run <run-id> --dataset <dataset-id>

Shots come from a real workspace, so run it against one that holds only data you are
happy to publish. Needs Chrome, Brave, Chromium or Edge for headless rendering, and
Pillow (installed by the `demo` extra) to trim and downscale the images.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    shutil.which("chromium") or "",
]


def browser_path():
    for path in BROWSERS:
        if path and Path(path).exists():
            return path
    sys.exit("No Chrome/Brave/Chromium/Edge found for headless rendering")


def capture(url, out, width=1400, height=1000, wait_ms=6000, scale=2, profile=None, timeout=90):
    """One headless screenshot.

    Chromium writes the PNG and then sometimes refuses to exit (a cold profile spends its
    first minute on first-run chores), so wait for the file rather than for the process,
    and reuse one profile for a whole batch.
    """
    out = Path(out)
    out.unlink(missing_ok=True)
    owned = profile is None
    profile = profile or tempfile.mkdtemp(prefix="laya-shot-")
    command = [
        browser_path(),
        "--headless=new",
        "--disable-gpu",
        "--disable-breakpad",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
        "--disable-extensions",
        "--disable-component-update",
        "--hide-scrollbars",
        f"--user-data-dir={profile}",
        f"--crash-dumps-dir={profile}",
        f"--force-device-scale-factor={scale}",
        f"--window-size={width},{height}",
        f"--virtual-time-budget={wait_ms}",
        f"--screenshot={out}",
        url,
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            if out.exists() and out.stat().st_size and time.time() - out.stat().st_mtime > 1.5:
                break
            if process.poll() is not None:
                break
            time.sleep(0.5)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
        if owned:
            shutil.rmtree(profile, ignore_errors=True)
    if not out.exists() or not out.stat().st_size:
        raise RuntimeError(f"No screenshot written for {url}")
    trim(out, scale)


def trim(path, scale, margin=24):
    """Crop the empty page below the content and downscale back to CSS pixels."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    background = image.getpixel((image.width - 5, image.height - 5))
    last = image.height
    for y in range(image.height - 1, 0, -8):
        row = image.crop((image.width // 3, y, image.width, y + 1))
        if any(abs(a - b) > 6 for pixel in row.getdata() for a, b in zip(pixel, background)):
            last = min(image.height, y + margin * scale)
            break
    image = image.crop((0, 0, image.width, last))
    image = image.resize((image.width // scale, image.height // scale), Image.LANCZOS)
    image.save(path, optimize=True)


SHOTS = {
    "results": ("#/runs/{run}", 1400, 1500),
    "gating": ("#/runs/{run}", 1400, 1000),
    "dataset": ("#/datasets/{dataset}", 1400, 1200),
    "playground": ("#/playground?run={run}&go=1&state={state}", 1400, 1400),
    "datasets": ("#/datasets", 1400, 1100),
    "guide": ("#/guide", 1400, 1000),
}
STATE = "I was charged twice this month and support never replied. I want the money back today."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--run", required=True, help="Run id to screenshot")
    parser.add_argument("--dataset", required=True, help="Dataset id to screenshot")
    parser.add_argument("--only", nargs="*", help="Subset of shots", choices=list(SHOTS))
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(tempfile.gettempdir()) / "layastudio-shots-profile",
        help="Browser profile to reuse; the very first initialization is slow",
    )
    args = parser.parse_args()
    # One profile, kept between runs: a brand new profile spends its first minutes on
    # first-run chores and never reaches the screenshot.
    profile = str(args.profile)
    if not args.profile.exists():
        print(f"Initializing the browser profile in {profile} (one-off, can take a minute)…")
        try:
            capture(
                "about:blank", str(HERE / ".warmup.png"), 400, 300, 500, 1, profile, timeout=180
            )
        except RuntimeError:
            pass
        (HERE / ".warmup.png").unlink(missing_ok=True)
    for name in args.only or SHOTS:
        path, width, height = SHOTS[name]
        # "?static=1" stops the page polling, which would otherwise keep the headless
        # browser's virtual clock busy forever.
        url = (
            args.url
            + "/?static=1"
            + path.format(run=args.run, dataset=args.dataset, state=quote(STATE))
        )
        out = HERE / f"{name}.png"
        capture(
            url,
            str(out),
            width,
            height,
            wait_ms=15000 if name == "playground" else 6000,
            profile=profile,
        )
        print(f"{out.name}: {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
