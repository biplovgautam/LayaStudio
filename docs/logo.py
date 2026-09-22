"""Draw the LayaStudio wordmark: python docs/logo.py

The logo is the name itself: "laya" in the page's ink, "studio" in Laya's blue, with the
line that says what the studio is for. Nothing to draw, nothing to misread at 16 pixels.

Writes docs/logo.svg and docs/logo-light.svg, plus transparent PNGs for the README when a
Chromium-based browser is available.
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLUE = "#2a78d6"
FONT = "Avenir Next,Helvetica Neue,Helvetica,Arial,sans-serif"
BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]


def lockup_svg(ink="#ecedf0", muted="#9aa0aa", width=620, height=150):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" \
width="{width}" height="{height}" role="img" aria-label="LayaStudio">
  <text x="20" y="92" font-family="{FONT}" font-size="84" font-weight="600" \
letter-spacing="-1.5"><tspan fill="{ink}">laya</tspan><tspan fill="{BLUE}">studio</tspan></text>
  <text x="24" y="126" font-family="{FONT}" font-size="17.5" letter-spacing="6.8" \
fill="{muted}">TUNE YOUR OWN DECISIONS</text>
</svg>
"""


def rasterize(svg, out, width, height, profile):
    """SVG to transparent PNG, so the README renders the same in both GitHub themes."""
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if not browser:
        return False
    page = Path(tempfile.mkdtemp()) / "logo.html"
    page.write_text(f"<html><body style='margin:0;background:transparent'>{svg}</body></html>")
    out = Path(out)
    out.unlink(missing_ok=True)
    process = subprocess.Popen(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--disable-breakpad",
            "--no-first-run",
            "--hide-scrollbars",
            "--default-background-color=00000000",
            f"--user-data-dir={profile}",
            f"--crash-dumps-dir={profile}",
            "--force-device-scale-factor=2",
            f"--window-size={width},{height}",
            "--virtual-time-budget=3000",
            f"--screenshot={out}",
            f"file://{page}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 90
    while time.time() < deadline:
        if out.exists() and out.stat().st_size and time.time() - out.stat().st_mtime > 1.5:
            break
        if process.poll() is not None:
            break
        time.sleep(0.5)
    if process.poll() is None:
        process.terminate()
    return out.exists()


if __name__ == "__main__":
    dark, light = lockup_svg(), lockup_svg(ink="#1b1b19", muted="#6b6b66")
    (HERE / "logo.svg").write_text(dark)
    (HERE / "logo-light.svg").write_text(light)
    print("wrote logo.svg, logo-light.svg")
    profile = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="laya-logo-")
    for name, svg in (("logo.png", dark), ("logo-light.png", light)):
        print(("wrote " if rasterize(svg, HERE / name, 620, 150, profile) else "skipped ") + name)
