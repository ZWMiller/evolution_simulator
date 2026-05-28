#!/usr/bin/env bash
# Usage: ./make_bimodality_gif.sh [output_dir]
# If no dir given, uses the most recent run under experiments/bimodality_output/.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -n "$1" ]]; then
    RUN_DIR="$1"
else
    RUN_DIR=$(ls -dt "$SCRIPT_DIR/bimodality_output"/2* 2>/dev/null | head -1)
fi

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "No bimodality output directory found. Run cladogenesis_bimodality.py first." >&2
    exit 1
fi

echo "Building GIF from: $RUN_DIR"

poetry -C "$SCRIPT_DIR/.." run python - "$RUN_DIR" <<'EOF'
import sys
from pathlib import Path
from PIL import Image

run_dir = Path(sys.argv[1])

frames = sorted(run_dir.glob("week_*.png"))
# keep only week_00010 through week_01500
frames = [f for f in frames if 10 <= int(f.stem.split("_")[1]) <= 1500]

if not frames:
    print("No week frames found in", run_dir)
    sys.exit(1)

timeseries = run_dir / "centroid_cosine.png"
if not timeseries.exists():
    print("centroid_cosine.png not found — gif will not include time series")
    timeseries = None

print(f"  {len(frames)} week frames  + {'1 time-series frame' if timeseries else 'no time-series'}")

imgs = [Image.open(f).convert("RGBA") for f in frames]
durations = [500] * len(imgs)   # 0.5 s per frame

if timeseries:
    imgs.append(Image.open(timeseries).convert("RGBA"))
    durations.append(10_000)    # 10 s for the summary chart

out_path = run_dir / "bimodality.gif"
imgs[0].save(
    out_path,
    save_all=True,
    append_images=imgs[1:],
    duration=durations,
    loop=0,
    optimize=False,
)
print(f"  Saved → {out_path}  ({out_path.stat().st_size // 1024} KB)")
EOF
