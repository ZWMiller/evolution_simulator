#!/usr/bin/env python3
"""
check_run_progress_from_logs.py  —  print a quick status snapshot for a running (or finished) sim.

Usage:
    python3 scripts/check_run_progress_from_logs.py                               # auto-picks the most recent log dir
    python3 scripts/check_run_progress_from_logs.py simulation_logs/2026-05-26_18-20-25
"""

import datetime
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def find_log_dir() -> Path:
    candidates = sorted((_REPO_ROOT / "simulation_logs").iterdir(), reverse=True)
    for d in candidates:
        if d.is_dir() and any(d.glob("week_*.json")):
            return d
    sys.exit(f"No simulation log directories found under {_REPO_ROOT / 'simulation_logs'}")


def latest_week_file(log_dir: Path) -> Path:
    files = sorted(log_dir.glob("week_*.json"))
    if not files:
        sys.exit(f"No week_*.json files found in {log_dir}")
    return files[-1]


def parse_start_time(log_dir: Path) -> datetime.datetime | None:
    try:
        name = log_dir.name  # e.g. 2026-05-26_18-20-25
        return datetime.datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def main():
    if len(sys.argv) > 1:
        log_dir = Path(sys.argv[1])
    else:
        log_dir = find_log_dir()

    week_file = latest_week_file(log_dir)
    file_mtime = datetime.datetime.fromtimestamp(week_file.stat().st_mtime)

    with open(week_file) as f:
        d = json.load(f)

    current_week = d["week"]
    total_weeks = None
    config_path = log_dir / "config.toml"
    if config_path.exists():
        for line in config_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("weeks") and "=" in stripped and "#" not in stripped.split("=")[0]:
                try:
                    total_weeks = int(stripped.split("=")[1].split("#")[0].strip())
                except ValueError:
                    pass

    global_pop = d["global_population"]
    total_species_ever = d["global_species_count"]
    living_dist = d.get("global_species_distribution", {})
    living_species = len(living_dist)

    # Timing estimate
    start_time = parse_start_time(log_dir)
    if start_time and total_weeks:
        elapsed_sec = (file_mtime - start_time).total_seconds()
        rate = current_week / (elapsed_sec / 60)  # weeks per minute
        remaining_weeks = total_weeks - current_week
        eta_min = remaining_weeks / rate if rate > 0 else None
    else:
        rate = eta_min = None

    finished = (log_dir / "summary.json").exists()

    # ── Header ──────────────────────────────────────────────────────────────
    print(f"\nRun:  {log_dir}")
    print(f"File: {week_file.name}  (written {file_mtime.strftime('%H:%M:%S')})")
    print()

    if finished:
        print(f"Status: COMPLETE  ({current_week} weeks)")
    elif total_weeks:
        pct = 100 * current_week / total_weeks
        print(f"Progress: week {current_week:,} / {total_weeks:,}  ({pct:.1f}%)")
    else:
        print(f"Progress: week {current_week:,}")

    if rate and eta_min is not None and not finished:
        eta_str = f"{eta_min:.0f} min"
        if eta_min > 90:
            eta_str += f"  (~{eta_min / 60:.1f} hr)"
        eta_clock = file_mtime + datetime.timedelta(minutes=eta_min)
        print(f"Rate:     {rate:.0f} weeks/min  →  ETA {eta_str}  (≈ {eta_clock.strftime('%H:%M')} local)")

    print()
    print(f"Population:          {global_pop:,}")
    print(f"Living species:      {living_species}")
    print(f"Total species ever:  {total_species_ever}")
    print()

    # ── Per-habitat table ────────────────────────────────────────────────────
    habitats = d.get("habitats", {})
    col_w = max((len(h.get("habitat_name", h.get("habitat_id", "?"))) for h in habitats.values()), default=10)
    header = f"  {'Habitat':<{col_w}}  {'Pop':>6}  {'Species':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for hab in habitats.values():
        name = hab.get("habitat_name", hab.get("habitat_id", "?"))
        pop = hab["population"]
        n_sp = len(hab.get("species_distribution", {}))
        print(f"  {name:<{col_w}}  {pop:>6,}  {n_sp:>7}")

    print()
    print("  (Species per habitat overlap — same species can appear in multiple habitats.)")
    print(f"  Unique living species across all habitats: {living_species}")
    print()


if __name__ == "__main__":
    main()
