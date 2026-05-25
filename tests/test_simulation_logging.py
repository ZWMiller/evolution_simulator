"""
Tests for the configurable logging cadence introduced in simulation.py.

These tests use a minimal in-memory TOML config written to a tmp_path so they
are isolated from the default simulation.toml and run fast (≤ 20 weeks each).
"""

import json
import pytest
from pathlib import Path

from evolution_simulator.simulation import SimulationRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_toml(path: Path, weeks: int, seed: int, stats_every: int,
                events_every: int, output_dir: str) -> None:
    """Write a minimal simulation TOML config."""
    path.write_text(f"""
[simulation]
weeks = {weeks}
seed = {seed}
founding_habitat_bias = 0.55
initial_species_per_habitat = 1
creatures_per_species = 10
initial_genome_noise = 0.05
isolation_probability = 0.0
stats_every = {stats_every}
events_every = {events_every}
output_dir = {output_dir!r}

[species]
threshold = 0.60

[habitats]
connections = []

[[habitats.instances]]
id = "hab_a"
type = "Plains"
seed = 1
name = "Test Plains"
""")


def make_runner(tmp_path: Path, weeks: int, stats_every: int,
                events_every: int, seed: int = 7) -> SimulationRunner:
    """Write a TOML config and return a set-up runner."""
    logs_dir = str(tmp_path / "logs")
    config_path = tmp_path / "config.toml"
    _write_toml(config_path, weeks=weeks, seed=seed,
                stats_every=stats_every, events_every=events_every,
                output_dir=logs_dir)
    runner = SimulationRunner(config_path)
    runner.setup()
    return runner


# ---------------------------------------------------------------------------
# Test 1 — Default cadence: every-week behavior is unchanged
# ---------------------------------------------------------------------------

def test_default_cadence_every_week(tmp_path):
    weeks = 10
    runner = make_runner(tmp_path, weeks=weeks, stats_every=1, events_every=1)

    for _ in range(weeks):
        runner.step()

    files = sorted(runner.log_dir.glob("week_*.json"))
    assert len(files) == weeks, f"Expected {weeks} files, got {len(files)}"

    assert (runner.log_dir / "week_00001.json").exists()

    with open(runner.log_dir / "week_00001.json") as fh:
        w1 = json.load(fh)

    for key in ("habitats", "habitat_stats", "species_stats", "global_species_distribution"):
        assert key in w1, f"Missing key '{key}' in week_00001.json"

    hab_entry = next(iter(w1["habitats"].values()))
    assert "births" in hab_entry


# ---------------------------------------------------------------------------
# Test 2 — Stats-only cadence: stats every 5 weeks, no events
# ---------------------------------------------------------------------------

def test_stats_only_cadence(tmp_path):
    weeks = 20
    runner = make_runner(tmp_path, weeks=weeks, stats_every=5, events_every=0)

    for _ in range(weeks):
        runner.step()

    files = {int(f.stem.split("_")[1]) for f in runner.log_dir.glob("week_*.json")}
    # week 1 (endpoint), 5, 10, 15, 20 (endpoint) → 5 files
    assert 1 in files
    assert 20 in files
    for w in (5, 10, 15):
        assert w in files

    # A non-endpoint stats week (e.g. week 5) has stats but no events
    with open(runner.log_dir / "week_00005.json") as fh:
        w5 = json.load(fh)

    assert "habitat_stats" in w5
    assert "species_stats" in w5
    assert "migrations" not in w5
    assert "speciation_events" not in w5

    hab_entry = next(iter(w5["habitats"].values()))
    assert "population" in hab_entry
    assert "births" not in hab_entry
    assert "mating_events" not in hab_entry


# ---------------------------------------------------------------------------
# Test 3 — Events-only cadence: events every week, stats never (except endpoints)
# ---------------------------------------------------------------------------

def test_events_only_cadence(tmp_path):
    weeks = 6
    runner = make_runner(tmp_path, weeks=weeks, stats_every=0, events_every=1)

    for _ in range(weeks):
        runner.step()

    files = sorted(runner.log_dir.glob("week_*.json"))
    assert len(files) == weeks

    assert (runner.log_dir / "week_00001.json").exists()

    # Mid-run week (e.g. week 3): has events, no stats
    with open(runner.log_dir / "week_00003.json") as fh:
        w3 = json.load(fh)

    hab_entry = next(iter(w3["habitats"].values()))
    assert "births" in hab_entry
    assert "mating_events" in hab_entry
    assert "population" in hab_entry
    assert "habitat_stats" not in w3
    assert "species_stats" not in w3
    assert "global_species_distribution" not in w3

    # Week 1 is an endpoint → also gets a stats snapshot
    with open(runner.log_dir / "week_00001.json") as fh:
        w1 = json.load(fh)
    assert "habitat_stats" in w1


# ---------------------------------------------------------------------------
# Test 4 — Fully-skipped week returns lightweight dict, no file written
# ---------------------------------------------------------------------------

def test_skipped_week_lightweight(tmp_path):
    weeks = 10
    runner = make_runner(tmp_path, weeks=weeks, stats_every=10, events_every=10)

    runner.step()  # week 1 — endpoint, will log
    result_w3 = None
    for _ in range(weeks - 1):
        r = runner.step()
        if runner.week == 3:
            result_w3 = r

    assert result_w3 is not None
    assert result_w3["logged"] is False
    assert result_w3["stats_logged"] is False
    assert result_w3["events_logged"] is False
    assert "week" in result_w3
    assert "global_population" in result_w3
    assert "global_species_count" in result_w3
    assert "births_this_week" in result_w3
    assert "deaths_this_week" in result_w3
    assert "habitats" not in result_w3

    assert not (runner.log_dir / "week_00003.json").exists()


# ---------------------------------------------------------------------------
# Test 5 — Every written file contains the visualizer-required keys
# ---------------------------------------------------------------------------

def test_every_file_has_required_keys(tmp_path):
    weeks = 20
    runner = make_runner(tmp_path, weeks=weeks, stats_every=7, events_every=3)

    for _ in range(weeks):
        runner.step()

    for path in runner.log_dir.glob("week_*.json"):
        with open(path) as fh:
            data = json.load(fh)
        for key in ("week", "global_population", "global_species_count"):
            assert key in data, f"Missing '{key}' in {path.name}"


# ---------------------------------------------------------------------------
# Test 6 — summary.json and metadata.json are always written
# ---------------------------------------------------------------------------

def test_summary_and_metadata_always_written(tmp_path):
    weeks = 15
    runner = make_runner(tmp_path, weeks=weeks, stats_every=7, events_every=0)

    for _ in range(weeks):
        runner.step()
    runner._write_summary(extinct=False)

    assert (runner.log_dir / "summary.json").exists()
    assert (runner.log_dir / "metadata.json").exists()

    with open(runner.log_dir / "summary.json") as fh:
        summary = json.load(fh)
    assert "weeks_simulated" in summary
    assert "extinct" in summary

    with open(runner.log_dir / "metadata.json") as fh:
        meta = json.load(fh)
    assert "parameters" in meta
