"""
Tests for the configurable logging cadence introduced in simulation.py.

These tests use a minimal in-memory TOML config written to a tmp_path so they
are isolated from the default simulation.toml and run fast (≤ 20 weeks each).
"""

import json
from pathlib import Path

from evolution_simulator.simulation import SimulationRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(
    path: Path, weeks: int, seed: int, stats_every: int, events_every: int, output_dir: str
) -> None:
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


def make_runner(
    tmp_path: Path, weeks: int, stats_every: int, events_every: int, seed: int = 7
) -> SimulationRunner:
    """Write a TOML config and return a set-up runner."""
    logs_dir = str(tmp_path / "logs")
    config_path = tmp_path / "config.toml"
    _write_toml(
        config_path,
        weeks=weeks,
        seed=seed,
        stats_every=stats_every,
        events_every=events_every,
        output_dir=logs_dir,
    )
    runner = SimulationRunner(config_path)
    runner.setup()
    return runner


def _write_two_hab_toml(
    path: Path,
    weeks: int,
    seed: int,
    stats_every: int,
    events_every: int,
    output_dir: str,
    isolation_probability: float = 0.0,
) -> None:
    """Write a minimal two-habitat connected config."""
    path.write_text(f"""
[simulation]
weeks = {weeks}
seed = {seed}
founding_habitat_bias = 0.55
initial_species_per_habitat = 1
creatures_per_species = 20
initial_genome_noise = 0.05
isolation_probability = {isolation_probability}
stats_every = {stats_every}
events_every = {events_every}
output_dir = {output_dir!r}

[species]
threshold = 0.60

[habitats]
connections = [["hab_a", "hab_b"]]

[[habitats.instances]]
id = "hab_a"
type = "Plains"
seed = 1
name = "Plains A"

[[habitats.instances]]
id = "hab_b"
type = "Forest"
seed = 2
name = "Forest B"
""")


def make_two_hab_runner(
    tmp_path: Path,
    weeks: int,
    stats_every: int,
    events_every: int,
    seed: int = 7,
    isolation_probability: float = 0.0,
) -> SimulationRunner:
    logs_dir = str(tmp_path / "logs")
    config_path = tmp_path / "config.toml"
    _write_two_hab_toml(
        config_path,
        weeks=weeks,
        seed=seed,
        stats_every=stats_every,
        events_every=events_every,
        output_dir=logs_dir,
        isolation_probability=isolation_probability,
    )
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


# ---------------------------------------------------------------------------
# Test 7 — interval_stats is present on stats-snapshot weeks and absent otherwise
# ---------------------------------------------------------------------------


def test_interval_stats_present_on_stats_weeks(tmp_path):
    weeks = 11
    runner = make_runner(tmp_path, weeks=weeks, stats_every=5, events_every=0)
    for _ in range(weeks):
        runner.step()

    # Week 1 is an endpoint — always gets a stats snapshot.
    with open(runner.log_dir / "week_00001.json") as fh:
        w1 = json.load(fh)
    assert "interval_stats" in w1, "Week 1 endpoint missing interval_stats"

    # Week 5 and week 10 are on the cadence.
    for w in (5, 10):
        path = runner.log_dir / f"week_{w:05d}.json"
        assert path.exists(), f"week_{w:05d}.json not written"
        with open(path) as fh:
            data = json.load(fh)
        assert "interval_stats" in data, f"week_{w:05d}.json missing interval_stats"

    # Week 3 is not logged at all (not a stats or events week).
    assert not (runner.log_dir / "week_00003.json").exists()


# ---------------------------------------------------------------------------
# Test 8 — interval_stats has the expected nested structure
# ---------------------------------------------------------------------------


def test_interval_stats_structure(tmp_path):
    runner = make_runner(tmp_path, weeks=10, stats_every=10, events_every=0)
    for _ in range(10):
        runner.step()

    with open(runner.log_dir / "week_00010.json") as fh:
        data = json.load(fh)

    ist = data["interval_stats"]
    assert "weeks_covered" in ist
    assert "deaths" in ist
    assert "births" in ist
    assert "migrations" in ist
    assert "mating" in ist
    assert "isolation_events" in ist

    deaths = ist["deaths"]
    assert "total" in deaths
    assert "by_cause" in deaths
    assert "by_species" in deaths

    births = ist["births"]
    assert "total" in births
    assert "by_habitat" in births
    assert "by_species" in births
    assert "by_habitat_and_species" in births

    migrations = ist["migrations"]
    assert "total" in migrations
    assert "by_route" in migrations

    mating = ist["mating"]
    assert "total_pairings" in mating
    assert "fertilizations" in mating
    assert "hybrid_conceptions" in mating


# ---------------------------------------------------------------------------
# Test 9 — interval_stats accumulates real death/birth counts over the bin
# ---------------------------------------------------------------------------


def test_interval_stats_accumulates_deaths_and_births(tmp_path):
    # 30 creatures, 20 weeks — enough for starvation/dehydration deaths and births.
    runner = make_runner(tmp_path, weeks=20, stats_every=20, events_every=0, seed=42)
    for _ in range(20):
        runner.step()

    with open(runner.log_dir / "week_00020.json") as fh:
        data = json.load(fh)

    ist = data["interval_stats"]
    # Week 1 is always an endpoint flush, so bin 2 covers [2, 20].
    assert ist["weeks_covered"] == [2, 20]
    assert ist["deaths"]["total"] > 0, "Expected some deaths over 20 weeks"
    assert ist["births"]["total"] > 0, "Expected some births over 20 weeks"

    # Cause keys should only be valid death causes.
    valid_causes = {"starvation", "dehydration", "old_age", "predation", "unknown"}
    for cause in ist["deaths"]["by_cause"]:
        assert cause in valid_causes, f"Unexpected death cause: {cause!r}"

    # Deaths by_species should name actual species, not just "unknown".
    assert any(k != "unknown" for k in ist["deaths"]["by_species"]), (
        "All deaths recorded as 'unknown' species — species not captured at death time"
    )

    # Births should be attributed to the single habitat.
    assert "hab_a" in ist["births"]["by_habitat"]
    assert "hab_a" in ist["births"]["by_habitat_and_species"]


# ---------------------------------------------------------------------------
# Test 10 — accumulators reset between bins; weeks_covered spans correctly
# ---------------------------------------------------------------------------


def test_interval_stats_weeks_covered_resets_between_bins(tmp_path):
    runner = make_runner(tmp_path, weeks=20, stats_every=10, events_every=0)
    for _ in range(20):
        runner.step()

    with open(runner.log_dir / "week_00010.json") as fh:
        w10 = json.load(fh)
    with open(runner.log_dir / "week_00020.json") as fh:
        w20 = json.load(fh)

    # Week 1 is always an endpoint flush with its own [1,1] bin, so bin 2 starts at 2.
    assert w10["interval_stats"]["weeks_covered"] == [2, 10]
    assert w20["interval_stats"]["weeks_covered"] == [11, 20]

    # Each bin's death total should be non-negative and finite.
    assert w10["interval_stats"]["deaths"]["total"] >= 0
    assert w20["interval_stats"]["deaths"]["total"] >= 0


# ---------------------------------------------------------------------------
# Test 11 — migrations are captured by route when habitats are connected
# ---------------------------------------------------------------------------


def test_interval_stats_captures_migrations(tmp_path):
    # Two connected habitats — some migration will happen over 20 weeks.
    runner = make_two_hab_runner(tmp_path, weeks=20, stats_every=20, events_every=0, seed=99)
    for _ in range(20):
        runner.step()

    with open(runner.log_dir / "week_00020.json") as fh:
        data = json.load(fh)

    migrations = data["interval_stats"]["migrations"]
    # Route keys use the "from→to" format; both directions are possible.
    for route in migrations["by_route"]:
        assert "→" in route, f"Route key missing arrow: {route!r}"
    assert migrations["total"] == sum(migrations["by_route"].values())


# ---------------------------------------------------------------------------
# Test 12 — isolation events appear in interval_stats and in summary.json
# ---------------------------------------------------------------------------


def test_isolation_events_logged(tmp_path):
    # isolation_probability=1.0 guarantees the single link closes on week 1.
    # Week 1 is always an endpoint flush, so events from week 1 land in week_00001.json.
    runner = make_two_hab_runner(
        tmp_path, weeks=10, stats_every=10, events_every=0, seed=5, isolation_probability=1.0
    )
    for _ in range(10):
        runner.step()
    runner._write_summary(extinct=False)

    # Week 1's bin [1,1] captures the closure events — check that file.
    with open(runner.log_dir / "week_00001.json") as fh:
        w1 = json.load(fh)

    iso_events = w1["interval_stats"]["isolation_events"]
    assert len(iso_events) > 0, "Expected isolation events with probability=1.0 in week 1 bin"

    for ev in iso_events:
        assert "week" in ev
        assert "from_habitat" in ev
        assert "blocked_neighbor" in ev

    # summary.json accumulates all isolation events across the full run.
    with open(runner.log_dir / "summary.json") as fh:
        summary = json.load(fh)

    assert "all_isolation_events" in summary
    assert len(summary["all_isolation_events"]) > 0


# ---------------------------------------------------------------------------
# Test 13 — interval_stats absent from events-only weeks (no stats)
# ---------------------------------------------------------------------------


def test_interval_stats_absent_from_events_only_week(tmp_path):
    # stats_every=10 means only weeks 1, 10 get stats; events fire every week.
    runner = make_runner(tmp_path, weeks=10, stats_every=10, events_every=1)
    for _ in range(10):
        runner.step()

    # Week 5: events logged, no stats → interval_stats must be absent.
    with open(runner.log_dir / "week_00005.json") as fh:
        w5 = json.load(fh)

    assert "events_logged" in w5 or "births" in next(iter(w5["habitats"].values()))
    assert "interval_stats" not in w5, "interval_stats leaked into an events-only week"
