"""
Reproducibility tests.

A run with a fixed config + seed must be deterministic: running it twice produces
identical logs.  We compare the actual on-disk log files (the artefacts a
researcher would diff), excluding only the wall-clock timestamp fields, which
record *when* the process ran rather than any simulation state and so are
inherently non-reproducible by design.

These guard the seeding/ordering invariants:
  - the population is iterated in a deterministic (insertion) order, not set order;
  - creature IDs are a deterministic counter, not uuid4;
  - logic-bearing set iterations are sorted before use;
  - every RNG stream is seeded from the run seed.
"""

import json

import pytest

from evolution_simulator.simulation import SimulationRunner

# Fields that record wall-clock time, not simulation state — excluded from the
# log comparison (they differ between any two runs by construction).
_VOLATILE_KEYS = {"timestamp", "simulation_start", "simulation_end"}

# Two small connected habitats, full per-week logging, a seed, and gates low
# enough that births / migrations / speciation candidates all exercise within a
# short run.
_CONFIG_TEMPLATE = """
[simulation]
weeks = {weeks}
seed = {seed}
founding_habitat_bias = 0.55
output_dir = "{output_dir}"
initial_species_per_habitat = 2
creatures_per_species = 8
initial_genome_noise = 0.12
isolation_probability = 0.02
stats_every = 1
events_every = 1
respeciate_every = 5
mating_strategy = "{mating_strategy}"

[species]
compatibility_threshold = 0.65
min_species_population = 4
min_species_weeks = 8
split_max_k = 4
split_isolation_threshold = 0.75
anagenesis_threshold = 0.93
anagenesis_weeks = 8

[habitats]
connections = [["a", "b"]]

[[habitats.instances]]
id = "a"
type = "Forest"
seed = 1

[[habitats.instances]]
id = "b"
type = "Plains"
seed = 2
"""


def _strip_volatile(obj):
    """Recursively drop wall-clock keys so only simulation content is compared."""
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _run(tmp_path, label, *, weeks, seed, mating_strategy="weighted_matrix"):
    """Run a full simulation and return its log directory."""
    out = tmp_path / label
    cfg = tmp_path / f"{label}.toml"
    cfg.write_text(
        _CONFIG_TEMPLATE.format(
            weeks=weeks, seed=seed, output_dir=out.as_posix(), mating_strategy=mating_strategy
        )
    )
    runner = SimulationRunner(cfg)
    runner.setup()
    runner.run()
    return runner.log_dir


def _load_logs(log_dir):
    """Load every JSON log file, keyed by filename, with volatile fields removed."""
    paths = sorted(log_dir.glob("week_*.json"))
    paths += [log_dir / "summary.json", log_dir / "metadata.json"]
    return {p.name: _strip_volatile(json.loads(p.read_text())) for p in paths if p.exists()}


@pytest.mark.parametrize(
    "mating_strategy",
    ["weighted_matrix", "zip", "species_priority", "stable_matching"],
)
def test_same_seed_produces_identical_logs(tmp_path, mating_strategy):
    """Two runs with the same config + seed write identical logs (modulo timestamps)."""
    dir_a = _run(tmp_path, "run_a", weeks=30, seed=123, mating_strategy=mating_strategy)
    dir_b = _run(tmp_path, "run_b", weeks=30, seed=123, mating_strategy=mating_strategy)

    logs_a = _load_logs(dir_a)
    logs_b = _load_logs(dir_b)

    assert logs_a, "no logs were written"
    assert logs_a.keys() == logs_b.keys(), "the two runs wrote different sets of log files"
    for name in logs_a:
        assert logs_a[name] == logs_b[name], (
            f"{name} differs between two identical-seed runs ({mating_strategy})"
        )


def test_summary_is_reproducible(tmp_path):
    """The end-of-run summary (species counts, events, populations) is identical."""
    dir_a = _run(tmp_path, "sum_a", weeks=40, seed=7)
    dir_b = _run(tmp_path, "sum_b", weeks=40, seed=7)

    sa = _strip_volatile(json.loads((dir_a / "summary.json").read_text()))
    sb = _strip_volatile(json.loads((dir_b / "summary.json").read_text()))
    assert sa == sb


def test_different_seed_produces_different_logs(tmp_path):
    """Sanity check: the comparison can actually detect divergence (different seed)."""
    dir_a = _run(tmp_path, "diff_a", weeks=30, seed=1)
    dir_b = _run(tmp_path, "diff_b", weeks=30, seed=2)

    logs_a = _load_logs(dir_a)
    logs_b = _load_logs(dir_b)
    assert logs_a != logs_b, "different seeds unexpectedly produced identical logs"


def test_no_global_np_random_in_hot_path(tmp_path, monkeypatch):
    """The stepping hot path must never touch the global np.random singleton.

    Every stochastic call (resource finding, predation, migration, mating,
    reproduction, isolation, species naming) draws from one explicit numpy
    Generator threaded from SimulationRunner.  Here we replace each global draw
    function with one that raises; a clean full run proves the simulation reads
    only the threaded Generator.  This guards against a future call site
    silently reintroducing a global draw (which would be invisible to the
    same-seed comparison until it happened to diverge).
    """
    import numpy as np

    def _forbidden(*args, **kwargs):
        raise AssertionError("simulation logic touched the global np.random singleton")

    # The legacy global singleton functions.  Generator methods (e.g.
    # gen.standard_normal) are bound to the threaded instance and are unaffected
    # by patching these module-level names, so threaded draws still work.
    for name in ("random", "randint", "shuffle", "choice", "normal", "poisson", "randn", "standard_normal"):
        monkeypatch.setattr(np.random, name, _forbidden)

    # A full run that exercises mating, reproduction, migration, predation,
    # isolation, and species naming without raising.
    _run(tmp_path, "no_global", weeks=20, seed=42, mating_strategy="weighted_matrix")


def test_deterministic_creature_ids(tmp_path):
    """Creature IDs are the deterministic counter, identical across same-seed runs."""
    dir_a = _run(tmp_path, "id_a", weeks=20, seed=99)
    dir_b = _run(tmp_path, "id_b", weeks=20, seed=99)

    meta_a = json.loads((dir_a / "metadata.json").read_text())
    meta_b = json.loads((dir_b / "metadata.json").read_text())
    ids_a = [f["creature_id"] for hab in meta_a["founders_by_hab"].values() for f in hab]
    ids_b = [f["creature_id"] for hab in meta_b["founders_by_hab"].values() for f in hab]
    assert ids_a == ids_b
    assert ids_a, "no founders recorded"
    assert all(cid.startswith("c") for cid in ids_a)
