# EvoSim

```
  ____________________________________________________
 |                                                    |
 |   _____  __   __   ___    ____   _____    __  __   |
 |  | ____| \ \ / /  / _ \  / ___| |_  __|  |  \/  |  |
 |  | |__    \ V /  | | | | | (__    | |    | |\/| |  |
 |  |  __|    \_/   | | | | \___ \   | |    | |  | |  |
 |  |  |__    ---   | |_| |  ___) | _| |_   | |  | |  |
 |  |_____|          \___/  |____/ |_____|  |_|  |_|  |
 |                                                    |
 |   500-dimensional · polygenic · multi-habitat      |
 |____________________________________________________|
```

A genetic evolution simulation engine written in Python. Creatures carry 500-dimensional gene
vectors encoding dozens of polygenic traits. They inhabit typed biomes, compete for resources,
reproduce sexually with Mendelian inheritance, migrate across connected regions, and diverge
into distinct species — all driven by the geometry of their genes against their environment.
The simulation advances in **weeks**. For a full technical reference see
[`docs/TECHNICAL_DOCS.md`](docs/TECHNICAL_DOCS.md); for the history of design decisions see
[`docs/DECISION_LOG.md`](docs/DECISION_LOG.md).

---

## What makes this different

Most evolution simulations compress genetics into a handful of explicit parameters and define a
fitness function that directly scores each individual. EvoSim instead poses evolution as a dual
optimization problem that creatures solve implicitly: geometric alignment of a 500-dim gene
vector with the habitat environment vector determines resource discovery probability, while a
separate set of polygenic traits — fecundity, metabolism, lifespan, predation vulnerability, and
others — shapes survivability and reproduction. Neither axis alone determines success, and the
two interact through pleiotropy. Speciation is detected via the same reproductive-isolation
signal that gates mating, not a fixed threshold on arbitrary genetic distance.

---

## Installation

```bash
# Install dependencies (requires Python 3.13+)
poetry install
```

---

## Quick Start

```bash
# Run with the bundled default config (4 habitats, 200 weeks)
python runner.py

# Use a custom config
python runner.py path/to/my_config.toml

# Override the number of weeks
python runner.py --weeks 5000
```

Logs are written to `simulation_logs/YYYY-MM-DD_HH-MM-SS/` and are git-ignored.
Ready-made configs for longer experiments live in `simulation_configs/`.

---

## Core Components

**`Creature` (`evolution_simulator/creature.py`)** — carries a 500-dim gene vector. All traits
are polygenic, computed via Ordered Weighted Averaging (OWA). Sex, mutation rate, compatibility,
and phenotype are all gene-encoded and heritable.

**`Habitat` (`evolution_simulator/habitat.py`)** — a biome with its own 500-dim environment
vector. Resource discovery is dot-product geometry: creatures aligned with the habitat find food
and water with high probability; misaligned creatures struggle. Supports 4 configurable mating
strategies and returns migration events rather than applying them, preventing double-simulation.

**`SpeciesRegistry` (`evolution_simulator/species.py`)** — detects and tracks species via
reproductive isolation on a 245-locus compatibility subset. Supports two speciation mechanisms:
cladogenesis (population splits detected via spherical k-means) and anagenesis (in-place
phenotype transformation of an intact lineage).

**`SimulationRunner` (`evolution_simulator/simulation.py`)** — orchestrates habitat steps,
migrations, speciation detection, and JSON log output.

---

## Visualizing Results

```bash
# Static summary charts
python visualizer/visualizer_basic.py simulation_logs/<run>/

# Interactive Dash app (phylogeny, habitat map, family tree, trait compare)
python visualizer/visualizer_advanced.py simulation_logs/<run>/
```

---

## Tests

```bash
poetry run pytest
```

263 tests covering gene/trait computation, reproduction, resource geometry, predation, migration,
the full speciation stack, and all four mating strategies.
