# EvoSim

```
  _____________________________________________________________
 |                                                             |
 |   _____  __   __   ___    ____   ___    __  __             |
 |  | ____| \ \ / /  / _ \  / ___| |_ _| |  \/  |            |
 |  | |__    \ V /  | | | | \___ \  | |  | |\/| |            |
 |  |  __|    \_/   | |_| |  ___) |  | |  | |  | |           |
 |  |_____|         \___/  |____/  |___|  |_|  |_|           |
 |                                                             |
 |   500-dimensional · polygenic · multi-habitat              |
 |_____________________________________________________________|
```

A genetic evolution simulation engine written in Python. Creatures carry 500-dimensional gene
vectors encoding dozens of polygenic traits. They inhabit typed biomes, compete for resources,
reproduce sexually with Mendelian inheritance, migrate across connected regions, and diverge
into distinct species — all driven by the geometry of their genes against their environment.

---

## Quick Start

```bash
# Install dependencies
poetry install

# Run with the bundled default config (365 days, 4 habitats)
python runner.py

# Use a custom config
python runner.py path/to/my_config.toml

# Override number of days
python runner.py --days 500
```

Logs are written to `simulation_logs/YYYY-MM-DD_HH-MM-SS/` and are ignored by git.

---

## Core Concepts

### Genes and Traits

Each creature carries a **500-dimensional float gene vector**. Traits are **polygenic** —
each is computed via **Ordered Weighted Averaging (OWA)** over a distributed, overlapping
set of loci:

```
1. vals    = genes[trait_indices]
2. sorted  = sort(vals, descending)
3. w_i     = α · (1 − α)^i  (normalised to sum = 1,  default α = 0.6)
4. raw     = dot(weights, sorted)
5. trait   = sigmoid(raw)   → [0, 1]
```

The highest-valued locus receives weight α ≈ 0.6, the next α(1−α) ≈ 0.24, and so on.
A beneficial mutation that pushes a locus to the top of the ranking gains immediate
phenotypic weight rather than being diluted 1/N by a plain mean — making individual
mutations visible to selection. `OWA_ALPHA` is a class attribute that species subclasses
can override to tune how strongly the dominant locus controls expression.

Many loci contribute to multiple traits (**pleiotropy**), creating correlated selection pressure.

**Inheritance** is Mendelian: each offspring locus is drawn 50/50 from either parent. Mutation
is applied per-locus at that parent's heritable `mutation_rate` — so mutation rate itself evolves.

### Resource Discovery

Food and water are found via **cross-product geometry**. Each habitat and each creature has a
vector in the same 500-dimensional space. The daily probability of finding a resource is:

```
P(resource) = sin(θ)   where θ = angle between creature genes and habitat vector
```

A creature **parallel** to the habitat finds nothing — `sin(0°) = 0`.
A creature **orthogonal** to it exploits it maximally — `sin(90°) = 1`.

This drives niche differentiation: different genetic directions thrive in different environments,
and creatures whose genes align with the habitat's orthogonal complement out-compete others.

### Species Detection

A `SpeciesRegistry` tracks all species by their progenitor gene vector. When a creature is
born, its full 500-dim genome is compared against **every registered progenitor** via vectorized
cosine similarity. It is assigned to the closest existing species if the score exceeds the
threshold (default 0.95). If no species is close enough, a speciation event is declared.

Checking all progenitors prevents two failure modes:

- **Drift-back**: a lineage diverges then converges toward an ancestral genetic region — it is
  re-absorbed rather than logged as a new species.
- **Convergent evolution**: two independent lineages evolving toward the same genetic region are
  recognized as the same species, not declared new ones.

---

## Components

### `Creature` — `evolution_simulator/creature.py`

| Attribute | Description |
|---|---|
| `genes` | `np.ndarray (500,)` — full gene vector |
| `sex` | `"male"` or `"female"` — determined at birth from genes |
| `species` | Species name — inherited from parents, updated by `SpeciesRegistry` |
| `age` | Days lived |
| `energy` | `[0, 1]` — set daily by `Habitat` based on food discovery |
| `hydration` | `[0, 1]` — set daily by `Habitat` based on water discovery |
| `is_alive` | `False` after starvation, dehydration, or old age |
| `parents` | Direct parents (empty list for founding individuals) |

Key methods:

- `is_compatible(other)` → `(bool, float, str)` — checks sex, viability, pregnancy status, and
  cosine similarity of the 245-locus compatibility gene subset against the threshold.
- `reproduce(other)` → `list[Creature]` — runs a fertility check, creates a litter of
  `max(1, Poisson(fecundity))` offspring via Mendelian draws with per-locus mutation, stores
  them in `female._pending_offspring` until gestation completes.
- `simulate_day()` → `dict` — checks starvation/dehydration, increments age, advances
  pregnancy, returns a log with `cause_of_death`.

**Genetic traits** (each determined by multiple distributed gene loci):

| Trait | Range | Effect |
|---|---|---|
| `size` | [0, 1] | Influences aggression and resource needs |
| `strength` | [0, 1] | Competition / combat ability |
| `speed` | [0, 1] | Foraging effectiveness |
| `aggression` | [0, 1] | Intraspecific conflict intensity |
| `metabolism` | [0, 1] | Scales daily energy cost when food is absent |
| `water_efficiency` | [0, 1] | Scales daily hydration cost when water is absent |
| `fecundity` | [0, 1] | Poisson mean for litter size |
| `reproduction_likelihood` | [0, 1] | Bernoulli probability fertilization succeeds |
| `reproduction_time` | 10–500 days | Gestation / incubation period |
| `days_to_sexual_viability` | 30–1000 days | Age before reproduction is possible |
| `max_lifespan` | 100–5000 days | Maximum age before death |
| `migration_likelihood` | [0, 1] | Multiplied by `DAILY_MIGRATION_BASE` (0.01) |
| `intelligence` | [0, 1] | General fitness modifier |
| `immune_strength` | [0, 1] | Disease and stress resistance |
| `camouflage` | [0, 1] | Predation avoidance |
| `selectivity` | [0, 1] | Raises mate-compatibility threshold above the floor |
| `mutation_rate` | [0, 1] | Per-locus mutation probability (heritable and evolvable) |
| `sex_determination` | — | `sigmoid ≥ 0.5` → female |
| `compatibility_genes` | 245 loci | Used for cosine-similarity mating and species checks |

---

### `Habitat` — `evolution_simulator/habitat.py`

A geographic region with a 500-dim environment vector. Each simulated day:

1. Computes food and water likelihoods for all creatures via vectorized `sin θ`.
2. Updates energy and hydration; marks starvation / dehydration deaths.
3. Advances each creature's day (aging, pregnancy timer).
4. Collects litters from females that reached term.
5. Removes the dead.
6. Migrates willing creatures to passable neighbours.
7. Pairs viable males and females for mating (1:1 random shuffle).
8. Adds newborns and assigns species via the registry.
9. Attempts spontaneous route isolation.

`simulate_day()` returns a dict with `births`, `deaths`, `mating_events`, `migrations`, and
`isolations`. Migration events are returned rather than applied immediately so that the runner
can process all habitats before moving any creature, preventing double-simulation on the same day.

---

### Habitat Types — `evolution_simulator/habitats/types.py`

Each type has a **fixed characteristic center vector** (computed once at class definition from
`TYPE_SEED` via `__init_subclass__`) plus Gaussian per-instance noise. Resource constants are
tuned to create distinct selection pressures.

| Type | Food↑ | Food↓ | Water↑ | Water↓ | Migration | Notes |
|---|---|---|---|---|---|---|
| `Desert` | 0.25 | 0.20 | 0.30 | **0.35** | 1× | Punishes low water efficiency |
| `Forest` | 0.30 | 0.15 | 0.30 | 0.20 | 1× | Balanced baseline |
| `Rainforest` | **0.40** | 0.15 | **0.45** | 0.20 | 1× | Abundant; high competition |
| `Plains` | 0.30 | 0.15 | 0.30 | 0.20 | **1.5×** | Easy dispersal |
| `Tundra` | 0.20 | 0.20 | 0.25 | 0.20 | 1× | Sparse food; cold cost |
| `Ocean` | 0.35 | 0.15 | 0.50 | **0.00** | **2×** | No dehydration; fast dispersal |
| `CoralReef` | **0.45** | 0.15 | 0.50 | **0.00** | 1× | Highest productivity |
| `Wetlands` | 0.30 | 0.15 | **0.45** | **0.00** | 1× | Water everywhere |
| `Alpine` | 0.18 | **0.25** | 0.30 | 0.20 | **0.5×** | Scarce food; terrain limits movement |
| `Volcanic` | 0.22 | **0.25** | 0.20 | **0.30** | **0.5×** | Extreme mortality pressure |
| `Cave` | **0.15** | 0.15 | 0.30 | 0.20 | **0.3×** | Scarce everything |
| `Arctic` | **0.15** | **0.25** | 0.35 | 0.20 | 1× | Extreme cold |
| `River` | 0.30 | 0.15 | **0.45** | **0.00** | **1.5×** | Current-aided movement |
| `Savanna` | 0.28 | 0.15 | 0.30 | **0.25** | **1.2×** | Seasonal water stress |

Food↑/↓ = energy gain/cost per missed day. Water↑/↓ = hydration gain/cost per missed day.
Migration = multiple of the default 1% daily base rate.

---

### `SpeciesRegistry` — `evolution_simulator/species.py`

```python
from evolution_simulator.species import SpeciesRegistry

registry = SpeciesRegistry(species_threshold=0.95)

# Register a founding individual
name = registry.register_founding_species(founder.genes)
founder.species = name

# Assign species at birth (called automatically by SimulationRunner)
registry.assign_species(newborn)

# Inspect
registry.species_count                         # int
registry.all_species                           # list[str]
registry.speciation_events                     # list[dict]
registry.similarity_to_all_progenitors(c)      # dict[str, float]
```

Species names are drawn from an adjective + noun vocabulary in
`evolution_simulator/config/species_names.toml` (100 × 100 = 10,000 combinations).
The file is user-editable; each list must be non-empty with unique strings.

---

### `SimulationRunner` — `evolution_simulator/simulation.py`

```python
from pathlib import Path
from evolution_simulator.simulation import SimulationRunner

runner = SimulationRunner(Path("my_config.toml"))
log_dir = runner.setup()   # builds habitats, seeds population, writes metadata.json
runner.run()               # simulates all days, writes per-day JSON + summary.json
```

Step manually for custom control:
```python
runner.setup()
for _ in range(100):
    day_log = runner.step()   # full event dict for the day
```

**Population seeding**: each habitat gets `initial_species_per_habitat` distinct founding
genomes. `creatures_per_species` individuals are placed near each genome (within
`initial_genome_noise`), split 50/50 male/female. Founding creatures start at
`age = days_to_sexual_viability + 1` so mating begins immediately. Both parameters
can be overridden per habitat in the config, so different habitats can start with
different populations.

**Early termination**: `run()` halts before the configured number of days if the global
population reaches zero. `summary.json` records `"extinct": true` in that case.

---

## Configuration

Copy `evolution_simulator/config/simulation.toml` and edit as needed:

```toml
[simulation]
days                        = 365
seed                        = 42        # remove for a random seed each run
output_dir                  = "simulation_logs"
initial_species_per_habitat = 3         # distinct founding genomes per habitat
creatures_per_species       = 10        # creatures per founding genome (50/50 sex split)
initial_genome_noise        = 0.05      # gene noise around founding genome (keep ≤ 0.1)
isolation_probability       = 0.001     # per-link per-day route severance probability

[species]
threshold = 0.95                        # cosine similarity floor for same-species membership

[habitats]
connections = [
    ["desert_1", "plains_1"],
    ["plains_1", "forest_1"],
]

[[habitats.instances]]
id   = "desert_1"
type = "Desert"           # any type from the table above
seed = 1                  # controls per-instance variation around the type center
name = "Northern Desert"  # optional human-readable label

[[habitats.instances]]
id   = "plains_1"
type = "Plains"
seed = 2
# initial_species_per_habitat = 1   # per-habitat overrides
# creatures_per_species = 50        # → 25 female + 25 male, one founding genome
```

**Single-species isolation experiment**: set `initial_species_per_habitat = 1` and a large
`creatures_per_species` on a habitat with no connections to watch divergence from a single
ancestor. The simulation halts automatically if the population collapses to zero.

---

## Simulation Logs

```
simulation_logs/
└── 2024-01-15_14-30-00/
    ├── config.toml         ← exact copy of the config used
    ├── metadata.json       ← habitat topology, parameters, seed
    ├── day_00001.json
    ├── day_00002.json
    ├── ...
    └── summary.json        ← final state, speciation history, extinction flag
```

Each `day_NNNNN.json` captures every event for replay in a visualizer:

```jsonc
{
  "day": 42,
  "global_population": 247,
  "global_species_count": 5,
  "global_species_distribution": { "blazing nomad": 80, "vivid surger": 167 },
  "speciation_events": [{ "new_species": "...", "parent_species": "...", "creature_id": "..." }],
  "migrations": [{ "creature_id": "...", "species": "...", "from_habitat": "...", "to_habitat": "..." }],
  "habitats": {
    "desert_1": {
      "habitat_type": "Desert",
      "population": 62,
      "species_distribution": { "blazing nomad": 62 },
      "births": [{ "creature_id": "...", "species": "...", "sex": "female", "parents": ["...", "..."] }],
      "deaths": [{ "creature_id": "...", "cause": "starvation", "age": 183 }],
      "mating_events": [
        {
          "male_id": "...", "female_id": "...",
          "compatibility_score": 0.9731,
          "compatible": true, "fertilized": true,
          "litter_size": 2, "offspring_ids": ["...", "..."]
        },
        {
          "male_id": "...", "female_id": "...",
          "compatibility_score": 0.7812,
          "compatible": false, "fertilized": false,
          "reason": "incompatible_genes"
        }
      ],
      "migrations_out": [{ "creature_id": "...", "species": "...", "to_habitat": "plains_1" }],
      "isolations": []
    }
  }
}
```

---

## Project Structure

```
evolution_simulator/
├── runner.py                          ← stand-alone entry point
├── evolution_simulator/
│   ├── creature.py                    ← Creature class, gene/trait system, reproduction
│   ├── habitat.py                     ← Habitat base class, resource geometry, migration
│   ├── habitats/
│   │   ├── __init__.py
│   │   └── types.py                   ← 14 typed biome subclasses + HABITAT_TYPE_REGISTRY
│   ├── species.py                     ← SpeciesRegistry, speciation detection
│   ├── simulation.py                  ← SimulationRunner, JSON logging
│   └── config/
│       ├── simulation.toml            ← default simulation config (copy and edit)
│       └── species_names.toml         ← adjective/noun vocabulary for species names
└── tests/
    ├── test_creature.py
    ├── test_habitat.py
    └── test_species.py
```

## Tests

```bash
poetry run pytest
```

185 tests covering creature genetics, trait computation, Mendelian reproduction, habitat
resource geometry, neighbour management, species assignment, drift-back protection, and
convergent evolution detection.
