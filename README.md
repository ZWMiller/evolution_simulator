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
The simulation advances in **weeks**.

---

## Quick Start

```bash
# Install dependencies
poetry install

# Run with the bundled default config
python runner.py

# Use a custom config
python runner.py path/to/my_config.toml

# Override the number of weeks
python runner.py --weeks 5000
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
can override.

Many loci contribute to multiple traits (**pleiotropy**), creating correlated selection pressure.

**Inheritance** is Mendelian: each offspring locus is drawn 50/50 from either parent. Mutation
is applied per-locus at that parent's heritable `mutation_rate` — so mutation rate itself evolves.

### Resource Discovery

Food and water are found via **dot-product geometry**. Each habitat and each creature has a
vector in the same 500-dimensional space. The weekly probability of finding a resource is:

```
P(resource) = (cos θ + 1) / 2   where θ = angle between creature genes and habitat vector
```

Aligned with the habitat → P=1 (perfectly adapted); orthogonal → P=0.5 (baseline); anti-aligned
→ P=0 (maladapted). This drives local adaptation without any explicit fitness function.

### Mortality

Beyond starvation, dehydration, and old age, each creature faces **density-dependent predation**
each week: `base_predation_rate` (an intrinsic, heritable vulnerability) plus a crowding term
`PREDATION_ALPHA · N / POPULATION_SUPPORT`. `base_predation_rate` shares loci with `fecundity`,
encoding an r/K trade-off (high-fecundity genotypes are also more conspicuous).

### Speciation

This is the heart of the engine, and it is built on one principle: **a species boundary means
reproductive isolation**, measured on the *same* 245-locus `compatibility_genes` subset that
gates mating. Each species keeps **two references**:

- a frozen full-genome **type** (anchors the name and the anagenesis axis), and
- a **living compatibility centroid**, refreshed every `respeciate_every` weeks from current
  members — so a whole population drifting together never speciates (the reference moves with it).

Three mechanisms run on top of this:

1. **Membership.** A newborn joins the nearest living species whose centroid is within
   `compatibility_threshold` (compat-subset cosine); otherwise it enters a two-stage **candidate**
   that is promoted to a confirmed species only once it has `min_species_population` members and
   has persisted `min_species_weeks` weeks.

2. **Cladogenesis (splits).** Periodically, each species' members are clustered in compatibility
   space (spherical k-means, K = 1..`split_max_k`); the largest K whose sub-clusters are mutually
   below `split_isolation_threshold` (the mating floor) wins. The sub-cluster nearest the frozen
   type keeps the name; the others split off. This catches divergence a single centroid would mask
   — e.g. one species adapting separately in two disconnected habitats, whose shared centroid sits
   at their midpoint and hides the split until they are nearly anti-aligned.

3. **Anagenesis (transformation).** A lineage can stay one interbreeding population yet drift far
   from its ancestral *form* over time (think dogs from wolves). When a species' living **phenotype**
   centroid drifts below `anagenesis_threshold` (centred cosine) from its frozen type and *persists*
   for `anagenesis_weeks`, a descendant species is minted and living members are re-sorted to
   whichever (ancestor vs descendant) they now resemble. Dead ancestors keep their labels — the log
   is the fossil record.

Every speciation event carries an `event_type` of `"cladogenesis"` or `"anagenesis"`, which the
visualizers render distinctly in the phylogeny.

---

## Components

### `Creature` — `evolution_simulator/creature.py`

| Attribute | Description |
|---|---|
| `genes` | `np.ndarray (500,)` — full gene vector |
| `sex` | `"male"`/`"female"` — `genes[sex_locus] >= 0 → female` |
| `species` | Inherited from the first parent; set by `SpeciesRegistry` |
| `age` | Weeks lived |
| `energy` / `hydration` | `[0, 1]` — set each week by `Habitat` |
| `is_alive` / `cause_of_death` | starvation, dehydration, old_age, or predation |

Key methods: `is_compatible(other)` (sex/viability/pregnancy + compat-subset cosine vs
`COMPATIBILITY_FLOOR + 0.15·selectivity`), `reproduce(other)` (fertility check, Poisson litter,
per-locus Mendelian inheritance + mutation), `simulate_week()`. Module helpers `compute_phenotype`
/ `compute_phenotype_matrix` produce the raw `PHENOTYPE_TRAITS` vector used by the anagenesis axis.

**Selected traits** (each polygenic; values are the scaled property ranges):

| Trait | Range | Effect |
|---|---|---|
| `fecundity` | 1–8 | Poisson mean for litter size |
| `reproduction_time` | 1–20 wk | Gestation period |
| `weeks_to_sexual_viability` | 4–50 wk | Age before reproduction |
| `max_lifespan` | 40–400 wk | Maximum age |
| `metabolism` | 0.5–2.0 | Scales weekly energy cost when food is missed |
| `water_efficiency` | [0, 1] | Reduces hydration cost when water is missed |
| `migration_likelihood` | [0, 1] | × `WEEKLY_MIGRATION_BASE` (0.01) |
| `base_predation_rate` | 0–0.005 | Intrinsic weekly predation vulnerability |
| `selectivity` | [0, 1] | Raises mate-compatibility floor; also controls mating sharpness in `weighted_matrix`/`stable_matching` |
| `mutation_rate` | 0.001–0.05 | Per-locus mutation probability (heritable) |
| `compatibility_genes` | 245 loci | Cosine basis for mating **and** speciation |

Plus the physical / physiological / behavioural / cognitive traits (`size`, `strength`, `speed`,
tolerances, `intelligence`, etc.) that make up the 32-trait `PHENOTYPE_TRAITS` vector.

### `Habitat` — `evolution_simulator/habitat.py`

A region with a 500-dim environment vector. `simulate_week(mating_strategy=...)` runs, in order:
batch food/water likelihoods → energy/hydration update and starvation/dehydration/old-age deaths
→ aging & pregnancy → litter collection → density-dependent predation → remove dead → collect
migrations → pair males/females for mating → add newborns (assign species) → spontaneous route
isolation. Migrations are **returned, not applied**, so `SimulationRunner` can process all habitats
before moving any creature (no double-simulation). `compute_stats()` reports per-species trait means
and adaptation (`mean_food_prob` / `mean_water_prob`).

**Mating strategies** (set via `mating_strategy` in `[simulation]`):

| Strategy | Behaviour | Complexity |
|---|---|---|
| `zip` | Shuffle all males + females, zip 1:1. Legacy default; penalises minority species with wasted cross-species encounters (Allee effect). | O(N) |
| `species_priority` | Pair within-species first; surplus individuals go to a cross-species spillover pool, preserving some hybridisation. | O(N) |
| `weighted_matrix` | Vectorised M×F compatibility-score matrix; each female samples a male via power-law weights `max(0, score − floor)^sharpness`. Sharpness = `1 + 3·mean(selectivity)` — low-selectivity creatures hybridise liberally; high-selectivity creatures strongly prefer same-species mates. | O(N²) |
| `stable_matching` | Gale-Shapley deferred acceptance on the score matrix — produces a stable matching (no blocking pair exists). Hybridisation only occurs when a cross-species partner is genuinely preferred over available same-species options. See `_gale_shapley()` docstring for the six documented assumptions. | O(N²) |

In `weighted_matrix` and `stable_matching`, cross-species cosine scores in 245-dimensional space average near zero (std ≈ 0.064, far below the 0.70 mating floor), so cross-species weights are effectively zero unless the populations have converged or are hybridising intentionally.

### Habitat Types — `evolution_simulator/habitats/types.py`

14 typed biomes, each with a fixed characteristic centre vector (seeded once via
`__init_subclass__`) plus Gaussian per-instance noise, and resource constants tuned for distinct
selection pressures: `Desert`, `Forest`, `Rainforest`, `Plains`, `Tundra`, `Ocean`, `CoralReef`,
`Wetlands`, `Alpine`, `Volcanic`, `Cave`, `Arctic`, `River`, `Savanna`. See `types.py` for the
per-type gain/cost/migration/predation constants.

### `SpeciesRegistry` — `evolution_simulator/species.py`

```python
from evolution_simulator.species import SpeciesRegistry

registry = SpeciesRegistry(compatibility_threshold=0.65)
name = registry.register_founding_species(founder.genes)   # founder.species = name

# Per birth (called automatically by SimulationRunner):
registry.assign_species(newborn)

# Each step, after migrations (SimulationRunner does this on the respeciate_every cadence):
registry.detect_subcluster_splits(all_alive, week)   # cladogenesis (run before refresh)
registry.refresh_centroids(all_alive)                # move living centroids
registry.detect_anagenesis(all_alive, week)          # in-place transformation
registry.promote_candidates(all_alive, week)         # two-stage gate

registry.speciation_events    # list[dict], each with "event_type"
registry.all_species          # list[str]   (incl. extinct)
registry.living_species_count # species with living members
```

Species names are drawn from an adjective + noun vocabulary in
`evolution_simulator/config/species_names.toml` (100 × 100 = 10,000 combinations).

### `SimulationRunner` — `evolution_simulator/simulation.py`

```python
from pathlib import Path
from evolution_simulator.simulation import SimulationRunner

runner = SimulationRunner(Path("my_config.toml"))
runner.setup()             # build habitats, seed population, write metadata.json
runner.run()               # simulate all weeks, write per-week JSON + summary.json
# or: for _ in range(100): runner.step()
```

Founders start at `age = weeks_to_sexual_viability + 1` so mating begins on week 1. `run()` halts
early on global extinction (`summary.json` records `"extinct": true`). Two independent cadences
control output volume: `stats_every` (per-species/habitat statistics) and `events_every`
(individual births/deaths/matings/migrations/speciations).

---

## Configuration

Copy `simulation_configs/simulation.toml` and edit:

```toml
[simulation]
weeks                       = 5000
seed                        = 42        # remove for a random seed
output_dir                  = "simulation_logs"
initial_species_per_habitat = 3
creatures_per_species       = 10        # 50/50 sex split
initial_genome_noise        = 0.05
founding_habitat_bias       = 0.0       # 0 = random genome, 1 = aligned to habitat
isolation_probability       = 0.001     # per-link weekly route severance
stats_every                 = 10        # weeks between statistics snapshots
events_every                = 0         # 0 = no per-event logging
respeciate_every            = 10        # weeks between living-centroid refresh + detectors
mating_strategy             = "zip"     # zip | species_priority | weighted_matrix | stable_matching

[species]
compatibility_threshold   = 0.65        # compat-subset cosine vs living centroid (≈ below 0.70 mating floor)
min_species_population     = 10         # candidate members required to promote
min_species_weeks          = 60         # weeks a candidate must persist (~2 generations)
split_max_k                = 5          # max sub-clusters considered per species
split_isolation_threshold  = 0.70       # sub-clusters below this are different species (= mating floor)
anagenesis_threshold       = 0.93       # centred phenotype cosine drift that counts as transformation
anagenesis_weeks           = 60         # weeks that drift must persist before naming

[habitats]
connections = [["desert_1", "plains_1"], ["plains_1", "forest_1"]]

[[habitats.instances]]
id   = "desert_1"
type = "Desert"
seed = 1
name = "Northern Desert"
# initial_species_per_habitat / creatures_per_species can be overridden per habitat
```

**On timescales:** a generation is roughly `weeks_to_sexual_viability` (~25–35 weeks), so the
generation-scaled gates above make speciation rare and mostly legible in long runs (thousands of
weeks). Smoke-test for correctness; observe emergence over long runs. Ready-made long configs live
in `simulation_configs/`.

---

## Simulation Logs

```
simulation_logs/2024-01-15_14-30-00/
├── config.toml         ← exact copy of the config used
├── metadata.json       ← habitat topology, parameters, seed, speciation knobs
├── week_00001.json     ← written when stats_every OR events_every is due
├── ...
└── summary.json        ← final state, all_speciation_events (with event_type), extinction flag
```

Speciation events (per-week when logged, and complete in `summary.json`) look like:

```jsonc
{ "new_species": "...", "parent_species": "...", "creature_id": "...",
  "week": 1240, "event_type": "cladogenesis_newborn" }
// event_type values:
//   "cladogenesis_newborn"          – newborn outside all living species; promoted from candidate
//   "cladogenesis_kmeans_subcluster"– k-means split detector found reproductively-isolated sub-clusters
//   "cladogenesis_bootstrap"        – founding species registration
//   "anagenesis"                    – in-place phenotype transformation (chronospecies)
```

Visualize a run with `python visualizer_basic.py` (static charts) or `python visualizer_advanced.py`
(interactive Dash app). The advanced visualizer has four modes: **HABITAT** (cytoscape species map),
**PHYLOGENY** (species tree with distinct colours and line styles per event type), **FAMILY TREE**
(per-creature ancestry wheel), and **TRAIT COMPARE** (multi-species, multi-metric time-series with
optional anagenesis-descendant overlay).

---

## Project Structure

```
evolution_simulator/
├── runner.py                     ← stand-alone entry point
├── visualizer_basic.py           ← static plotly charts
├── visualizer_advanced.py        ← interactive Dash app
├── visualizer/                   ← data loading, figures, panels for the advanced UI
├── experiments/                  ← recorded calibration experiments (e.g. drift_trajectory.py)
├── simulation_configs/           ← ready-made long-run configs
├── evolution_simulator/
│   ├── creature.py               ← gene/trait system, phenotype, reproduction
│   ├── habitat.py                ← resource geometry, predation, migration, stats
│   ├── habitats/types.py         ← 14 typed biomes + HABITAT_TYPE_REGISTRY
│   ├── species.py                ← SpeciesRegistry: membership + cladogenesis + anagenesis
│   ├── simulation.py             ← SimulationRunner, JSON logging
│   └── config/                   ← simulation.toml, species_names.toml
└── tests/                        ← test_creature.py, test_habitat.py, test_species.py, ...
```

## Tests

```bash
poetry run pytest
```

265 tests covering gene/trait computation, Mendelian reproduction, resource geometry, predation,
migration, the full speciation stack (compatibility-centroid membership, living-centroid drift
handling, spherical-k-means sub-cluster splits, phenotype-drift anagenesis with persistence gate),
and all four mating strategies (zip, species_priority, weighted_matrix, stable_matching) including
the Gale-Shapley stability guarantee and minority-species protection.
