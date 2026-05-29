# EvoSim Technical Reference

---

## Table of Contents

1. [How to Run](#1-how-to-run)
2. [Mental Model of the Simulation](#2-mental-model-of-the-simulation)
3. [Architecture Overview](#3-architecture-overview)
4. [Gene and Trait System](#4-gene-and-trait-system)
   - 4.1 [Gene Vector](#41-gene-vector)
   - 4.2 [Ordered Weighted Averaging (OWA)](#42-ordered-weighted-averaging-owa)
   - 4.3 [Trait Definitions and Loci](#43-trait-definitions-and-loci)
   - 4.4 [Pleiotropy](#44-pleiotropy)
   - 4.5 [Inheritance and Mutation](#45-inheritance-and-mutation)
5. [Resource Geometry](#5-resource-geometry)
   - 5.1 [Dot-Product Adaptation](#51-dot-product-adaptation)
   - 5.2 [Energy and Hydration Update Rules](#52-energy-and-hydration-update-rules)
6. [Creature Class](#6-creature-class)
   - 6.1 [Attributes](#61-attributes)
   - 6.2 [Key Methods](#62-key-methods)
   - 6.3 [Compatibility and Mating](#63-compatibility-and-mating)
7. [Mortality](#7-mortality)
8. [Habitat Class](#8-habitat-class)
   - 8.1 [Weekly Simulation Order](#81-weekly-simulation-order)
   - 8.2 [Habitat Type Constants](#82-habitat-type-constants)
9. [Mating Strategies](#9-mating-strategies)
10. [Species and Speciation](#10-species-and-speciation)
    - 10.1 [Species Representation](#101-species-representation)
    - 10.2 [Newborn Assignment](#102-newborn-assignment)
    - 10.3 [Cladogenesis (Population Splits)](#103-cladogenesis-population-splits)
    - 10.4 [Anagenesis (In-Place Transformation)](#104-anagenesis-in-place-transformation)
    - 10.5 [Speciation Event Types](#105-speciation-event-types)
    - 10.6 [SpeciesRegistry API](#106-speciesregistry-api)
11. [Simulation Runner](#11-simulation-runner)
12. [Configuration Reference](#12-configuration-reference)
13. [Simulation Log Format](#13-simulation-log-format)
14. [Key Constants and Invariants](#14-key-constants-and-invariants)

---

## 1. How to Run

### Command-line entry point

```bash
# Run with a config file
python runner.py simulation_configs/simulation.toml

# Override the week count from the command line
python runner.py simulation_configs/simulation.toml --weeks 10000
```

Pass the path to a TOML config. Ready-made configs — the default 4-habitat setup
(`simulation.toml`) plus several long-run scenarios — live in `simulation_configs/`.

`runner.py` drives its own step loop, writes per-week JSON files on the configured cadences, and
calls `_write_summary()` at completion (or early extinction). Logs are written to
`simulation_logs/YYYY-MM-DD_HH-MM-SS/` relative to the working directory.

### Programmatic API

```python
from pathlib import Path
from evolution_simulator.simulation import SimulationRunner

runner = SimulationRunner(Path("simulation_configs/my_experiment.toml"))
runner.setup()   # build habitats, seed population, write metadata.json

# Option A: run to completion
runner.run()

# Option B: step manually — step() returns the week-log dict
for week in range(5000):
    week_log = runner.step()
    if week_log["global_population"] == 0:
        break   # global extinction
```

### Visualizers

```bash
# Static summary charts (population, trait means, adaptation over time)
python visualizer/visualizer_basic.py simulation_logs/<run>/

# Interactive Dash app
python visualizer/visualizer_advanced.py simulation_logs/<run>/
```

The advanced visualizer has four modes: **HABITAT** (cytoscape species map per habitat),
**PHYLOGENY** (species tree with distinct line styles per event type), **FAMILY TREE**
(per-creature ancestry wheel), and **TRAIT COMPARE** (multi-species, multi-metric time-series
with optional anagenesis-descendant overlay).

### Tests

```bash
poetry run pytest                          # full suite
poetry run pytest tests/test_creature.py   # single file
poetry run pytest tests/test_creature.py::test_function_name   # single test
```

---

## 2. Mental Model of the Simulation

The simulation advances in discrete **weeks**. Each week, every creature in every habitat goes
through the same sequence: it searches for food and water, succeeds or fails based on how well
its gene vector aligns with the habitat's environment vector, pays an energy or hydration cost
for each miss, ages by one week, and may die of starvation, dehydration, old age, or
predation. Survivors may mate and produce offspring. Some creatures migrate to adjacent habitats.
Then the process repeats.

The selection pressure on any individual creature has two components. The first is **geometric**:
a creature whose gene vector points in roughly the same direction as the habitat vector finds
food and water with high probability, while one pointing the opposite direction is near-certain
to starve or dehydrate. This drives local adaptation — over many generations, populations tend
to drift toward alignment with their habitat's environment vector. The second component is
**trait-level**: once a creature finds food, whether it has the energy budget to survive a bad
week depends on its `metabolism`; whether it produces many offspring depends on `fecundity` and
`reproduction_time`; whether it survives crowding depends on `base_predation_rate`. These traits
are encoded in overlapping sets of the same gene vector, so selection on one axis (geometric
alignment) creates indirect pressure on the traits sharing those loci, and vice versa. There is
no single "fitness" number — a creature is simultaneously optimizing both axes under whatever
mortality pressures its habitat imposes.

Speciation emerges from reproductive isolation. Mating is gated by a cosine similarity test on a
245-locus compatibility subset of the gene vector. As populations adapt to different habitats,
or as subpopulations within a connected habitat drift apart, their compatibility vectors diverge.
When two subgroups can no longer interbreed — either because they've been physically separated
long enough to drift apart, or because k-means clustering detects that a single population has
become bimodal in compatibility space — a new species candidate is created. That candidate must
sustain a minimum population for a minimum number of weeks before it is promoted to a confirmed
species. Dead lineages remain in the log as the fossil record; their entries are never deleted.

---

## 3. Architecture Overview

The simulation is built around four classes that interact in a fixed data-flow order each week:

```
SimulationRunner
    └── step()
          ├── Habitat.simulate_week()  ← called for each habitat in sequence
          │       returns migration_events (not applied internally)
          ├── apply all migrations     ← moves creatures between habitats
          ├── SpeciesRegistry.detect_subcluster_splits()
          ├── SpeciesRegistry.refresh_centroids()
          ├── SpeciesRegistry.detect_anagenesis()
          └── SpeciesRegistry.promote_candidates()   ← every week
```

**Critical invariant:** migrations are returned as events from `Habitat.simulate_week()`, never
applied inside it. `SimulationRunner.step()` processes all habitats first, then moves migrants.
This prevents a creature from being simulated twice in the same week.

---

## 4. Gene and Trait System

### 4.1 Gene Vector

Every `Creature` carries a 500-dimensional float gene vector (`np.ndarray`, shape `(500,)`).
All phenotypic properties are derived from this vector; the vector itself is the only persistent
genetic state. Values are real numbers. A founding species' genome is drawn from a standard
normal distribution (mean 0, standard deviation 1); the individuals seeded around it differ only
by a small per-locus Gaussian perturbation (standard deviation `initial_genome_noise`, default
0.12). From there the vector is reshaped by selection across generations.

### 4.2 Ordered Weighted Averaging (OWA)

Traits are **polygenic**: each is computed from a named set of loci using Ordered Weighted
Averaging rather than a plain mean.

**Algorithm:**

```
1. vals   = genes[trait_loci]           # extract values at this trait's loci
2. sorted = sort(vals, descending)      # rank from highest to lowest
3. w_i    = α · (1 − α)^i              # exponential decay weights (i = 0, 1, 2, …)
4. w      = w / sum(w)                  # normalize to sum = 1
5. raw    = dot(w, sorted)              # weighted sum
6. trait  = sigmoid(raw)               # squash to [0, 1]
```

`α = OWA_ALPHA = 0.6` (class attribute on `Creature`; subclasses may override).

**Weight distribution at α = 0.6:**

| Rank | Weight |
|------|--------|
| 1st (highest locus) | 0.60 |
| 2nd | 0.24 |
| 3rd | 0.096 |
| 4th | 0.038 |
| … | … |

The consequence: a single beneficial mutation that pushes one locus to the top of the ranking
immediately captures ~60% of the trait's phenotypic weight, rather than being diluted by 1/N.
Individual mutations are selectable from their first generation.

Most traits are then **scaled to a biological range** via linear mapping from `[0, 1]`:

```python
trait_value = trait_min + sigmoid_value * (trait_max - trait_min)
```

### 4.3 Trait Definitions and Loci

**Reproduction traits:**

| Trait | Locus count | Biological range | Effect |
|-------|-------------|-----------------|--------|
| `fecundity` | 18 | 1–8 | Poisson mean litter size |
| `reproduction_time` | 13 | 1–20 wk | Gestation period |
| `weeks_to_sexual_viability` | 13 | 4–50 wk | Age before first reproduction |
| `parental_investment` | 12 | [0, 1] | Post-birth energy investment |
| `reproduction_likelihood` | 16 | [0, 1] | Per-encounter conception probability (averaged between the two parents) |

**Behavioral traits:**

| Trait | Notes |
|-------|-------|
| `aggression` | Shares loci with `size` and `strength` (bigger → more aggressive) |
| `migration_likelihood` | Shares loci with `speed` and `risk_tolerance` |
| `territorial` | — |
| `social_tendency` | — |
| `pack_hunting` | — |
| `scavenging_tendency` | — |
| `nocturnal_tendency` | — |
| `risk_tolerance` | — |

**Physical traits:**

| Trait | Notes |
|-------|-------|
| `size` | — |
| `strength` | — |
| `speed` | — |
| `camouflage` | — |

**Physiological traits:**

| Trait | Biological range | Effect in simulation |
|-------|-----------------|---------------------|
| `metabolism` | 0.5–2.0 | Scales weekly energy loss when food is missed |
| `water_efficiency` | [0, 1] | Reduces the `WATER_EFFICIENCY_COST` term |
| `max_lifespan` | 40–400 wk | Hard age cap |
| `base_predation_rate` | 0–0.005 | Intrinsic weekly predation vulnerability |
| `mutation_rate` | 0.001–0.05 | Per-locus mutation probability (heritable) |

**Tolerance and cognitive traits:**

`disease_resistance`, `immune_response`, `stress_tolerance`, `heat_tolerance`,
`cold_tolerance`, `drought_tolerance`, `hibernation_tendency`, `intelligence`,
`adaptability`, `communication` — these traits exist as phenotypic outputs but do not
currently drive direct simulation mechanics. They are part of the 32-trait `PHENOTYPE_TRAITS`
vector used by the anagenesis axis.

**Special gene subsets:**

| Name | Size | Role |
|------|------|------|
| `compatibility_genes` | 245 loci | Cosine basis for mating compatibility **and** speciation detection |
| `sex_determination` | 1 locus | `sigmoid(locus) ≥ 0.5` → female |

### 4.4 Pleiotropy

Many loci appear in more than one trait's locus list. Selection on one trait therefore creates
correlated pressure on others. For example, `aggression` shares loci with `size` and `strength`;
`migration_likelihood` shares loci with `speed` and `risk_tolerance`; `fecundity` shares loci
with `base_predation_rate`, encoding an r/K trade-off (high-fecundity genotypes are also more
conspicuous to predators).

### 4.5 Inheritance and Mutation

Offspring receive genes via **Mendelian inheritance**: for each locus, one parent is chosen
uniformly at random and the offspring inherits that parent's value at that position.

Mutation is then applied independently per locus, using the `mutation_rate` of whichever parent
contributed that locus. When a locus mutates, its value is **replaced outright** by a fresh draw
from the standard normal distribution — it is not nudged by a small delta:

```python
if rng.random() < contributing_parent.mutation_rate:
    locus_value = rng.standard_normal()   # full resample, N(0, 1)
```

Because `mutation_rate` is itself an OWA-computed, heritable trait, the mutation rate evolves
under selection. Lineages in stable environments tend toward lower rates; lineages under strong
directional selection may drift toward higher rates.

---

## 5. Resource Geometry

### 5.1 Dot-Product Adaptation

Each habitat has a fixed 500-dimensional environment vector. A creature's weekly probability of
finding food or water is:

```
P(resource) = (cos θ + 1) / 2
```

where θ is the angle between the creature's genes and the habitat vector, taken over the loci
relevant to that resource (see below).

| Alignment | cos θ | P(resource) |
|-----------|-------|-------------|
| Perfectly aligned | +1 | 1.0 |
| Orthogonal (random genome) | 0 | 0.5 |
| Anti-aligned | −1 | 0.0 |

**Food and water use different gene subspaces.** The cosine is *not* taken over the full
500-dim vector. Food-finding compares a 158-locus subset (`FOOD_GENE_INDICES`) of the creature's
genes against the same loci of the habitat vector; water-finding uses a separate 175-locus subset
(`WATER_GENE_INDICES`). The two subsets only partly overlap, so a creature can be well-adapted to
find food in a habitat while still struggling to find water there — food adaptation and water
adaptation are partly independent axes.

Resource discovery for each creature each week is an independent Bernoulli draw at this
probability. There is no inter-species competition for food or water — each creature's draw is
independent. The only population-dependent mortality is density-dependent predation.

**Why `(cos θ + 1) / 2` instead of raw cosine?**  In high-dimensional spaces (500-dim), random
vectors are nearly orthogonal with high probability. Raw cosine would concentrate all unselected
genomes near P=0.5 but with very little variance, making selection extremely slow. Shifting and
scaling to `[0, 1]` preserves the full range of selection pressure.

### 5.2 Energy and Hydration Update Rules

Each week, per creature:

```
if food_found:
    energy += FOOD_ENERGY_GAIN
else:
    energy -= FOOD_ENERGY_COST × creature.metabolism

if water_found:
    hydration += WATER_HYDRATION_GAIN
else:
    hydration -= WATER_BASE_COST + WATER_EFFICIENCY_COST × (1 − creature.water_efficiency)

energy    = clamp(energy, 0, 1)
hydration = clamp(hydration, 0, 1)
```

Death thresholds: `energy ≤ 0` → starvation; `hydration ≤ 0` → dehydration.

Fully aquatic habitats (Ocean, CoralReef) set both `WATER_BASE_COST` and `WATER_EFFICIENCY_COST`
to 0, removing water stress entirely. The semi-aquatic Wetlands and River keep small nonzero water
costs (Wetlands 0.04 / 0.06, River 0.07 / 0.10) — water is abundant but not free. See the table in
§8.2 for the full per-biome values.

---

## 6. Creature Class

**Module:** `evolution_simulator/creature.py`

### 6.1 Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `genes` | `np.ndarray (500,)` | Full gene vector |
| `sex` | `str` | `"male"` or `"female"` |
| `species` | `str` | Assigned by `SpeciesRegistry` |
| `age` | `int` | Weeks lived |
| `energy` | `float` | Current energy level `[0, 1]` |
| `hydration` | `float` | Current hydration level `[0, 1]` |
| `is_alive` | `bool` | — |
| `cause_of_death` | `str \| None` | `"starvation"`, `"dehydration"`, `"old_age"`, `"predation"` |
| `is_pregnant` | `bool` | — |
| `weeks_pregnant` | `int` | Weeks since conception |

### 6.2 Key Methods

| Method / property | Description |
|--------|-------------|
| `metabolism`, `fecundity`, … (properties) | Each trait is exposed as a property. On first access it computes its OWA value via the private `_compute_trait(name)`, caches it (genes are immutable after birth), and scales the raw `[0, 1]` value into the trait's biological range. There is no public `get_trait()`. |
| `compatibility_score(other)` | Cosine similarity of the two creatures' 245-locus `compatibility_genes` subsets, in `[-1, 1]`. |
| `is_compatible(other)` | Sex, viability, pregnancy, and cosine-threshold check. Returns `(compatible, score, reason)`. |
| `reproduce(other)` | Fertility check, Poisson litter, Mendelian inheritance + mutation |
| `simulate_week(environment=None)` | Checks starvation/dehydration from the already-updated energy/hydration, increments age, checks old age, advances the pregnancy timer. Energy and hydration deltas are applied by the `Habitat` **before** this is called, not here. |

### 6.3 Compatibility and Mating

**Compatibility check** (`is_compatible(other)`):

1. Must be opposite sex, both sexually viable (`age ≥ weeks_to_sexual_viability`), and female
   must not already be pregnant.
2. Cosine similarity is computed on the 245-locus `compatibility_genes` subset.
3. Threshold: `COMPATIBILITY_FLOOR + 0.15 × mean(self.selectivity, other.selectivity)`

`COMPATIBILITY_FLOOR = 0.70`. The threshold uses the **pair-averaged** selectivity: if both
partners have `selectivity = 1.0` the effective floor is 0.85; at `selectivity = 0.0` it stays at
0.70.

**Reproduction** (`reproduce(other)`):

- Conception probability: the **average** of the two parents' `reproduction_likelihood` traits.
  A uniform draw above that value means no conception this encounter.
- Litter size: `Poisson(mean = mother.fecundity)`, clamped to a minimum of 1.
- Each offspring locus: 50/50 draw from either parent, then possible mutation at the contributing
  parent's `mutation_rate`.

---

## 7. Mortality

Each creature faces up to four death causes per week, evaluated in order:

| Cause | Condition | Notes |
|-------|-----------|-------|
| Starvation | `energy ≤ 0` | — |
| Dehydration | `hydration ≤ 0` | — |
| Old age | `age >= max_lifespan` | Checked after the weekly age increment |
| Predation | Bernoulli draw | Rate = `base_predation_rate + PREDATION_ALPHA × N / POPULATION_SUPPORT` |

**Density-dependent predation:** `N` is the current habitat population. At carrying capacity
(`N = POPULATION_SUPPORT`), predation adds `PREDATION_ALPHA` to the creature's intrinsic
`base_predation_rate`. Above carrying capacity, predation scales linearly. The `base_predation_rate`
trait shares loci with `fecundity`, creating an r/K trade-off: fast-reproducing genotypes carry
higher intrinsic predation risk.

---

## 8. Habitat Class

**Module:** `evolution_simulator/habitat.py`

### 8.1 Weekly Simulation Order

`Habitat.simulate_week(mating_strategy)` executes the following steps in fixed order:

1. Compute food and water discovery probabilities for all creatures (vectorized dot-product).
2. Draw Bernoulli outcomes; update energy and hydration; mark starvation/dehydration deaths.
3. Advance creature age and pregnancy timers; mark old-age deaths (checked after the age increment).
4. Collect litters from females that reached gestation term (`weeks_pregnant ≥ reproduction_time`).
5. Apply density-dependent predation.
6. Remove dead creatures from the habitat population.
7. Collect migration candidates (but **do not apply** migrations).
8. Pair males and females for mating via the configured strategy; call `_attempt_mating()` for
   each pair; add newborns and assign species via `SpeciesRegistry`.
9. Attempt spontaneous route isolation (`isolation_probability` per link per week).

Steps 7 and 9 return data to `SimulationRunner`; they do not modify habitat state themselves.

### 8.2 Habitat Type Constants

**Defaults (Forest / `Habitat` base class):**

| Constant | Default | Description |
|----------|---------|-------------|
| `FOOD_ENERGY_GAIN` | 0.47 | Energy gained when food is found |
| `FOOD_ENERGY_COST` | 0.22 | Energy lost when food is missed (× `metabolism`) |
| `WATER_HYDRATION_GAIN` | 0.26 | Hydration gained when water is found |
| `WATER_BASE_COST` | 0.20 | Irreducible per-week hydration loss |
| `WATER_EFFICIENCY_COST` | 0.30 | Hydration loss multiplied by `(1 − water_efficiency)` |
| `WEEKLY_MIGRATION_BASE` | 0.01 | Base per-creature migration probability |
| `PREDATION_ALPHA` | 0.010 | Density-dependent predation coefficient |
| `POPULATION_SUPPORT` | 400 | Carrying capacity for density-predation scaling |

**Per-biome overrides:**

| Biome | TYPE_SEED | FOOD_GAIN | FOOD_COST | WATER_GAIN | WATER_BASE | WATER_EFF | MIGRATION | PRED_ALPHA | POP_SUPPORT |
|-------|-----------|-----------|-----------|------------|------------|-----------|-----------|------------|-------------|
| Desert | 1001 | 0.25 | 0.30 | default | 0.30 | 0.45 | default | 0.003 | 120 |
| Forest | 1002 | default | default | default | default | default | default | 0.010 | 400 |
| Rainforest | 1003 | 0.40 | default | 0.45 | default | default | default | 0.015 | 600 |
| Plains | 1004 | default | default | default | default | default | 0.015 | 0.010 | 500 |
| Tundra | 1005 | 0.20 | 0.30 | 0.25 | default | default | default | 0.004 | 150 |
| Ocean | 1006 | 0.35 | default | 0.50 | 0.0 | 0.0 | 0.020 | 0.008 | 300 |
| CoralReef | 1007 | 0.45 | default | 0.50 | 0.0 | 0.0 | default | 0.020 | 400 |
| Wetlands | 1008 | default | default | 0.45 | 0.04 | 0.06 | default | 0.012 | 500 |
| Alpine | 1009 | 0.18 | 0.35 | default | default | default | 0.005 | 0.002 | 80 |
| Volcanic | 1010 | 0.22 | 0.35 | 0.20 | 0.27 | 0.40 | 0.005 | 0.001 | 50 |
| Cave | 1011 | 0.15 | default | 0.30 | default | default | 0.003 | 0.002 | 80 |
| Arctic | 1012 | 0.15 | 0.35 | 0.35 | default | default | default | 0.001 | 60 |
| River | 1013 | default | default | 0.45 | 0.07 | 0.10 | 0.015 | 0.010 | 400 |
| Savanna | 1014 | 0.28 | default | default | 0.24 | 0.33 | 0.012 | 0.008 | 350 |

`default` = inherits the base-class value shown above. `0.0` = feature disabled.

**Habitat center vectors** are generated deterministically at class definition time from each
biome's `TYPE_SEED` via `np.random.default_rng(TYPE_SEED).standard_normal(500)`. Every instance
of the same biome type shares this center vector and adds Gaussian per-instance noise
(`σ = 0.15`), modelling intra-type geographic variation.

---

## 9. Mating Strategies

The mating strategy is selected via `mating_strategy` in `[simulation]` config and dispatched
inside `Habitat.simulate_week()`. All strategies call `_attempt_mating(male, female)`, which
runs `is_compatible()` and, on success, `reproduce()`.

### `zip` (legacy default)

Shuffles all viable males and all viable females habitat-wide, then zips 1:1. A cross-species
pair that fails `is_compatible()` wastes both individuals' mating opportunity for that week. This
creates a **minority-species Allee effect**: a minority of 5 pairs competing against a majority
of 50 gets only ~9% of pairings within-species. Retained for backwards compatibility. O(N).

### `species_priority`

Groups males and females by `creature.species`. Pairs each species' own pool first (shuffled
within-species), then routes surplus unpaired individuals to a shared cross-species spillover
pass. Each species gets mating proportional to its own sex ratio, regardless of relative
abundance. Hybridisation still occurs for leftover surplus individuals. O(N).

### `weighted_matrix`

Builds the full M×F compatibility score matrix in one vectorized numpy pass
(`(M, 245) @ (245, F)` + norm division). Iterates females in random order; each female samples
a male with probability proportional to:

```
weight_i = max(0, score_i − threshold) ^ sharpness

threshold  = COMPATIBILITY_FLOOR + 0.15 × female.selectivity
sharpness  = 1 + MATING_SHARPNESS_K × mean(male.selectivity, female.selectivity)
           = 1 + 3 × mean(selectivities)     [sharpness ∈ 1..4]
```

Low `selectivity` → nearly uniform sampling above the floor (liberal hybridisation). High
`selectivity` → sharply peaked at the best available mate. Each claimed male is removed from
the pool. Hybridisation propensity is an evolved trait driven by `selectivity`. O(N²).

Cross-species cosine scores in 245-dim space average near zero (std ≈ 1/√245 ≈ 0.064), well
below the 0.70 floor, so cross-species sampling weights are effectively zero unless populations
have genuinely converged.

### `stable_matching`

Runs **Gale-Shapley deferred acceptance** (male-proposing) on the same M×F score matrix. Each
male proposes down his ranked preference list; each female tentatively holds her best offer and
releases prior partners if a better proposer arrives. Terminates when no free male has remaining
candidates. Produces a **stable matching** — no unmatched (male, female) pair exists where both
prefer each other over their current partner. Hybridisation occurs only when a cross-species
individual genuinely outranks all same-species alternatives for both parties. O(N²).

**Documented assumptions in `_gale_shapley()`** (see its docstring):
1. **Male-proposing direction** (females hold tentative offers). This gives every male his best
   stable partner and every female her worst; the direction is biologically arbitrary.
2. **Bilateral threshold pre-filtering:** a female enters a male's preference list only if the
   score clears *both* their individual thresholds (each `FLOOR + 0.15 × own_selectivity`).
3. **Tie-breaking via small Gaussian noise** (`σ = 0.005`) added to scores before ranking. The
   noise is drawn once per call so rankings stay consistent; threshold checks still use the
   noise-free scores.
4. A female compares a new proposer against her current partner using those same noisy scores.
5. **Threshold asymmetry vs `is_compatible()`:** matching uses each creature's *own* selectivity,
   but `is_compatible()` (re-run by `_attempt_mating`) uses the *pair-averaged* selectivity — so a
   pair that clears the matching can still fail the final check, producing a `compatible=False`
   event rather than a birth.
6. Queue order (FIFO) influences which stable matching is found when several exist; combined with
   the noise this adds stochasticity without biasing toward any fixed outcome.

---

## 10. Species and Speciation

Speciation is built on one principle: **a species boundary means reproductive isolation**,
measured on the same 245-locus `compatibility_genes` subset that gates mating.

### 10.1 Species Representation

Each confirmed species keeps two references:

| Reference | Type | Purpose |
|-----------|------|---------|
| `progenitor_genes` | 500-dim frozen genome | Anchors the species name; provides the anagenesis axis |
| `centroid` | 245-dim living vector | Current mean of compatibility genes across living members; used for detection |

The living centroid is refreshed every `respeciate_every` weeks (`refresh_centroids`). When an
entire interbreeding population drifts together, the centroid moves with it — ordinary genetic
drift never triggers a speciation event. This is the fix for the historical runaway-speciation
bug.

### 10.2 Newborn Assignment

Every newborn's 245-locus compatibility vector is compared by cosine similarity to all living
species centroids. The newborn is assigned to the **nearest** species unconditionally (no
minimum threshold for assignment). An outlier newborn assigned to a species will not mate back
into it if its actual compatibility score falls below `COMPATIBILITY_FLOOR = 0.70`, but it is
counted as a member of that species and does not create a speciation candidate. Only a
population-level k-means split can initiate a candidate.

### 10.3 Cladogenesis (Population Splits)

**Detector:** `detect_subcluster_splits()` — runs before `refresh_centroids` each cycle.

A single living centroid can mask a population that has split into two reproductively-isolated
modes: the two clusters sit symmetrically around their shared centroid, so no individual newborn
falls below threshold, and the species appears healthy.

**Algorithm:**

1. For each species with ≥ `2 × min_species_population` non-candidate members:
2. Cluster the members in compatibility space via spherical k-means, sweeping K = 2..`split_max_k`.
3. Select the **largest K** whose cluster centroids are **all mutually below**
   `split_isolation_threshold` (0.75 in the bundled configs — above the 0.70 mating floor, so
   detected splits represent genuinely non-interbreeding populations. The in-code class default is
   0.70, equal to the floor; the configs deliberately raise it).
4. K = 1 (no split found) → nothing happens.
5. K ≥ 2: the cluster nearest the frozen type keeps the species name. Each other cluster seeds
   or extends a **candidate**.

Candidates must pass a two-stage gate before promotion: ≥ `min_species_population` living members
and ≥ `min_species_weeks` weeks of persistence.

**Event type:** `"cladogenesis_kmeans_subcluster"`

### 10.4 Anagenesis (In-Place Transformation)

**Detector:** `detect_anagenesis()` — runs after `refresh_centroids` each cycle, and also polls
weekly via `detect_anagenesis(pending_only=True)` when `_anagenesis_pending` is non-empty.

A lineage can remain one interbreeding population while drifting far from its ancestral form over
many generations (a chronospecies — the biological analogue of dogs from wolves). The compatibility
axis cannot detect this because the population still interbreeds freely.

Anagenesis measures a separate **phenotype axis**: the 32-trait `PHENOTYPE_TRAITS` raw-OWA vector
(`compute_phenotype_matrix()`), compared to the frozen-type phenotype by **centered cosine**:

```
centered cosine = cosine(phenotype − 0.5, type_phenotype − 0.5)
```

Centering on 0.5 is necessary because phenotype values live in `[0, 1]`. Raw cosine in `[0, 1]`
space is compressed toward 1; centering measures the deviation pattern over the full `[-1, 1]`
range and provides meaningful directional sensitivity.

**Trigger conditions:**

1. Centered cosine < `anagenesis_threshold` (default 0.93).
2. Persists for ≥ `anagenesis_weeks` weeks without rebounding (a transient dip does not trigger).

**On trigger:** living members are re-partitioned between the old type and the new centroid based
on which phenotype they more closely resemble. The new species' `_type_phenotype` is anchored to
the **mover population centroid** (not a single individual) to prevent the re-fire bug where a
misanchored phenotype immediately restarts the persistence clock.

Anagenesis does not run for species that have an active split candidate in the current cycle.

**Why 0.93 threshold?** Phenotype drift is selection-bounded: it plateaus around 0.94 because
adaptation chases a fixed habitat optimum. A threshold of 0.93 catches meaningful transformation
without false positives from normal adaptation noise.

**Event type:** `"anagenesis"`

### 10.5 Speciation Event Types

| `event_type` | Meaning |
|---|---|
| `"cladogenesis_bootstrap"` | Founding species registration at simulation start |
| `"cladogenesis_kmeans_subcluster"` | K-means split detector found reproductively-isolated sub-clusters |
| `"anagenesis"` | In-place phenotype transformation of an intact lineage |

Note: `"cladogenesis_newborn"` may appear in log files from runs before 2026-05-27. It is no
longer emitted by current code.

### 10.6 SpeciesRegistry API

```python
from evolution_simulator.species import SpeciesRegistry

registry = SpeciesRegistry(compatibility_threshold=0.65)
name = registry.register_founding_species(founder.genes)   # returns species name

# Per birth (called automatically by SimulationRunner):
registry.assign_species(newborn)

# Each step, after migrations — on the respeciate_every cadence:
registry.detect_subcluster_splits(all_alive, week)   # cladogenesis (run before refresh)
registry.refresh_centroids(all_alive)                # move living centroids
registry.detect_anagenesis(all_alive, week)          # in-place transformation
registry.promote_candidates(all_alive, week)         # two-stage gate (runs every week)

# Data access:
registry.speciation_events      # list[dict], each with "event_type"
registry.all_species            # list[str], including extinct
registry.living_species_count   # int, species with at least one living member
```

Species names are drawn from an adjective + noun vocabulary in
`evolution_simulator/config/species_names.toml` (100 adjectives × 100 nouns = 10,000 combinations).

---

## 11. Simulation Runner

**Module:** `evolution_simulator/simulation.py`

```python
from pathlib import Path
from evolution_simulator.simulation import SimulationRunner

runner = SimulationRunner(Path("my_config.toml"))
runner.setup()        # build habitats, seed population, write metadata.json
runner.run()          # simulate all weeks, write per-week JSON + summary.json
# or:
for _ in range(100):
    week_log = runner.step()   # single week; returns the week-log dict
    if week_log["global_population"] == 0:
        break                  # global extinction
```

Founding creatures start at `age = weeks_to_sexual_viability + 1` so mating can begin on week 1.

`run()` checks global population after each step and halts early on extinction. `runner.py` runs
its own step loop and does not call `run()`; it calls `_write_summary()` directly.

**Output cadences:**

| Setting | Controls |
|---------|---------|
| `stats_every` | Per-species/habitat trait statistics snapshots |
| `events_every` | Individual birth/death/mating/migration/speciation event records (0 = disabled) |

Both cadences produce entries in the weekly JSON files. The speciation event list is always
complete in `summary.json` regardless of cadence.

---

## 12. Configuration Reference

Copy `simulation_configs/simulation.toml` and edit as needed.

```toml
[simulation]
weeks                       = 200
seed                        = 42          # remove for a random seed each run
output_dir                  = "simulation_logs"
initial_species_per_habitat = 3
creatures_per_species       = 10          # 50/50 sex split enforced at founding
initial_genome_noise        = 0.12        # std of per-locus Gaussian noise around the founding genome
founding_habitat_bias       = 0.55        # 0 = random genome, 1 = genome aligned to habitat
isolation_probability       = 0.008       # per-link weekly probability of route severance
stats_every                 = 10          # weeks between statistics snapshots
events_every                = 0           # 0 = no per-event logging
respeciate_every            = 10          # cadence for centroid refresh + periodic detectors
mating_strategy             = "zip"       # zip | species_priority | weighted_matrix | stable_matching

[species]
compatibility_threshold   = 0.65          # newborn-vs-centroid cosine to join a species
min_species_population    = 10            # living members a candidate needs before promotion
min_species_weeks         = 60            # weeks a candidate must persist (~2 generations)
split_max_k               = 5             # max sub-clusters the split detector sweeps to
split_isolation_threshold = 0.75          # sub-cluster centroids must be mutually below this
anagenesis_threshold      = 0.93          # centred phenotype cosine below which anagenesis triggers
anagenesis_weeks          = 60            # persistence weeks required before the lineage is renamed

[habitats]
connections = [["desert_1", "plains_1"], ["plains_1", "forest_1"]]

[[habitats.instances]]
id   = "desert_1"
type = "Desert"
seed = 1                    # per-instance noise seed
name = "Northern Desert"    # optional display name
# initial_species_per_habitat and creatures_per_species can be overridden per habitat
```

**Speciation knob guide:**

| Knob | Faster speciation | Slower speciation |
|------|------------------|------------------|
| `min_species_population` | Lower | Higher |
| `min_species_weeks` | Lower | Higher |
| `split_isolation_threshold` | Lower (looser split criterion) | Higher |
| `anagenesis_threshold` | Higher (triggers sooner) | Lower |
| `anagenesis_weeks` | Lower | Higher |

A generation is approximately `weeks_to_sexual_viability` weeks (~25–35 at default parameters).
The bundled configs set `min_species_weeks = 60` (~2 generations) to make speciation rare and
legible in long runs.

---

## 13. Simulation Log Format

```
simulation_logs/2024-01-15_14-30-00/
├── config.toml         ← exact copy of the config used
├── metadata.json       ← habitat topology, parameters, seed, speciation knobs
├── week_00001.json     ← written when stats_every OR events_every is due
├── week_00010.json
├── …
└── summary.json        ← final state, all speciation events, extinction flag
```

A `week_NNNNN.json` is written only when the `stats_every` **or** `events_every` cadence fires
that week (plus week 1 and the final week, which always get at least a statistics snapshot). Each
block below appears only when its own cadence fires; there are no top-level `stats` or `events`
keys.

**Per-week JSON structure:**

```jsonc
{
  // --- always present ---
  "week": 1240,
  "timestamp": "2024-01-15T14:30:00.123456",
  "global_population": 1875,
  "global_species_count": 12,            // total species ever registered, incl. extinct

  // --- present only when the stats cadence fires ---
  "global_species_distribution": { "Common Wanderer": 420, "Agile Treader": 95 },
  "habitat_stats": {
    "forest_1": {
      "habitat_id": "forest_1",
      "habitat_name": "Eastern Forest",
      "habitat_type": "Forest",
      "by_species": {
        "Common Wanderer": {
          "count": 210,                  // note: "count", not "population"
          "mean_food_prob": 0.73,
          "mean_water_prob": 0.68,
          "mean_generation": 41.2,
          "mean_traits": { "metabolism": 1.12, "fecundity": 3.4, … }
        }
      }
    }
  },
  "species_stats": {                      // same species aggregated across all habitats
    "Common Wanderer": {
      "total_count": 420,
      "habitat_counts": { "forest_1": 210, "plains_1": 210 },
      "mean_generation": 41.0,
      "mean_traits": { … }
    }
  },

  // --- present only when the events cadence fires ---
  "speciation_events": [ … ],
  "migrations": [
    { "creature_id": "…", "species": "…", "from_habitat": "plains_1", "to_habitat": "forest_1" }
  ],

  // --- always present; per-habitat detail (births/deaths/mating/etc. added only on events weeks) ---
  "habitats": {
    "forest_1": {
      "habitat_id": "forest_1",
      "habitat_type": "Forest",
      "habitat_name": "Eastern Forest",
      "population": 420,
      "species_distribution": { "Common Wanderer": 210, … },
      // when the events cadence fires, also: "births", "deaths", "mating_events",
      // "migrations_out", "isolations"
    }
  }
}
```

**Speciation event record:**

```jsonc
{
  "new_species": "Agile Treader",
  "parent_species": "Common Wanderer",
  "creature_id": "3f2a…",                // representative individual of the new lineage
  "week": 1240,
  "event_type": "cladogenesis_kmeans_subcluster"
}
```

**summary.json** is written once at the end. Key fields: `weeks_simulated`, `extinct` (bool),
`final_population`, `total_species_ever`, `total_speciation_events`, `total_hybridization_events`,
`all_speciation_events` (the complete list, regardless of `events_every`),
`all_failed_speciation_attempts`, `final_species_distribution` (per habitat), and
`final_population_per_habitat`.

---

## 14. Key Constants and Invariants

| Constant | Value | Location |
|----------|-------|----------|
| `HABITAT_VECTOR_DIMS` | 500 | `habitat.py` |
| `OWA_ALPHA` | 0.6 | `Creature` class attribute |
| `COMPATIBILITY_FLOOR` | 0.70 | `Creature` class attribute |
| `compatibility_genes` loci | 245 | `DEFAULT_TRAIT_GENE_INDICES` in `creature.py` |
| `PHENOTYPE_TRAITS` | 32 traits | `creature.py` module constant |
| `MATING_SHARPNESS_K` | 3 | `habitat.py` (used by `weighted_matrix`) |
| `_INSTANCE_NOISE` | 0.15 (σ) | `habitats/types.py` |

**Invariants that must not be violated:**

- The `(cos θ + 1) / 2` resource geometry is central to local adaptation. Do not replace it with
  explicit fitness scores or cross-product geometry.
- OWA with `α = 0.6` is a deliberate design choice; change it only on subclasses, not the base
  `Creature` class.
- Migrations must not be applied within `Habitat.simulate_week()`.
- Speciation detection uses the 245-dim compatibility subset against **living centroids**. Do not
  revert to full-genome distance or frozen progenitor comparison.
- The two periodic detectors must run in fixed order: `detect_subcluster_splits` →
  `refresh_centroids` → `detect_anagenesis`. The split detector must precede the refresh so that
  split-off members are excluded from the parent centroid.
- The anagenesis axis uses the **phenotype** vector (raw OWA, centered on 0.5), not the
  compatibility axis. These are deliberately different signals.
- Founding creatures must start sexually viable (`age = weeks_to_sexual_viability + 1`).
