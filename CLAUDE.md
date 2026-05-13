# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install

# Run simulation with default config (365 days, 4 habitats)
python runner.py

# Run with custom config or day override
python runner.py path/to/config.toml --days 500

# Run all tests
poetry run pytest

# Run a single test file
poetry run pytest tests/test_creature.py

# Run a single test by name
poetry run pytest tests/test_creature.py::test_function_name
```

## Architecture

The simulation is built around four core classes that interact in a strict data-flow order each day.

### Gene/Trait System (`creature.py`)

Every `Creature` carries a **500-dimensional float gene vector**. All phenotypic traits are polygenic, computed via **Ordered Weighted Averaging (OWA)**:

1. Extract gene values at the trait's loci.
2. Sort descending; assign weights `w_i = α·(1−α)^i` (normalised), where `α = OWA_ALPHA` (default 0.6).
3. `raw = dot(weights, sorted_genes)` → `sigmoid(raw)` → `[0, 1]`.

The top-ranked locus carries ~60% of the weight, the next ~24%, and so on. A beneficial mutation that rises to the top of the ranking gains immediate phenotypic influence rather than being diluted 1/N by a plain mean. `OWA_ALPHA` is a class attribute that species subclasses can override.

Many loci feed multiple traits (pleiotropy), which means selection on one trait creates correlated pressure on others. The `mutation_rate` trait is itself encoded in genes and heritable, so mutation rates evolve.

Sex is gene-determined: `sigmoid(sex_determination_genes) >= 0.5` → female.

### Resource Geometry (`habitat.py`, `habitats/types.py`)

Each `Habitat` has its own 500-dim environment vector. Food/water discovery uses **cross-product geometry**: `P(resource) = sin(θ)` where θ is the angle between a creature's genes and the habitat vector. A creature parallel to the habitat finds nothing; a creature orthogonal to it exploits resources maximally. This drives niche differentiation without any explicit fitness function.

Each habitat type (`habitats/types.py`) has a fixed characteristic center vector seeded from a `TYPE_SEED` constant via `__init_subclass__`, with Gaussian per-instance noise layered on top.

### Daily Simulation Order (`habitat.py:simulate_day`)

Within a single day, `Habitat.simulate_day()` runs in this fixed order:
1. Compute resource probabilities for all creatures (vectorized)
2. Update energy/hydration; mark deaths
3. Advance creature age and pregnancy timers
4. Collect litters from females that reached gestation term
5. Remove dead creatures
6. Collect migration candidates (but do **not** apply migrations yet)
7. Pair males and females for mating
8. Add newborns and assign species via `SpeciesRegistry`
9. Attempt spontaneous route isolation

**Critical invariant**: migrations are returned as events, not applied inside `Habitat.simulate_day()`. `SimulationRunner.step()` processes all habitats first, then moves migrants — this prevents a creature from being simulated twice on the same day.

### Species Detection (`species.py`)

`SpeciesRegistry` maintains a list of progenitor gene vectors. At birth, a newborn's genome is compared to **every** registered progenitor via vectorized cosine similarity. It joins the closest existing species if the score exceeds `species_threshold` (default 0.95), otherwise a speciation event is declared.

Checking all progenitors (not just the parent's species) prevents two failure modes: drift-back (a lineage re-approaching an ancestral region would otherwise be logged as a new species) and convergent evolution (two lineages converging on the same genetic region would otherwise be counted twice).

### Simulation Runner (`simulation.py`)

`SimulationRunner` owns habitat construction, population seeding, and JSON log output. Founding creatures start at `age = days_to_sexual_viability + 1` so mating begins on day 1. Each day writes a `day_NNNNN.json` with full event data; a `summary.json` is written at the end (including an `"extinct": bool` field).

`run()` checks global population after each step and halts early on extinction. `runner.py` does the same in its own step loop and also calls `_write_summary()` directly — note that `runner.py` drives its own loop and does NOT call `run()`.

## Configuration

The default config lives at `evolution_simulator/config/simulation.toml`. Copy and edit it for custom runs — it controls habitat topology, connection graph, creature counts, species threshold, and per-habitat type/seed overrides. Species name vocabulary is in `evolution_simulator/config/species_names.toml` (100 adjectives × 100 nouns).

Each `[[habitats.instances]]` block can override `initial_species_per_habitat` and `creatures_per_species` locally. For a single-species isolation experiment, set `initial_species_per_habitat = 1` and a larger `creatures_per_species` on a habitat with no connections.

## Key Invariants to Preserve

- `sin(θ)` resource geometry is central to niche differentiation — don't replace it with dot product or explicit fitness scores
- OWA trait aggregation (not plain mean) is deliberate: it makes individual mutations selectable by giving higher-valued loci more phenotypic weight. `OWA_ALPHA = 0.6` is the class-level default; change it on subclasses, not the base class
- Migration events must not be applied within `Habitat.simulate_day()`; they must flow through `SimulationRunner.step()` to avoid double-simulation
- Species assignment must compare against all progenitors, not just parent lineage
- Founding creatures must start sexually viable so the first day produces mating events
