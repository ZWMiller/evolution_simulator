# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active Work

`docs/TODO.md` tracks open tasks across machines. Check it at the start of a session to see what's in progress or queued up, and update it when tasks are completed or new ones are identified.

## Commands

```bash
# Install dependencies
poetry install

# Run simulation with default config (4 habitats)
python runner.py

# Run with custom config or week override
python runner.py path/to/config.toml --weeks 500

# Run all tests
poetry run pytest

# Run a single test file
poetry run pytest tests/test_creature.py

# Run a single test by name
poetry run pytest tests/test_creature.py::test_function_name
```

## Architecture

The simulation is built around four core classes that interact in a strict data-flow order each week.

### Gene/Trait System (`creature.py`)

Every `Creature` carries a **500-dimensional float gene vector**. All phenotypic traits are polygenic, computed via **Ordered Weighted Averaging (OWA)**:

1. Extract gene values at the trait's loci.
2. Sort descending; assign weights `w_i = α·(1−α)^i` (normalised), where `α = OWA_ALPHA` (default 0.6).
3. `raw = dot(weights, sorted_genes)` → `sigmoid(raw)` → `[0, 1]`.

The top-ranked locus carries ~60% of the weight, the next ~24%, and so on. A beneficial mutation that rises to the top of the ranking gains immediate phenotypic influence rather than being diluted 1/N by a plain mean. `OWA_ALPHA` is a class attribute that species subclasses can override.

Many loci feed multiple traits (pleiotropy), which means selection on one trait creates correlated pressure on others. The `mutation_rate` trait is itself encoded in genes and heritable, so mutation rates evolve.

Sex is gene-determined: `sigmoid(sex_determination_genes) >= 0.5` → female.

### Resource Geometry (`habitat.py`, `habitats/types.py`)

Each `Habitat` has its own 500-dim environment vector. Food/water discovery uses **dot-product geometry**: `P(resource) = (cos θ + 1) / 2` where θ is the angle between a creature's genes and the habitat vector. A creature aligned with the habitat finds resources with P=1 (perfectly adapted); orthogonal gives P=0.5 (unadapted baseline); anti-aligned gives P=0 (maladapted). This drives local adaptation pressure without any explicit fitness function.

The cosine is computed over **resource-specific gene subspaces**, not the full 500-dim genome: food-finding slices `FOOD_GENE_INDICES` (158 loci) from both the creature genes and the habitat vector; water-finding slices the separate `WATER_GENE_INDICES` (175 loci). The two subsets overlap only partly, so food and water adaptation are partly independent axes (`Habitat.food_likelihoods` / `water_likelihoods`).

Each habitat type (`habitats/types.py`) has a fixed characteristic center vector seeded from a `TYPE_SEED` constant via `__init_subclass__`, with Gaussian per-instance noise layered on top.

### Weekly Simulation Order (`habitat.py:simulate_week`)

Within a single week, `Habitat.simulate_week()` runs in this fixed order:
1. Compute resource probabilities for all creatures (vectorized)
2. Update energy/hydration; mark deaths
3. Advance creature age and pregnancy timers
4. Collect litters from females that reached gestation term
5. Remove dead creatures
6. Collect migration candidates (but do **not** apply migrations yet)
7. Pair males and females for mating (dispatch to the configured `mating_strategy`)
8. Add newborns and assign species via `SpeciesRegistry`
9. Attempt spontaneous route isolation

**Critical invariant**: migrations are returned as events, not applied inside `Habitat.simulate_week()`. `SimulationRunner.step()` processes all habitats first, then moves migrants — this prevents a creature from being simulated twice in the same week.

### Mating Strategies (`habitat.py`)

The pairing algorithm is controlled by `mating_strategy` in `[simulation]` config and dispatched inside `simulate_week`. All four strategies call `_attempt_mating(male, female)` for each pair, which runs `is_compatible()` and, if compatible, `reproduce()`. The strategy only determines *who gets paired with whom* — conception probability and litter size are always downstream of the pairing.

**`zip` (default/legacy):** Shuffles all viable males and all viable females habitat-wide, then zips them 1:1. A cross-species pairing that fails `is_compatible()` wastes both individuals' mating opportunity that week, creating a severe **minority-species Allee effect**: a minority of 5 pairs competing against a majority of 50 expects only ~9% of pairings to land within-species. Kept as the default for backwards compatibility with old experiments.

**`species_priority`:** Groups males and females by `creature.species`, pairs each species' own pool first (shuffled within-species), then sends surplus unpaired individuals to a shared cross-species spillover pass. Eliminates the Allee effect entirely: each species gets mating proportional to its own sex ratio regardless of relative abundance. Hybridisation still occurs for leftover surplus individuals. O(N).

**`weighted_matrix`:** Builds the full M×F compatibility score matrix in one vectorized numpy pass (`(M,245) @ (245,F)` + norm division). Iterates females in random order; each female samples a male with probability proportional to `max(0, score − threshold)^sharpness`, where `threshold = COMPATIBILITY_FLOOR + 0.15·female.selectivity` and `sharpness = 1 + MATING_SHARPNESS_K · mean(male.selectivity, female.selectivity)` (default K=3, so sharpness ∈ [1,4]). Each claimed male is removed from the pool. Low selectivity → nearly uniform above the floor (liberal hybridisation); high selectivity → sharply peaked at the best available mate. Makes hybridisation propensity an **evolved trait** driven by `selectivity`. O(N²) but fast in practice under the `POPULATION_SUPPORT` cap.

**`stable_matching`:** Runs **Gale-Shapley deferred acceptance** (male-proposing) on the same M×F matrix. Each male proposes down his ranked preference list; each female tentatively holds her best offer and releases prior partners if a better proposer arrives. Terminates when no free male has remaining candidates. Produces a **stable matching** — no unmatched (male, female) pair both prefer each other over their current partners. Hybridisation occurs only when a cross-species individual genuinely ranks above all same-species alternatives for both parties. See `_gale_shapley()` in `habitat.py` for the full algorithm description and documentation of all six baked-in assumptions (proposing direction, bilateral threshold pre-filtering, tie-breaking noise, etc.). O(N²).

**Key note on cross-species immunity:** Under `weighted_matrix` and `stable_matching`, a majority female cannot accidentally claim a minority male. Cross-species cosine scores in 245-dim space are near 0 (std ≈ 1/√245 ≈ 0.064), well below the 0.70 `COMPATIBILITY_FLOOR`, so their sampling weights are zero. Minority males are protected regardless of iteration order.

### Species Detection (`species.py`)

Speciation is detected on the **same signal that gates mating**: the 245-locus `compatibility_genes` subset. A newborn's compatibility sub-vector is compared by cosine similarity against every living species centroid; it is assigned to the **nearest** species unconditionally (no threshold floor for assignment). The newborn will not mate back into that species if its actual compatibility score falls below the `COMPATIBILITY_FLOOR` — but it is counted under that species and creates no candidate. Only a population-level k-means split can initiate a new species candidate. This is the Biological Species Concept — a species boundary means "can no longer interbreed with that population."

**Why living centroids, not frozen progenitors**: the reproductive reference is the *current* population, refreshed every `respeciate_every` weeks (`SpeciesRegistry.refresh_centroids`, called from `SimulationRunner.step()` after migrations). When a whole interbreeding population drifts together the centroid drifts with it, so ordinary drift never trips a speciation event — this is the fix for the historical runaway-speciation bug. Candidate members are *excluded* from their parent species' centroid so an incipient split can pull away cleanly.

**Two references per species**: each confirmed species keeps a frozen full-genome **type** (`progenitor_genes`, anchors the name and feeds the anagenesis axis) *and* a living 245-dim compatibility **centroid** (`centroid`, used for detection). Comparison is against **all living species** (not just the parent), preventing drift-back and convergent-evolution false positives.

**Two speciation mechanisms.** Both are periodic detectors that run on the `respeciate_every` cadence (`SimulationRunner.step()`, week 1 and every Nth week). Order matters and is fixed: `detect_subcluster_splits` → `refresh_centroids` → `detect_anagenesis`, then `promote_candidates` runs **every** week.

1. **Cladogenesis split detector** (`detect_subcluster_splits`): a single living centroid masks a population that has split into two reproductively-isolated modes — the two clusters sit symmetrically around their midpoint centroid, so no individual newborn ever falls below threshold. For each species with ≥ `2·min_species_population` non-candidate members, the members are clustered in compatibility space via spherical k-means (`_select_clusters` sweeps K=2..`split_max_k` and picks the **largest K whose cluster centroids are all mutually below** `split_isolation_threshold`; K=1 = no split). `split_isolation_threshold` is set above the 0.70 mating floor (default 0.75) so detected splits represent cleanly non-interbreeding populations. The cluster nearest the frozen type keeps the name; each other cluster seeds or extends a **candidate**, so promotion flows through the same two-stage gate (`min_species_population` + `min_species_weeks`). Runs *before* `refresh_centroids` so split-off members are excluded from the parent centroid. Event `event_type: "cladogenesis_kmeans_subcluster"`.

2. **Anagenesis detector** (`detect_anagenesis`): catches a lineage that has **transformed in place** without ever splitting (chronospecies — "dogs from wolves" still interbreed, so the BSC/compatibility axis cannot separate them). It measures a separate **phenotype** axis: the 32-trait `PHENOTYPE_TRAITS` raw-OWA vector (`compute_phenotype_matrix`), compared to the species' frozen-type phenotype by cosine **centered on 0.5** (phenotype values live in `[0,1]`, where raw cosine compresses toward 1; centering measures the trait-deviation pattern over the full `[-1,1]` range). When that cosine falls below `anagenesis_threshold` *and* stays below for `anagenesis_weeks` (a **persistence gate** — `_anagenesis_pending` tracks the first week it dipped, and a rebound resets the clock so transient dips aren't named), the lineage is respeciated: members are partitioned by closest phenotype between the new centroid and the frozen type (the name follows the type). The new species' `_type_phenotype` is anchored to the **mover population centroid** (not a single individual) to prevent the re-fire bug where a misanchored phenotype immediately restarts the persistence clock. Species with an active split candidate this cycle are skipped. Event `event_type: "anagenesis"`. `SimulationRunner.step()` also polls `detect_anagenesis(pending_only=True)` every week when `_anagenesis_pending` is non-empty, so the persistence gate resolves at weekly resolution rather than waiting for the next `respeciate_every` cadence.

Phenotype drift is selection-bounded, not unbounded: it plateaus around 0.94 because adaptation chases a fixed habitat optimum, which is why `anagenesis_threshold` is 0.93 (not, say, 0.90) and why anagenesis only surfaces on long runs.

Every speciation event carries an `event_type` field. Valid values: `"cladogenesis_kmeans_subcluster"` (k-means split detector found reproductively-isolated sub-clusters), `"cladogenesis_bootstrap"` (founding species registration), `"anagenesis"` (in-place phenotype transformation). The string `"cladogenesis_newborn"` is retired — it may appear in old log files but is never emitted by current code. The phylogeny renders these types distinctly.

### Simulation Runner (`simulation.py`)

`SimulationRunner` owns habitat construction, population seeding, and JSON log output. Founding creatures start at `age = weeks_to_sexual_viability + 1` so mating begins in week 1. Each week writes a `week_NNNNN.json` (subject to the `stats_every` / `events_every` logging cadences); a `summary.json` is written at the end (including an `"extinct": bool` field).

`run()` checks global population after each step and halts early on extinction. `runner.py` does the same in its own step loop and also calls `_write_summary()` directly — note that `runner.py` drives its own loop and does NOT call `run()`.

## Configuration

The default config lives at `simulation_configs/simulation.toml`. Copy and edit it for custom runs — it controls habitat topology, connection graph, creature counts, the speciation knobs, and per-habitat type/seed overrides. Ready-made long-run configs (30k-week ring, parallel divergence, identical-habitat control) also live in `simulation_configs/`. Species name vocabulary is in `evolution_simulator/config/species_names.toml` (100 adjectives × 100 nouns) — this file must stay in its current location as it is loaded via `__file__`-relative path in `species.py`.

Speciation knobs (defaults shown are the *class* defaults in `SpeciesRegistry`; the bundled configs deliberately raise the gates to generation-scale, ~2 generations at ~25–35 weeks each):

| Key | Section | Class default | Config value | Meaning |
|---|---|---|---|---|
| `compatibility_threshold` | `[species]` | 0.65 | 0.65 | newborn-vs-living-centroid cosine to join a species (just below the 0.70 mating floor) |
| `respeciate_every` | `[simulation]` | 10 | 10 | weeks between centroid refreshes + the cadence on which both periodic detectors run |
| `min_species_population` | `[species]` | 3 | 10 | living members a candidate needs before promotion (and ≥2× this to attempt a split) |
| `min_species_weeks` | `[species]` | 5 | 60 | weeks a candidate must persist before promotion |
| `split_max_k` | `[species]` | 5 | 5 | max sub-clusters the split detector will sweep K up to |
| `split_isolation_threshold` | `[species]` | 0.75 | 0.75 | sub-cluster centroids must be mutually below this to count as a split (above the 0.70 mating floor for clean isolation) |
| `anagenesis_threshold` | `[species]` | 0.93 | 0.93 | centered phenotype cosine (type vs living centroid) below which anagenesis triggers |
| `anagenesis_weeks` | `[species]` | 60 | 60 | persistence: weeks the phenotype must stay diverged before the lineage is respeciated |

Each `[[habitats.instances]]` block can override `initial_species_per_habitat` and `creatures_per_species` locally. For a single-species isolation experiment, set `initial_species_per_habitat = 1` and a larger `creatures_per_species` on a habitat with no connections.

## Key Invariants to Preserve

- `(cos θ + 1) / 2` resource geometry is central to local adaptation — don't replace it with cross-product/sin or explicit fitness scores. Aligned genes → P=1, orthogonal → P=0.5, anti-aligned → P=0
- OWA trait aggregation (not plain mean) is deliberate: it makes individual mutations selectable by giving higher-valued loci more phenotypic weight. `OWA_ALPHA = 0.6` is the class-level default; change it on subclasses, not the base class
- Migration events must not be applied within `Habitat.simulate_week()`; they must flow through `SimulationRunner.step()` to avoid double-simulation
- Speciation detection keys on the 245-dim `compatibility_genes` subset (the mating signal), compared against **living centroids** refreshed periodically — not full-genome frozen progenitors. Don't revert detection to the full genome or a frozen reference; that reintroduces drift-driven runaway speciation. The frozen full-genome type is retained for naming + the anagenesis axis
- Species assignment must compare against all living species centroids, not just parent lineage
- The two periodic detectors run in a fixed order: `detect_subcluster_splits` → `refresh_centroids` → `detect_anagenesis`. The split detector must run *before* the refresh so split-off members are excluded from the parent centroid
- The anagenesis axis is the **phenotype** vector (raw OWA, centered on 0.5 for cosine), measured against the frozen type — it is deliberately a *different* signal from the compatibility axis, because a transformed-but-still-interfertile lineage (chronospecies) is invisible to the reproductive-isolation test
- Founding creatures must start sexually viable so the first week produces mating events
