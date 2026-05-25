# TODO

Cross-machine task tracking for active development items.

---

## In Progress

### Configurable logging cadence (PLANNING_TEMP.md Phase 1 — DONE)

`stats_every` and `events_every` knobs added to `simulation.toml` and
`SimulationRunner.step()` / `_build_week_log()`.  Default config now logs
stats every 10 weeks (`stats_every = 10`) and disables detailed event logging
(`events_every = 0`).  `runner.py` updated to use `births_this_week` /
`deaths_this_week` from the step return dict.  6 new tests in
`tests/test_simulation_logging.py`; all 206 tests pass.

### Vectorized compute_stats (PLANNING_TEMP.md Phase 2 — DONE)

`_batch_compute_traits()` in `habitat.py` stacks creature genes into a (N, 500)
matrix and computes all 34 traits in one vectorized pass.  `compute_stats` now
uses this instead of N × 34 individual `getattr` calls.  Total runtime dropped
from 37.7 s → 11.75 s (69% reduction, measured on default 200-week config).

### Speciation redesign — C2 (reproductive isolation) DONE

Root cause of runaway speciation was a signal mismatch: detection compared
newborns to a *frozen full-genome progenitor*, while mating is gated by the
245-dim `compatibility_genes` subset.  Ordinary full-genome drift away from the
ancient anchor speciated coherent, still-interbreeding populations (~2,150
species vs ~1,650 individuals at week 14k).

**C2 fix (implemented):** detection now keys on the compatibility subset,
compared against each species' **living centroid** (refreshed every
`respeciate_every` weeks, default 10) instead of a frozen progenitor.  A species
boundary now means "can no longer interbreed with that population" (Biological
Species Concept).  Knobs: `compatibility_threshold = 0.65` (a touch below the
0.70 mating floor), `respeciate_every = 10`.  Each species keeps a frozen
full-genome **type** (for naming + the anagenesis axis below) plus the living
centroid.  Speciation events now carry an `event_type` field (`"cladogenesis"`).

### Anagenesis + cladogenesis-split layer — DONE

Two periodic detectors added in `species.py`, run on the `respeciate_every`
cadence in the fixed order `detect_subcluster_splits` → `refresh_centroids` →
`detect_anagenesis` (`promote_candidates` still runs every week):

- **Cladogenesis split detector** (`detect_subcluster_splits`): catches a
  population that has split into two reproductively-isolated modes that a single
  averaged centroid masks (the modes sit symmetrically around the midpoint, so no
  individual newborn trips the membership test).  Members are clustered in
  compatibility space via a numpy **spherical k-means** (`_select_clusters`
  sweeps K=2..`split_max_k`, picks the largest K whose centroids are all mutually
  below `split_isolation_threshold`).  Cluster nearest the frozen type keeps the
  name; others seed/extend candidates and flow through the existing two-stage
  gate.  `event_type: "cladogenesis"`.
- **Anagenesis detector** (`detect_anagenesis`): catches in-place lineage
  transformation that stays interfertile (chronospecies — dogs/wolves).  Measures
  a separate 32-trait **phenotype** axis (`PHENOTYPE_TRAITS`,
  `compute_phenotype_matrix`), cosine **centered on 0.5**, type vs living
  centroid.  Below `anagenesis_threshold` (0.93) held for `anagenesis_weeks` (60,
  a **persistence gate** so transient dips aren't named) → respeciate members by
  closest phenotype between new centroid and frozen type (name follows the type).
  `event_type: "anagenesis"`.

Calibration: phenotype drift is selection-bounded (plateaus ~0.94 because
adaptation chases a fixed habitat optimum), measured via
`experiments/drift_trajectory.py`; hence 0.93 + persistence rather than 0.90.
`event_type` is plumbed through the logs and both visualizers (splits solid,
anagenesis dashed/amber).  Generation-scaled gates (`min_species_population=10`,
`min_species_weeks=60`) live in the bundled configs; class defaults stay 3/5.

### Next layer — species merging (BSC is non-monotonic)

Still open: interfertile species that drift back together should **re-merge**;
today the species count only ever grows.  The BSC says two populations that can
once again interbreed are one species, so a periodic merge check (living
centroids within the mating floor of each other → collapse to the older name)
is the symmetric counterpart to the split detector.

### Fix runaway speciation (PLANNING_TEMP.md Phase 3 — DONE)

Two-stage speciation implemented in `species.py`.  Diverged newborns now enter
a **candidate** stage instead of immediately creating a confirmed species.  A
candidate is promoted only when it has `min_species_population` (default 3)
living members AND has existed for `min_species_weeks` (default 5) weeks.
Candidates whose members all die before meeting criteria are silently evaporated.
Config knobs in `[species]` section of `simulation.toml`.

Next up:
- Phase 10: multi-run sweep script (`sweep.py`) + global RNG seeding

### Multiprocessing / parallelization

NOTE: Per PLANNING_TEMP.md Section 1, within-run parallelism cannot speed up a
single long run (weeks are sequentially dependent). The relevant multiprocessing
opportunity is Phase 10: running N independent full simulations in parallel
(parameter sweeps / replicates). The habitat-level parallelism investigation
below is lower-priority given that conclusion.

1. **Habitat-level parallelization** — each habitat's `simulate_week()` is
   independent until migrations are collected. Run all habitats in parallel via
   `multiprocessing.Pool` in `SimulationRunner.step()`, then merge migration
   events in the main process. Likely net-neutral or negative for typical configs
   (few habitats, process-overhead cost).

2. **Mating pair evaluation** — the male/female pairing loop may be
   parallelizable if pair compatibility checks are independent. Evaluate whether
   lock contention on shared state (offspring list, species registry) makes this
   impractical.

Branch: `feature/multiprocessing-core-loops`

## Backlog

### Intermediate / stepping-stone habitats between connected pairs

Add a new habitat type whose characteristic center vector is the average of
the two habitats it bridges.  This creates a realistic selective gradient
between connected biomes — e.g. a Forest-Alpine corridor whose resource
geometry sits exactly between forest and alpine, so a forest-adapted species
must first become viable in the intermediate before it can thrive in the full
alpine habitat.

This enables multi-generation range expansion: a lineage that never could
have survived a direct Forest→Alpine jump can instead adapt toward the
intermediate over many generations, then eventually colonise the alpine side.

Implementation sketch:
- Add an `intermediate = true` flag (or a `derive_from = ["hab_a", "hab_b"]`
  field) to a `[[habitats.instances]]` entry in the TOML config.
- At construction time, compute the habitat vector as
  `(vec_a + vec_b) / 2`, normalised, then add the usual per-instance
  Gaussian noise on top.
- The habitat is otherwise a full first-class habitat: creatures live, die,
  reproduce, and speciate there normally.
- Connections are still explicit in the TOML, e.g.:
    connections = [["forest_1", "corridor_1"], ["corridor_1", "alpine_1"]]
- The visualizer can optionally render corridor habitats differently
  (smaller node, dashed edges) once the data model is wired up.

---

### Intelligence as predation offset
Use the `intelligence` trait as an offset to predation rate — smarter creatures
should be better at avoiding predators. Needs to be wired into the predation
calculation in `habitat.py` alongside the existing `PREDATION_ALPHA` /
`base_predation_rate` logic.

---

### Log pre-parser / configurable analyzer loading

The visualizer loads all day logs into memory, which becomes impractical for
long runs. Add a pre-parsing layer with configurable filtering so the analyzer
only loads what it needs.

Requirements:
- **Date range filter**: load only logs between day X and day Y
- **Frequency downsampling**: load every Nth day (e.g. every 70th day ≈ weekly
  for a 365-day-year sim, every 10th "week" between week 1000 and 2000)
- **Configurable via CLI args or a small config block** so the user doesn't need
  to edit source to change the filter
- Output should be the same in-memory structure the visualizer already expects,
  so no downstream changes are needed

The pre-parser could live as a standalone utility (`log_loader.py` or similar)
that the visualizer imports, or as a flag added to the existing visualizer
entry points (`visualizer_basic.py`, `visualizer_advanced.py`).

---

## Done
