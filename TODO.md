# TODO

Cross-machine task tracking for active development items.

---

## In Progress

## Backlog

### Intelligence as predation offset
Use the `intelligence` trait as an offset to predation rate — smarter creatures
should be better at avoiding predators. Needs to be wired into the predation
calculation in `habitat.py` alongside the existing `PREDATION_ALPHA` /
`base_predation_rate` logic.

### Multiprocessing / parallelization

Speed up the simulation by running independent work in parallel. Confirmed
parallelization opportunities to investigate:

1. **Habitat-level parallelization** — each habitat's `simulate_day()` is
   independent until migrations are collected. Run all habitats in parallel via
   `multiprocessing.Pool` in `SimulationRunner.step()`, then merge migration
   events in the main process. This is likely the highest-leverage change.

2. **Creature-level updates within a habitat** — the per-creature energy/hydration
   updates and death checks (step 1–2 of `simulate_day`) are vectorized already,
   but confirm whether any loops remain that could benefit from parallelism.

3. **Mating pair evaluation** — the male/female pairing loop (step 7 of
   `simulate_day`) may be parallelizable if pair compatibility checks are
   independent. Evaluate whether lock contention on shared state (offspring list,
   species registry) makes this impractical.

4. **Species assignment at birth** — `SpeciesRegistry.assign_species()` compares
   each newborn against all progenitor vectors (vectorized cosine similarity).
   If progenitor count grows large, this could be farmed out per-newborn.

Watch for the migration invariant: migrations must still be collected from all
habitats before any are applied. Parallelism must not let a creature be
simulated twice in one day. Relevant code: `SimulationRunner.step()` in
`simulation.py`, `Habitat.simulate_day()` in `habitat.py`.

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

### Fix runaway speciation
3,253 species from a single founding genome over 5,000 weeks is far too many.
Two likely fixes to research and implement:

1. **Minimum population before speciation is declared** — a single-birth
   divergence should not count as a new species. Require the candidate lineage
   to sustain some minimum headcount (e.g. N individuals) before the registry
   promotes it.

2. **Stability window** — require the diverged lineage to persist for some
   minimum number of weeks without going extinct before it is recorded as a
   new species. Prevents ephemeral genetic outliers from inflating the count.

Relevant code: `SpeciesRegistry.assign_species()` in `species.py`.

---

## Done
