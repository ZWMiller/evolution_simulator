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
