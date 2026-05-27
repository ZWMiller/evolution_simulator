# Active Planning Notes

---

## NEXT: Remove newborn cladogenesis, rely solely on k-means

### Motivation

The newborn cladogenesis route (`cladogenesis_newborn`) is epistemically weak:
it promotes a cluster to a new species based on one outlier newborn plus enough
similar individuals accumulating over 60 weeks. It cannot distinguish drift,
a high-mutation-rate pocket, or a genuine reproductively-isolated sub-population
— it just guesses "new species" and lets selection sort it out. In the 30k-week
ring run this produced 1,782 newborn speciation events and 14,482 failed
candidates — overwhelmingly noise.

The k-means subcluster split detector makes the same call with far more
information: it looks at the full living population, requires ≥ 2 ×
`min_species_population` members, finds a genuine bimodal distribution in
compatibility space, and measures centroid-to-centroid distance rather than
a single newborn's distance to a centroid. It is the right tool for this job.

Secondary motivation: newborn cladogenesis continuously peels the most-diverged
individuals off a species into new species, which resets the phenotype-drift
clock and prevents anagenesis from ever accumulating enough signal to fire.
In the 30k run, anagenesis fired zero times. Removing newborn cladogenesis lets
species live long enough for genuine in-place transformation to be named.

### Design

**Remove:**
- The per-newborn species-assignment fallback path that creates candidates when
  a newborn scores below `compatibility_threshold` for all living species.
- The candidate-promotion logic for newborn-initiated clusters (distinct from
  k-means-initiated candidates).
- The `cladogenesis_newborn` event type (or retire it — keep the string defined
  but never emit it, so old log files still parse).

**Keep:**
- `cladogenesis_kmeans_subcluster`: the primary speciation mechanism.
- `cladogenesis_bootstrap`: founding species registration — unchanged.
- `anagenesis`: in-place phenotype transformation — unchanged, and should now
  fire meaningfully without newborn cladogenesis consuming the drift signal.
- The two-stage candidate gate (`min_species_population`, `min_species_weeks`)
  — retain for k-means-initiated candidates, which still flow through it.

**Newborn assignment without cladogenesis:**
A newborn that falls below `compatibility_threshold` for all living species
currently becomes a candidate. Without that route, two options:
- **Option A (recommended):** assign to the nearest species regardless of
  threshold (closest centroid wins, no floor). These creatures won't mate back
  into their assigned species (mating still checks the floor), but they'll be
  counted under it and not create a spurious candidate. Their lineage only
  diverges if a k-means split detects them as a bimodal cluster later.
- **Option B:** assign to the parent species unconditionally. Simpler but
  ignores compatibility centroid entirely.

Option A is preferred — preserves centroid-comparison logic, keeps assignment
meaningful.

**k-means cadence:**
Currently runs on `respeciate_every` (50 weeks in the ring run). With newborn
cladogenesis removed, this is the sole path to speciation. Consider tightening
to 10–20 weeks in new configs so splits are detected while still fresh.

**split_isolation_threshold:**
Raise from 0.70 (= mating floor) to 0.75–0.78 so detected splits represent
genuinely clean reproductive isolation, not marginal centroid separation where
boundary individuals can still mate across the split. Ship together with the
newborn removal.

### Files to touch

- `evolution_simulator/species.py`: remove/stub candidate creation on newborn
  miss; update `assign_species` to use Option A fallback.
- `evolution_simulator/species.py`: verify `promote_candidates` only handles
  k-means-sourced candidates.
- `evolution_simulator/config/simulation.toml`: update `split_isolation_threshold`
  default to 0.75.
- Long-run configs: same threshold update, consider tightening `respeciate_every`.
- `CLAUDE.md`: update the speciation mechanisms section.
- Tests: update/remove tests that rely on newborn cladogenesis; add tests for
  Option A fallback assignment.

### Acceptance criteria

- [ ] A newborn scoring below `compatibility_threshold` for all species is
      assigned to the nearest-centroid species, creates no candidate, logs no
      speciation event.
- [ ] K-means split continues to detect bimodal splits and flow candidates
      through the two-stage gate.
- [ ] Anagenesis fires correctly in existing test cases.
- [ ] `split_isolation_threshold = 0.75` is the new default.
- [ ] All existing tests pass (minus newborn-cladogenesis-specific ones, updated).
- [ ] A long smoke-test run produces fewer total speciation events and longer
      mean species lifespan than an equivalent run with newborn cladogenesis.

---

## Pending: sweep.py (multi-run multiprocessing)

Goal: run N independent full simulations in parallel (parameter sweeps /
replicates) using all CPU cores. Each run is a separate `SimulationRunner` in a
separate OS process with zero shared state. Near-linear speedup up to core count.

**Prerequisite (10.0) — seed the global RNGs** (file `simulation.py`):

After `rng = np.random.default_rng(seed)` in `SimulationRunner.setup()`, add:

```python
if seed is not None:
    np.random.seed(seed)    # seeds global MT19937 used in habitat.py / creature.py
    random.seed(seed)       # seeds stdlib random used for species names
```

Requires `import random` at top of file. Makes a full run reproducible from
`seed` (currently only founding genomes are seeded; week dynamics are not).

**Script:** create `sweep.py` at repo root. Key design points:
- `concurrent.futures.ProcessPoolExecutor` with `spawn` context
- Each worker overrides `seed` and `output_dir` in-memory (doesn't edit the
  base config file)
- Output lands in `sweep_logs/seed_<S>/<timestamp>/`
- `sweep_logs/sweep_summary.json` indexes all runs on completion
- `if __name__ == "__main__":` guard is mandatory (spawn re-imports the module)
- Set `OMP/OPENBLAS/MKL_NUM_THREADS=1` in each worker before importing numpy

**Usage:**
```bash
poetry run python sweep.py evolution_simulator/config/simulation.toml --seeds 1 2 3 4 --weeks 5000
poetry run python sweep.py evolution_simulator/config/simulation.toml --n 8 --workers 4
```
