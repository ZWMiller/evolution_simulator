# TODO

Cross-machine task tracking for active development items.

---

## Immediate / Next Session

### Validation runs for food/water retune + cladogenesis changes (branch: feature/food-water-retune-cladogenesis)

The branch raises food/water selection pressure and removes newborn cladogenesis
(k-means subcluster splits are now the sole speciation path). Need longer runs to
validate the changes behave as expected before merging.

Things to look for:
- Food/water pressure: are metabolism and water_efficiency now under meaningful
  selection? Mean values should drift downward (metabolism) and upward
  (water_efficiency) over thousands of weeks vs. the old flat trajectories.
- Cladogenesis: far fewer total speciation events; no `cladogenesis_newborn`
  entries in summary.json. K-means splits should still fire when populations
  genuinely diverge across disconnected habitats.
- Anagenesis: should now fire meaningfully in long runs — the newborn peeling
  that was continuously resetting the phenotype-drift clock is gone.
- Species lifespan: mean species lifetime should be longer than equivalent runs
  on main (less churn from spurious candidates).

---

### Run visualizer on parallel divergence experiment

Log dir: `simulation_logs/2026-05-25_16-09-11/`
Report:  `reports/2026-05-25_16-09-11/2026-05-25_16-09-11_Report.md`

Run `python visualizer_advanced.py` (or `visualizer_basic.py`), load that run,
and review the phylogeny, species timeline, and trait trajectories.  Key things
to look for:
- The burst of 6 anagenesis events in Forest (weeks 6050–6950) — does the
  phylogeny render them as a clean chain of in-place transformations?
- The r/K life-history split across Plains/Wetlands/Forest in the trait panels
- Desert and Tundra extinction timing
Update the report (`reports/2026-05-25_16-09-11/2026-05-25_16-09-11_Report.md`)
with any additional findings from the visual review.

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

Next up: **sweep.py — multi-run multiprocessing**

Goal: run N independent full simulations in parallel (parameter sweeps /
replicates) using all CPU cores. Each run is a separate `SimulationRunner` in a
separate OS process with zero shared state. Near-linear speedup up to core count.

**Prerequisite — seed the global RNGs** (`simulation.py`):

After `rng = np.random.default_rng(seed)` in `SimulationRunner.setup()`, add:

```python
if seed is not None:
    np.random.seed(seed)    # seeds global MT19937 used in habitat.py / creature.py
    random.seed(seed)       # seeds stdlib random used for species names
```

Makes a full run reproducible from `seed` (currently only founding genomes are
seeded; week dynamics are not).

**Script:** create `scripts/sweep.py`. Key design points:
- `concurrent.futures.ProcessPoolExecutor` with `spawn` context
- Each worker overrides `seed` and `output_dir` in-memory (doesn't edit the
  base config file)
- Output lands in `sweep_logs/seed_<S>/<timestamp>/`
- `sweep_logs/sweep_summary.json` indexes all runs on completion
- `if __name__ == "__main__":` guard is mandatory (spawn re-imports the module)
- Set `OMP/OPENBLAS/MKL_NUM_THREADS=1` in each worker before importing numpy

**Usage:**
```bash
poetry run python scripts/sweep.py simulation_configs/simulation.toml --seeds 1 2 3 4 --weeks 5000
poetry run python scripts/sweep.py simulation_configs/simulation.toml --n 8 --workers 4
```

### Multiprocessing / parallelization

NOTE: Within-run parallelism cannot speed up a single long run (weeks are
sequentially dependent). The relevant multiprocessing opportunity is sweep.py:
running N independent full simulations in parallel (parameter sweeps /
replicates). The habitat-level parallelism investigation below is lower-priority
given that conclusion.

1. **Habitat-level parallelization** — each habitat's `simulate_week()` is
   independent until migrations are collected. Run all habitats in parallel via
   `multiprocessing.Pool` in `SimulationRunner.step()`, then merge migration
   events in the main process. Likely net-neutral or negative for typical configs
   (few habitats, process-overhead cost).

2. **Mating pair evaluation** — the male/female pairing loop may be
   parallelizable if pair compatibility checks are independent. Evaluate whether
   lock contention on shared state (offspring list, species registry) makes this
   impractical.

Branch: `feature/multiprocessing-core-loops` — merged to main (PR #2)

## Experiments

### Fecundity vs. density-driven predation rate (proposed)

**Hypothesis:** There is an optimal fecundity value for any given density-driven predation rate, and isolated populations under different predation pressures will independently converge on different fecundity values — visible as a consistent, density-correlated split across lineages despite identical habitat selection geometry.

**Design:** Five isolated, identical Forest habitats with the same founding gene pool (`mirror_founding_population = true`), but each habitat assigned a distinct `POPULATION_SUPPORT` value that spans a wide range (e.g. 100, 250, 500, 1000, 2000). Lower support capacity → higher density term → higher effective predation rate at any given population size.

**What to measure:**
- Mean fecundity per habitat at weeks 10k, 20k, 30k
- Mean `base_predation_rate` per habitat (the heritable component, which shares loci with fecundity)
- Population equilibrium size relative to support capacity in each habitat
- Whether the fecundity ordering matches the predation-rate ordering across habitats

**Why this is interesting:** The parallel divergence and identical-habitat control runs ([emergent natural selection report](reports/general_findings/emergent_natural_selection.md)) established that metabolism and water efficiency are under direct selection from the food/water miss-penalty mechanics. Fecundity is also mechanically active (litter size = Poisson(fecundity)) but the selection direction is less clean because high-fecundity genotypes also carry higher intrinsic predation vulnerability via shared loci. This experiment isolates the predation axis: if different support capacities produce consistently different equilibrium fecundity values across independent replicates, that is direct evidence that the density-predation channel is resolving the r/K tradeoff in a population-density-dependent way. Controlling the habitat selection vector (identical Forest for all five) removes the metabolism/water-efficiency confound.

**Implementation note:** `POPULATION_SUPPORT` is currently a class attribute on `Habitat` subtypes, settable via `population_support` in the `[[habitats.instances]]` TOML block. Verify this override works before running.

---

### Harsh vs. permissive environment selection pressure (proposed)

**Scientific question:** Does the compound harshness of a Desert-type habitat (lower food/water discovery probability AND lower energy rewards per meal AND higher hydration costs) create a different overall selection pressure and evolutionary response than a permissive Forest — or does the dominant driver reduce to the alignment geometry alone?

**Why this is hard to run naively:** Desert went fully extinct in the parallel divergence run because random founding genes are too misaligned to the Desert CENTER vector to survive long enough for selection to act. We cannot directly compare Forest vs. Desert outcomes without first solving the shared-survivor founding problem.

**Why this is worth solving:** Harshness in this simulation has two separable axes that are currently conflated in every real habitat type:

1. **Selection direction** — how far the habitat CENTER is from a given founding genome in gene space. This determines food/water discovery probability via `cos θ`. Any two habitats with different TYPE_SEEDs differ on this axis.
2. **Energy economics** — the per-meal reward and per-miss penalty. Desert specifically overrides: `FOOD_ENERGY_GAIN = 0.25` (vs. Forest's 0.30), `FOOD_ENERGY_COST = 0.21` (vs. 0.15), `WATER_HYDRATION_COST = 0.44` (vs. 0.25). These make Desert punishing even for a well-aligned creature, because each missed meal costs more and each found meal recovers less.

The experiment should isolate these two axes. A creature adapted to Forest has good `cos θ` in Forest but poor `cos θ` in Desert. The question is whether the energy economics layer on top of that geometry creates a *qualitatively different* selection response — not just faster death, but different trait equilibria in survivors.

**Prerequisite implementation — founding population checkpoint loading:**

The cleanest approach requires a new config option: `founding_genes_checkpoint = "simulation_logs/RUNID/week_NNNNN.json"`. When set, the runner loads that week's surviving creature gene vectors and uses them as founding stock (with fresh per-creature noise), bypassing the random generation step. `mirror_founding_population` and `founding_habitat_bias` are ignored when a checkpoint is specified.

This enables:
- Run Forest-only for 10,000–15,000 weeks to get a well-adapted gene pool
- Use that checkpoint to seed Forest + Desert simultaneously
- Forest-adapted genes have decent `cos θ` in Forest; their `cos θ` in Desert depends on how similar Desert.CENTER is to Forest.CENTER. Even if initial survival rate in Desert is low, *some* creatures will survive by chance variation, seeding adaptation.

**Experimental design (once checkpoint loading exists):**

*Arm 1 — Geometry-only harshness:*
- Forest habitat (permissive, standard energy params)
- A synthetic "HarshForest" variant: same TYPE_SEED as Forest (identical CENTER), but Desert's energy economics (`FOOD_ENERGY_GAIN=0.25`, `FOOD_ENERGY_COST=0.21`, `WATER_HYDRATION_COST=0.44`)
- Same founding checkpoint seeded from a Forest run
- What differs: only the per-meal economics, not the selection direction
- Measure: do the two lineages evolve to the same gene-space direction but different metabolic/water-efficiency equilibria?

*Arm 2 — Direction-only harshness:*
- Forest habitat (permissive)
- Desert habitat with Forest's energy economics (keep Desert TYPE_SEED, override energy params to Forest defaults)
- Same founding checkpoint
- What differs: only the selection direction geometry, not the per-meal economics
- Measure: do the two lineages diverge in the same way as in the parallel divergence run, or does removing the energy-economics penalty change the trajectory?

*Arm 3 — Full Desert (compound harshness):*
- Forest habitat (permissive)
- Standard Desert (different CENTER AND harsh energy economics)
- Same founding checkpoint
- Measure: is the combined effect additive, synergistic, or redundant with the geometry effect alone?

**What to measure across arms:**
- Survival rate through first 1,000 weeks (early-mortality signature of harshness)
- Mean metabolism at weeks 5k, 15k, 30k — hypothesis: energy-economics harshness selects for lower metabolism more strongly than geometry alone
- Mean water efficiency — hypothesis: higher WATER_HYDRATION_COST creates stronger directional selection toward higher water efficiency
- Food and water discovery probability trajectory — does the harsh arm adapt faster (stronger selection) or slower (higher early mortality thins the gene pool)?
- Final population size relative to POPULATION_SUPPORT

**Implementation notes:**
- `FOOD_ENERGY_GAIN`, `FOOD_ENERGY_COST`, `WATER_HYDRATION_GAIN`, `WATER_HYDRATION_COST` are currently class-level constants with no per-instance TOML override path. Adding TOML overrides for these (alongside the existing `population_support` override) is needed for Arms 1 and 2.
- Arm 3 can be run immediately once checkpoint loading exists, using standard Desert + Forest types.
- Per-habitat energy overrides in TOML are independently useful for the fecundity/predation experiment above and for future biome design work.

---

## Experiments

### Cladogenesis bimodality diagnostic script

Build `experiments/cladogenesis_bimodality.py` — a standalone script that makes
the k-means split detector's bimodality *visible* as a density histogram.

**Setup:**
- Single founding species mirrored into two isolated habitats (different types,
  e.g. Forest + Tundra, no migration routes).
- Run a custom simulation loop (don't use `SimulationRunner.run()` — drive
  `habitat.simulate_week()` manually so we can intercept at each speciation check).

**What to plot (every `respeciate_every` weeks):**
The user's instinct for cosine similarity is correct but needs the right framing.
Cosine similarity of each creature to the *shared centroid* compresses bimodality
(both clusters pull the centroid to their midpoint, so everyone looks equally
far — unimodal distribution). Instead, use the **k-means separation axis**:

1. Run k-means K=2 on the 245-dim compatibility gene vectors of all living members
   of the species (reuse `_spherical_kmeans` from `species.py`).
2. Compute the unit separation axis: `axis = normalise(centroid_2 − centroid_1)`.
3. Project each creature: `score_i = dot(compat_genes_i, axis)`.
4. Plot a density histogram of the N scores.

Interpretation:
- **Unimodal** (single peak): population is coherent, no split emerging yet.
- **Bimodal** (two peaks): two sub-clusters are pulling apart along this axis —
  exactly what the k-means detector is measuring. Peaks separate over time as
  the habitats drive divergence.

This is cleaner than pairwise cosine similarity (N values vs N² pairs) and
directly visualises the signal the split detector acts on.

**Also useful:** plot a second panel showing within-cluster vs cross-cluster mean
cosine similarity over time — when the cross-cluster line falls through
`split_isolation_threshold` (0.75), that's the moment the detector fires.

**Output:** Save one PNG (or plotly HTML) per speciation-check week to
`experiments/bimodality_output/<timestamp>/week_NNNNN.png`. Print a summary of
when (if) the split fires and what K was selected.

**Dependencies:** plotly is already available; no new deps needed. Reuse
`_spherical_kmeans` and `Creature.COMPATIBILITY_GENE_INDICES` from the main
package rather than reimplementing.

---

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

## Derek's Suggestions

Feedback from a PhD geneticist/biologist. Items flagged with "already present" are accounted for in the current implementation.

### Decouple mutation rate from selective pressure

Derek's point: in real biology the per-locus error rate during replication is roughly constant; what natural selection does is speed the *fixation* of beneficial alleles, not alter the mutation rate itself. Currently `mutation_rate` is a heritable, gene-encoded trait (creature.py:904–911) that itself evolves under selection — so populations can drift toward higher or lower mutation rates, which conflates two distinct biological forces. Options:
- Make `mutation_rate` a fixed config parameter (simplest; removes the confound entirely).
- Keep it heritable but add a floor/ceiling config so it can't evolve far from a biological baseline.
- Document it as a deliberate "mutator strain" mechanism (bacteria do evolve elevated mutation rates under sustained stress via the SOS response), and be explicit that the simulation is modelling that, not the typical metazoan case.

### Drift

- **Already present**: genetic drift is not a separate process — it is the statistical consequence of finite population sampling. Every generation, which individuals happen to mate and survive is a random draw from the parent pool, and that sampling variance accumulates as a random walk in gene space. The simulation has all the necessary ingredients (finite populations, probabilistic death, shuffled mating). Anagenesis in long runs is evidence of this: lineages phenotypically transform even without strong directional selection, which is drift acting.
- What the "neutral phenotypes" suggestion below would add is a way to *observe and measure* drift magnitude directly, by tracking loci where selection pressure is zero — any allele frequency change there is pure drift.
- Drift strength scales as 1/N_e (effective population size), so small isolated habitats already experience stronger drift than large ones. If you want to make this more tunable, the lever is `POPULATION_SUPPORT` (smaller cap → stronger drift) rather than any new code.

### Neutral phenotypes / neutral loci

Derek suggests adding some gene loci (and corresponding traits) that are not wired into any selection pressure — not food discovery, water discovery, mating compatibility, predation, or reproduction. These "neutral" axes serve as a direct measurement probe: the rate at which neutral allele frequencies change over time is a pure signal of genetic drift strength, unconfounded by selection. This would let you separately quantify how much allele frequency change in the *adaptive* traits is drift vs. selection.

### Migration as evolutionary force

- **Already implemented**: `migration_likelihood` is a heritable gene-encoded trait (creature.py:768–770). Creatures with higher values emigrate more frequently; `weekly_migration_base` in habitat.py scales the actual per-week probability. The tension Derek describes — high migration → gene flow → homogenization → no speciation; low migration → divergence → speciation — is already present in the model architecture.
- Potential follow-up: add a visualizer panel showing how migration rate correlates with species count and genetic divergence across habitats, to make the gene-flow/isolation tradeoff legible in the output.

### Drifting habitat vector

Currently `self.vector` is fixed at construction (`CENTER + per-instance noise`) and never changes. Allowing the habitat vector to slowly drift over time would simulate environmental change — climate shifts, succession, resource composition changes — and create ongoing directional selection pressure instead of a fixed optimum.

Two drift modes worth distinguishing:
- **Random walk**: each week (or every N weeks), add a small Gaussian perturbation to `self.vector` in the FOOD/WATER subspace, then renormalize so the magnitude doesn't inflate. This simulates undirected environmental noise — the optimal phenotype wanders unpredictably. Strength knob: `habitat_drift_rate` (std per step in the config). Setting it to zero recovers current behaviour.
- **Directed drift**: slowly move `self.vector` toward a second target vector at a fixed rate, simulating a long-run environmental trend (e.g. desertification). Config: `drift_target_seed` + `drift_rate_per_week`.

Biologically interesting dynamics to expect:
- Slow drift → populations track the moving optimum via selection; fast drift → adaptation can't keep up, mean fitness declines (the "Red Queen" regime).
- If two connected habitats drift in different directions, migration between them becomes increasingly costly over time, which could drive allopatric speciation even without habitat isolation.
- If a habitat's drift rate is high relative to mutation rate, you'd expect lower mean adaptation scores and higher variance — testable against a static control run.

Implementation: add a `simulate_week()` call to `Habitat` (or a hook in `SimulationRunner.step()`) that mutates `self.vector[FOOD_GENE_INDICES]` and `self.vector[WATER_GENE_INDICES]` by `rng.standard_normal(len(indices)) * drift_rate`, then renormalizes those subspace slices. The rest of `self.vector` (compatibility, other loci) could optionally drift too or stay fixed.

### Rougher optimization surface (multi-peak habitat vectors)

Currently each habitat has a single center vector, creating a smooth single-peaked fitness landscape: one globally optimal phenotype direction, and any deviation from it lowers resource discovery probability. Real environments offer multiple viable strategies — a forest has generalists, canopy specialists, understory specialists, ground foragers — each a local optimum with a fitness valley between them.

**Idea**: replace the single `self.vector` with a bank of K peak vectors (`self.vectors`, shape `(K, 500)`), and change `_batch_resource_prob` to take the **max** cosine across all K peaks:

```
P_i = max_k( (cos θ_{i,k} + 1) / 2 )
```

The max formulation means each creature uses whichever strategy it's best aligned to, and is rewarded for that specialization without needing to be optimal across all peaks. Crucially, once a lineage adapts to peak *j*, it is unlikely to be drawn toward peak *k* unless it randomly traverses the fitness valley between them — this is what makes the surface *rugged*: local minima that selection cannot easily escape.

Alternative aggregations to consider:
- **Softmax-weighted mean**: smooth interpolation between peaks; reduces ruggedness but avoids hard discontinuities.
- **Harmonic mean**: penalizes specialization, rewards generalism.
- **Max with a generalism bonus**: `P_i = mean_k + λ·max_k` balances specialist vs. generalist payoffs.

The K peak vectors would be seeded from the config (e.g. `n_peaks = 3`, `peak_seeds = [1001, 1002, 1003]`), independently perturbed with the usual per-instance noise. Setting `n_peaks = 1` recovers the current single-vector behavior.

Biologically this would enable: **sympatric speciation** (two lineages in the same habitat locking onto different peaks), **competitive exclusion** (the first lineage to lock a peak raises the effective fitness valley for any other), and **niche partitioning** as a measurable outcome rather than an assumption.

### Diploid genome

Each creature currently has a single 500-dim gene vector. Diploid would mean two copies per locus (a maternal and paternal haplotype). This opens the door to dominance/recessiveness, heterozygote advantage (overdominance), and more realistic recombination during sexual reproduction. Major architectural change — would require rethinking how OWA aggregation works when you have two alleles per locus, how compatibility genes are expressed, and how the habitat cosine geometry is computed. Worth prototyping in isolation before committing.

### Chromosomes (linkage groups)

Requires diploid first. Genes on the same chromosome are inherited as a block (linkage disequilibrium), which means beneficial alleles near a positively-selected locus hitchhike to fixation — and deleterious alleles near such a locus can accumulate (genetic hitchhiking / selective sweeps). Would let the model produce realistic patterns like selective sweeps, clonal interference, and Muller's ratchet. Implementation would mean assigning loci to named chromosomes and enforcing block inheritance during recombination.

---

## Done
