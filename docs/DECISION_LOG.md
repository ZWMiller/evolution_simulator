# EvoSim Decision Log

A dated record of design decisions and methodology changes. Updated as part of feature
development and maintenance. Does not track file moves, dependency bumps, or tooling changes —
only decisions about the simulation system itself.

---

## 2026-05-27

- **Removed newborn-triggered cladogenesis; k-means subcluster split is now the sole cladogenesis path.**
  The newborn detector fired whenever a single offspring's compatibility vector fell outside its
  parent species' centroid by more than the threshold. This was noisy (a single unlucky genome
  could trigger a candidate) and redundant once the population-level k-means split detector was
  in place. Removing it makes cladogenesis a strictly population-level signal, consistent with
  the Biological Species Concept.

- **Retuned food and water pressure to create meaningful metabolic and hydration selection.**
  Previous constants produced near-universal food/water success, making `metabolism` and
  `water_efficiency` effectively neutral traits with no selection gradient. New per-biome
  constants (`FOOD_ENERGY_COST`, `WATER_BASE_COST`, `WATER_EFFICIENCY_COST`) are calibrated so
  that maladapted creatures face real attrition, giving those traits selective value.

## 2026-05-26

- **Added `stable_matching` mating strategy (Gale-Shapley deferred acceptance).**
  Produces a stable matching on the M×F compatibility score matrix — no blocking pair exists
  where both individuals prefer each other over their assigned partner. Hybridisation occurs only
  when a cross-species partner genuinely outranks all same-species alternatives for both parties.
  Six assumptions are documented in `_gale_shapley()`.

- **Added `weighted_matrix` mating strategy.**
  Each female samples a male via power-law weights over the vectorized M×F compatibility matrix.
  Mating sharpness is a function of `selectivity`, making hybridisation propensity an evolved
  trait rather than a fixed parameter. Minority males are protected: cross-species scores in
  245-dim space cluster near zero (std ≈ 0.064), far below the 0.70 mating floor.

- **Added `species_priority` mating strategy.**
  Within-species pairing runs first; surplus individuals go to a cross-species spillover pool.
  Eliminates the minority-species Allee effect present in `zip` while preserving hybridisation
  for surplus individuals.

- **Retained `zip` as the legacy default** for backwards compatibility with existing experiment
  logs. New experiments should use `species_priority` or better.

- **Introduced `selectivity` as a heritable trait controlling mate-choice sharpness.**
  Under `weighted_matrix` and `stable_matching`, high-selectivity creatures strongly prefer
  same-species mates; low-selectivity creatures hybridise liberally. This makes assortative
  mating an evolved phenotype, not a fixed parameter.

## 2026-05-25

- **Redesigned speciation to use living centroids refreshed on a cadence.**
  Previous design compared newborns against frozen progenitor centroids. As a whole population
  drifted, the frozen reference fell further behind, eventually causing every newborn to appear
  diverged — producing runaway speciation with hundreds of species. The fix: each species keeps
  a living centroid refreshed every `respeciate_every` weeks from current members. Ordinary
  drift moves the centroid with the population; only genuine reproductive isolation from the
  current population triggers a candidate.

- **Introduced two distinct speciation mechanisms: cladogenesis and anagenesis.**
  Cladogenesis (population splits) operates on the compatibility axis via k-means clustering.
  Anagenesis (in-place transformation) operates on a separate phenotype axis (32-trait OWA
  vector, centered cosine against the frozen type). The two axes are deliberately different
  signals: a transformed-but-still-interfertile lineage (chronospecies) is invisible to the
  reproductive-isolation test and requires the phenotype axis to surface.

- **Added k-means subcluster split detector for cladogenesis.**
  A single living centroid masks a bimodal population: the two clusters sit symmetrically around
  their shared mean, so no individual newborn falls outside threshold. The detector sweeps
  K = 2..`split_max_k` on the compatibility vectors and selects the largest K whose sub-cluster
  centroids are all mutually below `split_isolation_threshold` (0.75, above the 0.70 mating
  floor). This catches allopatric divergence that the centroid-assignment system would miss.

- **Added anagenesis persistence gate (`anagenesis_weeks`).**
  A single dip below `anagenesis_threshold` in phenotype cosine does not immediately rename the
  lineage. The dip must persist for `anagenesis_weeks` consecutive weeks. Transient selection
  spikes that later recover are ignored; only sustained drift triggers renaming.

- **Added `mirror_founding_population` seeding mode for controlled isolation experiments.**
  Allows two habitats to start with identical founding genomes, making divergence purely a
  product of selection rather than initial genetic differences.

- **Kept the frozen full-genome type (progenitor) as a secondary reference.**
  Used only for: (1) species naming, (2) the anagenesis phenotype axis. Detection and assignment
  use only the living compatibility centroid.

## 2026-05-24

- **Implemented two-stage candidate gate for speciation promotion.**
  Candidates must accumulate ≥ `min_species_population` living members AND persist for ≥
  `min_species_weeks` weeks before being promoted to a confirmed species. This prevents
  short-lived genetic outliers and small family groups from being named as new species.

- **Added performance optimization: vectorized batch computation for resource probabilities.**
  Replaced per-creature Python loops with numpy matrix operations for the dot-product
  probability computation across all creatures in a habitat simultaneously.

## 2026-05-20

- **Tuned default parameters from a stability search.**
  Adjusted `COMPATIBILITY_FLOOR`, `PREDATION_ALPHA`, and reproduction parameters to produce
  stable long-run populations without collapse or explosive growth. Added hybridisation event
  logging to make tuning observable.

## 2026-05-19

- **Added `founding_habitat_bias` parameter.**
  Controls how much founding genomes are biased toward the habitat's environment vector
  (0 = fully random, 1 = fully aligned). Random founding genomes produce near-orthogonal
  creatures with P(food) ≈ P(water) ≈ 0.5; positive bias improves founding viability without
  forcing adaptation.

- **Fixed population collapse caused by sex ratio, reproduction timing, and density interaction.**
  Early runs collapsed because: (a) random sex ratios left habitats with no viable pairs, (b)
  gestation timers prevented mating during pregnancy with no surplus population to absorb the
  cost, (c) density-dependent predation fired before population could stabilise. Founding
  creatures now start at `age = weeks_to_sexual_viability + 1` and sex ratios are enforced at
  50/50 during founding.

## 2026-05-16

- **Switched simulation time unit from days to weeks.**
  Days produced unrealistically fast generational turnover and required very small parameter
  values. Weeks align better with the intended generational timescale (~25–35 weeks per
  generation at default parameters) and make configuration values more interpretable.

- **Removed gene forcing from founding population.**
  An early implementation forced specific loci to fixed values in founding creatures to guarantee
  viable populations. This caused explosive speciation: all founders were genetically identical
  at forced loci, and any mutation away from those values was detected as a candidate. Removed
  in favor of `founding_habitat_bias` and calibrated cost/gain constants.

## 2026-05-14

- **Added density-dependent predation.**
  Intrinsic `base_predation_rate` alone produced no population ceiling. Added a crowding term:
  `PREDATION_ALPHA × N / POPULATION_SUPPORT`. `base_predation_rate` shares loci with `fecundity`
  to encode an r/K trade-off: fast-reproducing genotypes carry higher intrinsic predation risk.

- **Bootstrapped founding genomes for viable populations.**
  Calibrated `FOOD_ENERGY_GAIN`, `FOOD_ENERGY_COST`, `WATER_HYDRATION_GAIN`, `WATER_BASE_COST`,
  and per-biome overrides so that partially-adapted founding creatures survive long enough for
  selection to act, without making survival trivially easy.

## 2026-05-13

- **Replaced plain mean trait aggregation with Ordered Weighted Averaging (OWA).**
  A plain mean over N loci dilutes the effect of any single beneficial mutation by 1/N, making
  individual mutations nearly invisible to selection. OWA with α = 0.6 gives the highest-valued
  locus ~60% of the phenotypic weight, making each mutation immediately selectable from its
  first generation. The α parameter is a class attribute so species subclasses can tune
  selection sensitivity independently.

- **Adopted `(cos θ + 1) / 2` as the resource discovery probability function.**
  High-dimensional random vectors concentrate near orthogonality (cos θ ≈ 0), so raw cosine
  would give all unselected creatures P(resource) ≈ 0.5 with minimal variance, producing no
  selection gradient. The `(cos θ + 1) / 2` transform maps the full cosine range to `[0, 1]`
  and produces a usable gradient: aligned → P=1, orthogonal → P=0.5, anti-aligned → P=0. No
  explicit fitness function is defined; adaptation is purely geometric.

## 2026-05-12

- **Initial working build.** Core classes: `Creature` (gene vector, OWA traits, Mendelian
  reproduction), `Habitat` (resource geometry, mortality, migration), `SpeciesRegistry`
  (centroid-based membership), `SimulationRunner` (orchestration and JSON logging).
