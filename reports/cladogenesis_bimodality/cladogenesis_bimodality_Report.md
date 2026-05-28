# Allopatric Speciation via Natural Selection: Forest/Plains Isolation Experiment

---

## Experiment Summary

**Config:** `bimodality_experiment.toml` (copy in this folder)
**Script:** `experiments/cladogenesis_bimodality.py`
**Run date:** 2026-05-28
**Animation:** `allopatric_speciation_forest_plains.gif`

Two completely isolated habitats (Forest and Plains) were seeded with 500
near-identical creatures each — all descended from a single shared founding
genome biased toward the average of both habitat vectors (40% habitat
direction, 60% random variation), ensuring both populations were viable from
the start. No migration routes were configured; the two populations can never
exchange individuals. The simulation was run for 3,000 weeks (~82 generations)
and a compatibility-gene snapshot was taken every 10 weeks.

The experiment was designed to make the k-means split detector's signal
_visible_: rather than just watching a speciation event get logged, we plot
the full distribution of genetic compatibility vectors at each snapshot, so the
gradual bimodal separation can be observed frame by frame.

**Key non-defaults from the standard config:**

| Parameter | Value | Why |
|---|---|---|
| `creatures_per_species` | 500 | Larger N → smoother histograms, lower extinction risk |
| `initial_species_per_habitat` | 0 | Script seeds manually with a shared founding genome |
| `stats_every` / `events_every` | 0 | Experiment drives its own output; no per-week JSON logs needed |
| `mirror_founding_population` | false | Manual seeding used instead (mirror mode can't apply habitat bias) |
| Founding genome bias | 0.4 toward avg habitat vector | Ensures early survival without pre-adapting to either specific habitat |

**Starting state:** 500 creatures per habitat (250 female, 250 male), 1
founding species shared across both habitats, 0 connections.

---

## Report Summary

**Why this was interesting:** The k-means cladogenesis split detector identifies
speciation by finding bimodal structure in compatibility-gene space — but a
single summary statistic (centroid-to-centroid cosine) doesn't show _how_ that
bimodality develops. This experiment makes the full distribution visible.

![Allopatric speciation: Forest vs Plains](allopatric_speciation_forest_plains.gif)

*Weeks 10–1500 at 0.5 s/frame; centroid cosine time-series held for 10 s at the end.
The transition from a single unimodal peak (shared ancestor) to two fully
separated peaks (distinct species) is visible across the animation.*

**What was found:**
- Both populations remained genetically indistinguishable for the first
  ~10 generations (centroid cosine ≥ 0.99)
- Divergence was gradual but monotonic, driven purely by differential
  selection pressure in the two habitats
- The k-means split detector fired at **week 1090 (~30 generations)** when
  the centroid-to-centroid cosine fell below the 0.75 reproductive-isolation
  threshold
- After the split, divergence continued for another ~1,200 weeks before
  decelerating, with the cosine reaching ~0.30 by week 3,000
- Panel A (separation-axis projection, which cancels shared adaptations) showed
  cleaner bimodal separation than Panel B (cosine to Forest centroid), directly
  confirming the theoretical advantage of the axis approach
- The per-habitat centroid cosine and the k-means centroid cosine tracked each
  other almost identically once populations exceeded ~150 individuals, validating
  that the detector is reliably measuring the same signal as the ground truth

**Takeaway:** This is direct visual evidence of natural selection driving
allopatric speciation in the simulation. 500 creatures with an identical
founding genome, placed in two isolated environments, evolved to the point of
reproductive incompatibility in approximately 30 generations — driven entirely
by differential adaptation pressure, with no directed mutation or designer
fitness function.

---

## Report

### Setup and Methodology

The canonical approach to running this experiment with `mirror_founding_population = true`
fails under the raised food/water selection pressure on this branch because mirror
mode cannot apply `founding_habitat_bias` (a single genome cannot be aligned
to two different habitat vectors simultaneously). Instead, the script manually
constructs a founding genome as:

```
founding_genes = 0.6 × random_normal(500) + 0.4 × normalise(avg_habitat_vec) × √500
```

where `avg_habitat_vec` is the mean of the Forest and Plains characteristic
vectors. This gives all creatures a moderate initial food/water discovery
probability in both habitats without pre-adapting them to either environment
specifically. Both habitats then receive 500 copies of this genome with small
per-creature noise (`σ = 0.05`).

### Visualization Approach

At each 10-week snapshot, all living creatures' 245-locus compatibility gene
vectors are extracted and unit-normalised. Two projections are computed:

**Panel A (separation axis):** The per-habitat mean compatibility vectors
`c_Forest` and `c_Plains` are computed, and each creature is projected onto
`normalise(c_Plains − c_Forest)`. This direction cancels out genes that both
populations share (e.g., universal adaptations to food/water discovery) and
retains only the dimensions where they differ — producing the widest possible
peak separation. Forest creatures fall to the left (negative projection),
Plains creatures to the right.

**Panel B (cosine to Forest centroid):** Each creature's cosine similarity to
the current Forest population mean. Less separated than Panel A because the
shared adaptation signal is still present, but conceptually simpler.

A key engineering detail: using per-habitat means for the axis rather than
k-means centroid labels eliminates **label switching** — the problem where
k-means assigns `centroid_0` and `centroid_1` inconsistently between frames,
causing the histogram to visually flip sides. The k-means algorithm is still
run each frame but is used only for the centroid-to-centroid cosine in the
time-series, not for the histogram axis.

### Divergence Timeline

| Period | Hab cos | Description |
|---|---|---|
| Weeks 10–200 | 0.9998–0.9949 | Effectively one population; no visible separation |
| Weeks 200–500 | 0.9949–0.9282 | Slow early divergence as selection begins filtering |
| Weeks 500–900 | 0.9282–0.7987 | Accelerating divergence; bimodal peaks becoming visible |
| Week 1090 | 0.7272 | **Split detector fires** — populations reproductively isolated |
| Weeks 1090–2300 | 0.727–0.380 | Continued divergence post-speciation |
| Weeks 2300–3000 | 0.380–0.296 | Decelerating — populations approaching their respective habitat optima |

### What Drives the Speciation

The selection mechanism is the dot-product resource geometry:
`P(food) = (cos θ + 1) / 2` where θ is the angle between a creature's genes
and the habitat's characteristic vector. Creatures more aligned to Forest find
food and water more reliably; those better aligned to Plains fare better there.
Each generation, offspring inheriting genes closer to their habitat's optimum
are more likely to survive and reproduce. No fitness function is specified —
this emerges entirely from the energy/hydration accounting.

The compatibility genes (245 loci) are a subset of the full 500-locus genome.
As the two populations' compatibility sub-vectors diverge past the 0.75 cosine
threshold, the mating system treats them as incompatible — they would not
interbreed even if placed in the same habitat. This is the Biological Species
Concept implemented mechanically.

### Caveats

- The ~30-generation timeline is faster than typical biological speciation
  because `min_species_weeks = 60` (the candidate persistence gate) is short
  relative to generation length. Tuning this upward would require longer
  divergence before the split is confirmed.
- The founding genome bias (0.4) gives creatures a moderate initial fitness
  in both habitats. A purely random founding genome (bias = 0.0) results in
  rapid extinction under the current food/water selection pressure, so some
  initial alignment is a practical requirement for this experiment design.
- The Plains population consistently ran larger than Forest (~300–700 vs
  ~100–350 after week 500), suggesting Plains is slightly more permissive
  under these founding conditions. Both lineages survived robustly throughout.
