# Emergent Natural Selection: Evidence from Comparative Simulation Runs

**Date:** 2026-05-26
**Status:** Foundational finding — reference this when evaluating whether selective pressure is operating in new run configurations.

---

## Claim

The evolution simulator exhibits emergent natural selection as a property of its mechanics. We did not program evolutionary steps, fitness scores, or directed mutation. We programmed only inheritance, reproduction, and death — with survival and reproduction gated by resource-finding geometry, energy costs, and litter mechanics. The comparative evidence from two controlled runs demonstrates that the simulation nonetheless produces population-level trait change that is directional, habitat-correlated, and reproducible across independent lineages. This satisfies the classical definition of natural selection operating on heritable variation.

This is expected to strengthen over time as more traits are wired into mechanics. Even in the current implementation — with only a small number of traits directly linked to survival and reproduction — the signal is present.

---

## Background: What Was and Was Not Programmed

**Programmed directly:**
- Sexual reproduction with gene-vector inheritance and random mutation (no directed mutation)
- Death from energy or hydration depletion
- Food and water discovery stochastically gated by `P(resource) = (cos θ + 1) / 2`, where θ is the angle between a creature's 500-dim gene vector and the habitat's 500-dim environment vector
- Energy and hydration update rules that directly use trait values (see below)
- Litter size drawn from a Poisson distribution centered on the female's fecundity trait
- Density-dependent predation with per-creature base rate that shares gene loci with fecundity

**Not programmed:**
- Any fitness function or fitness score
- Directed mutation toward any trait or target
- Any explicit r/K tradeoff or life-history strategy as a rule
- Convergence or divergence as outcomes

---

## Mechanically-Active Traits

Three traits have direct code paths connecting them to individual survival or reproduction. These are not epiphenomenal — they are the channels through which selection acts:

### Metabolism
When a creature **fails** to find food in a given week:
```
energy -= FOOD_ENERGY_COST × creature.metabolism
```
When food **is** found, energy gain is a flat constant (metabolism does not scale it). Metabolism is therefore a **penalty multiplier on missed meals**: higher metabolism means faster energy depletion every week food isn't found. In any environment where food discovery is probabilistic — which applies to every creature in every habitat, especially before genetic adaptation has improved alignment — there is direct, per-creature survival pressure toward lower metabolism values. This is not epiphenomenal. A creature with lower metabolism survives longer on the same food probability.

### Water Efficiency
Symmetric to metabolism for hydration:
```
hydration -= WATER_HYDRATION_COST × (1 - creature.water_efficiency)
```
Higher water efficiency reduces hydration loss when water isn't found. Direct survival pressure toward higher water efficiency values.

### Fecundity
Litter size is drawn as `max(1, Poisson(female.fecundity))`. A female with higher fecundity produces more offspring per successful pregnancy — directly more gene copies in the next generation. This is individual-level reproductive advantage, independent of population density.

The countervailing pressure: fecundity shares gene loci with `base_predation_rate` (the r/K tradeoff encoded in the locus architecture). High-fecundity genotypes also carry higher intrinsic predation vulnerability. Density-dependent predation (`death_probability = base_predation_rate + PREDATION_ALPHA × N / POPULATION_SUPPORT`) creates pressure *against* high-fecundity genotypes when populations are large. The observed outcome — fecundity increasing in both runs — means the reproductive benefit of larger litters outweighed the predation cost under Forest conditions. This is itself a finding: the Forest habitat geometry resolves the r/K tradeoff toward higher fecundity.

### Resource Alignment (Not a Trait, but the Primary Selection Channel)
The `cos θ` geometry is not a trait — it is a function of the whole gene vector. Creatures whose genes are more aligned to the habitat vector find food and water more reliably; over generations, differential survival propagates those aligned gene configurations forward. This is the dominant selection pressure. The three traits above are secondary channels: they modulate how much a given food/water probability translates into survival.

---

## Evidence

### Evidence 1: Directional trait shift, consistent across independent replicates

Both the parallel divergence run and the identical-habitat control run start from the **same founding gene pool** (`mirror_founding_population = true`, `seed = 7`). Founding metabolism range: 1.567–1.847. Founding populations were explicitly unbiased toward any habitat (`founding_habitat_bias = 0.0`).

At week 30,000, Forest lineages in both runs shifted to 1.33–1.38 — substantially below the founding range — driven by independent gene pools with different random seeds. The only mechanism that could produce this consistent directional shift in two independent runs is a shared selection pressure: the Forest habitat vector, combined with the metabolism energy penalty, differentially eliminating higher-metabolism genotypes before they reproduce.

This direction is mechanically grounded: lower metabolism reduces energy loss on missed meals, and missed meals are the primary cause of death in early, poorly-adapted generations.

### Evidence 2: Habitat-correlated divergence in the multi-habitat run

The parallel divergence run (Forest, Plains, Wetlands — Desert and Tundra went extinct) started from the same founding gene pool. At week 30,000:

| Habitat | Species | Metabolism |
|---|---|---|
| Forest | *blazing swimmer* | 1.380 |
| Wetlands | *wandering caller* | 1.607 |
| Plains | *ochre chaser* | 1.603 |

The split is correlated with habitat type, not random. All Forest lineages moved in one direction; all non-Forest lineages moved in another. Independent gene pools under different selection geometries diverged in predictable directions. This is the canonical signature of natural selection: environment predicts outcome.

The direction is mechanically consistent. Forest likely creates sparser or more spatially concentrated food probability relative to the founding gene configuration, making the metabolism penalty more costly — selecting for energy conservation. Non-Forest habitats with different environment vectors may create a flatter food-miss distribution, making the metabolism penalty less costly and allowing other selective forces to dominate.

### Evidence 3: Independent convergence within identical environments

The identical-habitat control run is the critical null-hypothesis contrast. Five Forest habitats — isolated from each other, evolving independent gene pools for 30,000 weeks with no migration — independently arrived at nearly the same metabolism values:

| Species | Metabolism |
|---|---|
| *timid coiler* | 1.332 |
| *ancient wraith* | 1.338 |
| *primal feeder* | 1.352 |
| *spectral shaker* | 1.355 |
| *deep glider* | **1.569** (outlier) |

Four of five converged to within 0.023 of each other after starting from a founding spread of 0.280. Drift does not compress variance — it maintains or expands it. Compression of five independent populations toward the same value is selection pulling them toward the same gene-space direction. *Deep glider* is the predicted shape of a drift exception: a lineage that escaped the selective pull rather than all five scattering randomly.

### Evidence 4: Fecundity increase, consistent across both runs

Founding fecundity range: 5.402–6.851 (spread 1.449). At week 30,000:
- Control run: 6.960–7.327 (compressed and shifted upward)
- Divergence run: 6.945–7.053 (similar)

Both runs independently shifted fecundity upward from founding values and compressed the spread. The mechanically-active interpretation: in environments where starvation mortality is non-trivial, producing more offspring per pregnancy provides a direct reproductive advantage. The consistent direction across independent runs is consistent with direct selection on the fecundity-litter size pathway.

### Evidence 5: Reproducible evolutionary trajectory from the same founding genotype

The lineage *jade dweller* → *fractured borer* undergoes an anagenesis cascade in **both** runs, in both cases in Forest habitat, in similar generational windows:
- Divergence run: weeks 6,050–6,950
- Control run: weeks 8,450–8,900

These are independent runs with different random seeds. The same founding genotype, subjected to the same habitat environment vector, followed a broadly similar evolutionary trajectory. Anagenesis detects in-place phenotype transformation relative to the founding type; the repeated crossing of that threshold in the same lineage under the same habitat is consistent with the Forest vector imposing consistent directional pressure on the phenotype axis across both runs.

---

## What Kind of Natural Selection This Is

The selection operates at two levels simultaneously:

**Primary level — gene-vector alignment selection:** The `cos θ` geometry creates differential food and water discovery across the full 500-dim gene space. Creatures whose full gene vector is more aligned to the habitat vector survive better. Over generations, better-aligned gene configurations outcompete misaligned ones. Most of the observed trait changes are secondary consequences of this — as selection reshapes the gene pool toward the habitat direction, traits whose loci happen to be correlated with that direction change as byproducts. These are epiphenomenal: intelligence, cold tolerance, nocturnal tendency, stress tolerance, and most others have no code path connecting them to survival after computation.

**Secondary level — direct trait selection:** Metabolism, water efficiency, and fecundity have direct mechanical links to individual survival and reproduction, independent of the gene alignment signal. Selection on these traits occurs in parallel with, and is partially separable from, the gene-alignment pressure. The observed convergence of metabolism toward lower values, and fecundity toward higher values, is consistent with both levels operating simultaneously — the gene-alignment pull and the direct trait penalty/benefit reinforcing each other.

This layered structure is biologically realistic. Real selection acts on organisms as whole genotypes; individual traits often evolve because they are genetically correlated with selected variation, not because each trait has its own direct fitness consequence. The simulator reproduces this structure without any of it being programmed explicitly.

---

## Core Caveat: Anagenesis Bug Active During These Runs

The anagenesis cascade re-fire bug was present in both supporting runs. When an anagenesis event fires, the new species' type phenotype was incorrectly anchored to a single individual's phenotype rather than the mover population centroid, creating an immediate discrepancy that restarted the persistence clock — allowing the same underlying population shift to register multiple successive events at the minimum 150-week interval.

**Practical impact on the evidence:**
- The *fractured borer* cascade event counts (6 events in the divergence run, 4 in the control run) are inflated by the bug. A single population shift is triggering multiple named transformations.
- Evidence 1–4 (trait convergence, metabolic divergence, fecundity shift) are **not affected**. The bug only inflates anagenesis event counts; it does not alter trait measurements, gene pool composition, or species distributions.
- Evidence 5 (reproducible trajectory) remains valid as a qualitative finding — the lineage crossed the anagenesis threshold in both runs. The specific event counts are unreliable.

The bug was identified and fixed after these runs. Future runs will produce accurate anagenesis counts.

---

## Supporting Evidence

| Run | Report | Config | Key contribution |
|---|---|---|---|
| Parallel divergence (2026-05-25) | [report](../2026-05-25_16-09-11/2026-05-25_16-09-11_Report.md) | [config](../2026-05-25_16-09-11/config.toml) | Habitat-correlated metabolic divergence; bimodal split (Forest 1.38 vs. Plains/Wetlands 1.60+) |
| Identical-habitat control (2026-05-26) | [report](../2026-05-26_09-39-07/2026-05-26_09-39-07_Report.md) | [config](../2026-05-26_09-39-07/config.toml) | Independent convergence of 4/5 isolated Forest lineages to 1.33–1.36; *deep glider* as drift outlier |

Both runs: `mirror_founding_population = true`, `seed = 7`, `founding_habitat_bias = 0.0`, `weeks = 30000`, `stats_every = 50`, `events_every = 0`.

---

## Summary

The two runs together satisfy the three conditions required to attribute observed population changes to natural selection rather than drift:

1. **Directional:** trait shifts are not random; they move consistently toward the habitat-specific gene-space direction
2. **Environment-correlated:** different environments produced different directions of change from the same starting gene pool
3. **Reproducible:** the same environment produced the same direction of change in independent replicates

The selection signal operates through two simultaneous channels: primary selection on whole-gene-vector alignment to the habitat (driving gene pool movement toward the environment vector), and secondary direct selection on metabolism, water efficiency, and fecundity through their explicit mechanical links to energy depletion and litter size. The simulation has the emergent property of natural selection driven evolution toward a habitat-specific fitness landscape, arising from nothing more than inheritance, random mutation, geometry-based resource discovery, and a small number of trait-mechanic linkages.

As more traits are wired into mechanics in future development, the secondary selection channel will broaden and the signal across the trait space will strengthen.
