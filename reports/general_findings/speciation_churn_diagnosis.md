# Speciation Churn: the Cladogenesis Count Measures Turnover, Not Diversity

**Date:** 2026-06-04
**Status:** Diagnostic finding — reference this before tuning the k-means split detector or interpreting cumulative species counts. Root cause identified and directly confirmed: the churn is **post-split re-convergence with no merge mechanism**, not over-eager splitting. Calibration showed a member-level isolation gate would not reduce churn (splits are already isolated at detection) — though keying the split gate on mating-success is still the more defensible design (see Section 3 / Takeaway). The actual churn fix (a merge mechanism) is not yet implemented, and a larger design question is open (single habitat optimum → adaptively-neutral sympatric divergence).
**Runs analysed:**
- `simulation_logs/2026-06-02_19-15-21/` — the 50,000-week 5-biome wheel that flagged attention (`ring_5biome_50k.toml`, seed 42, `stable_matching`).
- `simulation_logs/diag_speciation/2026-06-04_21-06-08/` — a purpose-built 2,500-week confirmation run with per-week event logging (`events_every = 1`). Config copy: `speciation_churn_diagnosis_diag_config.toml` in this folder.
- **Calibration pass** — the same 2,500-week trajectory re-run with `experiments/calibrate_split_isolation.py`, a read-only observer that measures each detected split's member-level interbreed fraction at detection time. (Reproducible because the run is now byte-deterministic; the observer never perturbs the simulation RNG.)

---

## Why we were looking

Runaway speciation is the model's recurring failure mode. The simulation has **no fitness function and no explicit speciation rule** — species are *detected*, not declared, and the detector has historically been the most fragile part of the system. An earlier version compared each newborn to a *frozen full-genome progenitor*, so ordinary whole-genome drift away from an ancient anchor speciated coherent, still-interbreeding populations (~2,150 "species" vs ~1,650 living individuals at one point). The fix was to re-key detection onto the 245-dim `compatibility_genes` subset — the same signal that gates mating — compared against **living** centroids refreshed periodically, so a whole population drifting together no longer trips a split. On top of that, on the current branch the newborn-level cladogenesis path was retired entirely, leaving the **k-means sub-cluster split detector as the sole cladogenesis mechanism** (plus the separate anagenesis detector for in-place phenotype transformation).

So when the 50k wheel run finished with a large cumulative species count, the question was not "is speciation happening" but "**is this detector over-firing the way its predecessors did?**" A model whose central scientific claim is *emergent* speciation loses its value if the speciation count is an artifact of a too-eager detector rather than a real biological signal. That is what prompted the investigation.

The headline number that drew the eye: **~1,470 species over the run**, roughly one speciation event every ~30 weeks across a standing population of only ~2,500–3,900 creatures. On its face that looks like the old runaway bug resurfacing.

---

## What we found

### 1. It is churn, not diversity

The alarming count is **cumulative** (every species ever registered). The quantity that actually matters — **standing diversity, the number of species alive at any given time** — is small, stable, and never grows:

| Metric | 50k wheel run | 2.5k confirmation run |
|---|---|---|
| Species ever (cumulative) | 1,471 | 81 |
| **Standing diversity (median alive)** | **15** (range 9–31) | **15** (range 8–30) |
| Standing diversity trend over the run | flat | flat |
| Living species at the end | 9 | 19 |

Fifteen species across five habitats (~3/habitat) is entirely reasonable. The cumulative count is large only because species are **minted and go extinct at a matched, constant rate**, leaving the standing count pinned at ~15. In the 50k run the split rate was dead constant at **~145 events per 5,000 weeks** for the entire run — no acceleration, no saturation. This is a steady-state turnover process, not diversity accumulation.

Supporting the turnover picture:
- **Ephemeral species.** 50k run: median species lifetime **200 weeks** (~5–8 generations at this run's ~25–40 wk/generation); half of all species live ≤200 weeks. 2.5k run: **71% of split-children go extinct** by the end, median child lifetime 350 weeks.
- **One mechanism only.** 100% of events are `cladogenesis_kmeans_subcluster`. **Anagenesis never fired once** in either run — a separate observation worth flagging (the phenotype-transformation detector may be effectively dormant under these conditions).
- **Pervasive, not pathological.** The 1,456 splits in the 50k run came from **565 distinct parent lineages** (the most prolific parent split only 14 times; 258 parents split exactly once). It is a population-wide behaviour of the detector, not a handful of runaway lineages.
- **Borderline promotions.** Promotion rate 62% (1,456 confirmed vs 881 candidates that evaporated before the gate). Even the failures reached median peak 21 members (above the 10-member gate) and died around 69 weeks (right at the 60-week gate). The two-stage gate is filtering, but the candidates it admits are marginal.

### 2. The split boundary is reproductively leaky — directly observed

The confirmation run logged every mating, so we could test the core question: **do the two halves of a "speciation" event actually stop interbreeding?**

- **47% of split-children (31 of 66) continue to interbreed with their own parent species *after* the split** — median 6 distinct weeks of cross-species fertilization, up to 39 weeks. Under the Biological Species Concept the model is built on, those are not separate species.

### 3. Where the leakiness comes from — and what it is NOT

The post-split hybridizations are not marginal pairs hovering near the 0.70 mating floor. They sit high on the compatibility scale:

| Post-split (parent ↔ child) hybridization compatibility | value |
|---|---|
| min | 0.793 |
| 10th percentile | 0.831 |
| **median** | **0.878** |
| 90th percentile | 0.929 |
| share below 0.75 (the split-isolation threshold) | **0%** |
| *same-species* fertilization median (for comparison) | 0.893 |

The cross-label pairs that interbreed sit at cosine **~0.88 — statistically indistinguishable from same-species matings.** The first, tempting reading of this was that the split detector's *centroid* test (`centroid_a · centroid_b < 0.75`) was too coarse — that two clusters with centroids 0.75 apart still contain many member pairs at ~0.88 that freely interbreed, so the detector was declaring isolation that didn't exist at the individual level. **We built a member-level test to fix exactly that, then calibrated it before wiring it in — and the calibration refuted the hypothesis.**

#### Calibration update: splits are reproductively isolated *at birth*

`experiments/calibrate_split_isolation.py` re-runs the same (deterministic) 2,500-week trajectory with a read-only observer. At every respeciate cadence it reconstructs each detected split and computes the **member-level interbreed fraction** — the exhaustive fraction of cross-cluster *opposite-sex* pairs that clear the real `is_compatible` gate (cosine ≥ `0.70 + 0.15·mean(selectivity)`), in both directions (parent♂×child♀ and child♂×parent♀). If the centroid test were admitting leaky splits, these fractions would be high. They are not:

| Member-level interbreed fraction **at detection** (260 accepted splits) | value |
|---|---|
| median | **0.000** |
| max | **0.0036** |
| splits with fraction > 0.05 | **0 / 260** |
| centroid cosine of those splits | 0.519 – 0.750 |
| max member overlap even at the threshold boundary (centroid cos ∈ [0.72, 0.75], n=110) | **0.0008** |

The metric is not stuck at zero — the same observer's forced-K=2 probe of *coherent* species spans the full 0→1 range (12% of probes above 0.5). It reports overlap when overlap exists. **At the split boundary there is none:** by the time two k-means centroids reach 0.75, their members already do not interbreed, so the centroid test and a member-level test *agree completely*. A member-level split gate at any threshold from 0.02 to 0.30 would have removed **zero** of the 260 splits — i.e. it is *redundant with the centroid test at the current 0.75 threshold*, and adding it does not reduce churn (the churn is post-split re-convergence; see below).

That redundancy is an argument **for** the member-level test, not against it. The actual cross-cluster mating-success rate is the *defensible* split criterion — it is exactly "can these two groups still breed", the same gate mating uses and the same primitive a merge check needs — whereas centroid distance is only a proxy that happens to be well-calibrated at 0.75. The natural design is to **demote the centroid test to a cheap, loose pre-filter and let the member-level mating-success rate be the authoritative split gate**, regardless of which broader direction we take. It would change nothing today but makes the detector principled and robust to future threshold/geometry changes.

#### The real mechanism: post-split re-convergence

The ~0.88 hybridizations therefore do **not** happen at detection — they happen *afterward*. Splits are genuinely isolated when born, then **re-converge**. Two independent facts pin this down:

- **260/260 accepted splits are isolated at detection** (interbreed fraction ≈ 0).
- **47% of those children interbreed with their parent later, at cosine ~0.88** (Section 2).

A split that is fully isolated at birth but interbreeding at near-same-species compatibility a few cadences later has *moved back together*. Why it re-converges is the subject of Section 4. The point for the detector is: **it is firing correctly.** The clusters it splits really have stopped interbreeding at the moment it splits them. The defect is not over-eager detection — it is that nothing ever **merges** a pair of labels back when they re-coalesce.

> **Correction to an earlier draft of this report.** A prior version of this section claimed the centroid test was too coarse and admitted still-interbreeding splits "at the wrong granularity." The calibration above refutes that: at detection the centroid and member-level tests agree (fraction ≈ 0 for all 260 splits). The leakiness is post-split re-convergence, not a granularity gap in the split test.

### 4. Why isolated-at-birth splits don't persist — the split is adaptively neutral

If every split is genuinely isolated when born (Section 3), why does roughly half re-converge with the parent and the rest go extinct? Because **the split confers no advantage and faces no force holding it apart.** And, importantly, this is **not** resource competition — there is no competition for resources in this model. Food and water are per-creature binomial draws against the habitat geometry (`P = (cos θ + 1)/2`); they are effectively unlimited, and one creature finding food does not deny it to another. The only population-dependent mortality is habitat-wide density predation (`PREDATION_ALPHA · N / POPULATION_SUPPORT`), which applies equally to every creature in the habitat regardless of species.

The split is detected in **compatibility space, not resource-adaptation space**, so the daughter is **adaptively identical to its parent**: same habitat, essentially the same food/water alignment, the same per-creature survival odds. Nothing selects for the daughter's distinctness and nothing penalises re-mixing, so the only forces acting on the compatibility genes are mutation and drift — which are exactly as likely to pull the two clusters back together as to push them further apart. Two outcomes follow, both leaving standing diversity untouched at ~15:

- **Re-convergence (~half).** The daughter's compatibility genes drift back toward the parent's, the two labels start interbreeding again at ~0.88 (Section 2/3), and gene flow homogenises them — but the species **label never merges back**, so the split stays on the books.
- **Drift to extinction (the rest).** A small, advantage-less sub-population subject to the same population-wide density predation as everyone else simply **drifts stochastically to extinction** before it re-converges.

Either way the cumulative count ratchets up while nothing ever comes off it. The deeper question this raises — whether a single habitat optimum makes *all* compatibility-space divergence adaptively neutral and therefore transient — is taken up in the open questions.

---

## Takeaway

The "high speciation rate" is **not** a return of the old drift-driven runaway bug, and standing diversity is healthy and stable. The detector is also **not over-eager**: calibration shows every split it makes is genuinely reproductively isolated at the moment it fires. What the large cumulative number actually exposes:

1. **There is no merge mechanism.** The species count only ratchets upward. Splits that are isolated at birth re-converge (~half interbreed with the parent again at ~0.88) or drift to extinction, but the label is never retired or merged. The BSC is non-monotonic; the model currently is not. **This is the fix to build** (and it does not depend on resolving the design question below).
2. **The split gate should key on actual mating-success, not centroid distance.** The calibration showed the member-level interbreed fraction is *redundant with* the centroid test at 0.75 — but it is the *defensible* criterion ("can these two groups still breed", the same gate mating uses and the same primitive the merge check needs). Adopt it: relax the centroid test to a cheap pre-filter and let the cross-cluster mating-success rate be the authoritative split gate. Changes nothing today; future-proofs the detector. A member-level *churn* fix it is not — that is the merge check.
3. **Reporting standing diversity, not cumulative species count**, would on its own make every past run look far healthier — a measurement/interpretation lesson as much as a modelling one.

Underneath all of this sits a larger design question (see open questions): because each habitat has a *single* optimal gene vector, divergence in compatibility space is adaptively neutral, so the simulation may be structurally incapable of *sustaining* a sympatric split regardless of how the detector is tuned.

---

## Open questions / candidate fixes (for discussion, not yet decided)

### The larger design question (the one to think hardest about)

**Does a single optimal gene vector per habitat make sympatric speciation structurally impossible — i.e. is the "gravity well" too strong?** Each habitat has one characteristic center vector, and resource discovery is `P = (cos θ + 1)/2` against it. That creates a single global optimum: every lineage in a habitat is pulled toward the *same* point in gene space. Two species that start near each other are therefore both descending the same gradient toward the same solution, so any compatibility-space divergence between them is **adaptively neutral** — selection neither rewards nor defends it. Under that geometry, a split has nothing to hold it open and re-convergence/extinction (Section 4) is the *expected* outcome, not a bug. If so, no amount of detector tuning or merge logic will produce *durable* sympatric divergence — the model can only sustain **allopatric** speciation (different habitats = different optima), and within-habitat splits will always churn.

This would be a foundational modelling decision rather than a detector fix. Directions to weigh (some already sketched in the Derek's-suggestions section of `docs/TODO.md`):
- **Multi-peak / rugged habitat fitness** (a bank of K peak vectors per habitat, `P = max_k (cos θ_k + 1)/2`): multiple coexisting local optima within one habitat would give a split somewhere to *go* and stay, enabling sympatric niche partitioning as an emergent outcome.
- **Frequency-dependence / resource partitioning**: make a creature's payoff depend on how crowded its part of gene space is, so being different is itself rewarded.
- **Or accept it as correct biology**: real sympatric speciation is hard and usually needs disruptive selection; maybe the model *should* only do allopatric speciation, and the only bug is the counting/merging.

Resolving this first may reframe everything below — if sustained sympatric splits aren't a goal, the merge check is the whole fix.

### Detector-level items

- **Merge / re-coalescence check** *(the indicated fix given the diagnosis)*: two living species whose member-level interbreed fraction (the validated `interbreed_fraction` primitive from the calibration script) rises above a high threshold collapse to the older name. The symmetric counterpart to the split detector; makes the count non-monotonic and absorbs the re-converging ~47%. Set the merge threshold well above the split gate for hysteresis so a borderline pair can't oscillate.
- **Make mating-success the authoritative split gate** *(keep — more elegant/defensible than centroid)*: relax the centroid test to a cheap loose pre-filter and gate the actual split on the cross-cluster `interbreed_fraction`. Calibration showed it is redundant at the current 0.75 centroid threshold, so it changes nothing today, but it ties the split decision to the real mating mechanism (same primitive as the merge check) and is robust to any later threshold/geometry change. Adopt no matter which broader direction we take.
- **Why does anagenesis never fire?** Zero anagenesis events in either run. Is the phenotype-drift detector reachable at all under selection-bounded drift toward a fixed habitat optimum, or is its threshold/persistence gate effectively unreachable? (Note this is *also* downstream of the single-optimum question: a single optimum bounds phenotype drift.)
- **Report standing diversity** alongside (or instead of) cumulative species count in `summary.json` / the visualizers, so runs are read correctly going forward.

---

## How to reproduce the analysis

- Standing vs cumulative diversity and lifetimes: scan `week_*.json` `global_species_distribution` (or sum per-habitat `species_distribution`) for nonzero species per snapshot; species lifetime = last_seen − first_seen.
- Event-type breakdown, promotion rate, parent prolificacy: `summary.json` → `all_speciation_events` (`event_type`, `parent_species`, `week`) and `all_failed_speciation_attempts` (`detected_week`, `failed_week`, `peak_members`).
- Leaky-boundary test (needs `events_every = 1`): for each split `(child, parent, week)`, scan later weeks' `mating_events` for entries with a `hybridization` block whose species pair == `{child, parent}`; the event's `compatibility_score` is the cross-cluster compatibility.
- **Detection-time isolation calibration:** `poetry run python experiments/calibrate_split_isolation.py reports/general_findings/speciation_churn_diagnosis_diag_config.toml --weeks 2500`. Read-only observer over the deterministic trajectory; writes `experiments/calib_output/<ts>/split_isolation_calibration.csv` (one row per detected split: `centroid_cos`, member-level `interbreed_frac`, sizes, `rep_creature_id`) and prints the at-detection distribution. Note the genes themselves are not in the JSON logs, so this requires re-running the sim (cheap and exact, since it is byte-reproducible) rather than post-processing the logs.
