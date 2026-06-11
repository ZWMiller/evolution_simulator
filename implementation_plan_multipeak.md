# Multi-Peak Habitats as Ecological Niches — Implementation Plan

## Purpose of this document

This is a standing implementation plan, written to be picked up cold. It
captures every design decision, the *reasoning* behind each (so the README and
docs can be updated later), the parts of the original single-peak proposal that
survive, a Phase 2 list of deferred ideas, and a self-contained note on the one
open risk (`POPULATION_SUPPORT` occupancy) so it can be checked without
reconstructing the conversation that produced this plan.

No code has been written yet. This is the conceptual + architectural spec.

---

## 1. Goal (and why it is broader than speciation)

Today every habitat has a single characteristic gene vector (CENTER). Food and
water discovery use `P = (cos θ + 1) / 2` between a creature's genes and CENTER.
This is a **single-peaked fitness landscape**: there is exactly one optimal way
to solve a habitat, and every creature in it is pulled toward the same genetic
structure across a large fraction of its loci. Two consequences:

1. **No niche diversity.** A habitat cannot be solved in qualitatively different
   ways. We never see "this lineage got large and low-fecundity because the
   strategy it locked onto happens to be tied to size and fecundity loci." Every
   survivor converges on one phenotype envelope.
2. **No sustained sympatric speciation.** Any divergence in compatibility space
   is adaptively neutral — both lineages are dragged back to the same optimum —
   so same-habitat splits always re-coalesce (documented in
   `reports/general_findings/speciation_churn_diagnosis.md`).

**The primary goal of this feature is niche diversity, not speciation.** We want
a habitat to support N genuinely different survival strategies, each with its own
emergent trait signature, that can coexist. Speciation becomes an *optional,
controllable downstream consequence* rather than the point.

Why decouple them: the two outcomes have different requirements.
- **Coexisting distinct niches** require a *stabilizing force* (frequency
  dependence — see §5) so multiple peaks stay occupied. The geometry can be
  modest.
- **Speciation** additionally requires the niche divergence to land in the
  *compatibility* loci (the 245-locus mating subset). That is a separate,
  controllable overlap (§7).

Decoupling is the central design move: niches come cheaply from geometry +
frequency dependence; speciation is opted into with one knob.

**Design intent:** multi-peak is opt-in per habitat. `n_peaks = 1` is the
default and is byte-identical to the current model. Most habitats stay
single-peaked. This keeps existing experiments valid and keeps sympatric
speciation rare (as it is in real biology).

---

## 2. Background facts verified in the codebase (do not re-derive)

- Resource probability lives in `Habitat._batch_resource_prob` /
  `food_likelihoods` / `water_likelihoods` (`habitat.py` ~lines 335–416). It
  slices `self.vector[FOOD_GENE_INDICES]` (158 loci) and
  `self.vector[WATER_GENE_INDICES]` (175 loci) and applies `(cos θ + 1)/2`.
- Habitat vectors: `self.vector = CENTER + per-instance Gaussian noise`. CENTER
  is a class attribute generated from `TYPE_SEED` via `__init_subclass__`
  (`habitats/types.py`). Gene/vector entries are standard normal.
- **Trait loci overlap the food/water loci.** This is the fact that makes
  "peak → trait signature" real. Confirmed (`genetics.py`
  `DEFAULT_TRAIT_GENE_INDICES`):
  - `size`: 32, 59, 109, 159, 160, 262, 314, 326, 409 …
  - `metabolism`: 32, 34, 36, 63, 113, 164, 165, 267, 330, 413 …
  - `water_efficiency`: 38, 65, 115, 167, 269, 332, 415 …
  - `foraging_ability`: 37, 64, 114, 166, 268, 331, 414 …
  - `intelligence`: 31, 37, 74, 124, 176, 280, 341, 428 …
  - plus `speed`, `strength`, etc.
  - `FOOD_GENE_INDICES = range(37,80) ∪ range(110,170) ∪ range(230,285)`;
    `WATER_GENE_INDICES = range(38,78) ∪ range(115,175) ∪ range(270,345)`.
  - So resampling niche loci inside the food/water subspace *necessarily*
    perturbs size, metabolism, water_efficiency, foraging, speed, strength.
- **Food/water are currently unlimited** — per-creature independent binomial
  draws, no inter-species competition (only density predation is
  population-dependent). §5 deliberately changes this for multi-peak habitats.
- `compatibility_genes` = 245 loci: `range(80,130) ∪ range(140,190) ∪
  range(220,270) ∪ range(285,340) ∪ range(390,430)`. Food∩compat ≈ 90/158,
  water∩compat ≈ 105/175 (~57–60% overlap). Relevant to §7.
- Speciation detection is unchanged by this feature (see §7). Detectors live in
  `species.py` (`detect_subcluster_splits`, `detect_anagenesis`).

---

## 3. The geometry: what `niche_fraction` is, and the N-peak limit

A **peak** is the habitat's base vector (`self.vector`) with a *subset* of its
food/water loci overwritten by different values. `niche_fraction = f` is **the
fraction of the food/water loci allowed to differ between peaks**; the other
`(1 − f)` stay equal to the base vector across all peaks (they carry the shared
biome identity).

`f` is the **distinctness ↔ survivability** dial:
- `f` small → peaks nearly identical → niches barely differ (weak feature).
- `f` large → peaks strongly distinct, but a creature adapted to *no* peak (a
  fresh founder/migrant near the biome center) finds less food, because more of
  the subspace now demands peak-specific genes.

### Key quantitative result (governs every parameter choice)

Place the N peaks as far apart as the geometry allows (see §4 — a centered
simplex on the niche loci, pairwise niche cosine `−1/(N−1)`). Decomposing the
cosine into a shared component (norm² ∝ `(1−f)`) and a niche component (norm² ∝
`f`):

> **cross-peak cosine = `(1 − f) − f/(N − 1)`**

Setting this ≤ 0 (peaks at least orthogonal = genuinely different strategies)
gives the design rule:

> **N ≤ 1 / (1 − f)**, equivalently **`f ≥ (N − 1)/N`** for N distinct peaks.

| `f`  | max distinct peaks N | generalist-at-center food P = `(2−f)/2` |
|------|----------------------|------------------------------------------|
| 0.50 | 2 | 0.75 |
| 0.67 | 3 | 0.67 |
| 0.75 | 4 | 0.63 |
| 0.80 | 5 | 0.60 |

This is the answer to "how many peaks can we meaningfully support": distinctness
for N peaks needs `f ≥ (N−1)/N`. Want 4 real niches → `f ≈ 0.75`, and you accept
that an unadapted generalist forages at P≈0.63 until it commits to a peak. This
is a per-habitat config trade, not a global constant.

**Why the cosine geometry alone cannot carry this feature** (this is *why* §5
exists): even with maximally-separated peaks, the *valley* between two adjacent
peaks is shallow. A creature at the midpoint of two peaks has cosine
`sqrt(1 − f/2)` to each (e.g. f=0.5 → 0.87 → P=0.93; f=1.0 → 0.71 → P=0.85). The
disruptive-selection penalty for being an intermediate is only ~5–15% even in
the extreme. That is too close to neutral to *maintain* two occupied peaks on
its own — drift will collapse the population onto whichever peak it wanders to.
The geometry provides **trait distinctness**; the **coexistence force comes from
frequency dependence (§5)**, not from the landscape shape. Do not try to fix
coexistence by deepening the cosine valley — it geometrically caps at ~15%.

---

## 4. Peak generation (supports any N, anti-aligned)

At habitat construction, when `n_peaks > 1`:

1. Build the base vector `self.vector = CENTER + per-instance noise` exactly as
   today. **Peaks are built from `self.vector`, not the class `CENTER`**, so the
   `n_peaks = 1` case is byte-identical to current behavior (critical backward-
   compat seam).
2. **Choose niche loci.** Sample a subset of the food/water loci of size
   `round(f · |subspace|)`. The selection seed is derived deterministically from
   `TYPE_SEED` so runs are reproducible. Use **joint peaks**: one set of niche
   loci spanning the union of the food and water subspaces, so a peak is a single
   holistic strategy rather than independent food-strategy × water-strategy
   combinations. Niche-loci selection is biased by `niche_compat_overlap` (§7).
3. **Place N peaks as a centered simplex on the niche loci.** Draw N independent
   random niche sub-vectors, then **subtract their mean** so they sum to zero.
   Centering N roughly-orthogonal equal-norm vectors forces pairwise cosine to
   ≈ `−1/(N−1)` — a regular simplex. This is how the 2-peak "anti-aligned"
   intuition generalizes to any N in one operation:
   - N=2 → cosine −1 (true opposition)
   - N=3 → −0.5, N=4 → −0.33, degrading toward 0 as N grows (which is exactly
     why the N table in §3 caps out).
   Fallback `peak_separation = independent` skips centering (peaks ~orthogonal on
   the niche part) for large N where the simplex separation is negligible anyway.
4. **Assemble each peak:** copy `self.vector`; overwrite the niche loci with that
   peak's simplex sub-vector. Non-niche loci keep the base value across all
   peaks (shared biome identity). Store as `self.peaks` (shape `N × 500`, or the
   sliced food/water rows for vectorized scoring).

When `n_peaks = 1`, `self.peaks` has one row equal to `self.vector`; no behavior
changes.

**Why joint peaks (commit-to-best), not separate food/water peaks:** a creature
*commits* to its best single peak and has both its food and water finding scored
against that one peak's vector. This keeps a niche legible as one strategy ("the
heavy-bodied, water-thrifty niche") and avoids a combinatorial
food-peak × water-peak explosion that would be hard to read in the output. The
specialization trade-off (committing to a food-optimized peak may cost water
finding) is intentional and biological.

---

## 5. Frequency dependence (the force that makes niches coexist)

Without this, N peaks are N equivalent optima and the population drifts onto one
— no coexistence. The fix: **each peak is a limited resource pool**, so crowding
a peak makes it worse (negative frequency dependence). This is the mechanism that
turns "N optima" into "N occupied niches," and it is the textbook driver of
stable niche partitioning and competitive exclusion.

**Per habitat, each week, at the start of the resource step:**

1. **Assign** each living creature to its committed peak `k` = argmax over peaks
   of a combined alignment score (mean of food-cosine and water-cosine to that
   peak). MAX-by-commitment.
2. Each peak `k` has a weekly **capacity** `C_k`. Default
   `C_k = POPULATION_SUPPORT / n_peaks` (total capacity unchanged vs. today),
   overridable per peak so a habitat can have a rich peak + a marginal peak.
3. **Demand** `D_k` = number of creatures committed to peak `k`.
4. **Scale** each creature's food/water probability by occupancy, via one of two
   modes:

   - **`scramble` (egalitarian — the stable default).** Everyone on peak `k`
     gets `p_eff = p_geom × min(1, C_k / D_k)`. Smooth, robust, pure negative
     frequency dependence. Rare peak → no penalty; crowded peak → everyone's `p`
     dilutes proportionally. Ship this first to get coexistence dynamics stable.

   - **`contest` (intelligence-ranked — the trait-coupling).** Sort peak-`k`
     creatures by a priority score (default `intelligence`, plus tie-break
     noise). The top `C_k` get full `p_geom`; the overflow scavenges at
     `p_geom × spillover_floor` (e.g. 0.3). This makes a currently-inert trait
     (intelligence) directly fitness-relevant *under crowding* — smart creatures
     hold the good spots, others get pushed to the margin. Cost: a knife-edge at
     the `C_k` cutoff; can be softened later to a priority-weighted share. Layer
     this on after scramble is validated.

5. Apply the existing per-creature binomial draw with `p_eff`.

**Why this is the right design:**
- It supplies the *stabilizing* force the unlimited-resource model lacks. Real
  sympatric divergence (Dieckmann–Doebeli) needs disruptive selection *plus*
  competition for a shared limiting resource. The cosine valley gives weak
  disruptive selection; this gives the competition.
- It makes `intelligence` earn its keep (it is otherwise only a future
  "predation offset" idea in the TODO).
- Capacity is **per-peak, shared across all species on that peak**, which
  introduces the **first interspecific competition** in the simulation: two
  species crowding the same niche genuinely compete, which is what produces
  competitive exclusion and stable partitioning.

**Deliberate semantic change — confirm before building.** This breaks the
current invariant "food/water are unlimited, no inter-species competition." That
invariant is recorded in project memory. The change is intentional and confined
to multi-peak habitats (`n_peaks > 1` only); single-peak habitats keep the old
unlimited model untouched.

---

## 6. Trait signatures (making niches readable)

Because niche loci overlap the active-trait loci (§2), each peak carries an
emergent phenotype signature — but we should make it strong and observable:

- **Optionally bias niche-loci selection toward the active-trait loci** (size,
  metabolism, water_efficiency, fecundity, speed, strength, foraging). This
  guarantees peaks differ in the traits we care about, so we reliably get
  "peak 2 = large + low-fecundity" rather than peaks that differ only on
  biologically-inert loci.
- **Log per-peak phenotype centroids** each stats interval: the mean trait
  vector of creatures committed to each peak. This is the payoff — it lets us
  *see* niche partitioning ("peak 1 trended small/fast/fecund, peak 2 trended
  large/slow/thrifty") as a time series. This is the observability that makes
  the whole feature worth building.

---

## 7. Speciation coupling — one knob, on or off

Whether niche divergence *also* causes reproductive isolation depends entirely on
whether the niche loci overlap the 245 `compatibility_genes`. Make it explicit
via `niche_compat_overlap`:

- **`prefer`** → niche loci drawn to include compatibility loci → adapting to
  different peaks drifts the compat cosine apart → the existing
  `detect_subcluster_splits` eventually fires → sympatric speciation (rare,
  emergent, through the same Biological-Species-Concept filter as all other
  splits).
- **`avoid`** → niche loci drawn to skip compatibility loci → niches diverge in
  ecology and traits but stay fully interfertile → stable within-species
  polymorphism / ecotypes (a coastal vs inland morph that still interbreeds).
- **`neutral`** (default) → niche loci sampled without regard to compat overlap;
  speciation happens to the extent the random overlap allows.

**Why this is valuable:** `avoid` gives the "niches *without* speciation" case
you explicitly want to study; `prefer` turns the speciation pathway on when
wanted. **No changes to `species.py` are required** — the existing split detector
keys on the compat subset and will fire (or not) based purely on how much niche
divergence reaches those loci.

---

## 8. Config surface

Per `[[habitats.instances]]` (with `[habitats]` defaults so existing configs need
no edits; all defaults recover current behavior):

| Key | Default | Meaning |
|---|---|---|
| `n_peaks` | `1` | number of peaks; `1` = identical to today |
| `niche_fraction` | `0.5` | fraction of food/water loci that differ between peaks; pick from the §3 N-table |
| `peak_separation` | `simplex` | `simplex` (anti-aligned, centered) or `independent` (orthogonal, for large N) |
| `peak_capacities` | `POPULATION_SUPPORT / n_peaks` | per-peak weekly capacity; list-overridable for unequal niches |
| `competition_mode` | `scramble` | `scramble` (egalitarian) or `contest` (ranked) |
| `contest_priority_trait` | `intelligence` | trait used to rank under `contest` |
| `spillover_floor` | `0.3` | overflow multiplier under `contest` |
| `niche_compat_overlap` | `neutral` | `neutral` / `avoid` / `prefer` (§7) |

---

## 9. Implementation order (non-code spec)

1. **Config plumbing.** Add the §8 keys to the TOML loader and
   `[[habitats.instances]]` handling, with defaults in `[habitats]`. Verify
   `population_support`-style per-instance override path is reused.
2. **Peak construction** (§4) in habitat init: build `self.peaks` from
   `self.vector`. Guard so `n_peaks = 1` produces exactly the current single
   vector (byte-identical regression).
3. **Resource scoring** (§4 + §5): generalize `_batch_resource_prob` to score the
   `(N, K)` creature×peak cosine matrix (single matmul + norm division), commit
   each creature to argmax, then apply the frequency-dependence scaling. Keep
   `n_peaks = 1` on the existing code path (no capacity logic).
4. **Frequency dependence** (§5): `scramble` first, then `contest`.
5. **Observability** (§6): per-peak phenotype centroid logging + record `n_peaks`
   / `niche_fraction` per habitat in `summary.json`.
6. **Speciation overlap knob** (§7): wire `niche_compat_overlap` into niche-loci
   selection. No `species.py` changes.
7. **Tests:** `n_peaks=1` regression (identical probs); `n_peaks=2` distinct P by
   committed peak; cross-peak cosine matches `(1−f) − f/(N−1)`; capacity rationing
   reduces P when `D_k > C_k`; `contest` ranks by priority trait; config
   round-trips.

### Backward-compat seams (must preserve)
- Peaks built from `self.vector` (not class `CENTER`).
- `n_peaks = 1` skips the frequency-dependence layer entirely.
- No changes to `creature.py`, `species.py`, `simulation.py`, `runner.py`, or the
  visualizers for the core feature.

---

## 10. Decisions locked in (and why)

1. **Niches first, speciation second.** Easier to hit, more generally useful, and
   the two have different requirements (§1).
2. **Joint peaks, commit-to-best.** Legible single-strategy niches; avoids
   food×water combinatorial explosion (§4).
3. **MAX-by-commitment aggregation now.** Simple; the additive-cosine
   formulations clip/compress contrast among well-adapted creatures (rejected —
   see Phase 2 for the right home for a generalist benefit).
4. **Centered-simplex peak placement.** Generalizes anti-alignment to any N in
   one operation (§4).
5. **Frequency dependence is the coexistence force, not the geometry.** The
   cosine valley caps at ~15% depth; coexistence must come from limited per-peak
   resources (§3, §5).
6. **`scramble` default, `contest` opt-in.** Stability first, then activate
   `intelligence` (§5).
7. **Per-peak capacity shared across species → new interspecific competition.**
   Deliberate departure from the "unlimited food" invariant, confined to
   multi-peak habitats (§5).
8. **Speciation as an opt-in `niche_compat_overlap` knob, default neutral.** No
   `species.py` changes (§7).

## 11. Kept from the original single-peak proposal

- The core insight that **peaks must conflict at the *same* loci** (shared niche
  loci with different values), not resample disjoint loci. Disjoint loci create
  "additive niches" with no trade-off — a creature could be mediocre at all of
  them simultaneously — which produces coexisting generalists, not distinct
  strategies. (This survives as the niche-loci/simplex construction in §4.)
- Rejecting fully-independent per-peak random centers (Option A): peaks should
  share a biome identity, and fully-independent centers create fatal valleys.
  Shared non-niche loci provide that identity.
- The backward-compatibility philosophy: `n_peaks = 1` ≡ current model; new keys
  default to recover existing behavior; existing configs/tests/logs untouched.
- Vectorized resource scoring: the `(N, K)` cosine matrix is one matmul + norm
  division; K is small (2–5) so cost is negligible.
- Per-habitat `niche_fraction` / niche loci are *intended* to differ between
  biomes (a forest's strategic axes differ from a desert's), seeded
  deterministically from `TYPE_SEED`.

---

## 12. Phase 2 — deferred ideas to carry forward

- **Reconsider the angular response function (`(cos θ + 1)/2`) — foundational,
  would relax the §3 N-limit.** The peak-count ceiling `N ≤ 1/(1−f)` and the
  ~15% valley-depth cap (§3) are **not fundamental** — they are artifacts of the
  *response curve* being linear in `cos θ` and centered at 0.5 for orthogonal
  vectors. `(cos θ + 1)/2` is forgiving precisely where it matters: an orthogonal
  creature still gets P=0.5 (half credit), so mere orthogonality between peaks is
  not penalized, which is *why* the design is forced to push peaks all the way to
  negative cosine (anti-alignment, the simplex in §4) just to make them distinct.
  Cosine was the first and simplest way to drive the geometric habitat↔creature
  identity; it does not have to be the final one.

  *What a sharper response buys us:* if the response drops steeply near
  orthogonality, two *orthogonal* peaks already produce a deep valley (an
  orthogonal creature scores near 0, not 0.5). Then:
  - **Peak count decouples from `niche_fraction`.** You no longer need
    `cos ≤ 0` between peaks; orthogonal suffices. A 500-D space holds ~500 truly
    orthogonal directions (far more nearly-orthogonal), so you can pack many
    distinct peaks at *modest* `f` (better survival floor) instead of being
    capped at `1/(1−f)`. The §3 table is an artifact of the current map, not a
    law.
  - **Valley depth becomes a tunable lever** instead of a fixed ~15% ceiling,
    which means frequency dependence (§5) is no longer the *only* available
    coexistence force — geometry can contribute too.
  - **A real new biome axis:** the sharpness is itself a per-habitat lever. Low
    sharpness = forgiving, generalist-friendly environment (current behavior);
    high sharpness = harsh, specialist-only environment with many narrow niches
    (a stable rainforest vs. a homogeneous marginal habitat). This is exactly the
    "more levers for the approximation" goal — an oblate-spheroid cow that models
    specific, scientifically interesting regimes the spherical cow cannot.

  *Candidate formulations (all keep the geometric identity — still a monotonic
  function of `cos θ`, aligned→high / orthogonal→low / anti→0; no sin/cross-
  product, no explicit fitness landscape):*
  1. **Exponent (minimal, backward-compatible):**
     `P = ((cos θ + 1)/2) ** resource_sharpness`, default `1.0` (recovers current
     behavior *exactly*). `>1` sharpens. Endpoints preserved (cos=1→1, cos=−1→0);
     only the middle steepens. One knob. Caveat: `>1` lowers the whole curve
     below cos=1, so it needs energy retuning (per-habitat `FOOD_ENERGY_GAIN`).
  2. **Sigmoid in cos (most flexible):**
     `P = 1 / (1 + exp(−k·(cos θ − c)))` — two levers: a half-max **threshold**
     `c` (place the cliff at any angle) and **steepness** `k`. Lets you put
     selective resolution exactly where the population actually sits (see note
     below).
  3. **von Mises–Fisher concentration (most principled):** the directional-
     statistics analog of a Gaussian on the sphere, `∝ exp(κ·cos θ)`, with
     concentration `κ` as the single sensitivity parameter (κ→0 uniform/forgiving,
     κ→∞ only perfect alignment survives). This is the "correct" probabilistic
     object for angular dispersion if we want to be principled.

  *Why this is worth doing (resolution argument):* in high-D, random gene vectors
  concentrate near orthogonal (cos std ≈ 1/√D ≈ 0.045 for the 245-dim compat
  space; similar for food/water), so the population lives in a *narrow band* of
  cosine and the linear map wastes most of its `[0,1]` range on configurations
  that never occur. A response shaped to expand resolution where creatures
  actually sit (high cos, near their current adaptation) gives more selective
  contrast for the same genetic change.

  *Constraints / interactions to respect:*
  - **This revisits a documented core invariant.** CLAUDE.md lists
    "`(cos θ + 1)/2` resource geometry is central — don't replace it with
    cross-product/sin or explicit fitness scores." The spirit (angle drives
    probability, no explicit fitness function) is *preserved* by all three
    candidates; only the response *curve* is generalized. Update the invariant
    wording deliberately, framing it as "angular alignment drives probability via
    a tunable monotonic response," not a silent swap.
  - **Decouples cleanly from mating/compatibility geometry.** Resource response
    and mate-compatibility both read a cosine, but they are *separate* functions
    of *separate* cos values. We can sharpen the resource response while leaving
    `COMPATIBILITY_FLOOR = 0.70` and the split thresholds on raw cos untouched, so
    no speciation recalibration is forced by this change.
  - Default must recover current behavior (sharpness=1 / κ matched), and likely
    pairs with per-habitat energy retuning since sharper curves shift mean P.

- **Generalist benefit in the resource layer (not the cosine).** A creature with
  decent alignment to *two* peaks should be able to draw from *both* peaks'
  resource pools. Under crowding this is a real advantage (diversified foraging
  when every niche is saturated) and pays nothing when specialists have empty
  niches to themselves. This is the biologically-true "exploit multiple peaks
  without being a pure generalist" benefit — and the resource pool, not the
  cosine aggregation, is the correct home for it. Revisit after the commit-to-
  best model is validated.
- **Soften `contest` mode.** Replace the hard `C_k` cutoff with a
  priority-weighted share so allocation is continuous rather than knife-edge.
- **Cost coupling for the priority trait.** If `contest` drives `intelligence`
  (or whatever `contest_priority_trait` is) to the ceiling, couple it to a
  metabolic cost (smart is expensive) — expressible through existing trait loci —
  so there is a real trade-off rather than a free advantage.
- **Soft peak assignment.** Replace hard argmax commitment with a soft
  membership weight to multiple peaks. Smoother dynamics, but muddier niche
  legibility; only pursue if hard assignment produces jarring flip behavior.
- **Per-peak unequal capacities as a first-class experiment.** A rich peak + a
  marginal peak (set `peak_capacities` unequal) to study source-sink niche
  dynamics within a single habitat.
- **Merge / re-coalescence check (independent but synergistic).** The species
  count is still monotonic; interfertile species that drift back together should
  re-merge. This is the symmetric counterpart to the split detector and is the
  primary fix for speciation *churn*; multi-peak makes splits more durable, the
  merge check absorbs the ones that re-converge. Tracked separately in TODO.

---

## 13. How to find `POPULATION_SUPPORT` (the one open risk — check before building §5)

**The risk in one sentence:** capacity-based frequency dependence (§5) only bites
if populations actually press against `POPULATION_SUPPORT`. If habitats normally
sit *well under* their cap, then `C_k = POPULATION_SUPPORT / n_peaks` never
constrains anyone, no peak is ever "crowded," and the entire coexistence
mechanism is silently inert — peaks would just drift to one.

**What `POPULATION_SUPPORT` is:** a per-habitat-type class attribute (settable via
`population_support` in `[[habitats.instances]]`) that caps/scales the population;
it feeds the density-dependent predation term (higher population relative to
support → higher effective predation). It is the closest thing the model has to a
carrying capacity.

**How to check occupancy before committing to capacity-based competition:**
1. Take an existing long run's `summary.json` / `week_NNNNN.json` logs (e.g. one
   of the bundled long-run configs).
2. For each habitat, pull the per-week population and the configured
   `POPULATION_SUPPORT`.
3. Compute the **equilibrium occupancy ratio** = mean steady-state population /
   `POPULATION_SUPPORT` per habitat.
4. Interpretation:
   - Ratio near or above ~0.7–1.0 → populations do press the cap → capacity-based
     frequency dependence will bite → §5 works as designed with
     `C_k = SUPPORT/n_peaks`.
   - Ratio well below (~0.3 or less) → the cap is slack → §5 as written will be
     inert. **Mitigation:** set `C_k` from the *observed equilibrium* population,
     not the nominal cap (e.g. `C_k = (observed_equilibrium / n_peaks) × tightness`
     with a `tightness < 1` knob), so the per-peak pool is actually a binding
     constraint. The frequency-dependence math is unchanged; only the source of
     `C_k` changes.

**Decision:** run this occupancy check first. If habitats run slack, switch the
`C_k` default from "nominal cap / N" to "observed equilibrium / N × tightness"
before implementing §5. This is the highest-leverage thing to verify — it is the
one assumption that can quietly make the whole feature do nothing.
