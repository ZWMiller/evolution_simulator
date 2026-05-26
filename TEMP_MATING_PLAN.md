# Mating Pattern Refactor Plan

## Problem

The current zip-pairing mating system punishes minority species with a severe Allee effect.
All viable males and females in a habitat are shuffled together and zipped 1:1. A male from
species A paired with a female from species B produces a failed compatibility check and **neither
individual mates that week**. For a minority species with 5 pairs competing against a majority
of 50, expected same-species pairings drop to ~9% of capacity — the minority is driven extinct
much faster than differential adaptation alone would predict.

## Goal

Make the mating strategy **configurable** (one key in `[simulation]` config, or per-habitat
override). Implement four strategies. Zip-pairing is kept as the baseline/legacy option.

---

## Config Key

```toml
[simulation]
mating_strategy = "zip"  # options: "zip" | "species_priority" | "weighted_matrix" | "stable_matching"
```

The strategy name is passed through `SimulationRunner` → `Habitat.simulate_week()` (or a new
`Habitat.run_mating()` helper extracted from `simulate_week`).

---

## Strategy 0 — Zip (existing, baseline)

Shuffle all viable males, shuffle all viable females, zip them together.
For each pair run `is_compatible()`. If compatible, call `reproduce()`.

**Complexity:** O(N)
No changes needed. Kept as the default for reproducibility of old experiments.

---

## Strategy A — Species-First Priority with Cross-Species Spillover

### Algorithm

1. Group viable males and females by `creature.species`.
2. For each species (in random order), shuffle its males and its females independently,
   then zip and pair them (same compatibility check as zip). These pairings always pass
   the compatibility check (same species → cosine score near 1.0).
3. Collect all unpaired males and unpaired females (from unequal sex ratios, odd-sized
   species, or individuals whose within-species partner was already claimed).
4. Shuffle the spillover pools together and zip them — this is the cross-species
   hybridization pass, identical to the current zip strategy on the leftovers.

### Complexity
O(N) — one grouping pass + two zip passes.

### Pros
- Eliminates the minority penalty completely: each species mates proportional to its own sex ratio.
- Simple to implement and test.
- Hybridization still occurs for surplus individuals.

### Cons
- Hybridization is structurally deprioritized (leftovers only), not a continuous gradient.
- Selectivity trait is still used only as a binary threshold, not a preference signal.
- No mate choice — within-species pairings are still genetically random.

---

## Strategy B — Full Pairwise Score Matrix with Selectivity-Weighted Sampling

### Algorithm

1. Build the full M×F compatibility score matrix in one vectorized numpy operation:
   - `male_genes`:   (M, 245) — the `compatibility_genes` loci for each viable male
   - `female_genes`: (F, 245) — same for viable females
   - Raw scores: `(M, 245) @ (245, F)` (matrix multiply)
   - Normalize by L2 norms to get cosine similarities: score matrix `S` shape (M, F) ∈ [-1, 1]
2. For each female (iterated in random shuffled order):
   a. Retrieve her personal threshold: `t_f = COMPATIBILITY_FLOOR + 0.15 * female.selectivity`
   b. For each available male j: effective score = `max(0, S[j, f] − t_f)`
   c. Sharpness: `sharpness = 1 + k * (female.selectivity + male.selectivity) / 2`
      (suggested k=3 so sharpness ∈ [1, 4])
   d. Sampling weight for male j: `w_j = effective_score ^ sharpness`
   e. If all weights are zero (no compatible male), female goes unmated.
   f. Otherwise, sample one male j* from the weighted distribution (numpy.random.choice).
   g. Call `male.reproduce(female)` for the chosen pair.
   h. Remove male j* from the available pool.
3. Any remaining available males go unmated.

### Key behaviour of selectivity
- **Low selectivity** → sharpness ≈ 1 → nearly uniform distribution above the floor →
  hybrid pairings occur freely if cross-species scores clear the floor.
- **High selectivity** → sharpness ≈ 4 → strongly peaked at the highest-scoring available
  male → in practice mates same-species (highest cosine), rarely hybridizes.
- This makes hybridization an **evolved trait** rather than a hard toggle.

### Complexity
- Matrix build: O(M × F × 245) — vectorized; for habitat at POPULATION_SUPPORT=400
  (~200M × ~200F): `200 × 200 × 245 ≈ 9.8M` ops. Fast with numpy.
- Sampling: O(F × M) sequential — same order of magnitude.
- Overall: **O(N²)** in population size. Fine for capped habitats; watch at 2000+.

### Pros
- Selectivity becomes a continuous, evolvable hybridization propensity trait.
- No wasted pairings — every mate-attempt either succeeds or fails for a real reason.
- Fully vectorizable matrix build; fast in practice within current POPULATION_SUPPORT.
- Biologically richer: mate *preference* rather than random encounter.

### Cons
- O(N²) — bottleneck if habitat sizes grow large.
- Sequential sampling creates mild ordering artifacts (females shuffled first get first pick).
  Mitigate by doing two shuffle passes or running a second female-proposing round.
- Slightly more complex to implement and test.

### Implementation Notes
- Extract a `_build_compatibility_matrix(males, females)` helper to share between B and D.
- The `k` sharpness multiplier (suggested 3) should be a class constant on Habitat or a config knob.
- Sampling loop can be replaced with numpy batched weighted sampling if performance matters.

---

## Strategy D — Mutual Preference Stable Matching (Gale-Shapley Variant)

### Algorithm

1. Build the full M×F score matrix (same as Strategy B step 1).
2. Add small Gaussian noise (`σ ≈ 0.01`) to each score to break ties stochastically.
3. For each creature, apply its personal threshold: zero out scores below threshold.
4. Each creature's preference list is the rank-ordering of the opposite sex by score
   (descending), restricted to scores above zero.
5. Run **male-proposing Gale-Shapley**:
   - Each unmatched male with remaining candidates proposes to his top remaining female.
   - A free female accepts; an engaged female tentatively accepts if the proposer scores
     higher for her than her current partner, rejecting the old partner back to the pool.
   - Repeat until no unmatched male has remaining candidates.
6. All matched pairs call `reproduce()`.

### Selectivity integration
Selectivity raises the personal threshold (as today), shrinking a creature's preference list.
A high-selectivity creature proposes to / accepts from fewer individuals — effective
genetic protectiveness.

### Complexity
- Matrix build: O(M × F × 245) — same as Strategy B.
- Gale-Shapley: O(N²) worst case, O(N log N) for random inputs.
- Preference list sorting: O(N log N) per side.
- Overall: **O(N²)** with higher constant than B.

### Pros
- No individually "rational" deviation exists — the most efficient use of compatible pairings.
- Hybridization happens exactly when a cross-species individual is genuinely preferred over
  same-species alternatives.
- The most biologically motivated model of competitive mate choice.

### Cons
- Most complex to implement correctly (bookkeeping for proposals, engagements, rejections).
- Stability guarantee can **suppress hybridization**: a female who prefers a same-species
  male will never accept a cross-species proposer, even if nearly as compatible.
- Gale-Shapley is male-proposing → gives males their optimal stable match and females their
  pessimal. Running a second female-proposing pass and averaging is cleaner but doubles cost.
- Ties and creatures with empty preference lists need explicit handling.

### Implementation Notes
- Implement as `_gale_shapley(score_matrix, male_thresholds, female_thresholds)` returning
  a list of `(male_idx, female_idx)` pairs.
- Noise injection: `noisy = score_matrix + rng.normal(0, 0.01, score_matrix.shape)` before
  ranking; use original scores for the compatibility check, not the noisy ones.
- Consider offering a `stable_matching_proposer = "male" | "female" | "alternating"` sub-option.

---

## Implementation Checklist

### Phase 1 — Config plumbing + Strategy A (complete before B or D)
- [ ] Add `mating_strategy` key to `simulation.toml` (default `"zip"`)
- [ ] Thread the strategy string through `SimulationRunner.__init__` → store as `self.mating_strategy`
- [ ] Pass it into `Habitat.simulate_week()` as a parameter (or store on Habitat at construction)
- [ ] Extract the current zip mating block (lines ~593–636 of `habitat.py`) into
      `Habitat._mate_zip(viable_males, viable_females)` returning `list[dict]` (mating_events)
- [ ] Implement `Habitat._mate_species_priority(viable_males, viable_females)` (Strategy A)
- [ ] Add dispatch in `simulate_week` based on strategy string
- [ ] Add tests: minority species should survive longer under `species_priority` than `zip`

### Phase 2 — Strategy B
- [ ] Add `_build_compatibility_matrix(males, females)` vectorized helper to Habitat
- [ ] Implement `Habitat._mate_weighted_matrix(viable_males, viable_females)`
- [ ] Add `MATING_SHARPNESS_K` class constant to Habitat (default 3)
- [ ] Tests: check that low-selectivity populations hybridize at higher rates than high-selectivity

### Phase 3 — Strategy D
- [ ] Implement `_gale_shapley(score_matrix, male_thresholds, female_thresholds)` (standalone function)
- [ ] Implement `Habitat._mate_stable_matching(viable_males, viable_females)`
- [ ] Tests: verify no blocking pairs exist in output; verify cross-species matches only when
      outgroup score exceeds best same-species score

---

## Files to Touch

| File | Change |
|---|---|
| `evolution_simulator/habitat.py` | Extract zip → `_mate_zip`; add `_mate_species_priority`, `_mate_weighted_matrix`, `_mate_stable_matching`; dispatch in `simulate_week` |
| `evolution_simulator/simulation.py` | Read `mating_strategy` from config; pass to Habitat |
| `evolution_simulator/config/simulation.toml` | Add `mating_strategy = "zip"` under `[simulation]` |
| `tests/test_mating_strategies.py` | New test file covering all four strategies |
