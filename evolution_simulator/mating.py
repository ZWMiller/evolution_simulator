"""
Mating strategies: who gets paired with whom.

All four pairing algorithms plus their helpers live here as standalone
functions.  ``Habitat.simulate_week`` dispatches to one of the ``_mate_*``
functions by the configured ``mating_strategy`` name.  The strategies have no
state dependency on ``Habitat`` beyond the viable male/female lists and a couple
of constants:

  - ``Creature.COMPATIBILITY_FLOOR`` — read off the creature class.
  - ``MATING_SHARPNESS_K`` — a ``Habitat`` (subclass-overridable) attribute, so
    ``_mate_weighted_matrix`` takes it as an explicit ``mating_sharpness_k``
    argument rather than reaching back into the habitat.

The strategy only determines *who pairs with whom*; conception probability and
litter size are always downstream, inside ``_attempt_mating`` ->
``is_compatible`` / ``reproduce``.
"""

from collections import deque

import numpy as np

from .creature import Creature
from .genetics import DEFAULT_TRAIT_GENE_INDICES


def _build_compatibility_matrix(males: list, females: list) -> np.ndarray:
    """
    Vectorized pairwise cosine-similarity matrix on the compatibility_genes subset.

    Returns an (M, F) float64 array with values in [-1, 1], where entry [i, j]
    is the compatibility score between males[i] and females[j].  Shared by the
    weighted-sampling and stable-matching mating strategies.
    """
    indices = DEFAULT_TRAIT_GENE_INDICES["compatibility_genes"]

    male_mat = np.stack([m.genes[indices] for m in males])  # (M, 245)
    female_mat = np.stack([f.genes[indices] for f in females])  # (F, 245)

    dots = male_mat @ female_mat.T  # (M, F)

    male_norms = np.linalg.norm(male_mat, axis=1, keepdims=True)  # (M, 1)
    female_norms = np.linalg.norm(female_mat, axis=1, keepdims=True)  # (F, 1)
    denom = male_norms @ female_norms.T  # (M, F)

    safe = denom > 1e-10
    scores = np.where(safe, dots / np.where(safe, denom, 1.0), 0.0)
    return np.clip(scores, -1.0, 1.0)


def _attempt_mating(male, female, rng: np.random.Generator) -> dict:
    """Run one compatibility check + optional reproduce(); return the event dict."""
    compatible, score, reason = male.is_compatible(female)
    event: dict = {
        "male_id": male.creature_id,
        "female_id": female.creature_id,
        "compatibility_score": round(float(score), 4),
        "compatible": compatible,
        "fertilized": False,
        "litter_size": 0,
        "offspring_ids": [],
    }
    if compatible:
        litter = male.reproduce(female, rng)
        if litter:
            event["fertilized"] = True
            event["litter_size"] = len(litter)
            event["offspring_ids"] = [c.creature_id for c in litter]
            if male.species != female.species:
                event["hybridization"] = {
                    "male_species": male.species,
                    "female_species": female.species,
                }
        else:
            event["reason"] = "infertile"
    else:
        event["reason"] = reason
    return event


def _mate_zip(viable_males: list, viable_females: list, rng: np.random.Generator) -> list[dict]:
    """
    Legacy zip-pairing strategy.

    Shuffles all viable males and females habitat-wide, then zips them into 1:1
    pairs.  Any cross-species pairing that fails the compatibility check wastes
    both individuals' mating opportunity for that week, which creates a severe
    minority-species penalty (Allee effect).
    """
    rng.shuffle(viable_males)
    rng.shuffle(viable_females)
    mating_events: list[dict] = []
    for male, female in zip(viable_males, viable_females, strict=False):
        event = _attempt_mating(male, female, rng)
        mating_events.append(event)
    return mating_events


def _mate_species_priority(viable_males: list, viable_females: list, rng: np.random.Generator) -> list[dict]:
    """
    Species-first priority pairing with cross-species spillover.

    1. Group males and females by species.
    2. Pair each species' own males and females first (shuffled within-species).
    3. Collect all unpaired individuals into a shared spillover pool.
    4. Zip-pair the spillover pool for cross-species hybridisation.

    Eliminates the minority-species Allee effect: each species gets mating
    proportional to its own sex ratio, regardless of relative abundance.
    Hybridisation is still possible for surplus/unmatched individuals.
    """
    from collections import defaultdict

    males_by_species: dict[str, list] = defaultdict(list)
    females_by_species: dict[str, list] = defaultdict(list)
    for m in viable_males:
        males_by_species[m.species].append(m)
    for f in viable_females:
        females_by_species[f.species].append(f)

    mating_events: list[dict] = []
    spillover_males: list = []
    spillover_females: list = []

    # sorted() gives a canonical order independent of set/hash iteration order
    # (which varies per process); the seeded shuffle then yields a deterministic
    # permutation, so the pairing is reproducible run-to-run.
    all_species = set(males_by_species) | set(females_by_species)
    species_order = sorted(all_species)
    rng.shuffle(species_order)

    for sp in species_order:
        sp_males = males_by_species.get(sp, [])
        sp_females = females_by_species.get(sp, [])
        rng.shuffle(sp_males)
        rng.shuffle(sp_females)
        for male, female in zip(sp_males, sp_females, strict=False):
            event = _attempt_mating(male, female, rng)
            mating_events.append(event)
        # Surplus individuals go to spillover
        n_paired = min(len(sp_males), len(sp_females))
        spillover_males.extend(sp_males[n_paired:])
        spillover_females.extend(sp_females[n_paired:])

    # Cross-species hybridisation pass on leftovers
    rng.shuffle(spillover_males)
    rng.shuffle(spillover_females)
    for male, female in zip(spillover_males, spillover_females, strict=False):
        event = _attempt_mating(male, female, rng)
        mating_events.append(event)

    return mating_events


def _mate_weighted_matrix(
    viable_males: list,
    viable_females: list,
    mating_sharpness_k: float,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Full pairwise score matrix with selectivity-weighted sampling.

    1. Build the M×F compatibility score matrix in one vectorized pass.
    2. Iterate females in random order; each female samples a male using:
         weight_j = max(0, score[j,f] − threshold_f) ^ sharpness
       where threshold_f = COMPATIBILITY_FLOOR + 0.15 * female.selectivity
       and sharpness = 1 + mating_sharpness_k * mean(male.selectivity, female.selectivity)
    3. If no available male clears the female's threshold she goes unmated.
    4. The sampled male is removed from the pool (no double-mating).

    Low selectivity → exponent ≈ 1 → nearly uniform above the floor → liberal
    hybridisation.  High selectivity → exponent ≈ 4 → sharply peaked at the
    highest-scoring available male → near-exclusive same-species mating.
    """
    if not viable_males or not viable_females:
        return []

    score_matrix = _build_compatibility_matrix(viable_males, viable_females)
    # score_matrix[i, j] = compatibility between males[i] and females[j]

    female_order = list(range(len(viable_females)))
    rng.shuffle(female_order)

    available_males = list(range(len(viable_males)))
    male_selectivities = np.array([m.selectivity for m in viable_males])
    mating_events: list[dict] = []

    for f_idx in female_order:
        if not available_males:
            break
        female = viable_females[f_idx]

        threshold_f = Creature.COMPATIBILITY_FLOOR + 0.15 * female.selectivity
        scores_f = score_matrix[available_males, f_idx]
        above = np.maximum(0.0, scores_f - threshold_f)

        if above.max() == 0.0:
            continue

        # Per-pair sharpness: average selectivity of the specific male + this female
        sel_m = male_selectivities[available_males]
        sharpness = 1.0 + mating_sharpness_k * (sel_m + female.selectivity) / 2.0
        weights = above**sharpness

        total = weights.sum()
        if total == 0.0:
            continue
        probs = weights / total

        chosen_pool_idx = int(rng.choice(len(available_males), p=probs))
        chosen_male_idx = available_males[chosen_pool_idx]
        male = viable_males[chosen_male_idx]

        event = _attempt_mating(male, female, rng)
        mating_events.append(event)
        available_males.pop(chosen_pool_idx)

    return mating_events


def _mate_stable_matching(viable_males: list, viable_females: list, rng: np.random.Generator) -> list[dict]:
    """
    Mutual-preference stable matching via Gale-Shapley deferred acceptance.

    Builds the full M×F compatibility score matrix, computes each creature's
    personal acceptance threshold from its selectivity trait, runs the
    _gale_shapley() function to find a stable set of pairs, then calls
    _attempt_mating() on each pair.

    See _gale_shapley() for the full algorithm description, assumptions, and
    documented baked-in choices (male-proposing direction, bilateral threshold
    pre-filtering, tie-breaking noise, threshold asymmetry vs is_compatible).

    Unlike weighted_matrix, every matched pair is the best stable outcome for
    the male — no male would prefer an unmatched female who also prefers him.
    Hybridisation occurs only when a cross-species individual genuinely ranks
    above all available same-species candidates for both parties.
    """
    if not viable_males or not viable_females:
        return []

    score_matrix = _build_compatibility_matrix(viable_males, viable_females)

    # Per-creature thresholds mirror the formula in is_compatible(), but use
    # each creature's OWN selectivity rather than the pair average.  See
    # _gale_shapley assumption 5 for the implications of this asymmetry.
    male_thresholds = np.array([Creature.COMPATIBILITY_FLOOR + 0.15 * m.selectivity for m in viable_males])
    female_thresholds = np.array(
        [Creature.COMPATIBILITY_FLOOR + 0.15 * f.selectivity for f in viable_females]
    )

    pairs = _gale_shapley(score_matrix, male_thresholds, female_thresholds, rng)

    mating_events: list[dict] = []
    for male_idx, female_idx in pairs:
        event = _attempt_mating(viable_males[male_idx], viable_females[female_idx], rng)
        mating_events.append(event)

    return mating_events


def _gale_shapley(
    score_matrix: np.ndarray,
    male_thresholds: np.ndarray,
    female_thresholds: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """
    Male-proposing Gale-Shapley deferred-acceptance stable matching.

    Finds a stable matching between M males and F females given a pairwise
    compatibility score matrix and per-creature acceptance thresholds.  A
    matching is *stable* if no unmatched (male, female) pair both prefer each
    other over their current partners — i.e. no blocking pair exists.

    Parameters
    ----------
    score_matrix : (M, F) float array
        Pairwise cosine similarity scores in [-1, 1].  Entry [i, j] is the
        compatibility score between male i and female j.  Produced by
        _build_compatibility_matrix().
    male_thresholds : (M,) float array
        Each male's personal acceptance floor.  He will not propose to any
        female whose score is below this value.
    female_thresholds : (F,) float array
        Each female's personal acceptance floor.  She will auto-reject any
        proposer whose score is below this value.
    rng : np.random.Generator
        The single simulation generator, used for the tie-breaking noise
        (assumption 3).  Threaded explicitly so the matching is reproducible
        from the run seed.

    Returns
    -------
    list of (male_idx, female_idx) int tuples
        Indices into the original males/females lists passed to
        _mate_stable_matching.  Only matched pairs are returned; unmatched
        individuals are omitted.  Each matched pair is subsequently passed to
        _attempt_mating, which re-checks is_compatible() and calls reproduce().
        The algorithm itself does not trigger conception — it only determines
        who attempts to mate with whom.

    Algorithm
    ---------
    Each free male proposes to his top-ranked remaining candidate.  A free
    female tentatively accepts; an engaged female accepts if the new proposer
    scores higher for her than her current partner (releasing the old partner
    back to the free pool).  Rejected males advance to their next candidate.
    The loop terminates when every free male has exhausted his list.

    Assumptions and baked-in choices
    ---------------------------------
    1.  **Male-proposing direction.**  This gives every male his best possible
        partner across all stable matchings, and every female her worst.  The
        direction is biologically arbitrary — there is no neutral variant.
        Swapping to female-proposing simply flips which side is optimal vs.
        pessimal.

    2.  **Bilateral threshold pre-filtering.**  A female j is included in male
        i's preference list only if score[i,j] >= BOTH male_thresholds[i] AND
        female_thresholds[j].  This is an optimisation: a female will never
        accept a score below her floor, so proposals below it are provably
        wasted.  Skipping them reduces the worst-case proposal count without
        changing the set of possible stable matchings.

    3.  **Tie-breaking via noise.**  Preference lists are ranked on
        score_matrix + N(0, NOISE_SIGMA).  The noise is sampled once before the
        loop so rankings are consistent throughout a single call.  Threshold
        checks always use the *original* score_matrix — noise must never push a
        score across a compatibility floor.  NOISE_SIGMA = 0.005 is well below
        the minimum meaningful score gap (~0.01 between near-identical
        same-species pairs with genome noise = 0.05).

    4.  **Female preference comparison uses noisy scores.**  When a female
        decides whether to swap her current partner for a new proposer, she
        compares noisy[new, j] vs noisy[current, j].  Using the same noisy
        matrix that determined the initial ranking keeps comparisons consistent
        and ensures no two males have exactly equal appeal to the same female.

    5.  **Threshold asymmetry vs is_compatible().**  This function uses
        *individual* thresholds (FLOOR + 0.15 * own_selectivity) for
        pre-filtering and acceptance.  Creature.is_compatible() uses the
        *average* selectivity of the pair.  A pair that clears both individual
        thresholds here can still fail is_compatible() if the pair-average
        raises the combined bar above one partner's score.  _attempt_mating()
        re-checks is_compatible() after the matching, so these edge cases
        produce a mating_event with compatible=False rather than a birth.

    6.  **Deque order (FIFO) affects which stable matching is found.**  When
        multiple males become free in the same round, they are processed in the
        order they were released.  Multiple stable matchings can exist; FIFO
        does not guarantee any specific one.  Combined with noise (assumption 3)
        this introduces stochasticity without biasing toward any fixed outcome.
    """
    M, F = score_matrix.shape
    if M == 0 or F == 0:
        return []

    NOISE_SIGMA = 0.005  # see assumption 3
    noisy = score_matrix + rng.normal(0, NOISE_SIGMA, score_matrix.shape)

    # Build each male's ordered preference list (assumptions 2 and 3).
    # male_prefs[i] is sorted from most-preferred (index 0) to least-preferred.
    # Only females that clear BOTH individual thresholds are included.
    male_prefs: list[list[int]] = []
    for i in range(M):
        eligible_mask = (
            (score_matrix[i] >= male_thresholds[i])  # male's own floor
            & (score_matrix[i] >= female_thresholds)  # female's own floor
        )
        eligible_indices = np.where(eligible_mask)[0]
        if eligible_indices.size == 0:
            male_prefs.append([])
            continue
        order = np.argsort(noisy[i, eligible_indices])[::-1]  # highest noisy score first
        male_prefs.append(eligible_indices[order].tolist())

    # Algorithm state ----------------------------------------------------------
    # male_next[i]      : next index into male_prefs[i] for the next proposal.
    #                     Only ever increments — a male never re-proposes to a
    #                     female who has already rejected him.
    # male_partner[i]   : current female partner index, or -1 if unmatched.
    # female_partner[j] : current male partner index, or -1 if free.
    male_next: list[int] = [0] * M
    male_partner: list[int] = [-1] * M
    female_partner: list[int] = [-1] * F

    # Seed the free queue with every male who has at least one candidate.
    # Males with empty preference lists (no compatible females) are never queued
    # and remain permanently unmatched, which is correct.
    free: deque[int] = deque(i for i in range(M) if male_prefs[i])

    while free:
        i = free.popleft()

        # A male may have exhausted his list since he was last queued (e.g. he
        # was re-queued after a rejection, then his remaining candidates were
        # all already visited).  Guard prevents index-out-of-bounds.
        if male_next[i] >= len(male_prefs[i]):
            continue

        j = male_prefs[i][male_next[i]]
        male_next[i] += 1  # advance regardless of outcome; i never re-proposes to j

        if female_partner[j] == -1:
            # j is free: tentative acceptance
            female_partner[j] = i
            male_partner[i] = j

        else:
            current = female_partner[j]
            # Female j compares i to her current partner using noisy scores
            # (assumption 4) to break ties consistently
            if noisy[i, j] > noisy[current, j]:
                # j prefers i; release current partner back to the free pool
                male_partner[current] = -1
                female_partner[j] = i
                male_partner[i] = j
                if male_next[current] < len(male_prefs[current]):
                    free.append(current)
            else:
                # j rejects i; re-queue i only if he still has candidates
                if male_next[i] < len(male_prefs[i]):
                    free.append(i)

    # Return matched pairs.  _attempt_mating will re-run is_compatible() on
    # each pair (assumption 5), so pairs that pass Gale-Shapley thresholds but
    # fail the pair-averaged is_compatible() threshold produce a compatible=False
    # event rather than offspring.
    return [(i, male_partner[i]) for i in range(M) if male_partner[i] != -1]
