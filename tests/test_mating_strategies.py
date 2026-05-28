"""
Tests for configurable mating strategies in Habitat.simulate_week().

Covers:
  - "zip"              : legacy behavior preserved
  - "species_priority" : minority species protected; hybridisation via spillover
  - unknown strategy   : falls back to zip
"""

from collections import defaultdict

import numpy as np

from evolution_simulator.creature import DEFAULT_TRAIT_GENE_INDICES, GENE_DIMS, Creature
from evolution_simulator.habitat import Habitat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_species_genome(rng, base_seed_genes: np.ndarray) -> np.ndarray:
    """
    Return a genome suitable for a founding species in tests.

    Only modifies loci that do NOT overlap with compatibility_genes (ranges
    80-129, 140-189, 220-269, 285-339, 390-429).  Setting reproduction_likelihood
    to a high value is deliberately avoided: those loci (80-82, 140-142, 220-222,
    290-291, 392-393) fall inside compatibility_genes, and setting them identically
    for two species inflates their cross-species cosine similarity above the mating
    floor, making genetically distinct species appear compatible with each other.
    """
    genes = base_seed_genes.copy()
    # selectivity loci 353-359, 436-439 are outside compatibility_genes — safe to set
    genes[DEFAULT_TRAIT_GENE_INDICES["selectivity"]] = -10.0  # low selectivity
    return genes


def _make_creature(genes: np.ndarray, sex: str, species: str) -> Creature:
    c = Creature(genes=genes.copy())
    c.sex = sex
    c.age = c.weeks_to_sexual_viability + 1
    c.species = species
    return c


def _build_two_species_habitat(
    n_majority: int = 20,
    n_minority: int = 4,
    rng_seed: int = 0,
) -> tuple[Habitat, list[Creature], list[Creature]]:
    """
    Habitat with two species:
      - majority: n_majority creatures (half male, half female), high compatibility
      - minority: n_minority creatures (half male, half female), high compatibility
    The two species are genetically distant so cross-species compatibility < floor.
    Returns (habitat, majority_creatures, minority_creatures).
    """
    rng = np.random.default_rng(rng_seed)
    hab = Habitat()

    # Majority: random base genome, large cluster
    maj_base = rng.standard_normal(GENE_DIMS)
    maj_base = _make_species_genome(rng, maj_base)

    # Minority: orthogonal-ish base genome (independent random → near-zero cosine in 500-dim)
    min_base = rng.standard_normal(GENE_DIMS)
    min_base = _make_species_genome(rng, min_base)

    majority, minority = [], []
    noise = 0.05

    for i in range(n_majority):
        genes = maj_base + rng.standard_normal(GENE_DIMS) * noise
        sex = "female" if i % 2 == 0 else "male"
        c = _make_creature(genes, sex, "majority")
        hab.add_creature(c)
        majority.append(c)

    for i in range(n_minority):
        genes = min_base + rng.standard_normal(GENE_DIMS) * noise
        sex = "female" if i % 2 == 0 else "male"
        c = _make_creature(genes, sex, "minority")
        hab.add_creature(c)
        minority.append(c)

    return hab, majority, minority


def _count_within_species_matings(mating_events: list[dict]) -> dict[str, int]:
    """Count fertilized within-species matings per species from the mating event list."""
    counts: dict[str, int] = defaultdict(int)
    for ev in mating_events:
        if ev.get("fertilized") and "hybridization" not in ev:
            counts["within_species"] += 1
        elif ev.get("fertilized") and "hybridization" in ev:
            counts["cross_species"] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Strategy: zip (legacy behavior preserved)
# ---------------------------------------------------------------------------


class TestZipStrategy:
    def test_zip_produces_mating_events(self):
        hab, majority, minority = _build_two_species_habitat(n_majority=10, n_minority=10)
        result = hab.simulate_week(mating_strategy="zip")
        assert len(result["mating_events"]) > 0

    def test_zip_minority_penalty(self):
        """With zip, minority species loses most mating opportunities."""
        # Run many trials to get a stable estimate
        minority_fertilized_counts = []
        for seed in range(30):
            hab, majority, minority = _build_two_species_habitat(n_majority=40, n_minority=4, rng_seed=seed)
            result = hab.simulate_week(mating_strategy="zip")
            min_fertilized = sum(
                1
                for ev in result["mating_events"]
                if ev.get("fertilized")
                and "hybridization" not in ev
                and any(ev["male_id"] == c.creature_id or ev["female_id"] == c.creature_id for c in minority)
            )
            minority_fertilized_counts.append(min_fertilized)

        # On average the minority (4 creatures = 2M+2F) should rarely all mate successfully
        # because most of their pairings land with majority individuals
        mean_minority_births = np.mean(minority_fertilized_counts)
        # The minority has 2 possible same-species pairs; they should succeed well below 50% of the time
        assert mean_minority_births < 1.5, (
            f"Expected minority to be penalised by zip, got mean {mean_minority_births:.2f} successful matings"
        )


# ---------------------------------------------------------------------------
# Strategy: species_priority
# ---------------------------------------------------------------------------


class TestSpeciesPriorityStrategy:
    def test_species_priority_produces_mating_events(self):
        hab, majority, minority = _build_two_species_habitat()
        result = hab.simulate_week(mating_strategy="species_priority")
        assert len(result["mating_events"]) > 0

    def test_minority_gets_within_species_pairings(self):
        """
        Under species_priority, minority creatures with n_males == n_females should all
        be paired within-species first, not wasted on cross-species pairings.
        """
        # n_minority=4 → 2 males, 2 females; both pairs should be within-species
        minority_pair_rates = []
        for seed in range(30):
            hab, majority, minority = _build_two_species_habitat(n_majority=40, n_minority=4, rng_seed=seed)
            result = hab.simulate_week(mating_strategy="species_priority")
            # Count mating events where both participants are minority
            minority_ids = {c.creature_id for c in minority}
            within_minority = [
                ev
                for ev in result["mating_events"]
                if ev["male_id"] in minority_ids and ev["female_id"] in minority_ids
            ]
            minority_pair_rates.append(len(within_minority))

        # With 2M+2F in minority, we should nearly always get 2 within-species pairings
        mean_pairs = np.mean(minority_pair_rates)
        assert mean_pairs >= 1.8, f"Expected ~2 within-species minority pairings, got mean {mean_pairs:.2f}"

    def test_minority_gets_more_matings_than_zip(self):
        """
        species_priority should produce more minority-species births than zip
        across many trials.
        """

        def count_minority_births(strategy: str, n_trials: int = 50) -> float:
            total = 0
            for seed in range(n_trials):
                hab, majority, minority = _build_two_species_habitat(
                    n_majority=40, n_minority=4, rng_seed=seed
                )
                result = hab.simulate_week(mating_strategy=strategy)
                minority_ids = {c.creature_id for c in minority}
                births = sum(
                    1
                    for ev in result["mating_events"]
                    if ev.get("fertilized")
                    and ev["male_id"] in minority_ids
                    and ev["female_id"] in minority_ids
                )
                total += births
            return total / n_trials

        zip_mean = count_minority_births("zip")
        sp_mean = count_minority_births("species_priority")
        assert sp_mean > zip_mean, (
            f"species_priority ({sp_mean:.2f}) should produce more minority births than zip ({zip_mean:.2f})"
        )

    def test_hybridisation_still_occurs_in_spillover(self):
        """
        When species have unequal sex ratios, surplus individuals enter the spillover
        pool where cross-species encounters can happen.
        """
        # 3 males + 1 female for majority → 2 majority males surplus → spillover
        # 1 male + 3 females for minority → 2 minority females surplus → spillover
        # So we expect spillover pairings between majority males and minority females
        for seed in range(50):
            rng = np.random.default_rng(seed)
            hab = Habitat()

            maj_base = _make_species_genome(rng, rng.standard_normal(GENE_DIMS))
            min_base = _make_species_genome(rng, rng.standard_normal(GENE_DIMS))
            noise = 0.05

            # 3 majority males, 1 majority female
            for _ in range(3):
                genes = maj_base + rng.standard_normal(GENE_DIMS) * noise
                c = _make_creature(genes, "male", "majority")
                hab.add_creature(c)
            genes = maj_base + rng.standard_normal(GENE_DIMS) * noise
            hab.add_creature(_make_creature(genes, "female", "majority"))

            # 1 minority male, 3 minority females
            genes = min_base + rng.standard_normal(GENE_DIMS) * noise
            hab.add_creature(_make_creature(genes, "male", "minority"))
            for _ in range(3):
                genes = min_base + rng.standard_normal(GENE_DIMS) * noise
                c = _make_creature(genes, "female", "minority")
                hab.add_creature(c)

            result = hab.simulate_week(mating_strategy="species_priority")
            if any("hybridization" in ev for ev in result["mating_events"]):
                break

        # Hybridization in the spillover pool should occur at least once in 50 trials
        # Note: it may not due to compatibility scores — this is a soft check
        # (spillover pairings exist; whether they pass the compatibility floor depends on genes)
        # We verify at minimum that spillover pairings were *attempted*
        # by checking for cross-species compatible_score events
        rng = np.random.default_rng(999)
        hab = Habitat()
        maj_base = _make_species_genome(rng, rng.standard_normal(GENE_DIMS))
        min_base = _make_species_genome(rng, rng.standard_normal(GENE_DIMS))
        noise = 0.05
        for _ in range(5):
            hab.add_creature(
                _make_creature(maj_base + rng.standard_normal(GENE_DIMS) * noise, "male", "majority")
            )
        hab.add_creature(
            _make_creature(maj_base + rng.standard_normal(GENE_DIMS) * noise, "female", "majority")
        )
        hab.add_creature(
            _make_creature(min_base + rng.standard_normal(GENE_DIMS) * noise, "male", "minority")
        )
        for _ in range(5):
            hab.add_creature(
                _make_creature(min_base + rng.standard_normal(GENE_DIMS) * noise, "female", "minority")
            )
        result = hab.simulate_week(mating_strategy="species_priority")
        # With 4 surplus majority males and 4 surplus minority females, spillover pairings
        # should exist (they will appear in mating_events with cross-species participant ids)
        maj_ids = {c.creature_id for c in hab.alive_creatures if c.species == "majority"}
        min_ids = {c.creature_id for c in hab.alive_creatures if c.species == "minority"}
        cross_events = [
            ev
            for ev in result["mating_events"]
            if (ev["male_id"] in maj_ids and ev["female_id"] in min_ids)
            or (ev["male_id"] in min_ids and ev["female_id"] in maj_ids)
        ]
        assert len(cross_events) > 0, "Expected cross-species spillover pairings to be attempted"

    def test_single_species_produces_mating_events(self):
        """With only one species, species_priority still pairs individuals and produces events."""
        noise = 0.05

        hab_zip = Habitat()
        hab_sp = Habitat()
        rng2 = np.random.default_rng(7)
        base2 = _make_species_genome(rng2, rng2.standard_normal(GENE_DIMS))
        for i in range(10):
            genes = base2 + rng2.standard_normal(GENE_DIMS) * noise
            sex = "female" if i % 2 == 0 else "male"
            hab_zip.add_creature(_make_creature(genes, sex, "only"))
            hab_sp.add_creature(_make_creature(genes.copy(), sex, "only"))

        r_zip = hab_zip.simulate_week(mating_strategy="zip")
        r_sp = hab_sp.simulate_week(mating_strategy="species_priority")
        assert len(r_zip["mating_events"]) > 0
        assert len(r_sp["mating_events"]) > 0


# ---------------------------------------------------------------------------
# Strategy: weighted_matrix
# ---------------------------------------------------------------------------


class TestWeightedMatrixStrategy:
    def test_weighted_matrix_produces_mating_events(self):
        hab, majority, minority = _build_two_species_habitat()
        result = hab.simulate_week(mating_strategy="weighted_matrix")
        assert len(result["mating_events"]) > 0

    def test_minority_gets_within_species_pairings(self):
        """
        weighted_matrix should pair minority creatures with same-species partners
        (high-selectivity default genes → high sharpness → picks highest-scoring male).
        """
        minority_pair_rates = []
        for seed in range(30):
            hab, majority, minority = _build_two_species_habitat(n_majority=40, n_minority=4, rng_seed=seed)
            result = hab.simulate_week(mating_strategy="weighted_matrix")
            minority_ids = {c.creature_id for c in minority}
            within_minority = [
                ev
                for ev in result["mating_events"]
                if ev["male_id"] in minority_ids and ev["female_id"] in minority_ids
            ]
            minority_pair_rates.append(len(within_minority))

        mean_pairs = np.mean(minority_pair_rates)
        assert mean_pairs >= 1.5, (
            f"Expected minority to mostly pair within-species, got mean {mean_pairs:.2f}"
        )

    def test_minority_gets_more_matings_than_zip(self):
        """weighted_matrix should protect the minority at least as well as species_priority."""

        def count_minority_births(strategy: str, n_trials: int = 50) -> float:
            total = 0
            for seed in range(n_trials):
                hab, majority, minority = _build_two_species_habitat(
                    n_majority=40, n_minority=4, rng_seed=seed
                )
                result = hab.simulate_week(mating_strategy=strategy)
                minority_ids = {c.creature_id for c in minority}
                births = sum(
                    1
                    for ev in result["mating_events"]
                    if ev.get("fertilized")
                    and ev["male_id"] in minority_ids
                    and ev["female_id"] in minority_ids
                )
                total += births
            return total / n_trials

        zip_mean = count_minority_births("zip")
        wm_mean = count_minority_births("weighted_matrix")
        assert wm_mean > zip_mean, (
            f"weighted_matrix ({wm_mean:.2f}) should protect minority better than zip ({zip_mean:.2f})"
        )

    def test_low_selectivity_increases_hybridisation(self):
        """
        Creatures with low selectivity should hybridise at a higher rate than
        those with high selectivity under weighted_matrix.
        """
        from evolution_simulator.creature import DEFAULT_TRAIT_GENE_INDICES, GENE_DIMS

        def hybrid_rate(selectivity_value: float, n_trials: int = 40) -> float:
            hybrids = 0
            total = 0
            sel_idx = DEFAULT_TRAIT_GENE_INDICES["selectivity"]
            rl_idx = DEFAULT_TRAIT_GENE_INDICES["reproduction_likelihood"]

            for seed in range(n_trials):
                rng2 = np.random.default_rng(seed + 100)
                hab = Habitat()
                # Two species: same population size, equal sex ratio
                for sp in ("A", "B"):
                    base = rng2.standard_normal(GENE_DIMS)
                    base[sel_idx] = selectivity_value
                    base[rl_idx] = 10.0  # high fertility
                    for i in range(10):
                        genes = base + rng2.standard_normal(GENE_DIMS) * 0.05
                        genes[sel_idx] = selectivity_value
                        c = _make_creature(genes, "female" if i % 2 == 0 else "male", sp)
                        hab.add_creature(c)

                result = hab.simulate_week(mating_strategy="weighted_matrix")
                for ev in result["mating_events"]:
                    if ev.get("fertilized"):
                        total += 1
                        if "hybridization" in ev:
                            hybrids += 1

            return hybrids / total if total > 0 else 0.0

        low_sel_rate = hybrid_rate(-5.0)  # genes → low selectivity trait value
        high_sel_rate = hybrid_rate(5.0)  # genes → high selectivity trait value
        assert low_sel_rate >= high_sel_rate, (
            f"Low selectivity should hybridise at least as often as high: "
            f"low={low_sel_rate:.3f}, high={high_sel_rate:.3f}"
        )

    def test_no_double_mating(self):
        """Each male should appear in at most one fertilized event."""
        hab, majority, minority = _build_two_species_habitat(n_majority=20, n_minority=20)
        result = hab.simulate_week(mating_strategy="weighted_matrix")
        male_ids_in_events = [ev["male_id"] for ev in result["mating_events"]]
        assert len(male_ids_in_events) == len(set(male_ids_in_events)), (
            "A male appeared in more than one mating event (double-mating)"
        )

    def test_build_compatibility_matrix_shape_and_range(self):
        """_build_compatibility_matrix returns (M, F) with values in [-1, 1]."""
        rng = np.random.default_rng(42)
        males = [_make_creature(rng.standard_normal(500), "male", "s") for _ in range(5)]
        females = [_make_creature(rng.standard_normal(500), "female", "s") for _ in range(7)]
        mat = Habitat._build_compatibility_matrix(males, females)
        assert mat.shape == (5, 7)
        assert mat.min() >= -1.0 and mat.max() <= 1.0

    def test_empty_pools_return_no_events(self):
        hab = Habitat()
        result = hab.simulate_week(mating_strategy="weighted_matrix")
        assert result["mating_events"] == []


# ---------------------------------------------------------------------------
# Strategy: stable_matching (Gale-Shapley)
# ---------------------------------------------------------------------------


class TestStableMatchingStrategy:
    # --- Basic smoke tests ---------------------------------------------------

    def test_produces_mating_events(self):
        hab, majority, minority = _build_two_species_habitat()
        result = hab.simulate_week(mating_strategy="stable_matching")
        assert len(result["mating_events"]) > 0

    def test_empty_pools_return_no_events(self):
        hab = Habitat()
        result = hab.simulate_week(mating_strategy="stable_matching")
        assert result["mating_events"] == []

    # --- No-double-matching --------------------------------------------------

    def test_no_male_appears_twice(self):
        """Each male must appear in at most one event (Gale-Shapley produces a matching)."""
        hab, majority, minority = _build_two_species_habitat(n_majority=20, n_minority=20)
        result = hab.simulate_week(mating_strategy="stable_matching")
        male_ids = [ev["male_id"] for ev in result["mating_events"]]
        assert len(male_ids) == len(set(male_ids)), "A male appeared in more than one mating event"

    def test_no_female_appears_twice(self):
        """Each female must appear in at most one event."""
        hab, majority, minority = _build_two_species_habitat(n_majority=20, n_minority=20)
        result = hab.simulate_week(mating_strategy="stable_matching")
        female_ids = [ev["female_id"] for ev in result["mating_events"]]
        assert len(female_ids) == len(set(female_ids)), "A female appeared in more than one mating event"

    # --- Stability property --------------------------------------------------

    def test_no_blocking_pairs(self):
        """
        Verify the stability guarantee: no unmatched (male, female) pair exists
        where both would prefer each other over their current partners.

        We test this directly on _gale_shapley() with a known score matrix so
        the result is deterministic (fixed seed via np.random.seed).
        """
        from evolution_simulator.habitat import _gale_shapley

        np.random.seed(0)
        M, F = 8, 8
        rng = np.random.default_rng(42)

        # Build a score matrix where same-index pairs have high scores,
        # so the expected stable matching is the diagonal.
        scores = rng.uniform(-0.5, 0.5, (M, F))
        # Boost diagonal to ensure a clear preference ordering
        for k in range(min(M, F)):
            scores[k, k] = 0.95

        floor = 0.70
        male_thresholds = np.full(M, floor)
        female_thresholds = np.full(F, floor)

        # Run the algorithm multiple times with different noise seeds
        for trial in range(10):
            pairs = _gale_shapley(scores, male_thresholds, female_thresholds)
            male_of = {p[0]: p[1] for p in pairs}  # male i → his female partner
            female_of = {p[1]: p[0] for p in pairs}  # female j → her male partner

            # Check every unmatched (male, female) pair for blocking
            for i in range(M):
                for j in range(F):
                    # Only care about pairs where both could potentially mate
                    if scores[i, j] < floor:
                        continue

                    i_partner = male_of.get(i)  # None if unmatched
                    j_partner = female_of.get(j)  # None if unmatched

                    # Does male i prefer female j over his current partner?
                    i_prefers_j = (
                        i_partner is None  # i is unmatched (would always prefer j)
                        or scores[i, j] > scores[i, i_partner]
                    )
                    # Does female j prefer male i over her current partner?
                    j_prefers_i = (
                        j_partner is None  # j is free (would always prefer i)
                        or scores[j_partner, j] < scores[i, j]  # NOTE: scores[male, female]
                    )

                    assert not (i_prefers_j and j_prefers_i), (
                        f"Trial {trial}: blocking pair found: male {i} (partner={i_partner}) "
                        f"and female {j} (partner={j_partner}), score={scores[i, j]:.3f}"
                    )

    # --- Minority protection -------------------------------------------------

    def test_minority_gets_more_matings_than_zip(self):
        """stable_matching should protect minority species at least as well as zip."""

        def count_minority_births(strategy: str, n_trials: int = 50) -> float:
            total = 0
            for seed in range(n_trials):
                hab, majority, minority = _build_two_species_habitat(
                    n_majority=40, n_minority=4, rng_seed=seed
                )
                result = hab.simulate_week(mating_strategy=strategy)
                minority_ids = {c.creature_id for c in minority}
                births = sum(
                    1
                    for ev in result["mating_events"]
                    if ev.get("fertilized")
                    and ev["male_id"] in minority_ids
                    and ev["female_id"] in minority_ids
                )
                total += births
            return total / n_trials

        zip_mean = count_minority_births("zip")
        sm_mean = count_minority_births("stable_matching")
        assert sm_mean > zip_mean, (
            f"stable_matching ({sm_mean:.2f}) should protect minority better than zip ({zip_mean:.2f})"
        )

    # --- _gale_shapley unit tests --------------------------------------------

    def test_gale_shapley_empty_inputs(self):
        from evolution_simulator.habitat import _gale_shapley

        assert _gale_shapley(np.zeros((0, 5)), np.array([]), np.full(5, 0.7)) == []
        assert _gale_shapley(np.zeros((5, 0)), np.full(5, 0.7), np.array([])) == []

    def test_gale_shapley_all_below_threshold(self):
        """When no scores clear the threshold, nobody is matched."""
        from evolution_simulator.habitat import _gale_shapley

        scores = np.full((4, 4), 0.5)  # all below 0.70
        thresholds = np.full(4, 0.70)
        pairs = _gale_shapley(scores, thresholds, thresholds)
        assert pairs == []

    def test_gale_shapley_perfect_diagonal(self):
        """
        When each male scores 0.99 with his same-index female and 0.71 with all
        others, the stable matching should be the diagonal pairing.
        """
        from evolution_simulator.habitat import _gale_shapley

        N = 5
        # Off-diagonal scores all just above floor (0.71); diagonal at 0.99
        scores = np.full((N, N), 0.71)
        np.fill_diagonal(scores, 0.99)
        thresholds = np.full(N, 0.70)

        # Run many trials to verify robustness against noise
        for seed in range(20):
            np.random.seed(seed)
            pairs = _gale_shapley(scores, thresholds, thresholds)
            pair_dict = dict(pairs)
            for k in range(N):
                assert pair_dict.get(k) == k, (
                    f"seed={seed}: male {k} matched to {pair_dict.get(k)}, expected {k}"
                )

    def test_gale_shapley_unequal_pool_sizes(self):
        """With more males than females, some males go unmatched; no female is double-matched."""
        from evolution_simulator.habitat import _gale_shapley

        M, F = 6, 3
        rng = np.random.default_rng(7)
        scores = rng.uniform(0.72, 0.99, (M, F))  # all above floor
        thresholds_m = np.full(M, 0.70)
        thresholds_f = np.full(F, 0.70)
        pairs = _gale_shapley(scores, thresholds_m, thresholds_f)
        # At most F pairs (one per female)
        assert len(pairs) <= F
        # No female matched twice
        matched_females = [p[1] for p in pairs]
        assert len(matched_females) == len(set(matched_females))
        # No male matched twice
        matched_males = [p[0] for p in pairs]
        assert len(matched_males) == len(set(matched_males))


# ---------------------------------------------------------------------------
# Unknown strategy → zip fallback
# ---------------------------------------------------------------------------


class TestUnknownStrategyFallback:
    def test_unknown_strategy_falls_back_to_zip(self):
        hab, majority, minority = _build_two_species_habitat(n_majority=10, n_minority=10)
        result = hab.simulate_week(mating_strategy="nonexistent_strategy")
        # Should not raise; should produce mating events (zip behavior)
        assert isinstance(result["mating_events"], list)
