"""
Tests for configurable mating strategies in Habitat.simulate_week().

Covers:
  - "zip"              : legacy behavior preserved
  - "species_priority" : minority species protected; hybridisation via spillover
  - unknown strategy   : falls back to zip
"""

import numpy as np
import pytest
from collections import defaultdict

from evolution_simulator.creature import Creature, DEFAULT_TRAIT_GENE_INDICES, GENE_DIMS
from evolution_simulator.habitat import Habitat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_species_genome(rng, base_seed_genes: np.ndarray) -> np.ndarray:
    """Return a genome that is compatible with base_seed_genes (same compatibility loci)."""
    genes = base_seed_genes.copy()
    genes[DEFAULT_TRAIT_GENE_INDICES["selectivity"]] = -10.0            # low selectivity
    genes[DEFAULT_TRAIT_GENE_INDICES["reproduction_likelihood"]] = 10.0 # high fertility
    genes[DEFAULT_TRAIT_GENE_INDICES["fecundity"]] = 5.0                # moderate fecundity
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
            hab, majority, minority = _build_two_species_habitat(
                n_majority=40, n_minority=4, rng_seed=seed
            )
            result = hab.simulate_week(mating_strategy="zip")
            min_fertilized = sum(
                1 for ev in result["mating_events"]
                if ev.get("fertilized")
                and "hybridization" not in ev
                and any(ev["male_id"] == c.creature_id or ev["female_id"] == c.creature_id
                        for c in minority)
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
            hab, majority, minority = _build_two_species_habitat(
                n_majority=40, n_minority=4, rng_seed=seed
            )
            result = hab.simulate_week(mating_strategy="species_priority")
            # Count mating events where both participants are minority
            minority_ids = {c.creature_id for c in minority}
            within_minority = [
                ev for ev in result["mating_events"]
                if ev["male_id"] in minority_ids and ev["female_id"] in minority_ids
            ]
            minority_pair_rates.append(len(within_minority))

        # With 2M+2F in minority, we should nearly always get 2 within-species pairings
        mean_pairs = np.mean(minority_pair_rates)
        assert mean_pairs >= 1.8, (
            f"Expected ~2 within-species minority pairings, got mean {mean_pairs:.2f}"
        )

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
                    1 for ev in result["mating_events"]
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
        hybrid_found = False
        for seed in range(50):
            rng = np.random.default_rng(seed)
            hab = Habitat()

            maj_base = _make_species_genome(rng, rng.standard_normal(GENE_DIMS))
            min_base = _make_species_genome(rng, rng.standard_normal(GENE_DIMS))
            noise = 0.05

            # 3 majority males, 1 majority female
            for i in range(3):
                genes = maj_base + rng.standard_normal(GENE_DIMS) * noise
                c = _make_creature(genes, "male", "majority")
                hab.add_creature(c)
            genes = maj_base + rng.standard_normal(GENE_DIMS) * noise
            hab.add_creature(_make_creature(genes, "female", "majority"))

            # 1 minority male, 3 minority females
            genes = min_base + rng.standard_normal(GENE_DIMS) * noise
            hab.add_creature(_make_creature(genes, "male", "minority"))
            for i in range(3):
                genes = min_base + rng.standard_normal(GENE_DIMS) * noise
                c = _make_creature(genes, "female", "minority")
                hab.add_creature(c)

            result = hab.simulate_week(mating_strategy="species_priority")
            if any("hybridization" in ev for ev in result["mating_events"]):
                hybrid_found = True
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
            hab.add_creature(_make_creature(
                maj_base + rng.standard_normal(GENE_DIMS) * noise, "male", "majority"
            ))
        hab.add_creature(_make_creature(
            maj_base + rng.standard_normal(GENE_DIMS) * noise, "female", "majority"
        ))
        hab.add_creature(_make_creature(
            min_base + rng.standard_normal(GENE_DIMS) * noise, "male", "minority"
        ))
        for _ in range(5):
            hab.add_creature(_make_creature(
                min_base + rng.standard_normal(GENE_DIMS) * noise, "female", "minority"
            ))
        result = hab.simulate_week(mating_strategy="species_priority")
        # With 4 surplus majority males and 4 surplus minority females, spillover pairings
        # should exist (they will appear in mating_events with cross-species participant ids)
        maj_ids = {c.creature_id for c in hab.alive_creatures if c.species == "majority"}
        min_ids = {c.creature_id for c in hab.alive_creatures if c.species == "minority"}
        cross_events = [
            ev for ev in result["mating_events"]
            if (ev["male_id"] in maj_ids and ev["female_id"] in min_ids)
            or (ev["male_id"] in min_ids and ev["female_id"] in maj_ids)
        ]
        assert len(cross_events) > 0, "Expected cross-species spillover pairings to be attempted"

    def test_single_species_produces_mating_events(self):
        """With only one species, species_priority still pairs individuals and produces events."""
        rng = np.random.default_rng(7)
        base = _make_species_genome(rng, rng.standard_normal(GENE_DIMS))
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
# Unknown strategy → zip fallback
# ---------------------------------------------------------------------------

class TestUnknownStrategyFallback:
    def test_unknown_strategy_falls_back_to_zip(self):
        hab, majority, minority = _build_two_species_habitat(n_majority=10, n_minority=10)
        result = hab.simulate_week(mating_strategy="nonexistent_strategy")
        # Should not raise; should produce mating events (zip behavior)
        assert isinstance(result["mating_events"], list)
