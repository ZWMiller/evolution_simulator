import numpy as np
import pytest
from evolution_simulator.creature import Creature, DEFAULT_TRAIT_GENE_INDICES, GENE_DIMS
from evolution_simulator.species import SpeciesRegistry, ADJECTIVES, NOUNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_creature(genes: np.ndarray, parent_species: str = "unknown") -> Creature:
    """Creature with explicit genes and pre-set species label."""
    c = Creature(genes=genes.copy())
    c.species = parent_species
    return c


def near_genes(base: np.ndarray, noise: float = 0.01, seed: int = 0) -> np.ndarray:
    """Genes very close to base (high cosine similarity)."""
    rng = np.random.default_rng(seed)
    return base + rng.standard_normal(GENE_DIMS) * noise


def far_genes(seed: int = 99) -> np.ndarray:
    """Completely random genes, unlikely to be similar to anything."""
    return np.random.default_rng(seed).standard_normal(GENE_DIMS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return SpeciesRegistry()


@pytest.fixture
def registry_with_founder():
    rng = np.random.default_rng(1)
    reg = SpeciesRegistry()
    genes = rng.standard_normal(GENE_DIMS)
    name = reg.register_founding_species(genes, name="Primal Wanderer")
    return reg, genes, name


# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------

class TestNamePools:
    def test_adjectives_count(self):
        assert len(ADJECTIVES) == 100

    def test_nouns_count(self):
        assert len(NOUNS) == 100

    def test_adjectives_unique(self):
        assert len(set(ADJECTIVES)) == 100

    def test_nouns_unique(self):
        assert len(set(NOUNS)) == 100


# ---------------------------------------------------------------------------
# Registry initialisation
# ---------------------------------------------------------------------------

class TestRegistryInit:
    def test_starts_empty(self, registry):
        assert registry.species_count == 0
        assert registry.all_species == []
        assert registry.speciation_events == []

    def test_custom_threshold(self):
        reg = SpeciesRegistry(species_threshold=0.99)
        assert reg.species_threshold == 0.99


# ---------------------------------------------------------------------------
# Founding species registration
# ---------------------------------------------------------------------------

class TestFoundingSpecies:
    def test_register_with_explicit_name(self, registry):
        rng = np.random.default_rng(0)
        name = registry.register_founding_species(rng.standard_normal(GENE_DIMS), name="Alpha Seeker")
        assert name == "Alpha Seeker"
        assert "Alpha Seeker" in registry.all_species

    def test_register_auto_name(self, registry):
        rng = np.random.default_rng(0)
        name = registry.register_founding_species(rng.standard_normal(GENE_DIMS))
        assert isinstance(name, str)
        assert " " in name  # adjective + noun
        assert registry.species_count == 1

    def test_duplicate_name_raises(self, registry_with_founder):
        reg, genes, name = registry_with_founder
        with pytest.raises(ValueError, match="already registered"):
            reg.register_founding_species(genes, name=name)

    def test_progenitor_genes_stored(self, registry_with_founder):
        reg, genes, name = registry_with_founder
        stored = reg.progenitor_genes(name)
        np.testing.assert_array_equal(stored, genes)

    def test_progenitor_genes_returns_copy(self, registry_with_founder):
        reg, genes, name = registry_with_founder
        stored = reg.progenitor_genes(name)
        stored[0] += 999.0
        # Modifying the returned copy must not affect the registry
        np.testing.assert_array_equal(reg.progenitor_genes(name), genes)

    def test_unknown_species_returns_none(self, registry):
        assert registry.progenitor_genes("Nonexistent") is None

    def test_multiple_founders_stored(self, registry):
        rng = np.random.default_rng(5)
        for i in range(5):
            registry.register_founding_species(
                rng.standard_normal(GENE_DIMS), name=f"Species {i}"
            )
        assert registry.species_count == 5


# ---------------------------------------------------------------------------
# assign_species — same species
# ---------------------------------------------------------------------------

class TestAssignSpeciesSame:
    def test_similar_genome_stays_in_parent_species(self, registry_with_founder):
        reg, base_genes, founder_name = registry_with_founder
        child = make_creature(near_genes(base_genes, noise=0.001), parent_species=founder_name)
        name = reg.assign_species(child)
        assert name == founder_name
        assert child.species == founder_name
        assert reg.species_count == 1  # no new species
        assert reg.speciation_events == []

    def test_identical_genome_stays_in_parent_species(self, registry_with_founder):
        reg, base_genes, founder_name = registry_with_founder
        child = make_creature(base_genes, parent_species=founder_name)
        name = reg.assign_species(child)
        assert name == founder_name

    def test_assign_sets_creature_species(self, registry_with_founder):
        reg, base_genes, founder_name = registry_with_founder
        child = make_creature(near_genes(base_genes, noise=0.001))
        child.species = founder_name
        reg.assign_species(child)
        assert child.species == founder_name


# ---------------------------------------------------------------------------
# assign_species — candidate stage (two-stage speciation)
# ---------------------------------------------------------------------------

class TestAssignSpeciesNew:
    def test_diverged_genome_creates_candidate_not_immediate_species(self, registry_with_founder):
        """A diverged newborn should enter the candidate stage, not create a confirmed species."""
        reg, _, founder_name = registry_with_founder
        diverged = make_creature(far_genes(seed=42), parent_species=founder_name)
        returned_name = reg.assign_species(diverged)

        # No confirmed species added yet
        assert reg.species_count == 1
        assert len(reg.speciation_events) == 0
        # One pending candidate
        assert len(reg._candidates) == 1
        # Creature keeps the parent species label throughout the candidate period
        assert returned_name == founder_name
        assert diverged.species == founder_name

    def test_similar_diverged_creatures_join_same_candidate(self, registry_with_founder):
        """Multiple creatures near the same diverged genome should all join one candidate."""
        reg, _, founder_name = registry_with_founder
        diverged_base = far_genes(seed=42)
        for i in range(3):
            c = make_creature(near_genes(diverged_base, noise=0.001, seed=i), parent_species=founder_name)
            reg.assign_species(c)

        assert len(reg._candidates) == 1
        cand = next(iter(reg._candidates.values()))
        assert len(cand["members"]) == 3

    def test_very_different_diverged_creatures_create_separate_candidates(self, registry_with_founder):
        """Two truly different diverged genomes should create two separate candidates."""
        reg, _, founder_name = registry_with_founder
        c1 = make_creature(far_genes(seed=42), parent_species=founder_name)
        c2 = make_creature(far_genes(seed=99), parent_species=founder_name)
        reg.assign_species(c1)
        reg.assign_species(c2)

        assert len(reg._candidates) == 2

    def test_no_registry_entries_triggers_immediate_species(self):
        """Bootstrap case: first-ever creature gets a confirmed species immediately."""
        reg = SpeciesRegistry()
        c = make_creature(far_genes(seed=1), parent_species="unknown")
        name = reg.assign_species(c)
        assert reg.species_count == 1
        assert c.species == name
        assert len(reg._candidates) == 0


# ---------------------------------------------------------------------------
# Candidate promotion
# ---------------------------------------------------------------------------

def _add_members_to_candidate(reg, cid, n_extra, base_genes):
    """Add n_extra creatures as members of an existing candidate (for test setup)."""
    members = []
    for i in range(n_extra):
        c = make_creature(near_genes(base_genes, noise=0.001, seed=100 + i))
        reg._candidates[cid]["members"].add(c.creature_id)
        members.append(c)
    return members


class TestCandidatePromotion:
    def test_candidate_not_promoted_with_too_few_members(self, registry_with_founder):
        """A candidate with fewer than min_species_population members is not promoted."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        diverged = make_creature(diverged_genes, parent_species=founder_name)
        reg.assign_species(diverged)
        # Only 1 member; min_species_population=3 → should not promote
        reg.current_week = 99
        reg.promote_candidates([diverged], current_week=99)

        assert reg.species_count == 1
        assert len(reg.speciation_events) == 0
        assert len(reg._candidates) == 1

    def test_candidate_not_promoted_before_min_weeks(self, registry_with_founder):
        """A candidate with enough members but insufficient age is not promoted."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        first = make_creature(diverged_genes, parent_species=founder_name)
        reg.assign_species(first)
        cid = next(iter(reg._candidates))
        extras = _add_members_to_candidate(reg, cid, 2, diverged_genes)
        all_members = [first] + extras

        # detected_week=0, promote at week 3 → only 3 weeks elapsed < min_species_weeks=5
        reg.promote_candidates(all_members, current_week=3)

        assert reg.species_count == 1
        assert len(reg.speciation_events) == 0
        assert len(reg._candidates) == 1

    def test_candidate_promoted_when_both_criteria_met(self, registry_with_founder):
        """A candidate meeting population AND age requirements is promoted to confirmed species."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        first = make_creature(diverged_genes, parent_species=founder_name)
        reg.assign_species(first)
        cid = next(iter(reg._candidates))
        extras = _add_members_to_candidate(reg, cid, 2, diverged_genes)
        all_members = [first] + extras

        # 3 members >= 3 AND 10 weeks >= 5
        reg.promote_candidates(all_members, current_week=10)

        assert reg.species_count == 2
        assert len(reg.speciation_events) == 1
        assert len(reg._candidates) == 0

    def test_promoted_members_are_renamed(self, registry_with_founder):
        """All living members of a promoted candidate get the new species name."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        first = make_creature(diverged_genes, parent_species=founder_name)
        reg.assign_species(first)
        cid = next(iter(reg._candidates))
        extras = _add_members_to_candidate(reg, cid, 2, diverged_genes)
        all_members = [first] + extras

        reg.promote_candidates(all_members, current_week=10)

        new_name = reg.speciation_events[0]["new_species"]
        for c in all_members:
            assert c.species == new_name

    def test_promotion_speciation_event_has_correct_fields(self, registry_with_founder):
        """The speciation event logged on promotion has the right parent and creature_id."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        first = make_creature(diverged_genes, parent_species=founder_name)
        reg.assign_species(first)
        cid = next(iter(reg._candidates))
        extras = _add_members_to_candidate(reg, cid, 2, diverged_genes)

        reg.promote_candidates([first] + extras, current_week=10)

        ev = reg.speciation_events[0]
        assert ev["parent_species"] == founder_name
        assert ev["creature_id"] == first.creature_id
        assert ev["week"] == 10

    def test_promoted_species_name_is_adjective_noun(self, registry_with_founder):
        """The promoted species gets a valid adjective-noun name."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        first = make_creature(diverged_genes, parent_species=founder_name)
        reg.assign_species(first)
        cid = next(iter(reg._candidates))
        extras = _add_members_to_candidate(reg, cid, 2, diverged_genes)

        reg.promote_candidates([first] + extras, current_week=10)

        new_name = reg.speciation_events[0]["new_species"]
        adj, noun = new_name.split(" ", 1)
        assert adj in ADJECTIVES
        assert noun in NOUNS

    def test_promoted_progenitor_stored_in_registry(self, registry_with_founder):
        """After promotion the new species progenitor genes are stored."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        first = make_creature(diverged_genes, parent_species=founder_name)
        reg.assign_species(first)
        cid = next(iter(reg._candidates))
        extras = _add_members_to_candidate(reg, cid, 2, diverged_genes)

        reg.promote_candidates([first] + extras, current_week=10)

        new_name = reg.speciation_events[0]["new_species"]
        assert reg.progenitor_genes(new_name) is not None

    def test_candidate_evaporates_when_all_members_die(self, registry_with_founder):
        """A candidate with no living members is removed and records a failed attempt."""
        reg, _, founder_name = registry_with_founder
        diverged = make_creature(far_genes(seed=42), parent_species=founder_name)
        reg.assign_species(diverged)
        assert len(reg._candidates) == 1

        reg.promote_candidates([], current_week=99)

        assert len(reg._candidates) == 0
        assert len(reg.speciation_events) == 0
        assert reg.species_count == 1
        assert len(reg.failed_speciation_attempts) == 1

    def test_failed_attempt_records_correct_fields(self, registry_with_founder):
        """A failed attempt captures parent_species, detected_week, failed_week, peak_members."""
        reg, _, founder_name = registry_with_founder
        reg.current_week = 7
        diverged = make_creature(far_genes(seed=42), parent_species=founder_name)
        reg.assign_species(diverged)

        reg.promote_candidates([], current_week=12)

        attempt = reg.failed_speciation_attempts[0]
        assert attempt["parent_species"] == founder_name
        assert attempt["detected_week"] == 7
        assert attempt["failed_week"] == 12
        assert attempt["peak_members"] == 1
        assert attempt["first_creature_id"] == diverged.creature_id

    def test_peak_members_reflects_maximum_simultaneous_count(self, registry_with_founder):
        """peak_members is the maximum number of members the candidate ever had at once."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)

        # 3 creatures join → peak_members should reach 3
        members = []
        for i in range(3):
            c = make_creature(near_genes(diverged_genes, noise=0.001, seed=i), parent_species=founder_name)
            reg.assign_species(c)
            members.append(c)

        cand = next(iter(reg._candidates.values()))
        assert cand["peak_members"] == 3

        # Evaporate — peak_members should be preserved in the failure record
        reg.promote_candidates([], current_week=50)
        assert reg.failed_speciation_attempts[0]["peak_members"] == 3

    def test_successful_promotion_not_recorded_as_failure(self, registry_with_founder):
        """A candidate that meets criteria and is promoted should NOT appear in failed attempts."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        first = make_creature(diverged_genes, parent_species=founder_name)
        reg.assign_species(first)
        cid = next(iter(reg._candidates))
        extras = _add_members_to_candidate(reg, cid, 2, diverged_genes)

        reg.promote_candidates([first] + extras, current_week=10)

        assert len(reg.failed_speciation_attempts) == 0
        assert len(reg.speciation_events) == 1


# ---------------------------------------------------------------------------
# assign_species — drift-back / convergence protection
# ---------------------------------------------------------------------------

class TestDriftBack:
    def test_drift_back_to_ancestor_not_new_species(self):
        """
        A creature carrying a diverged species label but with genes close to
        an ancestral progenitor should be re-absorbed into the ancestral
        confirmed species, NOT trigger a new candidate or third species.
        Both species are registered as founders so we can test the assignment
        logic directly without depending on two-stage promotion.
        """
        rng = np.random.default_rng(7)
        reg = SpeciesRegistry()

        ancestor_genes = rng.standard_normal(GENE_DIMS)
        diverged_genes = rng.standard_normal(GENE_DIMS)

        ancestor_name = reg.register_founding_species(ancestor_genes, name="Ancient Wanderer")
        sp2_name = reg.register_founding_species(diverged_genes, name="Diverged Species")
        assert reg.species_count == 2

        # Creature inheriting sp2's label but genes drifted back near the ancestor
        drifted_back = make_creature(
            near_genes(ancestor_genes, noise=0.001, seed=10),
            parent_species=sp2_name,
        )
        assigned = reg.assign_species(drifted_back)

        # Re-classified as ancestor — no new species or candidate created
        assert assigned == ancestor_name
        assert reg.species_count == 2
        assert len(reg._candidates) == 0
        assert len(reg.speciation_events) == 0

    def test_convergent_lineages_share_species(self):
        """
        Two independently evolving lineages that converge toward the same
        genetic region should both be assigned to the same confirmed species,
        not create two separate candidates.
        """
        rng = np.random.default_rng(11)
        reg = SpeciesRegistry()

        base = rng.standard_normal(GENE_DIMS)
        target = rng.standard_normal(GENE_DIMS)

        founder_name = reg.register_founding_species(base, name="Origin Seeker")
        sp2_name = reg.register_founding_species(
            near_genes(target, noise=0.001, seed=20), name="Target Species"
        )

        # A second lineage independently converges toward the same target region
        c2 = make_creature(near_genes(target, noise=0.001, seed=21), parent_species=founder_name)
        assigned = reg.assign_species(c2)

        assert assigned == sp2_name
        assert reg.species_count == 2  # still only two confirmed species
        assert len(reg._candidates) == 0


# ---------------------------------------------------------------------------
# similarity_to_all_progenitors
# ---------------------------------------------------------------------------

class TestSimilarityToAll:
    def test_returns_dict_for_all_species(self, registry_with_founder):
        reg, base_genes, name = registry_with_founder
        c = make_creature(base_genes)
        scores = reg.similarity_to_all_progenitors(c)
        assert name in scores
        assert 0.99 <= scores[name] <= 1.0

    def test_empty_registry_returns_empty(self, registry):
        c = Creature()
        scores = registry.similarity_to_all_progenitors(c)
        assert scores == {}

    def test_scores_in_range(self, registry_with_founder):
        reg, base_genes, _ = registry_with_founder
        # Add a second species
        reg.register_founding_species(np.random.default_rng(5).standard_normal(GENE_DIMS))
        c = make_creature(base_genes)
        scores = reg.similarity_to_all_progenitors(c)
        for score in scores.values():
            assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Unique name generation
# ---------------------------------------------------------------------------

class TestUniqueNames:
    def test_generated_names_are_unique(self):
        reg = SpeciesRegistry()
        rng = np.random.default_rng(0)
        names = set()
        for _ in range(50):
            name = reg.register_founding_species(rng.standard_normal(GENE_DIMS))
            names.add(name)
        assert len(names) == 50

    def test_name_format(self):
        reg = SpeciesRegistry()
        rng = np.random.default_rng(3)
        name = reg.register_founding_species(rng.standard_normal(GENE_DIMS))
        parts = name.split(" ")
        assert len(parts) == 2
        assert parts[0] in ADJECTIVES
        assert parts[1] in NOUNS


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_species_count(self, registry_with_founder):
        reg, _, _ = registry_with_founder
        assert "1 confirmed species" in repr(reg)

    def test_repr_contains_candidate_count(self, registry_with_founder):
        reg, _, _ = registry_with_founder
        assert "0 candidates" in repr(reg)

    def test_repr_contains_event_count(self, registry_with_founder):
        reg, _, _ = registry_with_founder
        assert "0 speciation events" in repr(reg)
