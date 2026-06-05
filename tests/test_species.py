import numpy as np
import pytest

from evolution_simulator.creature import Creature
from evolution_simulator.genetics import DEFAULT_TRAIT_GENE_INDICES, GENE_DIMS
from evolution_simulator.speciation_math import unit_rows
from evolution_simulator.species import ADJECTIVES, NOUNS, SpeciesRegistry
from evolution_simulator.traits import (
    PHENOTYPE_TRAITS,
    compute_phenotype,
    compute_phenotype_matrix,
)

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


# Compatibility subset (the signal C2 speciation keys on) and its complement.
COMPAT_IDX = np.array(DEFAULT_TRAIT_GENE_INDICES["compatibility_genes"], dtype=int)
NON_COMPAT_IDX = np.array(sorted(set(range(GENE_DIMS)) - set(COMPAT_IDX.tolist())), dtype=int)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def with_new_compat(base: np.ndarray, seed: int) -> np.ndarray:
    """Copy of base with ONLY the compatibility loci replaced (far in compat space)."""
    g = base.copy()
    g[COMPAT_IDX] = np.random.default_rng(seed).standard_normal(len(COMPAT_IDX))
    return g


def with_new_noncompat(base: np.ndarray, seed: int) -> np.ndarray:
    """Copy of base with ONLY the non-compatibility loci replaced (identical in compat space)."""
    g = base.copy()
    g[NON_COMPAT_IDX] = np.random.default_rng(seed).standard_normal(len(NON_COMPAT_IDX))
    return g


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(rng):
    return SpeciesRegistry(rng)


@pytest.fixture
def registry_with_founder(rng):
    reg = SpeciesRegistry(rng)
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

    def test_custom_threshold(self, rng):
        reg = SpeciesRegistry(rng, compatibility_threshold=0.99)
        assert reg.compatibility_threshold == 0.99

    def test_founder_seeds_living_centroid(self, registry_with_founder):
        reg, _, name = registry_with_founder
        # Registering a founder makes it a living species immediately.
        assert reg.living_species_count == 1
        assert reg.centroid(name) is not None


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
            registry.register_founding_species(rng.standard_normal(GENE_DIMS), name=f"Species {i}")
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
# assign_species — isolation and bootstrap behaviour
# ---------------------------------------------------------------------------


class TestAssignSpeciesNew:
    def test_diverged_newborn_joins_nearest_species_no_candidate(self, registry_with_founder):
        """A newborn with diverged genes is assigned to the nearest living species.
        No candidate is created — population-level splits are detected by k-means."""
        reg, _, founder_name = registry_with_founder
        diverged = make_creature(far_genes(seed=42), parent_species=founder_name)
        returned_name = reg.assign_species(diverged)

        assert reg.species_count == 1
        assert len(reg.speciation_events) == 0
        assert len(reg._candidates) == 0
        # Assigned to the only living species (the founder)
        assert returned_name == founder_name
        assert diverged.species == founder_name

    def test_no_registry_entries_triggers_immediate_species(self, rng):
        """Bootstrap case: first-ever creature gets a confirmed species immediately."""
        reg = SpeciesRegistry(rng)
        c = make_creature(far_genes(seed=1), parent_species="unknown")
        name = reg.assign_species(c)
        assert reg.species_count == 1
        assert c.species == name
        assert len(reg._candidates) == 0


# ---------------------------------------------------------------------------
# Candidate promotion
# ---------------------------------------------------------------------------


def _inject_candidate(reg, genes, parent_species, detected_week=0):
    """
    Inject a k-means-style candidate directly into the registry.

    Simulates the result of detect_subcluster_splits() seeding a candidate
    without going through assign_species().  Returns (cid, seed_creature).
    """
    compat = reg._compat(genes)
    cid = f"cand_{reg._next_candidate_id}"
    reg._next_candidate_id += 1
    seed = make_creature(genes, parent_species=parent_species)
    reg._candidates[cid] = {
        "genes": genes.copy(),
        "compat": compat.copy(),
        "parent_species": parent_species,
        "detected_week": detected_week,
        "members": {seed.creature_id},
        "first_creature_id": seed.creature_id,
        "peak_members": 1,
        "origin": "kmeans_subcluster",
    }
    return cid, seed


def _add_members_to_candidate(reg, cid, n_extra, base_genes):
    """Add n_extra creatures as members of an existing candidate (for test setup)."""
    members = []
    for i in range(n_extra):
        c = make_creature(near_genes(base_genes, noise=0.001, seed=100 + i))
        cand = reg._candidates[cid]
        cand["members"].add(c.creature_id)
        cand["peak_members"] = max(cand["peak_members"], len(cand["members"]))
        members.append(c)
    return members


class TestCandidatePromotion:
    def test_candidate_not_promoted_with_too_few_members(self, registry_with_founder):
        """A candidate with fewer than min_species_population members is not promoted."""
        reg, _, founder_name = registry_with_founder
        cid, first = _inject_candidate(reg, far_genes(seed=42), founder_name, detected_week=0)
        # Only 1 member; min_species_population=3 → should not promote
        reg.promote_candidates([first], current_week=99)

        assert reg.species_count == 1
        assert len(reg.speciation_events) == 0
        assert len(reg._candidates) == 1

    def test_candidate_not_promoted_before_min_weeks(self, registry_with_founder):
        """A candidate with enough members but insufficient age is not promoted."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        cid, first = _inject_candidate(reg, diverged_genes, founder_name, detected_week=0)
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
        cid, first = _inject_candidate(reg, diverged_genes, founder_name, detected_week=0)
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
        cid, first = _inject_candidate(reg, diverged_genes, founder_name, detected_week=0)
        extras = _add_members_to_candidate(reg, cid, 2, diverged_genes)
        all_members = [first] + extras

        reg.promote_candidates(all_members, current_week=10)

        new_name = reg.speciation_events[0]["new_species"]
        for c in all_members:
            assert c.species == new_name

    def test_promotion_speciation_event_has_correct_fields(self, registry_with_founder):
        """The speciation event logged on promotion carries the k-means origin."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        cid, first = _inject_candidate(reg, diverged_genes, founder_name, detected_week=0)
        extras = _add_members_to_candidate(reg, cid, 2, diverged_genes)

        reg.promote_candidates([first] + extras, current_week=10)

        ev = reg.speciation_events[0]
        assert ev["parent_species"] == founder_name
        assert ev["creature_id"] == first.creature_id
        assert ev["week"] == 10
        assert ev["event_type"] == "cladogenesis_kmeans_subcluster"

    def test_promoted_species_name_is_adjective_noun(self, registry_with_founder):
        """The promoted species gets a valid adjective-noun name."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        cid, first = _inject_candidate(reg, diverged_genes, founder_name, detected_week=0)
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
        cid, first = _inject_candidate(reg, diverged_genes, founder_name, detected_week=0)
        extras = _add_members_to_candidate(reg, cid, 2, diverged_genes)

        reg.promote_candidates([first] + extras, current_week=10)

        new_name = reg.speciation_events[0]["new_species"]
        assert reg.progenitor_genes(new_name) is not None

    def test_candidate_evaporates_when_all_members_die(self, registry_with_founder):
        """A candidate with no living members is removed and records a failed attempt."""
        reg, _, founder_name = registry_with_founder
        _, seed = _inject_candidate(reg, far_genes(seed=42), founder_name, detected_week=0)
        assert len(reg._candidates) == 1

        reg.promote_candidates([], current_week=99)

        assert len(reg._candidates) == 0
        assert len(reg.speciation_events) == 0
        assert reg.species_count == 1
        assert len(reg.failed_speciation_attempts) == 1

    def test_failed_attempt_records_correct_fields(self, registry_with_founder):
        """A failed attempt captures parent_species, detected_week, failed_week, peak_members."""
        reg, _, founder_name = registry_with_founder
        _, seed = _inject_candidate(reg, far_genes(seed=42), founder_name, detected_week=7)

        reg.promote_candidates([], current_week=12)

        attempt = reg.failed_speciation_attempts[0]
        assert attempt["parent_species"] == founder_name
        assert attempt["detected_week"] == 7
        assert attempt["failed_week"] == 12
        assert attempt["peak_members"] == 1
        assert attempt["first_creature_id"] == seed.creature_id

    def test_peak_members_reflects_maximum_simultaneous_count(self, registry_with_founder):
        """peak_members is the maximum number of members the candidate ever had at once."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)

        # Inject with 1 seed then add 2 more → peak_members should reach 3
        cid, first = _inject_candidate(reg, diverged_genes, founder_name)
        _add_members_to_candidate(reg, cid, 2, diverged_genes)

        cand = next(iter(reg._candidates.values()))
        assert cand["peak_members"] == 3

        # Evaporate — peak_members should be preserved in the failure record
        reg.promote_candidates([], current_week=50)
        assert reg.failed_speciation_attempts[0]["peak_members"] == 3

    def test_successful_promotion_not_recorded_as_failure(self, registry_with_founder):
        """A candidate that meets criteria and is promoted should NOT appear in failed attempts."""
        reg, _, founder_name = registry_with_founder
        diverged_genes = far_genes(seed=42)
        cid, first = _inject_candidate(reg, diverged_genes, founder_name, detected_week=0)
        extras = _add_members_to_candidate(reg, cid, 2, diverged_genes)

        reg.promote_candidates([first] + extras, current_week=10)

        assert len(reg.failed_speciation_attempts) == 0
        assert len(reg.speciation_events) == 1


# ---------------------------------------------------------------------------
# assign_species — drift-back / convergence protection
# ---------------------------------------------------------------------------


class TestDriftBack:
    def test_drift_back_to_ancestor_not_new_species(self, rng):
        """
        A creature carrying a diverged species label but with genes close to
        an ancestral progenitor should be re-absorbed into the ancestral
        confirmed species, NOT trigger a new candidate or third species.
        Both species are registered as founders so we can test the assignment
        logic directly without depending on two-stage promotion.
        """
        rng = np.random.default_rng(7)
        reg = SpeciesRegistry(rng)

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

    def test_convergent_lineages_share_species(self, rng):
        """
        Two independently evolving lineages that converge toward the same
        genetic region should both be assigned to the same confirmed species,
        not create two separate candidates.
        """
        rng = np.random.default_rng(11)
        reg = SpeciesRegistry(rng)

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

# ---------------------------------------------------------------------------
# Detection keys on the compatibility subset only
# ---------------------------------------------------------------------------


class TestCompatibilitySignal:
    def test_divergence_outside_compat_subset_does_not_speciate(self, rng):
        """A genome that differs from the founder ONLY outside the compatibility
        loci stays the same species — full-genome drift is no longer the signal."""
        rng = np.random.default_rng(3)
        reg = SpeciesRegistry(rng)
        base = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(base, name="Base Dweller")

        # Identical in the compatibility subset, wildly different elsewhere.
        c = make_creature(with_new_noncompat(base, seed=50), parent_species=founder)
        assigned = reg.assign_species(c)

        assert assigned == founder
        assert reg.species_count == 1
        assert len(reg._candidates) == 0

    def test_divergence_in_compat_subset_assigns_to_nearest_species(self, rng):
        """A genome that differs in the compatibility loci is assigned to the nearest
        living species — population-level splits are detected by k-means, not per-newborn."""
        rng = np.random.default_rng(3)
        reg = SpeciesRegistry(rng)
        base = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(base, name="Base Dweller")

        c = make_creature(with_new_compat(base, seed=50), parent_species=founder)
        assigned = reg.assign_species(c)

        assert assigned == founder
        assert reg.species_count == 1
        assert len(reg._candidates) == 0


# ---------------------------------------------------------------------------
# refresh_centroids
# ---------------------------------------------------------------------------


class TestRefreshCentroids:
    def test_centroid_is_member_compat_mean(self, rng):
        rng = np.random.default_rng(4)
        reg = SpeciesRegistry(rng)
        base = rng.standard_normal(GENE_DIMS)
        name = reg.register_founding_species(base, name="Mean Seeker")

        members = [
            make_creature(near_genes(base, noise=0.001, seed=i), parent_species=name) for i in range(5)
        ]
        reg.refresh_centroids(members)

        expected = np.mean([m.genes[COMPAT_IDX] for m in members], axis=0)
        np.testing.assert_allclose(reg.centroid(name), expected, rtol=1e-6)

    def test_extinct_species_pruned_from_living_set(self, rng):
        rng = np.random.default_rng(4)
        reg = SpeciesRegistry(rng)
        a = reg.register_founding_species(rng.standard_normal(GENE_DIMS), name="Alive One")
        b = reg.register_founding_species(rng.standard_normal(GENE_DIMS), name="Dead One")

        # Only species A has living members at refresh time.
        living = [make_creature(reg.progenitor_genes(a), parent_species=a) for _ in range(3)]
        reg.refresh_centroids(living)

        assert reg.species_count == 2  # both remain in the historical registry
        assert reg.living_species_count == 1  # only A is a live comparison target
        assert reg.centroid(b) is None
        assert reg.centroid(a) is not None

    def test_candidate_members_excluded_from_parent_centroid(self, rng):
        """Incipiently-divergent candidate members must not drag the parent
        species' reproductive centre toward themselves."""
        rng = np.random.default_rng(8)
        reg = SpeciesRegistry(rng)
        base = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(base, name="Anchor Walker")

        # Normal members near the founder.
        normal = [
            make_creature(near_genes(base, noise=0.001, seed=i), parent_species=founder) for i in range(3)
        ]
        # A diverging cluster (far in compat space) — inject as a k-means candidate.
        far_compat = with_new_compat(base, seed=70)
        candidate_members = [
            make_creature(near_genes(far_compat, noise=0.001, seed=200 + i), parent_species=founder)
            for i in range(4)
        ]
        cid, _ = _inject_candidate(reg, candidate_members[0].genes, founder)
        # Replace the internal seed ID with the actual candidate_members' IDs so
        # refresh_centroids excludes the right creatures from the parent centroid.
        reg._candidates[cid]["members"] = {c.creature_id for c in candidate_members}
        assert len(reg._candidates) == 1

        reg.refresh_centroids(normal + candidate_members)

        centroid = reg.centroid(founder)
        # Centroid tracks the normal members, NOT the diverging candidate cluster.
        assert _cos(centroid, base[COMPAT_IDX]) > 0.99
        assert _cos(centroid, far_compat[COMPAT_IDX]) < reg.compatibility_threshold


# ---------------------------------------------------------------------------
# Regression: a uniformly drifting population stays ONE species
# ---------------------------------------------------------------------------


class TestDriftRegression:
    def test_population_wide_drift_does_not_speciate(self, rng):
        """The core bug: with a frozen progenitor, a whole population drifting
        together eventually falls below threshold and speciates.  With a living
        centroid the reference moves with the population, so it does not."""
        rng = np.random.default_rng(12)
        reg = SpeciesRegistry(rng)
        base = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(base, name="Origin Drifter")

        # The whole population has drifted far in the compatibility subset, but
        # remains internally coherent (all near the new location).
        drifted = with_new_compat(base, seed=33)
        living = [
            make_creature(near_genes(drifted, noise=0.001, seed=i), parent_species=founder) for i in range(6)
        ]
        newborn_genes = near_genes(drifted, noise=0.001, seed=500)

        # Before refresh: the newborn is far from the FOUNDING centroid — the old
        # frozen-reference logic would have flagged it as a new species.
        before_name, before_score = reg._closest_centroid(reg._compat(newborn_genes))
        assert before_score < reg.compatibility_threshold
        # And it has genuinely left the founding region.
        assert _cos(reg._compat(newborn_genes), base[COMPAT_IDX]) < reg.compatibility_threshold

        # Refresh moves the reference to the living population.
        reg.refresh_centroids(living)
        _, after_score = reg._closest_centroid(reg._compat(newborn_genes))
        assert after_score >= reg.compatibility_threshold

        # Assigning the newborn now keeps it in the same species — no candidate,
        # no speciation event.
        newborn = make_creature(newborn_genes, parent_species=founder)
        assigned = reg.assign_species(newborn)
        assert assigned == founder
        assert reg.species_count == 1
        assert len(reg._candidates) == 0
        assert len(reg.speciation_events) == 0


class TestUniqueNames:
    def test_generated_names_are_unique(self, rng):
        reg = SpeciesRegistry(rng)
        rng = np.random.default_rng(0)
        names = set()
        for _ in range(50):
            name = reg.register_founding_species(rng.standard_normal(GENE_DIMS))
            names.add(name)
        assert len(names) == 50

    def test_name_format(self, rng):
        reg = SpeciesRegistry(rng)
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
        assert "1 species" in repr(reg)

    def test_repr_contains_candidate_count(self, registry_with_founder):
        reg, _, _ = registry_with_founder
        assert "0 candidates" in repr(reg)

    def test_repr_contains_event_count(self, registry_with_founder):
        reg, _, _ = registry_with_founder
        assert "0 speciation events" in repr(reg)


# ---------------------------------------------------------------------------
# Phenotype computation (anagenesis axis)
# ---------------------------------------------------------------------------


class TestPhenotype:
    def test_matches_per_creature_owa(self):
        rng = np.random.default_rng(0)
        genes = rng.standard_normal(GENE_DIMS)
        ph = compute_phenotype(genes)
        c = Creature(genes=genes.copy())
        for j, t in enumerate(PHENOTYPE_TRAITS):
            assert abs(ph[j] - c._compute_trait(t)) < 1e-9

    def test_matrix_matches_single(self):
        rng = np.random.default_rng(1)
        M = rng.standard_normal((4, GENE_DIMS))
        mat = compute_phenotype_matrix(M)
        for i in range(4):
            np.testing.assert_allclose(mat[i], compute_phenotype(M[i]), rtol=1e-9)

    def test_excludes_genetic_machinery_traits(self, rng):
        for t in ("compatibility_genes", "sex_determination", "mutation_rate", "selectivity"):
            assert t not in PHENOTYPE_TRAITS


# ---------------------------------------------------------------------------
# Spherical k-means + K-selection
# ---------------------------------------------------------------------------


class TestClustering:
    def test_two_isolated_blobs_select_k2(self, rng):
        reg = SpeciesRegistry(rng)
        rng = np.random.default_rng(2)
        a = rng.standard_normal(245)
        b = rng.standard_normal(245)
        A = a + 0.01 * rng.standard_normal((6, 245))
        B = b + 0.01 * rng.standard_normal((6, 245))
        X = unit_rows(np.vstack([A, B]))
        labels, centroids = reg._select_clusters(X, seed=0)
        assert centroids.shape[0] == 2
        assert len(set(labels[:6].tolist())) == 1
        assert len(set(labels[6:].tolist())) == 1
        assert labels[0] != labels[6]

    def test_single_blob_stays_k1(self, rng):
        reg = SpeciesRegistry(rng)
        rng = np.random.default_rng(3)
        a = rng.standard_normal(245)
        X = unit_rows(a + 0.01 * rng.standard_normal((10, 245)))
        _, centroids = reg._select_clusters(X, seed=0)
        assert centroids.shape[0] == 1


# ---------------------------------------------------------------------------
# Sub-cluster split detector (cladogenesis)
# ---------------------------------------------------------------------------


class TestSubclusterSplit:
    def _two_clusters(self, reg, base, founder, n_near=5, n_far=5):
        near = [make_creature(near_genes(base, 0.001, seed=i), parent_species=founder) for i in range(n_near)]
        far_base = with_new_compat(base, seed=777)
        far = [
            make_creature(near_genes(far_base, 0.001, seed=100 + i), parent_species=founder)
            for i in range(n_far)
        ]
        return near, far

    def test_split_seeds_candidate_for_diverged_cluster(self, rng):
        rng = np.random.default_rng(5)
        reg = SpeciesRegistry(rng)
        base = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(base, name="Root")
        near, far = self._two_clusters(reg, base, founder)

        reg.detect_subcluster_splits(near + far, current_week=10)

        assert len(reg._candidates) == 1
        cand = next(iter(reg._candidates.values()))
        # The cluster nearest the type keeps the name; the far cluster splits off.
        assert cand["members"] <= {c.creature_id for c in far}
        assert cand["parent_species"] == founder

    def test_split_promotes_and_relabels_via_existing_gate(self, rng):
        rng = np.random.default_rng(6)
        reg = SpeciesRegistry(rng)
        base = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(base, name="Root")
        near, far = self._two_clusters(reg, base, founder)
        allc = near + far

        reg.detect_subcluster_splits(allc, current_week=0)
        reg.promote_candidates(allc, current_week=10)  # age >= min_species_weeks

        assert reg.species_count == 2
        ev = reg.speciation_events[-1]
        assert ev["event_type"] == "cladogenesis_kmeans_subcluster"
        assert all(c.species != founder for c in far)
        assert all(c.species == founder for c in near)

    def test_coherent_species_not_split(self, rng):
        rng = np.random.default_rng(7)
        reg = SpeciesRegistry(rng)
        base = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(base, name="Root")
        members = [make_creature(near_genes(base, 0.01, seed=i), parent_species=founder) for i in range(10)]
        reg.detect_subcluster_splits(members, current_week=10)
        assert len(reg._candidates) == 0


# ---------------------------------------------------------------------------
# Anagenesis detector (in-place transformation)
# ---------------------------------------------------------------------------


class TestAnagenesis:
    def test_phenotype_drift_mints_descendant_and_respeciates(self, rng):
        rng = np.random.default_rng(8)
        reg = SpeciesRegistry(rng)
        g0 = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(g0, name="Ancestor")

        delta = rng.standard_normal(GENE_DIMS)  # whole-lineage directional shift
        members = [
            make_creature(g0 + delta + 0.001 * rng.standard_normal(GENE_DIMS), parent_species=founder)
            for _ in range(6)
        ]
        type_ph = compute_phenotype(g0)
        centroid = compute_phenotype_matrix(np.stack([m.genes for m in members])).mean(axis=0)
        # Detector centres phenotype on 0.5 before the cosine; mirror that here.
        actual = _cos(centroid - 0.5, type_ph - 0.5)
        # Set the bar just above the observed drift so the trigger is deterministic.
        reg.anagenesis_threshold = actual + 0.005
        reg.anagenesis_weeks = 0  # no persistence wait for this single-call test

        events = reg.detect_anagenesis(members, current_week=20)

        assert len(events) == 1
        assert events[0]["event_type"] == "anagenesis"
        assert reg.species_count == 2
        # The whole lineage moved → all members carry the descendant name.
        assert all(m.species == events[0]["new_species"] for m in members)

    def test_no_drift_no_event(self, rng):
        rng = np.random.default_rng(9)
        reg = SpeciesRegistry(rng)
        g0 = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(g0, name="Ancestor")
        members = [make_creature(near_genes(g0, 0.001, seed=i), parent_species=founder) for i in range(6)]
        type_ph = compute_phenotype(g0)
        centroid = compute_phenotype_matrix(np.stack([m.genes for m in members])).mean(axis=0)
        # below the observed (centred) drift → no trigger
        reg.anagenesis_threshold = _cos(centroid - 0.5, type_ph - 0.5) - 0.01

        events = reg.detect_anagenesis(members, current_week=20)
        assert events == []
        assert reg.species_count == 1

    def test_species_with_active_split_candidate_is_skipped(self, rng):
        rng = np.random.default_rng(10)
        reg = SpeciesRegistry(rng)
        g0 = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(g0, name="Ancestor")
        members = [
            make_creature(g0 + rng.standard_normal(GENE_DIMS), parent_species=founder) for _ in range(6)
        ]
        # Force an active split candidate for the founder.
        reg._candidates["cand_x"] = {
            "genes": members[0].genes.copy(),
            "compat": reg._compat(members[0].genes).copy(),
            "parent_species": founder,
            "detected_week": 0,
            "members": {members[0].creature_id},
            "first_creature_id": members[0].creature_id,
            "peak_members": 1,
        }
        reg.anagenesis_threshold = 1.0  # would fire if not skipped
        events = reg.detect_anagenesis(members, current_week=20)
        assert events == []

    def test_persistence_gate_delays_minting(self, rng):
        rng = np.random.default_rng(11)
        reg = SpeciesRegistry(rng)
        g0 = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(g0, name="Ancestor")
        delta = rng.standard_normal(GENE_DIMS)
        members = [
            make_creature(g0 + delta + 0.001 * rng.standard_normal(GENE_DIMS), parent_species=founder)
            for _ in range(6)
        ]
        type_ph = compute_phenotype(g0)
        centroid = compute_phenotype_matrix(np.stack([m.genes for m in members])).mean(axis=0)
        reg.anagenesis_threshold = _cos(centroid - 0.5, type_ph - 0.5) + 0.005
        reg.anagenesis_weeks = 50

        # First detection starts the persistence clock — no event yet.
        assert reg.detect_anagenesis(members, current_week=100) == []
        assert reg.species_count == 1
        # Still within the window.
        assert reg.detect_anagenesis(members, current_week=140) == []
        # Past the window → descendant is minted.
        events = reg.detect_anagenesis(members, current_week=160)
        assert len(events) == 1
        assert events[0]["event_type"] == "anagenesis"
        assert reg.species_count == 2

    def test_rebound_resets_persistence_clock(self, rng):
        rng = np.random.default_rng(12)
        reg = SpeciesRegistry(rng)
        g0 = rng.standard_normal(GENE_DIMS)
        founder = reg.register_founding_species(g0, name="Ancestor")
        drifted = [
            make_creature(
                g0 + rng.standard_normal(GENE_DIMS) + 0.001 * rng.standard_normal(GENE_DIMS),
                parent_species=founder,
            )
            for _ in range(6)
        ]
        type_ph = compute_phenotype(g0)
        cen = compute_phenotype_matrix(np.stack([m.genes for m in drifted])).mean(axis=0)
        reg.anagenesis_threshold = _cos(cen - 0.5, type_ph - 0.5) + 0.005
        reg.anagenesis_weeks = 50

        reg.detect_anagenesis(drifted, current_week=100)  # clock starts
        # A coherent (near-type) population appears → above threshold → clock clears.
        coherent = [make_creature(near_genes(g0, 0.001, seed=i), parent_species=founder) for i in range(6)]
        assert reg.detect_anagenesis(coherent, current_week=120) == []
        assert "Ancestor" not in reg._anagenesis_pending
