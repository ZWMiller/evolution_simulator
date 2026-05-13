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
# assign_species — new species
# ---------------------------------------------------------------------------

class TestAssignSpeciesNew:
    def test_diverged_genome_creates_new_species(self, registry_with_founder):
        reg, _, founder_name = registry_with_founder
        diverged = make_creature(far_genes(seed=42), parent_species=founder_name)
        name = reg.assign_species(diverged)
        assert name != founder_name
        assert reg.species_count == 2
        assert len(reg.speciation_events) == 1

    def test_new_species_name_is_adjective_noun(self, registry_with_founder):
        reg, _, founder_name = registry_with_founder
        diverged = make_creature(far_genes(seed=43), parent_species=founder_name)
        name = reg.assign_species(diverged)
        adj, noun = name.split(" ", 1)
        assert adj in ADJECTIVES
        assert noun in NOUNS

    def test_speciation_event_logged(self, registry_with_founder):
        reg, _, founder_name = registry_with_founder
        diverged = make_creature(far_genes(seed=44), parent_species=founder_name)
        new_name = reg.assign_species(diverged)
        assert len(reg.speciation_events) == 1
        event = reg.speciation_events[0]
        assert event["new_species"] == new_name
        assert event["parent_species"] == founder_name
        assert event["creature_id"] == diverged.creature_id

    def test_new_species_progenitor_stored(self, registry_with_founder):
        reg, _, founder_name = registry_with_founder
        diverged = make_creature(far_genes(seed=45), parent_species=founder_name)
        new_name = reg.assign_species(diverged)
        np.testing.assert_array_equal(
            reg.progenitor_genes(new_name), diverged.genes
        )

    def test_no_registry_entries_triggers_new_species(self):
        reg = SpeciesRegistry()
        c = make_creature(far_genes(seed=1), parent_species="unknown")
        name = reg.assign_species(c)
        assert reg.species_count == 1
        assert c.species == name


# ---------------------------------------------------------------------------
# assign_species — drift-back / convergence protection
# ---------------------------------------------------------------------------

class TestDriftBack:
    def test_drift_back_to_ancestor_not_new_species(self):
        """
        A creature that speciated away and then drifts back toward the
        ancestral progenitor should be re-absorbed into the ancestral species,
        NOT logged as a third new species.
        """
        rng = np.random.default_rng(7)
        reg = SpeciesRegistry()

        # Register ancestor
        ancestor_genes = rng.standard_normal(GENE_DIMS)
        ancestor_name = reg.register_founding_species(ancestor_genes, name="Ancient Wanderer")

        # A diverged creature creates species 2
        diverged_genes = rng.standard_normal(GENE_DIMS)
        diverged = make_creature(diverged_genes, parent_species=ancestor_name)
        sp2_name = reg.assign_species(diverged)
        assert sp2_name != ancestor_name
        assert reg.species_count == 2

        # A new creature with genes very close to the ANCESTOR (drift-back)
        drifted_back = make_creature(
            near_genes(ancestor_genes, noise=0.001, seed=10),
            parent_species=sp2_name,  # inherits species 2 from its parent
        )
        assigned = reg.assign_species(drifted_back)

        # Should be re-classified as Ancient Wanderer, NOT a third species
        assert assigned == ancestor_name
        assert reg.species_count == 2          # no new species
        assert len(reg.speciation_events) == 1  # only the original divergence

    def test_convergent_lineages_share_species(self):
        """
        Two independently diverged lineages that both converge toward the
        same genetic region should be assigned to the same species.
        """
        rng = np.random.default_rng(11)
        reg = SpeciesRegistry()

        base = rng.standard_normal(GENE_DIMS)
        founder_name = reg.register_founding_species(base, name="Origin Seeker")

        # Two separate diverged creatures both converge toward the same target
        target = rng.standard_normal(GENE_DIMS)
        # Register the target as a second species (via first convergent)
        c1 = make_creature(near_genes(target, noise=0.001, seed=20), parent_species=founder_name)
        sp2 = reg.assign_species(c1)
        assert sp2 != founder_name

        # Second lineage also converges toward target — should join sp2
        c2 = make_creature(near_genes(target, noise=0.001, seed=21), parent_species=founder_name)
        sp2_again = reg.assign_species(c2)
        assert sp2_again == sp2
        assert reg.species_count == 2  # still only 2 species


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
        assert "1 species" in repr(reg)

    def test_repr_contains_event_count(self, registry_with_founder):
        reg, _, _ = registry_with_founder
        assert "0 speciation events" in repr(reg)
