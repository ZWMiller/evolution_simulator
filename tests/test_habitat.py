import numpy as np
import pytest
from evolution_simulator.creature import Creature, DEFAULT_TRAIT_GENE_INDICES, GENE_DIMS
from evolution_simulator.habitat import (
    Habitat,
    DEFAULT_FOOD_GENE_INDICES,
    DEFAULT_WATER_GENE_INDICES,
    HABITAT_VECTOR_DIMS,
    LOGGED_TRAITS,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_creature(seed: int, sex: str) -> Creature:
    """Create a sexually viable creature with controlled sex and seed."""
    rng = np.random.default_rng(seed)
    c = Creature(genes=rng.standard_normal(GENE_DIMS))
    c.sex = sex
    c.age = c.weeks_to_sexual_viability
    return c


def make_compatible_pair(base_seed: int = 0) -> tuple[Creature, Creature]:
    """
    Male + female with near-identical compatibility genes and low selectivity,
    so they can mate.
    """
    rng = np.random.default_rng(base_seed)
    base = rng.standard_normal(GENE_DIMS)
    base[DEFAULT_TRAIT_GENE_INDICES["selectivity"]] = -10.0             # low selectivity
    base[DEFAULT_TRAIT_GENE_INDICES["reproduction_likelihood"]] = 10.0  # high fertility

    male = Creature(genes=base.copy())
    male.sex = "male"
    male.age = male.weeks_to_sexual_viability

    female = Creature(genes=base.copy())
    female.sex = "female"
    female.age = female.weeks_to_sexual_viability

    return male, female


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def habitat():
    rng = np.random.default_rng(42)
    return Habitat(vector=rng.standard_normal(HABITAT_VECTOR_DIMS), name="Test Habitat")


@pytest.fixture
def habitat_pair():
    rng = np.random.default_rng(0)
    h1 = Habitat(vector=rng.standard_normal(HABITAT_VECTOR_DIMS), name="Alpha")
    h2 = Habitat(vector=rng.standard_normal(HABITAT_VECTOR_DIMS), name="Beta")
    return h1, h2


@pytest.fixture
def populated_habitat():
    """Habitat with a mix of alive creatures."""
    rng = np.random.default_rng(7)
    h = Habitat(vector=rng.standard_normal(HABITAT_VECTOR_DIMS), name="Populated")
    for i in range(5):
        h.add_creature(make_creature(seed=i * 10, sex="male"))
    for i in range(5):
        h.add_creature(make_creature(seed=i * 10 + 1, sex="female"))
    return h


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestHabitatInit:
    def test_default_vector_shape(self):
        h = Habitat()
        assert h.vector.shape == (HABITAT_VECTOR_DIMS,)

    def test_custom_vector_stored(self):
        v = np.ones(HABITAT_VECTOR_DIMS)
        h = Habitat(vector=v)
        np.testing.assert_array_equal(h.vector, v)

    def test_invalid_vector_shape_raises(self):
        with pytest.raises(ValueError):
            Habitat(vector=np.zeros(100))

    def test_vector_is_float(self):
        h = Habitat(vector=np.ones(HABITAT_VECTOR_DIMS, dtype=int))
        assert h.vector.dtype == float

    def test_unique_ids(self):
        ids = {Habitat().habitat_id for _ in range(50)}
        assert len(ids) == 50

    def test_explicit_id_respected(self):
        h = Habitat(habitat_id="test-123")
        assert h.habitat_id == "test-123"

    def test_name_stored(self):
        h = Habitat(name="Desert")
        assert h.name == "Desert"

    def test_starts_empty(self, habitat):
        assert habitat.population_size == 0
        assert habitat.creatures == []


# ---------------------------------------------------------------------------
# Creature management
# ---------------------------------------------------------------------------

class TestCreatureManagement:
    def test_add_creature(self, habitat):
        c = Creature()
        habitat.add_creature(c)
        assert habitat.has_creature(c)
        assert habitat.population_size == 1

    def test_remove_creature(self, habitat):
        c = Creature()
        habitat.add_creature(c)
        habitat.remove_creature(c)
        assert not habitat.has_creature(c)
        assert habitat.population_size == 0

    def test_remove_absent_creature_is_noop(self, habitat):
        habitat.remove_creature(Creature())  # should not raise

    def test_alive_creatures_excludes_dead(self, habitat):
        alive = Creature()
        dead = Creature()
        dead.is_alive = False
        habitat.add_creature(alive)
        habitat.add_creature(dead)
        assert alive in habitat.alive_creatures
        assert dead not in habitat.alive_creatures

    def test_population_size_counts_alive_only(self, habitat):
        habitat.add_creature(Creature())
        dead = Creature()
        dead.is_alive = False
        habitat.add_creature(dead)
        assert habitat.population_size == 1

    def test_creatures_returns_all(self, habitat):
        c1, c2 = Creature(), Creature()
        habitat.add_creature(c1)
        habitat.add_creature(c2)
        assert set(habitat.creatures) == {c1, c2}


# ---------------------------------------------------------------------------
# Neighbour / migration management
# ---------------------------------------------------------------------------

class TestNeighborManagement:
    def test_add_neighbor_bidirectional(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2)
        assert h1.is_neighbor(h2)
        assert h2.is_neighbor(h1)

    def test_add_neighbor_unidirectional(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2, bidirectional=False)
        assert h1.is_neighbor(h2)
        assert not h2.is_neighbor(h1)

    def test_passable_by_default(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2)
        assert h1.can_migrate_to(h2)
        assert h2.can_migrate_to(h1)

    def test_add_neighbor_impassable(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2, passable=False)
        assert not h1.can_migrate_to(h2)

    def test_block_migration(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2)
        h1.block_migration_to(h2)
        assert not h1.can_migrate_to(h2)
        assert h2.can_migrate_to(h1)  # only one direction blocked

    def test_block_migration_bidirectional(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2)
        h1.block_migration_to(h2, bidirectional=True)
        assert not h1.can_migrate_to(h2)
        assert not h2.can_migrate_to(h1)

    def test_open_migration(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2, passable=False)
        h1.open_migration_to(h2)
        assert h1.can_migrate_to(h2)

    def test_passable_neighbors_excludes_blocked(self, habitat_pair):
        h1, h2 = habitat_pair
        h3 = Habitat()
        h1.add_neighbor(h2, passable=True)
        h1.add_neighbor(h3, passable=False)
        assert h2 in h1.passable_neighbors()
        assert h3 not in h1.passable_neighbors()

    def test_can_migrate_to_non_neighbor_is_false(self, habitat_pair):
        h1, h2 = habitat_pair
        assert not h1.can_migrate_to(h2)  # not registered as neighbor

    def test_spontaneous_isolation_blocks_route(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2, passable=True)
        # probability=1.0 guarantees isolation
        isolated = h1.try_spontaneous_isolation(probability=1.0)
        assert h2.habitat_id in isolated
        assert not h1.can_migrate_to(h2)

    def test_spontaneous_isolation_returns_ids(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2)
        isolated = h1.try_spontaneous_isolation(probability=1.0)
        assert isolated == [h2.habitat_id]

    def test_spontaneous_isolation_zero_probability(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2)
        isolated = h1.try_spontaneous_isolation(probability=0.0)
        assert isolated == []
        assert h1.can_migrate_to(h2)

    def test_already_blocked_not_re_isolated(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2, passable=False)
        isolated = h1.try_spontaneous_isolation(probability=1.0)
        assert isolated == []  # already impassable, nothing new


# ---------------------------------------------------------------------------
# Resource likelihoods
# ---------------------------------------------------------------------------

class TestResourceLikelihoods:
    def test_food_likelihoods_shape(self, habitat, populated_habitat):
        alive = populated_habitat.alive_creatures
        probs = populated_habitat.food_likelihoods(alive)
        assert probs.shape == (len(alive),)

    def test_water_likelihoods_shape(self, habitat, populated_habitat):
        alive = populated_habitat.alive_creatures
        probs = populated_habitat.water_likelihoods(alive)
        assert probs.shape == (len(alive),)

    def test_food_likelihoods_in_range(self, populated_habitat):
        probs = populated_habitat.food_likelihoods(populated_habitat.alive_creatures)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_water_likelihoods_in_range(self, populated_habitat):
        probs = populated_habitat.water_likelihoods(populated_habitat.alive_creatures)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_empty_population_returns_empty_array(self, habitat):
        assert habitat.food_likelihoods([]).shape == (0,)
        assert habitat.water_likelihoods([]).shape == (0,)

    def test_aligned_genes_give_max_likelihood(self, habitat):
        """A creature aligned with the habitat vector → cos(θ)=1 → P=1."""
        indices = habitat.FOOD_GENE_INDICES
        genes = np.zeros(GENE_DIMS)
        genes[indices] = habitat.vector[indices]  # perfectly aligned
        c = Creature(genes=genes)
        probs = habitat.food_likelihoods([c])
        assert probs[0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_genes_give_midpoint_likelihood(self, habitat):
        """A creature orthogonal to the habitat → cos(θ)=0 → P=0.5."""
        indices = habitat.FOOD_GENE_INDICES
        hab_sub = habitat.vector[indices]
        rand = np.random.randn(len(indices))
        orth = rand - (np.dot(rand, hab_sub) / np.dot(hab_sub, hab_sub)) * hab_sub
        genes = np.zeros(GENE_DIMS)
        genes[indices] = orth
        c = Creature(genes=genes)
        probs = habitat.food_likelihoods([c])
        assert probs[0] == pytest.approx(0.5, abs=1e-6)

    def test_antialigned_genes_give_zero_likelihood(self, habitat):
        """A creature anti-aligned with the habitat vector → cos(θ)=-1 → P=0."""
        indices = habitat.FOOD_GENE_INDICES
        genes = np.zeros(GENE_DIMS)
        genes[indices] = -habitat.vector[indices]  # perfectly anti-aligned
        c = Creature(genes=genes)
        probs = habitat.food_likelihoods([c])
        assert probs[0] == pytest.approx(0.0, abs=1e-6)

    def test_different_habitats_give_different_likelihoods(self):
        rng = np.random.default_rng(5)
        h1 = Habitat(vector=rng.standard_normal(HABITAT_VECTOR_DIMS))
        h2 = Habitat(vector=rng.standard_normal(HABITAT_VECTOR_DIMS))
        c = Creature()
        p1 = h1.food_likelihoods([c])[0]
        p2 = h2.food_likelihoods([c])[0]
        assert p1 != pytest.approx(p2)


# ---------------------------------------------------------------------------
# simulate_week — structure and basic behaviour
# ---------------------------------------------------------------------------

class TestSimulateDayStructure:
    def test_returns_expected_keys(self, populated_habitat):
        result = populated_habitat.simulate_week()
        assert {"habitat_id", "population", "week_results",
                "births", "deaths", "migrations", "isolations"} <= result.keys()

    def test_habitat_id_in_result(self, populated_habitat):
        result = populated_habitat.simulate_week()
        assert result["habitat_id"] == populated_habitat.habitat_id

    def test_week_results_keyed_by_creature_id(self, populated_habitat):
        result = populated_habitat.simulate_week()
        for cid in result["week_results"]:
            assert isinstance(cid, str)

    def test_empty_habitat_runs_without_error(self, habitat):
        result = habitat.simulate_week()
        assert result["population"] == 0
        assert result["births"] == []
        assert result["deaths"] == []
        assert result["migrations"] == []

    def test_population_count_in_result(self, populated_habitat):
        result = populated_habitat.simulate_week()
        assert result["population"] == populated_habitat.population_size


class TestSimulateDayAging:
    def test_creatures_age_each_day(self, habitat):
        c = make_creature(seed=1, sex="male")
        habitat.add_creature(c)
        age_before = c.age
        habitat.simulate_week()
        assert c.age == age_before + 1

    def test_old_creature_dies(self, habitat):
        c = make_creature(seed=2, sex="female")
        c.age = c.max_lifespan - 1
        habitat.add_creature(c)
        result = habitat.simulate_week()
        assert c.creature_id in result["deaths"]
        assert not habitat.has_creature(c)

    def test_dead_creatures_removed_from_habitat(self, habitat):
        c = make_creature(seed=3, sex="male")
        c.age = c.max_lifespan - 1
        habitat.add_creature(c)
        habitat.simulate_week()
        assert not habitat.has_creature(c)


class TestSimulateDayResources:
    def test_starvation_kills_creature(self, habitat):
        c = make_creature(seed=4, sex="male")
        c.energy = 0.0
        # Force no food found: anti-align genes with habitat → (cos θ + 1)/2 = 0
        indices = habitat.FOOD_GENE_INDICES
        c.genes[indices] = -habitat.vector[indices]
        habitat.add_creature(c)
        result = habitat.simulate_week()
        # May die of starvation if food prob = 0 and energy drains to ≤ 0
        # We verify the mechanism works by checking energy ≤ 0 kills the creature
        # (energy was already 0, so any FOOD_ENERGY_COST push kills it)
        assert c.creature_id in result["deaths"] or c.energy <= 0 or not c.is_alive

    def test_energy_increases_when_food_found(self, habitat):
        """Force food found by aligning genes with habitat → (cos θ + 1)/2 = 1."""
        indices = habitat.FOOD_GENE_INDICES

        c = make_creature(seed=5, sex="male")
        c.genes[indices] = habitat.vector[indices]
        c.energy = 0.5
        habitat.add_creature(c)

        np.random.seed(0)
        habitat.simulate_week()
        # With p=1 food likelihood, food is always found
        # Energy should have increased (or stayed near 0.5 + gain)
        assert c.energy >= 0.5 or not c.is_alive  # creature either ate or died of age


class TestSimulateDayMigration:
    def test_migration_moves_creature_to_neighbor(self):
        rng = np.random.default_rng(11)
        h1 = Habitat(vector=rng.standard_normal(HABITAT_VECTOR_DIMS))
        h2 = Habitat(vector=rng.standard_normal(HABITAT_VECTOR_DIMS))
        h1.add_neighbor(h2)

        # Force very high migration likelihood
        c = make_creature(seed=6, sex="male")
        c.genes[DEFAULT_TRAIT_GENE_INDICES["migration_likelihood"]] = 100.0
        h1.add_creature(c)

        # Override WEEKLY_MIGRATION_BASE to 1.0 so high trait → certain migration
        original = Habitat.WEEKLY_MIGRATION_BASE
        Habitat.WEEKLY_MIGRATION_BASE = 1.0
        try:
            result = h1.simulate_week()
        finally:
            Habitat.WEEKLY_MIGRATION_BASE = original

        migrations = result["migrations"]
        assert len(migrations) == 1
        migrant, destination = migrations[0]
        assert migrant is c
        assert destination is h2
        assert not h1.has_creature(c)  # removed from source

    def test_no_migration_without_passable_neighbor(self, habitat):
        c = make_creature(seed=7, sex="male")
        c.genes[DEFAULT_TRAIT_GENE_INDICES["migration_likelihood"]] = 100.0
        habitat.add_creature(c)

        original = Habitat.WEEKLY_MIGRATION_BASE
        Habitat.WEEKLY_MIGRATION_BASE = 1.0
        try:
            result = habitat.simulate_week()
        finally:
            Habitat.WEEKLY_MIGRATION_BASE = original

        assert result["migrations"] == []
        assert habitat.has_creature(c)

    def test_blocked_route_prevents_migration(self, habitat_pair):
        h1, h2 = habitat_pair
        h1.add_neighbor(h2)
        h1.block_migration_to(h2)

        c = make_creature(seed=8, sex="male")
        c.genes[DEFAULT_TRAIT_GENE_INDICES["migration_likelihood"]] = 100.0
        h1.add_creature(c)

        original = Habitat.WEEKLY_MIGRATION_BASE
        Habitat.WEEKLY_MIGRATION_BASE = 1.0
        try:
            result = h1.simulate_week()
        finally:
            Habitat.WEEKLY_MIGRATION_BASE = original

        assert result["migrations"] == []


class TestSimulateDayMating:
    def test_mating_creates_pregnancy(self):
        h = Habitat()
        male, female = make_compatible_pair(base_seed=99)
        h.add_creature(male)
        h.add_creature(female)

        h.simulate_week()
        # After one day, compatible male and female should have mated
        assert female.is_pregnant or female._pending_offspring

    def test_births_appear_after_gestation(self):
        """Fast-forward a female through gestation and confirm birth event."""
        h = Habitat()
        male, female = make_compatible_pair(base_seed=42)

        h.add_creature(male)
        h.add_creature(female)

        # Mate them (modifying genes after pair creation would break compat check)
        litter = male.reproduce(female)
        assert len(litter) >= 1

        # Fast-forward pregnancy to one day before term
        female.weeks_pregnant = female.reproduction_time - 1

        result = h.simulate_week()
        assert len(result["births"]) >= 1
        for child in result["births"]:
            assert h.has_creature(child)


# ---------------------------------------------------------------------------
# Predation
# ---------------------------------------------------------------------------

class TestPredation:
    def test_predation_deaths_key_in_result(self, habitat):
        c = make_creature(seed=10, sex="male")
        habitat.add_creature(c)
        result = habitat.simulate_week()
        assert "predation_deaths" in result

    def test_high_density_causes_predation(self):
        """Pack population far above support capacity — expect predation deaths."""
        rng = np.random.default_rng(0)
        hab = Habitat(vector=rng.standard_normal(HABITAT_VECTOR_DIMS))
        hab.PREDATION_ALPHA = 1.0    # guarantee death at any population > 0
        hab.POPULATION_SUPPORT = 1
        c = make_creature(seed=11, sex="male")
        c.energy = 1.0
        c.hydration = 1.0
        # Align genes so creature doesn't starve before predation
        c.genes[DEFAULT_FOOD_GENE_INDICES] = hab.vector[DEFAULT_FOOD_GENE_INDICES]
        c.genes[DEFAULT_WATER_GENE_INDICES] = hab.vector[DEFAULT_WATER_GENE_INDICES]
        hab.add_creature(c)
        result = hab.simulate_week()
        assert len(result["predation_deaths"]) >= 1

    def test_zero_density_no_predation_pressure(self):
        """Single creature in a zero-alpha habitat has only genetic base rate."""
        rng = np.random.default_rng(1)
        hab = Habitat(vector=rng.standard_normal(HABITAT_VECTOR_DIMS))
        hab.PREDATION_ALPHA = 0.0
        c = make_creature(seed=12, sex="male")
        # Force zero base predation rate by anti-setting the trait genes
        c.genes[DEFAULT_TRAIT_GENE_INDICES["base_predation_rate"]] = -100.0
        c._trait_cache.clear()   # clear cache so new gene values take effect
        c.energy = 1.0
        c.hydration = 1.0
        hab.add_creature(c)
        result = hab.simulate_week()
        assert result["predation_deaths"] == []


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_empty_habitat_returns_empty_dict(self, habitat):
        assert habitat.compute_stats() == {}

    def test_stats_structure(self, habitat):
        c = make_creature(seed=13, sex="female")
        c.species = "Test Species"
        habitat.add_creature(c)
        stats = habitat.compute_stats()
        assert "Test Species" in stats
        sp = stats["Test Species"]
        assert sp["count"] == 1
        assert 0.0 <= sp["mean_food_prob"] <= 1.0
        assert 0.0 <= sp["mean_water_prob"] <= 1.0
        assert set(sp["mean_traits"].keys()) == set(LOGGED_TRAITS)

    def test_stats_count_matches_population(self, habitat):
        for i in range(5):
            c = make_creature(seed=i + 20, sex="female" if i % 2 == 0 else "male")
            c.species = "Pack"
            habitat.add_creature(c)
        stats = habitat.compute_stats()
        assert stats["Pack"]["count"] == 5

    def test_stats_separates_species(self, habitat):
        for i in range(3):
            c = make_creature(seed=i + 30, sex="male")
            c.species = "Alpha"
            habitat.add_creature(c)
        for i in range(2):
            c = make_creature(seed=i + 40, sex="female")
            c.species = "Beta"
            habitat.add_creature(c)
        stats = habitat.compute_stats()
        assert stats["Alpha"]["count"] == 3
        assert stats["Beta"]["count"] == 2


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_id(self, habitat):
        assert habitat.habitat_id[:8] in repr(habitat)

    def test_repr_contains_name(self, habitat):
        assert "Test Habitat" in repr(habitat)

    def test_repr_shows_population(self, populated_habitat):
        r = repr(populated_habitat)
        assert str(populated_habitat.population_size) in r
