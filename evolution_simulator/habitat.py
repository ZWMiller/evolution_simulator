import uuid
import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .creature import Creature

# Must match GENE_DIMS in creature.py
HABITAT_VECTOR_DIMS = 500

# ---------------------------------------------------------------------------
# Default gene index subsets for habitat–creature resource interactions
# ---------------------------------------------------------------------------
# These indices are drawn from BOTH the habitat vector and the creature gene
# vector when computing the resource-finding probability.
# Subclasses can override these to represent different habitat types (desert,
# ocean, forest, etc.) that interact with different genetic dimensions.
#
# Design note: resource probability uses (cos θ + 1) / 2, where θ is the
# angle between a creature's gene sub-vector and the habitat sub-vector at
# these indices.  A creature aligned with the habitat vector finds resources
# easily (P → 1); an orthogonal creature has a baseline P = 0.5; an
# anti-aligned creature cannot extract resources (P → 0).  This drives
# local adaptation: populations whose genes drift toward alignment with the
# local habitat gain a survival edge, while migrants entering a new habitat
# face immediate resource pressure until they adapt.

# ---------------------------------------------------------------------------
# Traits logged in per-habitat and per-species daily statistics
# ---------------------------------------------------------------------------
# These are property names on Creature.  Logged values use the actual scaled
# ranges (fecundity 1-8, metabolism 0.5-2.0, etc.) for interpretability.
LOGGED_TRAITS: tuple[str, ...] = (
    # Reproduction
    "fecundity", "reproduction_time", "weeks_to_sexual_viability",
    "parental_investment", "reproduction_likelihood",
    # Survival / physiology
    "metabolism", "water_efficiency", "max_lifespan",
    "disease_resistance", "immune_response", "stress_tolerance",
    # Environmental adaptation
    "heat_tolerance", "cold_tolerance", "drought_tolerance", "hibernation_tendency",
    # Movement / behaviour
    "migration_likelihood", "risk_tolerance", "aggression",
    "territorial", "social_tendency", "nocturnal_tendency",
    # Physical
    "size", "strength", "speed", "camouflage",
    # Cognitive / ecological
    "foraging_ability", "intelligence", "adaptability",
    "pack_hunting", "scavenging_tendency", "communication",
    # Genetics
    "mutation_rate", "selectivity",
    # Predation vulnerability (new)
    "base_predation_rate",
)


DEFAULT_FOOD_GENE_INDICES: list[int] = (
    list(range(37, 80))     # foraging ability, water efficiency, intelligence loci
    + list(range(110, 170)) # size, strength, speed, physiology loci
    + list(range(230, 285)) # broad genomic coverage
)  # 168 total indices

DEFAULT_WATER_GENE_INDICES: list[int] = (
    list(range(38, 78))     # water efficiency, drought tolerance loci
    + list(range(115, 175)) # immune, stress, environmental adaptation loci
    + list(range(270, 345)) # broad genomic coverage
)  # 175 total indices


class Habitat:
    """
    A geographic region that creatures inhabit.

    Each Habitat has a 500-dimensional float vector representing the
    environmental conditions of that region.  This vector interacts with
    creature gene vectors via cross-product geometry to determine daily
    resource-finding probabilities.

    Core responsibilities
    ---------------------
    - Track which creatures are present (O(1) add / remove via set).
    - Compute batched food and water likelihoods across the whole population
      using vectorised numpy operations.
    - Manage neighbour connections and migration routes, including support for
      spontaneous geographic isolation (speciation driver).
    - Simulate one full day for all contained creatures, returning a structured
      event log for the simulation runner to act on.

    Class attributes (override in subclasses for habitat-type specialisation)
    --------------------------------------------------------------------------
    FOOD_GENE_INDICES : list[int]
        Creature/habitat gene indices used for food-finding likelihood.
    WATER_GENE_INDICES : list[int]
        Creature/habitat gene indices used for water-finding likelihood.
    FOOD_ENERGY_GAIN : float
        Energy gained when a creature successfully finds food (per day).
    FOOD_ENERGY_COST : float
        Base energy lost when no food is found (scaled by creature.metabolism).
    WATER_HYDRATION_GAIN : float
        Hydration gained when water is found (per day).
    WATER_HYDRATION_COST : float
        Base hydration lost when no water is found
        (scaled by 1 - creature.water_efficiency).
    WEEKLY_MIGRATION_BASE : float
        Multiplier applied to creature.migration_likelihood to get the actual
        per-week migration probability.  Keeps average migration rare even when
        the trait value is moderate.
    """

    FOOD_GENE_INDICES: list[int] = DEFAULT_FOOD_GENE_INDICES
    WATER_GENE_INDICES: list[int] = DEFAULT_WATER_GENE_INDICES

    FOOD_ENERGY_GAIN: float = 0.30
    FOOD_ENERGY_COST: float = 0.15   # multiplied by creature.metabolism
    WATER_HYDRATION_GAIN: float = 0.30
    WATER_HYDRATION_COST: float = 0.25  # multiplied by (1 - creature.water_efficiency)

    # Raw migration_likelihood ∈ [0, 1] is multiplied by this so that a
    # creature with an average trait (~0.5) has only a 0.5% weekly chance of
    # migrating — keeping populations stable while still allowing spread.
    WEEKLY_MIGRATION_BASE: float = 0.01

    # Density-dependent mortality parameters.
    # Per-week death probability from crowding = PREDATION_ALPHA * N / POPULATION_SUPPORT,
    # added to each creature's intrinsic base_predation_rate.  When N = POPULATION_SUPPORT
    # the density term equals PREDATION_ALPHA (~1% for default Forest).  Harsh habitats
    # use lower values of both, rewarding adaptation with reduced crowding pressure.
    PREDATION_ALPHA: float = 0.010
    POPULATION_SUPPORT: int = 400

    def __init__(
        self,
        vector: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        habitat_id: Optional[str] = None,
        population_support: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        vector : np.ndarray, optional
            500-dimensional float array representing environmental conditions.
            Randomly initialised from N(0, 1) if not provided.
        name : str, optional
            Human-readable label (e.g. "Northern Savanna").
        habitat_id : str, optional
            Explicit ID string.  Auto-generated UUID4 if not provided.
        """
        if vector is not None:
            if np.asarray(vector).shape != (HABITAT_VECTOR_DIMS,):
                raise ValueError(
                    f"Habitat vector must have shape ({HABITAT_VECTOR_DIMS},), "
                    f"got {np.asarray(vector).shape}"
                )
            self.vector: np.ndarray = np.asarray(vector, dtype=float)
        else:
            self.vector = np.random.randn(HABITAT_VECTOR_DIMS)

        self.name: Optional[str] = name
        self.habitat_id: str = habitat_id or str(uuid.uuid4())
        if population_support is not None:
            self.POPULATION_SUPPORT = population_support

        # Population stored as a set for O(1) membership operations
        self._creatures: set = set()

        # Neighbour registry: neighbour_habitat_id → {"habitat": Habitat, "passable": bool}
        self._neighbors: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Creature management
    # ------------------------------------------------------------------

    @property
    def creatures(self) -> list:
        """All creatures currently registered in this habitat."""
        return list(self._creatures)

    @property
    def alive_creatures(self) -> list:
        """All living creatures in this habitat."""
        return [c for c in self._creatures if c.is_alive]

    @property
    def population_size(self) -> int:
        """Number of living creatures."""
        return sum(1 for c in self._creatures if c.is_alive)

    def add_creature(self, creature: "Creature") -> None:
        """Add a creature to this habitat."""
        self._creatures.add(creature)

    def remove_creature(self, creature: "Creature") -> None:
        """Remove a creature from this habitat (no-op if not present)."""
        self._creatures.discard(creature)

    def has_creature(self, creature: "Creature") -> bool:
        return creature in self._creatures

    # ------------------------------------------------------------------
    # Neighbour / migration management
    # ------------------------------------------------------------------

    def add_neighbor(
        self,
        other: "Habitat",
        bidirectional: bool = True,
        passable: bool = True,
    ) -> None:
        """
        Register *other* as a neighbouring habitat.

        Parameters
        ----------
        other : Habitat
        bidirectional : bool
            If True (default), also register self as a neighbour of other.
        passable : bool
            Whether creatures can currently migrate along this link.
        """
        self._neighbors[other.habitat_id] = {"habitat": other, "passable": passable}
        if bidirectional:
            other._neighbors[self.habitat_id] = {"habitat": self, "passable": passable}

    def block_migration_to(
        self, other: "Habitat", bidirectional: bool = False
    ) -> None:
        """
        Block migration between this habitat and *other*.

        Models a permanent geographic barrier (mountain range, river, etc.).
        Pass bidirectional=True to close both directions simultaneously.
        """
        if other.habitat_id in self._neighbors:
            self._neighbors[other.habitat_id]["passable"] = False
        if bidirectional and self.habitat_id in other._neighbors:
            other._neighbors[self.habitat_id]["passable"] = False

    def open_migration_to(
        self, other: "Habitat", bidirectional: bool = False
    ) -> None:
        """Re-open a previously blocked migration route."""
        if other.habitat_id in self._neighbors:
            self._neighbors[other.habitat_id]["passable"] = True
        if bidirectional and self.habitat_id in other._neighbors:
            other._neighbors[self.habitat_id]["passable"] = True

    def passable_neighbors(self) -> list["Habitat"]:
        """Return neighbour habitats that creatures can currently migrate to."""
        return [
            info["habitat"]
            for info in self._neighbors.values()
            if info["passable"]
        ]

    def is_neighbor(self, other: "Habitat") -> bool:
        """True if *other* is registered as a neighbour (passable or not)."""
        return other.habitat_id in self._neighbors

    def can_migrate_to(self, other: "Habitat") -> bool:
        """True if *other* is a neighbour and the route is currently open."""
        info = self._neighbors.get(other.habitat_id)
        return info is not None and info["passable"]

    def try_spontaneous_isolation(self, probability: float = 0.001) -> list[str]:
        """
        Randomly sever open migration routes with the given per-link probability.

        Models low-frequency geographic events — landslides, floods, lava flows
        — that cut populations off from one another, initiating the geographic
        isolation required for allopatric speciation.

        Parameters
        ----------
        probability : float
            Per-link chance of severance each time this is called (default 0.001).

        Returns
        -------
        list[str]
            habitat_ids of neighbours that were newly isolated this call.
        """
        newly_isolated: list[str] = []
        for hid, info in self._neighbors.items():
            if info["passable"] and np.random.random() < probability:
                info["passable"] = False
                newly_isolated.append(hid)
        return newly_isolated

    # ------------------------------------------------------------------
    # Resource likelihood (vectorised cross-product geometry)
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_resource_prob(
        gene_matrix: np.ndarray,  # shape (N, K)
        habitat_vec: np.ndarray,  # shape (K,)
    ) -> np.ndarray:
        """
        Compute the per-creature resource-finding probability for each creature.

        P = (cos θ + 1) / 2   ∈ [0, 1]

        where θ is the angle between each creature's gene sub-vector and the
        habitat sub-vector at the relevant loci.

          cos θ = +1  (aligned)    → P = 1.0  — fully exploits this habitat
          cos θ =  0  (orthogonal) → P = 0.5  — baseline pressure
          cos θ = −1  (opposed)    → P = 0.0  — cannot extract resources

        Computed entirely via numpy for O(N·K) efficiency with no Python loop.
        """
        dots: np.ndarray = gene_matrix @ habitat_vec                      # (N,)
        creature_norms: np.ndarray = np.linalg.norm(gene_matrix, axis=1)  # (N,)
        habitat_norm: float = float(np.linalg.norm(habitat_vec))

        denom = creature_norms * habitat_norm
        safe = denom > 1e-10
        cos_theta = np.where(safe, dots / np.where(safe, denom, 1.0), 0.0)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return (cos_theta + 1.0) / 2.0  # maps [-1, 1] → [0, 1]

    def food_likelihoods(self, creatures: list) -> np.ndarray:
        """
        Per-creature weekly food-finding probability via (cos θ + 1) / 2.

        Shape (N,), values in [0, 1].  Uses the FOOD_GENE_INDICES subspace.
        """
        if not creatures:
            return np.array([], dtype=float)
        indices = self.FOOD_GENE_INDICES
        gene_matrix = np.stack([c.genes[indices] for c in creatures])
        return self._batch_resource_prob(gene_matrix, self.vector[indices])

    def water_likelihoods(self, creatures: list) -> np.ndarray:
        """
        Per-creature weekly water-finding probability via (cos θ + 1) / 2.

        Shape (N,), values in [0, 1].  Uses the WATER_GENE_INDICES subspace.
        """
        if not creatures:
            return np.array([], dtype=float)
        indices = self.WATER_GENE_INDICES
        gene_matrix = np.stack([c.genes[indices] for c in creatures])
        return self._batch_resource_prob(gene_matrix, self.vector[indices])

    # ------------------------------------------------------------------
    # Daily simulation
    # ------------------------------------------------------------------

    def simulate_week(
        self,
        species_registry=None,
        isolation_probability: float = 0.001,
    ) -> dict:
        """
        Advance the habitat by one week.

        Processing order
        ----------------
        1. Compute food / water likelihoods for all living creatures (batched).
        2. Binomial resource draws — each creature either finds food/water or not.
        3. Update energy / hydration; mark starvation / dehydration / old-age deaths.
        4. Advance each creature's internal week (aging, pregnancy timer).
        5. Collect litters from females that reached term this week.
        6. Apply density-dependent predation to survivors of step 3.
        7. Remove all dead creatures from the population.
        8. Migration: willing creatures move to a random passable neighbour.
        9. Mating: randomly pair viable males and females; offspring stored at
           term in female._pending_offspring.
        10. Add newborns to the habitat population.
        11. Attempt spontaneous route isolation (very low probability).

        Migration events are *returned* rather than applied directly so that the
        simulation runner can process all habitats before moving creatures,
        avoiding double-simulation of migrants on the same week.

        Returns
        -------
        dict
            "habitat_id"       : str
            "population"       : int   – alive count after processing
            "week_results"     : dict[creature_id, week_log]
            "births"           : list[Creature]  – newborns added to this habitat
            "deaths"           : list[str]       – creature_ids killed by resource
                                                   stress or old age
            "predation_deaths" : list[str]       – creature_ids killed by predation
            "mating_events"    : list[dict]      – one entry per attempted pairing
            "migrations"       : list[tuple[Creature, Habitat]]
                                 Each tuple is (creature, destination_habitat).
                                 The creature has already been removed from this
                                 habitat; the runner must add it to the destination.
            "isolations"       : list[str]  – neighbour habitat_ids newly cut off
        """
        alive = self.alive_creatures

        # ------------------------------------------------------------------
        # 1. Batch resource likelihoods
        # ------------------------------------------------------------------
        food_probs = self.food_likelihoods(alive)
        water_probs = self.water_likelihoods(alive)
        food_found = np.random.random(len(alive)) < food_probs
        water_found = np.random.random(len(alive)) < water_probs

        week_results: dict = {}
        deaths: list[str] = []
        newborns: list = []

        # ------------------------------------------------------------------
        # 2–5. Per-creature resource update, aging, and birth collection
        # ------------------------------------------------------------------
        for i, creature in enumerate(alive):
            # --- Energy ---
            if food_found[i]:
                creature.energy = min(1.0, creature.energy + self.FOOD_ENERGY_GAIN)
            else:
                creature.energy = max(
                    0.0,
                    creature.energy - self.FOOD_ENERGY_COST * creature.metabolism,
                )

            # --- Hydration ---
            if water_found[i]:
                creature.hydration = min(
                    1.0, creature.hydration + self.WATER_HYDRATION_GAIN
                )
            else:
                creature.hydration = max(
                    0.0,
                    creature.hydration
                    - self.WATER_HYDRATION_COST * (1.0 - creature.water_efficiency),
                )

            # --- Advance creature's week ---
            # creature.simulate_week() handles starvation / dehydration checks
            # internally (energy/hydration were already updated above).
            log = creature.simulate_week()

            # --- Collect litter if pregnancy completed this week ---
            if "gave_birth" in log["events"] and creature._pending_offspring:
                newborns.extend(creature._pending_offspring)
                creature._pending_offspring = []

            if not creature.is_alive:
                deaths.append(creature.creature_id)

            week_results[creature.creature_id] = log

        # ------------------------------------------------------------------
        # 6. Density-dependent predation (applied to survivors of step 3)
        # ------------------------------------------------------------------
        predation_deaths: list[str] = []
        alive_after_resources = [c for c in alive if c.is_alive]
        if alive_after_resources:
            n_alive = len(alive_after_resources)
            density_term = self.PREDATION_ALPHA * n_alive / self.POPULATION_SUPPORT
            base_rates = np.array([c.base_predation_rate for c in alive_after_resources])
            death_mask = np.random.random(n_alive) < (base_rates + density_term)
            for i in np.where(death_mask)[0]:
                creature = alive_after_resources[i]
                creature.is_alive = False
                creature.cause_of_death = "predation"
                predation_deaths.append(creature.creature_id)
                week_results[creature.creature_id]["cause_of_death"] = "predation"

        # ------------------------------------------------------------------
        # 7. Remove the dead
        # ------------------------------------------------------------------
        for creature in alive:
            if not creature.is_alive:
                self._creatures.discard(creature)

        # ------------------------------------------------------------------
        # 8. Migration
        # ------------------------------------------------------------------
        pending_migrations: list[tuple] = []
        passable = self.passable_neighbors()
        if passable:
            weekly_migration_base = self.WEEKLY_MIGRATION_BASE
            for creature in list(self._creatures):
                if not creature.is_alive:
                    continue
                weekly_prob = creature.migration_likelihood * weekly_migration_base
                if np.random.random() < weekly_prob:
                    destination = passable[np.random.randint(len(passable))]
                    self._creatures.discard(creature)
                    pending_migrations.append((creature, destination))

        # ------------------------------------------------------------------
        # 9. Mating
        # ------------------------------------------------------------------
        viable_males = [
            c for c in self._creatures
            if c.is_alive and c.sex == "male" and c.is_sexually_viable
        ]
        viable_females = [
            c for c in self._creatures
            if c.is_alive
            and c.sex == "female"
            and c.is_sexually_viable
            and not c.is_pregnant
        ]
        np.random.shuffle(viable_males)
        np.random.shuffle(viable_females)

        mating_events: list[dict] = []
        for male, female in zip(viable_males, viable_females):
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
                # reproduce() re-checks compatibility internally (small overhead)
                # and applies the fertility Bernoulli draw
                litter = male.reproduce(female)
                if litter:
                    event["fertilized"] = True
                    event["litter_size"] = len(litter)
                    event["offspring_ids"] = [c.creature_id for c in litter]
                else:
                    event["reason"] = "infertile"
            else:
                event["reason"] = reason
            mating_events.append(event)

        # ------------------------------------------------------------------
        # 10. Add newborns to the population; check for speciation events
        # ------------------------------------------------------------------
        for child in newborns:
            if species_registry is not None:
                species_registry.assign_species(child)
            self._creatures.add(child)

        # ------------------------------------------------------------------
        # 11. Spontaneous isolation
        # ------------------------------------------------------------------
        isolations = self.try_spontaneous_isolation(probability=isolation_probability)

        return {
            "habitat_id": self.habitat_id,
            "population": self.population_size,
            "week_results": week_results,
            "births": newborns,
            "deaths": deaths,
            "predation_deaths": predation_deaths,
            "mating_events": mating_events,
            "migrations": pending_migrations,
            "isolations": isolations,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_stats(self) -> dict[str, dict]:
        """
        Compute per-species aggregate statistics for the current alive population.

        Called by SimulationRunner after migrations are applied so the snapshot
        reflects true end-of-day state.  Resource probabilities represent each
        species' current adaptation level to this habitat — a population drifting
        toward alignment will show rising mean_food_prob / mean_water_prob over time.

        Returns
        -------
        dict mapping species_name → {
            "count"           : int,
            "mean_food_prob"  : float,   # adaptation to food in this habitat [0,1]
            "mean_water_prob" : float,   # adaptation to water in this habitat [0,1]
            "mean_traits"     : {trait: float, ...}  # all LOGGED_TRAITS
        }
        Empty dict if no creatures are alive.
        """
        alive = self.alive_creatures
        if not alive:
            return {}

        food_probs = self.food_likelihoods(alive)
        water_probs = self.water_likelihoods(alive)

        groups: dict[str, list] = {}
        group_food: dict[str, list] = {}
        group_water: dict[str, list] = {}
        for i, c in enumerate(alive):
            sp = c.species
            if sp not in groups:
                groups[sp] = []
                group_food[sp] = []
                group_water[sp] = []
            groups[sp].append(c)
            group_food[sp].append(float(food_probs[i]))
            group_water[sp].append(float(water_probs[i]))

        stats: dict[str, dict] = {}
        for sp_name, creatures in groups.items():
            n = len(creatures)
            mean_traits = {
                t: round(sum(getattr(c, t) for c in creatures) / n, 4)
                for t in LOGGED_TRAITS
            }
            stats[sp_name] = {
                "count": n,
                "mean_food_prob": round(float(np.mean(group_food[sp_name])), 4),
                "mean_water_prob": round(float(np.mean(group_water[sp_name])), 4),
                "mean_generation": round(sum(c.generation for c in creatures) / n, 2),
                "mean_traits": mean_traits,
            }

        return stats

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        label = f" ({self.name})" if self.name else ""
        n_neighbors = len(self._neighbors)
        passable = len(self.passable_neighbors())
        return (
            f"Habitat{label}[id={self.habitat_id[:8]}…, "
            f"pop={self.population_size}, "
            f"neighbors={n_neighbors} ({passable} open)]"
        )
