"""
Concrete Habitat subclasses representing distinct biome types.

Each class has a fixed TYPE_SEED that deterministically generates a
characteristic 500-dimensional center vector for that biome at class
definition time (via __init_subclass__).  Every instance then adds a
small Gaussian perturbation around that center, so two deserts are
similar but not identical — modelling intra-type geographic variation.

The center vector is accessible as a class attribute:
    Desert.CENTER   # shape (500,), same across all Desert instances

Available types
---------------
Desert, Forest, Rainforest, Plains, Tundra, Ocean, CoralReef,
Wetlands, Alpine, Volcanic, Cave, Arctic, River, Savanna
"""

import numpy as np

from ..habitat import Habitat, HABITAT_VECTOR_DIMS

_INSTANCE_NOISE: float = 0.15  # default std of per-instance perturbation


class TypedHabitat(Habitat):
    """
    Base for all named biome types.

    Subclasses must define:
        TYPE_SEED : int  – used to generate CENTER deterministically
        TYPE_NAME : str  – human-readable biome label used in logs

    When a subclass is defined, __init_subclass__ automatically computes
    and stores a ``CENTER`` class attribute (shape (500,)) from TYPE_SEED.
    This is the canonical "center of mass" in gene space for that biome —
    all instances share it and vary only by the per-instance noise.

    Constructor parameters
    ----------------------
    habitat_id : str, optional
    name : str, optional  (defaults to TYPE_NAME)
    instance_seed : int or None
        Seeds the per-instance noise RNG.  Use different seeds for
        multiple instances of the same type.
    instance_noise : float
        Standard deviation of the Gaussian noise added to CENTER.
        Default 0.15 — enough variation to differentiate instances
        while keeping them recognisably the same biome.
    """

    TYPE_SEED: int = 0
    TYPE_NAME: str = "Unknown"
    CENTER: np.ndarray = np.zeros(HABITAT_VECTOR_DIMS)  # overwritten per subclass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Compute and freeze the center vector for every concrete subclass
        if cls.TYPE_SEED != 0:
            cls.CENTER = np.random.default_rng(cls.TYPE_SEED).standard_normal(
                HABITAT_VECTOR_DIMS
            )

    def __init__(
        self,
        habitat_id=None,
        name=None,
        instance_seed=None,
        instance_noise: float = _INSTANCE_NOISE,
        population_support: int = None,
    ):
        inst_rng = np.random.default_rng(instance_seed)
        noise = inst_rng.standard_normal(HABITAT_VECTOR_DIMS) * instance_noise
        vector = self.CENTER + noise

        super().__init__(
            vector=vector,
            habitat_id=habitat_id,
            name=name or self.TYPE_NAME,
            population_support=population_support,
        )

    @property
    def habitat_type(self) -> str:
        return self.TYPE_NAME


# ---------------------------------------------------------------------------
# Biome subclasses
#
# Each overrides only the resource constants that differ from Habitat defaults:
#   FOOD_ENERGY_GAIN     = 0.30
#   FOOD_ENERGY_COST     = 0.25  (× creature.metabolism)
#   WATER_HYDRATION_GAIN = 0.30
#   WATER_HYDRATION_COST = 0.40  (× (1 - creature.water_efficiency))
#   WEEKLY_MIGRATION_BASE = 0.01
#   PREDATION_ALPHA      = 0.010  (density term at N = POPULATION_SUPPORT)
#   POPULATION_SUPPORT   = 400    (carrying capacity for density-dependent mortality)
#
# Predation design: harsh habitats have low POPULATION_SUPPORT and low alpha —
# resource scarcity limits population, keeping crowding mortality minimal for
# adapted survivors.  Rich habitats have high support and higher alpha — easy
# resources allow large populations, which ramp up crowding pressure.
# ---------------------------------------------------------------------------

class Desert(TypedHabitat):
    """Hot, arid desert. Water is scarce; food requires specialist adaptation."""
    TYPE_SEED = 1001
    TYPE_NAME = "Desert"
    WATER_HYDRATION_COST: float = 0.44   # dehydrates fast without adaptation
    FOOD_ENERGY_COST: float = 0.21       # foraging requires more effort
    FOOD_ENERGY_GAIN: float = 0.25       # sparse food patches
    PREDATION_ALPHA: float = 0.003
    POPULATION_SUPPORT: int = 120


class Forest(TypedHabitat):
    """Temperate woodland. Balanced resources; moderate selection pressure."""
    TYPE_SEED = 1002
    TYPE_NAME = "Forest"
    # Uses Habitat defaults for resources — represents the baseline environment
    PREDATION_ALPHA: float = 0.010
    POPULATION_SUPPORT: int = 400


class Rainforest(TypedHabitat):
    """Tropical rainforest. Abundant food and water; intense competition."""
    TYPE_SEED = 1003
    TYPE_NAME = "Rainforest"
    FOOD_ENERGY_GAIN: float = 0.40
    WATER_HYDRATION_GAIN: float = 0.45
    PREDATION_ALPHA: float = 0.015       # rich habitat → high density → more crowding pressure
    POPULATION_SUPPORT: int = 600


class Plains(TypedHabitat):
    """Open grassland. Easy movement; moderate resources."""
    TYPE_SEED = 1004
    TYPE_NAME = "Plains"
    WEEKLY_MIGRATION_BASE: float = 0.015  # flat terrain aids dispersal
    PREDATION_ALPHA: float = 0.010
    POPULATION_SUPPORT: int = 500


class Tundra(TypedHabitat):
    """Cold, sparse tundra. Food scarce; water from seasonal ice melt."""
    TYPE_SEED = 1005
    TYPE_NAME = "Tundra"
    FOOD_ENERGY_GAIN: float = 0.20
    FOOD_ENERGY_COST: float = 0.21       # high metabolic cost in cold
    WATER_HYDRATION_GAIN: float = 0.25
    PREDATION_ALPHA: float = 0.004
    POPULATION_SUPPORT: int = 150


class Ocean(TypedHabitat):
    """Open pelagic ocean. No water cost; food rewards specialist traits."""
    TYPE_SEED = 1006
    TYPE_NAME = "Ocean"
    WATER_HYDRATION_COST: float = 0.0    # surrounded by water
    WATER_HYDRATION_GAIN: float = 0.50
    FOOD_ENERGY_GAIN: float = 0.35
    WEEKLY_MIGRATION_BASE: float = 0.02   # currents aid dispersal
    PREDATION_ALPHA: float = 0.008
    POPULATION_SUPPORT: int = 300


class CoralReef(TypedHabitat):
    """Shallow tropical reef. Very high productivity; tight specialist niche."""
    TYPE_SEED = 1007
    TYPE_NAME = "CoralReef"
    FOOD_ENERGY_GAIN: float = 0.45
    WATER_HYDRATION_GAIN: float = 0.50
    WATER_HYDRATION_COST: float = 0.0
    PREDATION_ALPHA: float = 0.020       # tight niche → intense density competition
    POPULATION_SUPPORT: int = 400


class Wetlands(TypedHabitat):
    """Swamps and marshes. Abundant water; moderate food."""
    TYPE_SEED = 1008
    TYPE_NAME = "Wetlands"
    WATER_HYDRATION_GAIN: float = 0.45
    WATER_HYDRATION_COST: float = 0.0    # water everywhere
    PREDATION_ALPHA: float = 0.012
    POPULATION_SUPPORT: int = 500


class Alpine(TypedHabitat):
    """High-altitude mountain. Thin air; scarce food; migration is hard."""
    TYPE_SEED = 1009
    TYPE_NAME = "Alpine"
    FOOD_ENERGY_GAIN: float = 0.18
    FOOD_ENERGY_COST: float = 0.24       # high-altitude exertion
    WEEKLY_MIGRATION_BASE: float = 0.005  # rugged terrain limits movement
    PREDATION_ALPHA: float = 0.002       # sparse population → low density pressure
    POPULATION_SUPPORT: int = 80


class Volcanic(TypedHabitat):
    """Geothermal volcanic zone. Extreme environment; very high mortality pressure."""
    TYPE_SEED = 1010
    TYPE_NAME = "Volcanic"
    FOOD_ENERGY_GAIN: float = 0.22
    FOOD_ENERGY_COST: float = 0.24
    WATER_HYDRATION_GAIN: float = 0.20
    WATER_HYDRATION_COST: float = 0.38
    WEEKLY_MIGRATION_BASE: float = 0.005  # inhospitable terrain
    PREDATION_ALPHA: float = 0.001
    POPULATION_SUPPORT: int = 50


class Cave(TypedHabitat):
    """Underground cave system. No light; very scarce resources."""
    TYPE_SEED = 1011
    TYPE_NAME = "Cave"
    FOOD_ENERGY_GAIN: float = 0.15
    WATER_HYDRATION_GAIN: float = 0.30   # underground water sources
    WEEKLY_MIGRATION_BASE: float = 0.003  # difficult to navigate
    PREDATION_ALPHA: float = 0.002
    POPULATION_SUPPORT: int = 80


class Arctic(TypedHabitat):
    """Polar ice sheet. Extreme cold; low food; water from melt."""
    TYPE_SEED = 1012
    TYPE_NAME = "Arctic"
    FOOD_ENERGY_GAIN: float = 0.15
    FOOD_ENERGY_COST: float = 0.24
    WATER_HYDRATION_GAIN: float = 0.35
    PREDATION_ALPHA: float = 0.001
    POPULATION_SUPPORT: int = 60


class River(TypedHabitat):
    """Freshwater river. Abundant water; good food for adapted creatures."""
    TYPE_SEED = 1013
    TYPE_NAME = "River"
    WATER_HYDRATION_GAIN: float = 0.45
    WATER_HYDRATION_COST: float = 0.0
    WEEKLY_MIGRATION_BASE: float = 0.015  # currents aid movement
    PREDATION_ALPHA: float = 0.010
    POPULATION_SUPPORT: int = 400


class Savanna(TypedHabitat):
    """Dry tropical grassland. Seasonal water stress; easy movement."""
    TYPE_SEED = 1014
    TYPE_NAME = "Savanna"
    FOOD_ENERGY_GAIN: float = 0.28
    WATER_HYDRATION_COST: float = 0.31
    WEEKLY_MIGRATION_BASE: float = 0.012
    PREDATION_ALPHA: float = 0.008
    POPULATION_SUPPORT: int = 350


# ---------------------------------------------------------------------------
# Registry — maps config type name → class for the simulation runner
# ---------------------------------------------------------------------------

HABITAT_TYPE_REGISTRY: dict[str, type] = {
    "Desert": Desert,
    "Forest": Forest,
    "Rainforest": Rainforest,
    "Plains": Plains,
    "Tundra": Tundra,
    "Ocean": Ocean,
    "CoralReef": CoralReef,
    "Wetlands": Wetlands,
    "Alpine": Alpine,
    "Volcanic": Volcanic,
    "Cave": Cave,
    "Arctic": Arctic,
    "River": River,
    "Savanna": Savanna,
}
