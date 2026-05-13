from .creature import Creature, DEFAULT_TRAIT_GENE_INDICES, GENE_DIMS
from .habitat import Habitat, DEFAULT_FOOD_GENE_INDICES, DEFAULT_WATER_GENE_INDICES
from .habitats import (
    TypedHabitat,
    Desert, Forest, Rainforest, Plains, Tundra,
    Ocean, CoralReef, Wetlands, Alpine, Volcanic,
    Cave, Arctic, River, Savanna,
    HABITAT_TYPE_REGISTRY,
)
from .species import SpeciesRegistry, ADJECTIVES, NOUNS
from .simulation import SimulationRunner
