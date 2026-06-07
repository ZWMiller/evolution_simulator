<a id="evolution_simulator.habitats.types"></a>

# evolution\_simulator.habitats.types

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

<a id="evolution_simulator.habitats.types.TypedHabitat"></a>

## TypedHabitat Objects

```python
class TypedHabitat(Habitat)
```

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

<a id="evolution_simulator.habitats.types.TypedHabitat.CENTER"></a>

#### CENTER

overwritten per subclass

<a id="evolution_simulator.habitats.types.Desert"></a>

## Desert Objects

```python
class Desert(TypedHabitat)
```

Hot, arid desert. Water is scarce; food requires specialist adaptation.

<a id="evolution_simulator.habitats.types.Desert.WATER_BASE_COST"></a>

#### WATER\_BASE\_COST

high irreducible dehydration

<a id="evolution_simulator.habitats.types.Desert.WATER_EFFICIENCY_COST"></a>

#### WATER\_EFFICIENCY\_COST

efficiency helps but desert is still lethal

<a id="evolution_simulator.habitats.types.Desert.FOOD_ENERGY_COST"></a>

#### FOOD\_ENERGY\_COST

foraging requires more effort (scaled from 0.21)

<a id="evolution_simulator.habitats.types.Desert.FOOD_ENERGY_GAIN"></a>

#### FOOD\_ENERGY\_GAIN

sparse food patches

<a id="evolution_simulator.habitats.types.Forest"></a>

## Forest Objects

```python
class Forest(TypedHabitat)
```

Temperate woodland. Balanced resources; moderate selection pressure.

<a id="evolution_simulator.habitats.types.Rainforest"></a>

## Rainforest Objects

```python
class Rainforest(TypedHabitat)
```

Tropical rainforest. Abundant food and water; intense competition.

<a id="evolution_simulator.habitats.types.Rainforest.PREDATION_ALPHA"></a>

#### PREDATION\_ALPHA

rich habitat → high density → more crowding pressure

<a id="evolution_simulator.habitats.types.Plains"></a>

## Plains Objects

```python
class Plains(TypedHabitat)
```

Open grassland. Easy movement; moderate resources.

<a id="evolution_simulator.habitats.types.Plains.WEEKLY_MIGRATION_BASE"></a>

#### WEEKLY\_MIGRATION\_BASE

flat terrain aids dispersal

<a id="evolution_simulator.habitats.types.Tundra"></a>

## Tundra Objects

```python
class Tundra(TypedHabitat)
```

Cold, sparse tundra. Food scarce; water from seasonal ice melt.

<a id="evolution_simulator.habitats.types.Tundra.FOOD_ENERGY_COST"></a>

#### FOOD\_ENERGY\_COST

high metabolic cost in cold (scaled from 0.21)

<a id="evolution_simulator.habitats.types.Ocean"></a>

## Ocean Objects

```python
class Ocean(TypedHabitat)
```

Open pelagic ocean. No water cost; food rewards specialist traits.

<a id="evolution_simulator.habitats.types.Ocean.WATER_BASE_COST"></a>

#### WATER\_BASE\_COST

surrounded by water

<a id="evolution_simulator.habitats.types.Ocean.WEEKLY_MIGRATION_BASE"></a>

#### WEEKLY\_MIGRATION\_BASE

currents aid dispersal

<a id="evolution_simulator.habitats.types.CoralReef"></a>

## CoralReef Objects

```python
class CoralReef(TypedHabitat)
```

Shallow tropical reef. Very high productivity; tight specialist niche.

<a id="evolution_simulator.habitats.types.CoralReef.WATER_BASE_COST"></a>

#### WATER\_BASE\_COST

submerged — no water stress

<a id="evolution_simulator.habitats.types.CoralReef.PREDATION_ALPHA"></a>

#### PREDATION\_ALPHA

tight niche → intense density competition

<a id="evolution_simulator.habitats.types.Wetlands"></a>

## Wetlands Objects

```python
class Wetlands(TypedHabitat)
```

Swamps and marshes. Abundant water; moderate food.

<a id="evolution_simulator.habitats.types.Wetlands.WATER_BASE_COST"></a>

#### WATER\_BASE\_COST

minimal — mostly submerged but not zero

<a id="evolution_simulator.habitats.types.Alpine"></a>

## Alpine Objects

```python
class Alpine(TypedHabitat)
```

High-altitude mountain. Thin air; scarce food; migration is hard.

<a id="evolution_simulator.habitats.types.Alpine.FOOD_ENERGY_COST"></a>

#### FOOD\_ENERGY\_COST

high-altitude exertion (scaled from 0.24)

<a id="evolution_simulator.habitats.types.Alpine.WEEKLY_MIGRATION_BASE"></a>

#### WEEKLY\_MIGRATION\_BASE

rugged terrain limits movement

<a id="evolution_simulator.habitats.types.Alpine.PREDATION_ALPHA"></a>

#### PREDATION\_ALPHA

sparse population → low density pressure

<a id="evolution_simulator.habitats.types.Volcanic"></a>

## Volcanic Objects

```python
class Volcanic(TypedHabitat)
```

Geothermal volcanic zone. Extreme environment; very high mortality pressure.

<a id="evolution_simulator.habitats.types.Volcanic.FOOD_ENERGY_COST"></a>

#### FOOD\_ENERGY\_COST

extreme exertion (scaled from 0.24)

<a id="evolution_simulator.habitats.types.Volcanic.WATER_BASE_COST"></a>

#### WATER\_BASE\_COST

high evaporative loss from geothermal heat

<a id="evolution_simulator.habitats.types.Volcanic.WEEKLY_MIGRATION_BASE"></a>

#### WEEKLY\_MIGRATION\_BASE

inhospitable terrain

<a id="evolution_simulator.habitats.types.Cave"></a>

## Cave Objects

```python
class Cave(TypedHabitat)
```

Underground cave system. No light; very scarce resources.

<a id="evolution_simulator.habitats.types.Cave.WATER_HYDRATION_GAIN"></a>

#### WATER\_HYDRATION\_GAIN

underground water sources

<a id="evolution_simulator.habitats.types.Cave.WEEKLY_MIGRATION_BASE"></a>

#### WEEKLY\_MIGRATION\_BASE

difficult to navigate

<a id="evolution_simulator.habitats.types.Arctic"></a>

## Arctic Objects

```python
class Arctic(TypedHabitat)
```

Polar ice sheet. Extreme cold; low food; water from melt.

<a id="evolution_simulator.habitats.types.Arctic.FOOD_ENERGY_COST"></a>

#### FOOD\_ENERGY\_COST

extreme cold metabolic cost (scaled from 0.24)

<a id="evolution_simulator.habitats.types.River"></a>

## River Objects

```python
class River(TypedHabitat)
```

Freshwater river. Abundant water; good food for adapted creatures.

<a id="evolution_simulator.habitats.types.River.WATER_BASE_COST"></a>

#### WATER\_BASE\_COST

low — water is near but creatures must reach it

<a id="evolution_simulator.habitats.types.River.WEEKLY_MIGRATION_BASE"></a>

#### WEEKLY\_MIGRATION\_BASE

currents aid movement

<a id="evolution_simulator.habitats.types.Savanna"></a>

## Savanna Objects

```python
class Savanna(TypedHabitat)
```

Dry tropical grassland. Seasonal water stress; easy movement.

<a id="evolution_simulator.habitats.types.Savanna.WATER_BASE_COST"></a>

#### WATER\_BASE\_COST

dry season water stress
