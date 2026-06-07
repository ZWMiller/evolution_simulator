<a id="evolution_simulator.habitat"></a>

# evolution\_simulator.habitat

Habitat model and weekly simulation loop for the evolution simulator.

A ``Habitat`` is a geographic region represented by a 500-dimensional
environment vector.  Creature fitness is determined entirely by how well each
creature's gene sub-vector aligns with the habitat vector, via the
``(cos θ + 1) / 2`` resource-geometry formula.

Key public API
--------------
``Habitat(vector, name)``
    Construct a habitat.  If *vector* is omitted it is drawn from N(0, 1).

``habitat.simulate_week(rng, species_registry, mating_strategy) -> dict``
    Advance all creatures by one week in a fixed processing order:
    resource draws → energy/hydration update → aging → birth collection →
    predation → remove dead → migration → mating → add newborns →
    spontaneous isolation.  Returns a structured event dict.

``habitat.food_likelihoods(creatures) -> ndarray``
``habitat.water_likelihoods(creatures) -> ndarray``
    Batched ``(cos θ + 1) / 2`` resource probabilities for a list of creatures.

``habitat.add_neighbor(other) / block_migration_to / open_migration_to``
    Manage the migration graph between habitats.

``habitat.compute_stats() -> dict``
    Per-species aggregate statistics for the current alive population.

Concrete biome subclasses live in ``habitats/types.py``.

<a id="evolution_simulator.habitat.Habitat"></a>

## Habitat Objects

```python
class Habitat()
```

A geographic region that creatures inhabit.

Each Habitat has a 500-dimensional float vector representing the
environmental conditions of that region.  This vector interacts with
creature gene vectors via cross-product geometry to determine daily
resource-finding probabilities.

Core responsibilities
---------------------
- Track which creatures are present (O(1) add / remove via an insertion-
  ordered dict keyed by creature_id, for deterministic iteration).
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
WATER_BASE_COST : float
    Fixed hydration lost per missed water week (irreducible by water_efficiency).
WATER_EFFICIENCY_COST : float
    Variable hydration cost scaled by (1 - creature.water_efficiency); added to
    WATER_BASE_COST so total cost = WATER_BASE_COST + WATER_EFFICIENCY_COST * (1 - eff).
WEEKLY_MIGRATION_BASE : float
    Multiplier applied to creature.migration_likelihood to get the actual
    per-week migration probability.  Keeps average migration rare even when
    the trait value is moderate.

<a id="evolution_simulator.habitat.Habitat.FOOD_ENERGY_COST"></a>

#### FOOD\_ENERGY\_COST

multiplied by creature.metabolism

<a id="evolution_simulator.habitat.Habitat.__init__"></a>

#### \_\_init\_\_

```python
def __init__(vector: np.ndarray | None = None,
             name: str | None = None,
             habitat_id: str | None = None,
             population_support: int | None = None)
```

Parameters
----------
vector : np.ndarray, optional
    500-dimensional float array representing this habitat's
    environmental conditions.  Gene sub-vectors are compared against
    this via ``(cos θ + 1) / 2`` to produce resource probabilities.
    Randomly initialised from N(0, 1) if not provided.
name : str, optional
    Human-readable label used in logs and visualisation (e.g.
    ``"Northern Savanna"``).
habitat_id : str, optional
    Explicit ID string.  Auto-generated UUID4 if not provided.
population_support : int, optional
    Carrying capacity used in density-dependent mortality:
    ``death_prob = PREDATION_ALPHA * N / population_support``.
    Overrides the class-level ``POPULATION_SUPPORT`` for this
    instance only.

<a id="evolution_simulator.habitat.Habitat.creatures"></a>

#### creatures

```python
@property
def creatures() -> list
```

All creatures currently registered in this habitat.

<a id="evolution_simulator.habitat.Habitat.alive_creatures"></a>

#### alive\_creatures

```python
@property
def alive_creatures() -> list
```

All living creatures in this habitat.

<a id="evolution_simulator.habitat.Habitat.population_size"></a>

#### population\_size

```python
@property
def population_size() -> int
```

Number of living creatures.

<a id="evolution_simulator.habitat.Habitat.add_creature"></a>

#### add\_creature

```python
def add_creature(creature: "Creature") -> None
```

Add a creature to this habitat.

<a id="evolution_simulator.habitat.Habitat.remove_creature"></a>

#### remove\_creature

```python
def remove_creature(creature: "Creature") -> None
```

Remove a creature from this habitat (no-op if not present).

<a id="evolution_simulator.habitat.Habitat.has_creature"></a>

#### has\_creature

```python
def has_creature(creature: "Creature") -> bool
```

True if *creature* is currently registered in this habitat.

<a id="evolution_simulator.habitat.Habitat.add_neighbor"></a>

#### add\_neighbor

```python
def add_neighbor(other: "Habitat",
                 bidirectional: bool = True,
                 passable: bool = True) -> None
```

Register *other* as a neighbouring habitat.

Parameters
----------
other : Habitat
    The habitat to link to.  Must not already be registered as a
    neighbour (re-adding silently overwrites the existing entry).
bidirectional : bool, optional
    If ``True`` (default), also register ``self`` as a neighbour of
    *other* so creatures can migrate in both directions.
passable : bool, optional
    Initial passability of the link.  ``False`` registers the
    connection in the topology without opening it to migration — useful
    when building a graph that starts isolated.  Default ``True``.

<a id="evolution_simulator.habitat.Habitat.block_migration_to"></a>

#### block\_migration\_to

```python
def block_migration_to(other: "Habitat", bidirectional: bool = False) -> None
```

Block migration from this habitat to *other*.

Models a permanent geographic barrier (mountain range, river, flood,
etc.) that cuts creatures off from the neighbouring habitat.  If
*other* is not a registered neighbour, this is a no-op.

Parameters
----------
other : Habitat
    The neighbouring habitat whose migration link should be closed.
bidirectional : bool, optional
    If ``True``, also block migration from *other* back to ``self``.
    Default ``False`` (one-directional block).

<a id="evolution_simulator.habitat.Habitat.open_migration_to"></a>

#### open\_migration\_to

```python
def open_migration_to(other: "Habitat", bidirectional: bool = False) -> None
```

Re-open a previously blocked migration route to *other*.

Parameters
----------
other : Habitat
    The neighbouring habitat whose migration link should be re-opened.
bidirectional : bool, optional
    If ``True``, also re-open migration from *other* back to ``self``.
    Default ``False``.

<a id="evolution_simulator.habitat.Habitat.passable_neighbors"></a>

#### passable\_neighbors

```python
def passable_neighbors() -> list["Habitat"]
```

Return neighbour habitats that creatures can currently migrate to.

<a id="evolution_simulator.habitat.Habitat.is_neighbor"></a>

#### is\_neighbor

```python
def is_neighbor(other: "Habitat") -> bool
```

True if *other* is registered as a neighbour (passable or not).

<a id="evolution_simulator.habitat.Habitat.can_migrate_to"></a>

#### can\_migrate\_to

```python
def can_migrate_to(other: "Habitat") -> bool
```

True if *other* is a neighbour and the route is currently open.

<a id="evolution_simulator.habitat.Habitat.try_spontaneous_isolation"></a>

#### try\_spontaneous\_isolation

```python
def try_spontaneous_isolation(rng: np.random.Generator,
                              probability: float = 0.001) -> list[str]
```

Randomly sever open migration routes with the given per-link probability.

Models low-frequency geographic events — landslides, floods, lava flows
— that cut populations off from one another, initiating the geographic
isolation required for allopatric speciation.

Parameters
----------
rng : np.random.Generator
    The single simulation generator (threaded from SimulationRunner).
probability : float
    Per-link chance of severance each time this is called (default 0.001).

Returns
-------
list[str]
    habitat_ids of neighbours that were newly isolated this call.

<a id="evolution_simulator.habitat.Habitat.food_likelihoods"></a>

#### food\_likelihoods

```python
def food_likelihoods(creatures: list) -> np.ndarray
```

Per-creature weekly food-finding probability via ``(cos θ + 1) / 2``.

Parameters
----------
creatures : list[Creature]
    The creatures to compute probabilities for.  An empty list returns
    an empty array.

Returns
-------
np.ndarray, shape (N,)
    Food-finding probability for each creature in [0, 1], computed
    over the ``FOOD_GENE_INDICES`` subspace (158 loci by default).

<a id="evolution_simulator.habitat.Habitat.water_likelihoods"></a>

#### water\_likelihoods

```python
def water_likelihoods(creatures: list) -> np.ndarray
```

Per-creature weekly water-finding probability via ``(cos θ + 1) / 2``.

Parameters
----------
creatures : list[Creature]
    The creatures to compute probabilities for.  An empty list returns
    an empty array.

Returns
-------
np.ndarray, shape (N,)
    Water-finding probability for each creature in [0, 1], computed
    over the ``WATER_GENE_INDICES`` subspace (175 loci by default).

<a id="evolution_simulator.habitat.Habitat.simulate_week"></a>

#### simulate\_week

```python
def simulate_week(rng: np.random.Generator,
                  species_registry=None,
                  isolation_probability: float = 0.001,
                  mating_strategy: str = "zip") -> dict
```

Advance the habitat by one week.

Parameters
----------
rng : np.random.Generator
    The single simulation generator threaded from ``SimulationRunner``.
    Every stochastic draw (resource finding, predation, migration,
    mating, reproduction, isolation) comes from this stream.  Required
    — no global-RNG fallback exists in simulation logic.
species_registry : SpeciesRegistry, optional
    If provided, ``assign_species()`` is called on every newborn so
    they receive the correct species label before being added to the
    population.  Pass ``None`` in tests that don't need speciation.
isolation_probability : float, optional
    Per-link probability of a spontaneous migration-route severance
    each week.  Default 0.001 (0.1 % per link per week).
mating_strategy : str, optional
    Pairing algorithm for the mating step.  One of ``"zip"``
    (default), ``"species_priority"``, ``"weighted_matrix"``, or
    ``"stable_matching"``.  Unknown strings fall back to ``"zip"``.

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
9. Mating: pair viable males and females according to *mating_strategy*;
   offspring stored at term in female._pending_offspring.
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

<a id="evolution_simulator.habitat.Habitat.compute_stats"></a>

#### compute\_stats

```python
def compute_stats() -> dict[str, dict]
```

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
