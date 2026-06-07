<a id="evolution_simulator.creature"></a>

# evolution\_simulator.creature

Core creature model for the evolution simulator.

A ``Creature`` is a simulated organism whose entire biology is encoded in a
500-dimensional float gene vector.  Every phenotypic trait is computed from a
polygenic subset of that vector using Ordered Weighted Averaging (see
``Creature._compute_trait``).  Reproduction is Mendelian with per-locus
mutation; speciation emerges from drift in the ``compatibility_genes`` subset.

Key public API
--------------
``Creature(genes, parents)``
    Construct a new individual.  If genes are omitted, they are drawn from
    N(0, 1) — suitable for founding populations.

``creature.reproduce(other, rng) -> list[Creature]``
    Attempt to produce a litter.  Returns [] if the pair is incompatible or
    the conception draw fails.

``creature.is_compatible(other) -> (bool, float, str)``
    Full mating check: opposite sex, reproductive age, not pregnant, cosine
    similarity above the compatibility floor.

``creature.compatibility_score(other) -> float``
    Raw cosine similarity of the 245-locus compatibility gene subset.

``creature.simulate_week() -> dict``
    Advance age, pregnancy timers, and survival; return a week-log dict.
    Energy and hydration are set by the ``Habitat`` before this is called.

Module-level utilities
----------------------
``reset_creature_ids(start)``
    Reset the monotonic ID counter; called once at simulation setup.

<a id="evolution_simulator.creature.reset_creature_ids"></a>

#### reset\_creature\_ids

```python
def reset_creature_ids(start: int = 1) -> None
```

Reset the monotonic creature-ID counter.

Called once at simulation setup so creature IDs are deterministic and
byte-identical across runs that share the same config and seed.

Parameters
----------
start : int, optional
    First integer value the counter will produce.  Default 1, so the
    first ID is ``c0000000001``.

<a id="evolution_simulator.creature.Creature"></a>

## Creature Objects

```python
class Creature()
```

A simulated creature whose genome is a 500-dimensional float vector.

Each named biological trait is computed from a distributed subset of gene
indices (see TRAIT_GENE_INDICES).  The same index may contribute to
multiple traits, modelling pleiotropy.  Subclasses can override
TRAIT_GENE_INDICES to represent species with different genetic
architectures.

Trait value formula — Ordered Weighted Averaging (OWA)
-------------------------------------------------------
1. vals       = genes[trait_indices]
2. sorted     = sort(vals, descending)
3. w_i        = OWA_ALPHA * (1 - OWA_ALPHA)^i  (then normalised to sum = 1)
4. raw_signal = dot(weights, sorted)
5. trait_value = sigmoid(raw_signal)            → [0, 1]

Higher-valued loci receive exponentially more weight, so a beneficial
mutation that pushes a locus to the top of the ranking gains immediate
phenotypic influence rather than being diluted 1/N by a plain mean.

Properties then scale [0, 1] into biologically meaningful ranges.

Parameters
----------
genes : np.ndarray, optional
    A 500-dimensional float array.  If None, genes are sampled from a
    standard normal distribution.
parents : list[Creature], optional
    Direct parent Creature objects.  Pass [] for a founding individual.
creature_id : str, optional
    Explicit ID string.  Auto-assigned from a monotonic counter if not
    provided; call ``reset_creature_ids()`` at simulation setup to
    restart the counter.

<a id="evolution_simulator.creature.Creature.trait_indices"></a>

#### trait\_indices

```python
def trait_indices(name: str) -> list[int]
```

Return the gene loci that contribute to the named trait.

Parameters
----------
name : str
    Trait name; must be a key in ``TRAIT_GENE_INDICES``.

Returns
-------
list[int]
    Indices into the 500-dimensional gene vector for this trait.

<a id="evolution_simulator.creature.Creature.fecundity"></a>

#### fecundity

```python
@property
def fecundity() -> float
```

Expected offspring per birth event (1 – 8).

<a id="evolution_simulator.creature.Creature.reproduction_time"></a>

#### reproduction\_time

```python
@property
def reproduction_time() -> int
```

Weeks required to gestate / develop offspring (1 – 20).

<a id="evolution_simulator.creature.Creature.weeks_to_sexual_viability"></a>

#### weeks\_to\_sexual\_viability

```python
@property
def weeks_to_sexual_viability() -> int
```

Weeks before the creature can reproduce (4 – 50).

<a id="evolution_simulator.creature.Creature.parental_investment"></a>

#### parental\_investment

```python
@property
def parental_investment() -> float
```

Parental care score [0, 1]; higher → better offspring survival.

<a id="evolution_simulator.creature.Creature.aggression"></a>

#### aggression

```python
@property
def aggression() -> float
```

Aggression score [0, 1].

<a id="evolution_simulator.creature.Creature.migration_likelihood"></a>

#### migration\_likelihood

```python
@property
def migration_likelihood() -> float
```

Weekly probability of migrating to a new environment [0, 1].

<a id="evolution_simulator.creature.Creature.territorial"></a>

#### territorial

```python
@property
def territorial() -> float
```

Territorial tendency [0, 1].

<a id="evolution_simulator.creature.Creature.social_tendency"></a>

#### social\_tendency

```python
@property
def social_tendency() -> float
```

Social / cooperative tendency [0, 1].

<a id="evolution_simulator.creature.Creature.pack_hunting"></a>

#### pack\_hunting

```python
@property
def pack_hunting() -> float
```

Tendency to hunt in coordinated groups [0, 1].

<a id="evolution_simulator.creature.Creature.scavenging_tendency"></a>

#### scavenging\_tendency

```python
@property
def scavenging_tendency() -> float
```

Tendency to scavenge rather than actively hunt [0, 1].

<a id="evolution_simulator.creature.Creature.nocturnal_tendency"></a>

#### nocturnal\_tendency

```python
@property
def nocturnal_tendency() -> float
```

Tendency to be active at night [0, 1].

<a id="evolution_simulator.creature.Creature.risk_tolerance"></a>

#### risk\_tolerance

```python
@property
def risk_tolerance() -> float
```

Willingness to accept risk in pursuit of resources [0, 1].

<a id="evolution_simulator.creature.Creature.size"></a>

#### size

```python
@property
def size() -> float
```

Body size score [0, 1]; scales food requirement and combat.

<a id="evolution_simulator.creature.Creature.strength"></a>

#### strength

```python
@property
def strength() -> float
```

Physical strength [0, 1].

<a id="evolution_simulator.creature.Creature.speed"></a>

#### speed

```python
@property
def speed() -> float
```

Movement speed [0, 1].

<a id="evolution_simulator.creature.Creature.camouflage"></a>

#### camouflage

```python
@property
def camouflage() -> float
```

Predator-avoidance / camouflage score [0, 1].

<a id="evolution_simulator.creature.Creature.metabolism"></a>

#### metabolism

```python
@property
def metabolism() -> float
```

Weekly resource consumption multiplier [0.5 – 2.0].

<a id="evolution_simulator.creature.Creature.foraging_ability"></a>

#### foraging\_ability

```python
@property
def foraging_ability() -> float
```

Effectiveness at locating food [0, 1].

<a id="evolution_simulator.creature.Creature.water_efficiency"></a>

#### water\_efficiency

```python
@property
def water_efficiency() -> float
```

Water-use efficiency [0, 1]; higher → needs less water per week.

<a id="evolution_simulator.creature.Creature.max_lifespan"></a>

#### max\_lifespan

```python
@property
def max_lifespan() -> int
```

Maximum lifespan in weeks (40 – 400 weeks, i.e. ~1 – 8 years).

<a id="evolution_simulator.creature.Creature.disease_resistance"></a>

#### disease\_resistance

```python
@property
def disease_resistance() -> float
```

Disease resistance [0, 1].

<a id="evolution_simulator.creature.Creature.immune_response"></a>

#### immune\_response

```python
@property
def immune_response() -> float
```

Immune-response strength [0, 1].

<a id="evolution_simulator.creature.Creature.stress_tolerance"></a>

#### stress\_tolerance

```python
@property
def stress_tolerance() -> float
```

Tolerance to general environmental stressors [0, 1].

<a id="evolution_simulator.creature.Creature.heat_tolerance"></a>

#### heat\_tolerance

```python
@property
def heat_tolerance() -> float
```

Tolerance to high temperatures [0, 1].

<a id="evolution_simulator.creature.Creature.cold_tolerance"></a>

#### cold\_tolerance

```python
@property
def cold_tolerance() -> float
```

Tolerance to low temperatures [0, 1].

<a id="evolution_simulator.creature.Creature.drought_tolerance"></a>

#### drought\_tolerance

```python
@property
def drought_tolerance() -> float
```

Tolerance to low water availability [0, 1].

<a id="evolution_simulator.creature.Creature.hibernation_tendency"></a>

#### hibernation\_tendency

```python
@property
def hibernation_tendency() -> float
```

Tendency to hibernate during harsh periods [0, 1].

<a id="evolution_simulator.creature.Creature.intelligence"></a>

#### intelligence

```python
@property
def intelligence() -> float
```

Problem-solving and learning ability [0, 1].

<a id="evolution_simulator.creature.Creature.adaptability"></a>

#### adaptability

```python
@property
def adaptability() -> float
```

Behavioral adaptability [0, 1].

<a id="evolution_simulator.creature.Creature.communication"></a>

#### communication

```python
@property
def communication() -> float
```

Ability to communicate with conspecifics [0, 1].

<a id="evolution_simulator.creature.Creature.reproduction_likelihood"></a>

#### reproduction\_likelihood

```python
@property
def reproduction_likelihood() -> float
```

Probability that a compatible mating attempt results in conception [0, 1].

Distinct from fecundity (litter size) — this is the per-encounter
conception probability.  Low values model sub-fertility; high values
model high fertility.  Both are under natural selection pressure.

<a id="evolution_simulator.creature.Creature.mutation_rate"></a>

#### mutation\_rate

```python
@property
def mutation_rate() -> float
```

Per-locus probability of mutation during reproduction [0.001 – 0.05].

Low values → offspring inherit genes faithfully; high values → more
genetic noise.  Both are under natural selection pressure.

<a id="evolution_simulator.creature.Creature.selectivity"></a>

#### selectivity

```python
@property
def selectivity() -> float
```

How much the compatibility threshold is raised above COMPATIBILITY_FLOOR [0, 1].
Higher selectivity → harder to find a compatible mate → more genetically protective.

<a id="evolution_simulator.creature.Creature.base_predation_rate"></a>

#### base\_predation\_rate

```python
@property
def base_predation_rate() -> float
```

Intrinsic per-week probability of death from exogenous causes [0, MAX_BASE_PREDATION_RATE].

Encodes the creature's inherent vulnerability to all mortality sources not
modelled explicitly — predation, accidents, disease outbreaks, etc. — rather
than simulating predator–prey interactions directly.  Shares loci with fecundity,
so high-fecundity genotypes also tend toward higher vulnerability (r/K tradeoff).
Density-dependent pressure from the habitat's PREDATION_ALPHA adds on top.

<a id="evolution_simulator.creature.Creature.is_sexually_viable"></a>

#### is\_sexually\_viable

```python
@property
def is_sexually_viable() -> bool
```

True when the creature has reached reproductive age.

<a id="evolution_simulator.creature.Creature.trace_lineage"></a>

#### trace\_lineage

```python
def trace_lineage(n: int) -> dict
```

Return the ancestry tree up to *n* generations back.

Parameters
----------
n : int
    Depth of ancestry to retrieve.  n=1 → direct parents only;
    n=2 → parents and grandparents; etc.

Returns
-------
dict
    {"id": str, "parents": [<same structure>, ...]}
    "parents" is empty when n=0 or no parents are recorded.

<a id="evolution_simulator.creature.Creature.compatibility_score"></a>

#### compatibility\_score

```python
def compatibility_score(other: "Creature") -> float
```

Cosine similarity of the two creatures' compatibility gene subsets.

Operates on the 245-locus ``compatibility_genes`` subspace, which
models a major-histocompatibility-complex-like signal.

Parameters
----------
other : Creature
    The individual to compare compatibility with.

Returns
-------
float
    Cosine similarity in [-1, 1]:
      -1  completely anti-correlated (maximally incompatible)
       0  orthogonal (unrelated; typical for random founders)
      +1  identical gene vectors (perfect genetic match)

    Creatures from a shared lineage cluster near +1; diverged
    populations drift toward 0 or below, producing reproductive
    isolation (speciation).

<a id="evolution_simulator.creature.Creature.is_compatible"></a>

#### is\_compatible

```python
def is_compatible(other: "Creature") -> tuple[bool, float, str]
```

Full mating-compatibility check between this creature and *other*.

Tests in order: opposite sex, both sexually viable, female not already
pregnant, and compatibility score above the combined selectivity
threshold.  The threshold is ``COMPATIBILITY_FLOOR + 0.15 * mean
selectivity`` of the pair, capped so high-OWA selectivity values
cannot make same-species mating impossible.

Parameters
----------
other : Creature
    The candidate mate to check against.

Returns
-------
compatible : bool
    ``True`` if all checks pass and mating can proceed.
score : float
    Raw cosine similarity from ``compatibility_score()`` in [-1, 1].
    Zero when the pair is rejected before the genetic check (e.g.
    same sex).
reason : str
    Empty string when compatible; otherwise a short token describing
    the first failing check (``"same_sex"``, ``"self_not_viable"``,
    ``"other_not_viable"``, ``"female_already_pregnant"``, or
    ``"genetic_incompatibility (score=… < threshold=…)"``.

<a id="evolution_simulator.creature.Creature.reproduce"></a>

#### reproduce

```python
def reproduce(other: "Creature", rng: np.random.Generator) -> list["Creature"]
```

Attempt to produce a litter with *other*.

Litter size is a Poisson draw centred on the female's fecundity trait
(minimum 1 if the draw is zero).  Each offspring receives an
independent per-locus Mendelian draw from the two parents, and each
locus is mutated at the rate of whichever parent contributed it,
keeping both mutation rates under independent selection pressure.

Parameters
----------
other : Creature
    The other parent.  One of the pair must be male and the other
    female; the method resolves which is which internally.
rng : np.random.Generator
    The single simulation generator threaded from ``SimulationRunner``.
    All reproductive stochasticity — fertility draw, litter size,
    parent-allele selection, mutation — comes from this stream.
    Required; there is no global-RNG fallback in simulation logic.

Returns
-------
list[Creature]
    Empty if the compatibility check fails or the fertility draw
    misses.  Non-empty on a successful conception: the female is
    marked pregnant and the litter is stored in
    ``female._pending_offspring`` until gestation completes, at which
    point ``Habitat.simulate_week()`` collects and releases them.

<a id="evolution_simulator.creature.Creature.simulate_week"></a>

#### simulate\_week

```python
def simulate_week(environment=None) -> dict
```

Advance the creature's internal state by one week.

Checks starvation and dehydration (energy/hydration set externally by
``Habitat`` before this call), increments age, tests for death from old
age, and advances the pregnancy timer.

Parameters
----------
environment : optional
    Currently unused; reserved for a future refactor where resource
    probabilities would be computed inside this method rather than by
    the owning ``Habitat``.

Returns
-------
dict
    Week log with keys:

    - ``"creature_id"``    : str
    - ``"age"``            : int — post-increment value
    - ``"is_alive"``       : bool
    - ``"cause_of_death"`` : str or None — ``"starvation"``,
      ``"dehydration"``, or ``"old_age"`` as applicable
    - ``"events"``         : list[str] — e.g. ``["gave_birth"]``
