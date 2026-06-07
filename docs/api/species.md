<a id="evolution_simulator.species"></a>

# evolution\_simulator.species

Species tracking for the evolution simulator.

Speciation is detected on the **same signal that governs mating**: the
245-locus ``compatibility_genes`` subset (see Creature.compatibility_score).
Every newborn is assigned to the nearest living species by centroid cosine
similarity.  Genuine reproductive isolation — and thus a new species — is
declared only when a *population-level* detector finds a bimodal split.

Why living centroids (not frozen progenitors)
---------------------------------------------
A species' reproductive reference point is the *current* population, refreshed
every ``respeciate_every`` weeks from its living members.  This is the fix for
runaway speciation: when an entire interbreeding population drifts together,
the centroid drifts with it, so ordinary drift never trips a speciation event.
A new species is declared only when a sub-cluster diverges in the compatibility
subset far enough that it can no longer breed with any living population — i.e.
genuine reproductive isolation (the Biological Species Concept).

Why the comparison is against ALL living species (not just the parent)
----------------------------------------------------------------------
Checking every living centroid prevents two failure modes:
  1. Drift-back: a lineage briefly diverges, then converges back toward an
     ancestral species — re-absorbed rather than logged as a new species.
  2. Convergent evolution: two independent lineages evolving toward the same
     compatibility region are recognised as one species.

Two references per species
---------------------------
Each confirmed species keeps two genomes:
  - **type** (full 500-dim, frozen at founding/promotion): anchors the name
    ("the name follows the type") and feeds the anagenesis axis, which measures
    how far a lineage has drifted from its own past.
  - **living centroid** (245-dim compatibility subset, refreshed periodically):
    the reproductive reference used for population-level split detection.

<a id="evolution_simulator.species.load_name_config"></a>

#### load\_name\_config

```python
def load_name_config(
        path: Path = DEFAULT_CONFIG_PATH) -> tuple[list[str], list[str]]
```

Load adjective and noun lists from a TOML config file.

The file must contain two keys at the top level:
    adjectives = ["word1", "word2", ...]
    nouns      = ["word1", "word2", ...]

Parameters
----------
path : Path
    Path to the TOML file.  Defaults to the bundled
    ``config/species_names.toml`` inside this package.

Returns
-------
(adjectives, nouns) — two lists of strings.

<a id="evolution_simulator.species.SpeciesRegistry"></a>

## SpeciesRegistry Objects

```python
class SpeciesRegistry()
```

Tracks species and detects speciation events on the mating-compatibility signal.

Usage
-----
1. Register founding individuals before the simulation begins.  The
   registry shares the simulation's single explicit Generator (passed by
   SimulationRunner) so its species-name draws stay on the same seeded
   stream as the rest of the run:

       registry = SpeciesRegistry(rng)
       name = registry.register_founding_species(founder.genes)
       founder.species = name

2. At each birth, assign_species() is called automatically by
   Habitat.simulate_week() when a registry is passed:

       result = habitat.simulate_week(rng, species_registry=registry)

3. Each week, after migrations are applied, refresh the living centroids
   (on the configured cadence) and promote any eligible candidates:

       all_alive = [c for hab in habitats.values() for c in hab.alive_creatures]
       registry.refresh_centroids(all_alive)        # periodic
       registry.promote_candidates(all_alive, current_week)

4. Inspect the history:

       registry.speciation_events  →  list[dict]   (each carries event_type)
       registry.all_species        →  list[str]

Detection algorithm (two-stage, compatibility space)
----------------------------------------------------
Detection: a newborn's 245-dim compatibility sub-vector is compared by
cosine similarity to the living centroid of every species with living
members.  If the best score is below compatibility_threshold for all of
them, the creature enters a candidate (joining the nearest candidate if
close enough, else founding a new one).  It keeps its parent species label
during the candidate period.

Promotion: each week, promote_candidates() promotes a candidate to a
confirmed species only when it has at least min_species_population living
members AND has existed for at least min_species_weeks weeks.  On promotion
its full-genome type is frozen, its living centroid is seeded from its
members, and all living members are renamed.  Candidates whose members all
die before promotion are silently evaporated.

Parameters
----------
compatibility_threshold : float
    Cosine similarity (on the 245-dim compatibility subset) to a species'
    living centroid required to be counted as that species.  Set a touch
    below the mating floor (Creature.COMPATIBILITY_FLOOR) so a species
    boundary means "cannot breed with that population".  Default 0.65.
min_species_population : int
    Minimum number of living candidate members required for promotion.
min_species_weeks : int
    Minimum number of weeks a candidate must exist before promotion.
compat_indices : list[int], optional
    Gene loci that define the compatibility subset.  Defaults to
    Creature's ``compatibility_genes`` index set.

<a id="evolution_simulator.species.SpeciesRegistry.register_founding_species"></a>

#### register\_founding\_species

```python
def register_founding_species(genes: np.ndarray,
                              name: str | None = None) -> str
```

Register a founding species.

Stores the full genome as the frozen type and seeds the species' living
centroid from the genome's compatibility subset.

Parameters
----------
genes : np.ndarray (500-dim)
name  : str, optional.  Auto-generated if not provided.

Returns the species name; set ``creature.species`` to this on founders.

<a id="evolution_simulator.species.SpeciesRegistry.assign_species"></a>

#### assign\_species

```python
def assign_species(creature: "Creature") -> str
```

Assign the species for a newborn creature.

Compares the newborn's compatibility sub-vector against every living
species centroid and assigns the nearest one, regardless of whether
the cosine score clears ``compatibility_threshold``.  Genuine
reproductive isolation is detected at the population level by
``detect_subcluster_splits()``, not per-newborn.

If the registry has no living centroids (empty or fully extinct),
the newborn's genome is immediately registered as a bootstrap species.

Parameters
----------
creature : Creature
    The newly born individual.  ``creature.species`` is updated as
    a side effect.

Returns
-------
str
    The species name assigned to *creature*.

<a id="evolution_simulator.species.SpeciesRegistry.refresh_centroids"></a>

#### refresh\_centroids

```python
def refresh_centroids(alive_creatures: "list[Creature]") -> None
```

Recompute each species' living centroid from its current members.

Call periodically (every ``respeciate_every`` weeks) after migrations
so migrated creatures are counted toward their new habitat's species
distribution.

Candidate members are excluded from their parent species' centroid:
they are incipiently divergent and must not drag the parent's
reproductive reference toward themselves.  Species whose only living
members are candidates are dropped from the centroid comparison set
until non-candidate individuals remain; they stay in the historical
registry.

Parameters
----------
alive_creatures : list[Creature]
    All living creatures across all habitats.

<a id="evolution_simulator.species.SpeciesRegistry.promote_candidates"></a>

#### promote\_candidates

```python
def promote_candidates(alive_creatures: "list[Creature]",
                       current_week: int) -> list[dict]
```

Promote any candidates that meet both population and age criteria.

Call once per simulation step, after all habitats have run and
migrations have been applied.  Candidates whose members all died before
promotion are recorded in ``failed_speciation_attempts`` and removed.

Parameters
----------
alive_creatures : list[Creature]
    All living creatures across all habitats.
current_week : int
    The current simulation week, used to compute how long each
    candidate has existed (must exceed ``min_species_weeks`` for
    promotion).

Returns
-------
list[dict]
    New speciation events created this call.  Each event dict has keys
    ``new_species``, ``parent_species``, ``creature_id``, ``week``,
    and ``event_type``.  Also appended to ``self.speciation_events``.

<a id="evolution_simulator.species.SpeciesRegistry.detect_subcluster_splits"></a>

#### detect\_subcluster\_splits

```python
def detect_subcluster_splits(alive_creatures: "list[Creature]",
                             current_week: int) -> None
```

Detect reproductive-isolation splits within a species (cladogenesis).

Must run **before** ``refresh_centroids`` so any sub-cluster seeded as
a candidate is excluded from the parent's refreshed centroid.

For each species with at least ``2 * min_species_population``
non-candidate members, the members are clustered in compatibility space
via spherical k-means.  If two or more mutually reproductively-isolated
sub-clusters are found (all pairwise centroid cosines below
``split_isolation_threshold``), the cluster nearest the frozen type
keeps the species name and each other cluster seeds or extends a
candidate.

This catches allopatric/sympatric divergence that a single shared
centroid masks: two habitat-adapted populations sit symmetrically
around their midpoint centroid so newborns never individually fall
below the assignment threshold — but their members form two distinct
clusters, which this detector finds directly.

Parameters
----------
alive_creatures : list[Creature]
    All living creatures across all habitats.
current_week : int
    The current simulation week, recorded as the ``detected_week``
    on any newly created candidates.

<a id="evolution_simulator.species.SpeciesRegistry.detect_anagenesis"></a>

#### detect\_anagenesis

```python
def detect_anagenesis(alive_creatures: "list[Creature]",
                      current_week: int,
                      pending_only: bool = False) -> list[dict]
```

Detect a lineage that has transformed in place without splitting.

Anagenesis (chronospecies) is visible on the **phenotype** axis —
the 32-trait raw OWA vector — not the compatibility axis.  A lineage
that drifts far from its frozen type phenotype (centred cosine below
``anagenesis_threshold``) and stays there for ``anagenesis_weeks``
is respeciated: members are partitioned by closest phenotype between
the new centroid and the old type; the name follows the type.

Species with an active split candidate are skipped (cladogenesis
takes priority); species with fewer than ``min_species_population``
living members are skipped.

Parameters
----------
alive_creatures : list[Creature]
    All living creatures across all habitats.
current_week : int
    The current simulation week, used to advance the persistence
    clock and to timestamp any new events.
pending_only : bool, optional
    If ``True``, only process species already in
    ``_anagenesis_pending`` (rebound-check or fire).  Used for the
    weekly between-cadence poll so the persistence clock is
    week-precise.  Default ``False`` (full scan).

Returns
-------
list[dict]
    Anagenesis events created this call.  Each dict has keys
    ``new_species``, ``parent_species``, ``creature_id``, ``week``,
    and ``event_type: "anagenesis"``.  Also appended to
    ``self.speciation_events``.

<a id="evolution_simulator.species.SpeciesRegistry.similarity_to_all_progenitors"></a>

#### similarity\_to\_all\_progenitors

```python
def similarity_to_all_progenitors(creature: "Creature") -> dict[str, float]
```

Full-genome cosine similarity between *creature* and every species' frozen type.

Diagnostics helper; uses the full 500-dim genome, not the 245-dim
compatibility subset used for mating and centroid detection.

Parameters
----------
creature : Creature
    The individual to compare against all registered progenitor genomes.

Returns
-------
dict[str, float]
    Mapping of species name → cosine similarity in [-1, 1].
    Empty dict if no species are registered.

<a id="evolution_simulator.species.SpeciesRegistry.progenitor_genes"></a>

#### progenitor\_genes

```python
def progenitor_genes(species_name: str) -> np.ndarray | None
```

Return a copy of the frozen TYPE genome for the named species.

Parameters
----------
species_name : str
    Name of a registered species (living or extinct).

Returns
-------
np.ndarray or None
    A copy of the 500-dim founding genome, or ``None`` if the species
    is not found.

<a id="evolution_simulator.species.SpeciesRegistry.centroid"></a>

#### centroid

```python
def centroid(species_name: str) -> np.ndarray | None
```

Return a copy of the living compatibility centroid for *species_name*.

Parameters
----------
species_name : str
    Name of a registered species.

Returns
-------
np.ndarray or None
    The 245-dim compatibility centroid as of the last
    ``refresh_centroids()`` call, or ``None`` if the species has no
    living members (extinct or centroid not yet computed).

<a id="evolution_simulator.species.SpeciesRegistry.species_count"></a>

#### species\_count

```python
@property
def species_count() -> int
```

Number of distinct species ever registered (including extinct).

<a id="evolution_simulator.species.SpeciesRegistry.living_species_count"></a>

#### living\_species\_count

```python
@property
def living_species_count() -> int
```

Number of species with living members as of the last centroid refresh.

<a id="evolution_simulator.species.SpeciesRegistry.all_species"></a>

#### all\_species

```python
@property
def all_species() -> list[str]
```

Names of all species ever registered.
