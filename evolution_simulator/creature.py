"""
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
"""

import itertools

import numpy as np

# Genome layout + the OWA genotype->phenotype kernel live in genetics.py
# (single source of truth).
from .genetics import DEFAULT_OWA_ALPHA, DEFAULT_TRAIT_GENE_INDICES, GENE_DIMS, owa_value

# Deterministic creature-ID source.  A monotonic counter (not uuid4, which draws
# OS entropy and is therefore non-reproducible) so that two runs with the same
# config + seed produce byte-identical IDs and hence byte-identical logs.
# SimulationRunner.setup() calls reset_creature_ids() at the start of each run.
_id_counter = itertools.count(1)


def _next_creature_id() -> str:
    return f"c{next(_id_counter):010d}"


def reset_creature_ids(start: int = 1) -> None:
    """
    Reset the monotonic creature-ID counter.

    Called once at simulation setup so creature IDs are deterministic and
    byte-identical across runs that share the same config and seed.

    Parameters
    ----------
    start : int, optional
        First integer value the counter will produce.  Default 1, so the
        first ID is ``c0000000001``.
    """
    global _id_counter
    _id_counter = itertools.count(start)


class Creature:
    """
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
    """

    # Class-level attribute so species subclasses can override the mapping
    TRAIT_GENE_INDICES: dict[str, list[int]] = DEFAULT_TRAIT_GENE_INDICES

    # OWA weight-decay rate.  The highest-ranked gene locus receives weight
    # alpha, the next alpha*(1-alpha), and so on (normalised to sum to 1).
    # Higher values → faster decay → top locus dominates more strongly.
    # Range (0, 1); species subclasses can override to tune selectivity.
    # Defaults to genetics.DEFAULT_OWA_ALPHA but stays an overridable class attr.
    OWA_ALPHA: float = DEFAULT_OWA_ALPHA

    # Scales the [0,1] base_predation_rate trait into a per-week probability.
    # This is an abstraction for all exogenous mortality not modelled explicitly:
    # being eaten by a predator, falling to a fatal injury, disease, extreme
    # weather events, etc.  Rather than simulate predator–prey interactions
    # directly (which would require a second population of agents), we encode
    # the creature's intrinsic vulnerability as a heritable trait under
    # selection pressure.  Density-dependent mortality from the habitat's
    # PREDATION_ALPHA adds on top.
    MAX_BASE_PREDATION_RATE: float = 0.005

    # Minimum cosine-similarity score (on [-1, 1] scale) required for mating.
    # The selectivity trait can raise this up toward 1.0.
    # Override in subclasses to tune how permissive mating is for a species.
    COMPATIBILITY_FLOOR: float = 0.7

    def __init__(
        self,
        genes: np.ndarray | None = None,
        parents: list | None = None,
        creature_id: str | None = None,
    ):
        if genes is not None:
            if genes.shape != (GENE_DIMS,):
                raise ValueError(f"genes must be a 1-D array of length {GENE_DIMS}, got {genes.shape}")
            self.genes: np.ndarray = genes.astype(float)
        else:
            self.genes = np.random.randn(GENE_DIMS)

        self.creature_id: str = creature_id if creature_id is not None else _next_creature_id()
        self.parents: list[Creature] = parents if parents is not None else []
        self.generation: int = (max(p.generation for p in self.parents) + 1) if self.parents else 0

        # Must be initialised before any property access (sex determination calls _compute_trait)
        self._trait_cache: dict[str, float] = {}

        # Vital state
        self.age: int = 0
        self.is_alive: bool = True
        self.cause_of_death: str | None = None

        # Sex is fixed at birth from a single raw gene (heritable, no OWA bias).
        # A raw N(0,1) gene gives exactly 50/50 male/female for random populations.
        _sex_locus = self.TRAIT_GENE_INDICES["sex_determination"][0]
        self.sex: str = "female" if self.genes[_sex_locus] >= 0 else "male"

        # Species is inherited from the first parent; "unknown" for founders.
        # The SpeciesRegistry validates this at birth and triggers a speciation
        # event if the genome has diverged too far from the parent species' progenitor.
        self.species: str = self.parents[0].species if self.parents else "unknown"

        # Reproductive state
        self.is_pregnant: bool = False
        self.weeks_pregnant: int = 0
        # Litter created at fertilisation; released into the population at term
        self._pending_offspring: list[Creature] = []

        # Resource state (driven by habitat cross-product calculations)
        self.energy: float = 1.0  # 0 = starving, 1 = fully fed
        self.hydration: float = 1.0  # 0 = dehydrated, 1 = fully hydrated

    # ------------------------------------------------------------------
    # Core trait computation
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _compute_trait(self, name: str) -> float:
        """
        Compute the normalised [0, 1] value for a named trait via OWA.

        Gene values at contributing loci are sorted descending and combined
        with exponentially decaying weights (OWA_ALPHA controls the decay),
        then passed through a sigmoid.  Higher-valued loci receive more weight,
        so beneficial mutations are immediately visible to selection.

        Results are memoised in ``_trait_cache``: since genes are immutable
        after birth, each trait value is computed at most once per creature
        lifetime.  Subclasses can tune OWA_ALPHA or override
        TRAIT_GENE_INDICES[name].

        Parameters
        ----------
        name : str
            Trait name; must be a key in ``TRAIT_GENE_INDICES``.

        Returns
        -------
        float
            Normalised sigmoid value in [0, 1].
        """
        cached = self._trait_cache.get(name)
        if cached is not None:
            return cached
        indices = self.TRAIT_GENE_INDICES[name]
        result = owa_value(self.genes[indices], self.__class__.OWA_ALPHA)
        self._trait_cache[name] = result
        return result

    def trait_indices(self, name: str) -> list[int]:
        """
        Return the gene loci that contribute to the named trait.

        Parameters
        ----------
        name : str
            Trait name; must be a key in ``TRAIT_GENE_INDICES``.

        Returns
        -------
        list[int]
            Indices into the 500-dimensional gene vector for this trait.
        """
        return self.TRAIT_GENE_INDICES[name]

    # ------------------------------------------------------------------
    # Trait properties  (each scales the [0,1] signal to a useful range)
    # ------------------------------------------------------------------

    @property
    def fecundity(self) -> float:
        """Expected offspring per birth event (1 – 8)."""
        return 1.0 + 7.0 * self._compute_trait("fecundity")

    @property
    def reproduction_time(self) -> int:
        """Weeks required to gestate / develop offspring (1 – 20)."""
        return int(1 + 19 * self._compute_trait("reproduction_time"))

    @property
    def weeks_to_sexual_viability(self) -> int:
        """Weeks before the creature can reproduce (4 – 50)."""
        return int(4 + 46 * self._compute_trait("weeks_to_sexual_viability"))

    @property
    def parental_investment(self) -> float:
        """Parental care score [0, 1]; higher → better offspring survival."""
        return self._compute_trait("parental_investment")

    @property
    def aggression(self) -> float:
        """Aggression score [0, 1]."""
        return self._compute_trait("aggression")

    @property
    def migration_likelihood(self) -> float:
        """Weekly probability of migrating to a new environment [0, 1]."""
        return self._compute_trait("migration_likelihood")

    @property
    def territorial(self) -> float:
        """Territorial tendency [0, 1]."""
        return self._compute_trait("territorial")

    @property
    def social_tendency(self) -> float:
        """Social / cooperative tendency [0, 1]."""
        return self._compute_trait("social_tendency")

    @property
    def pack_hunting(self) -> float:
        """Tendency to hunt in coordinated groups [0, 1]."""
        return self._compute_trait("pack_hunting")

    @property
    def scavenging_tendency(self) -> float:
        """Tendency to scavenge rather than actively hunt [0, 1]."""
        return self._compute_trait("scavenging_tendency")

    @property
    def nocturnal_tendency(self) -> float:
        """Tendency to be active at night [0, 1]."""
        return self._compute_trait("nocturnal_tendency")

    @property
    def risk_tolerance(self) -> float:
        """Willingness to accept risk in pursuit of resources [0, 1]."""
        return self._compute_trait("risk_tolerance")

    @property
    def size(self) -> float:
        """Body size score [0, 1]; scales food requirement and combat."""
        return self._compute_trait("size")

    @property
    def strength(self) -> float:
        """Physical strength [0, 1]."""
        return self._compute_trait("strength")

    @property
    def speed(self) -> float:
        """Movement speed [0, 1]."""
        return self._compute_trait("speed")

    @property
    def camouflage(self) -> float:
        """Predator-avoidance / camouflage score [0, 1]."""
        return self._compute_trait("camouflage")

    @property
    def metabolism(self) -> float:
        """Weekly resource consumption multiplier [0.5 – 2.0]."""
        return 0.5 + 1.5 * self._compute_trait("metabolism")

    @property
    def foraging_ability(self) -> float:
        """Effectiveness at locating food [0, 1]."""
        return self._compute_trait("foraging_ability")

    @property
    def water_efficiency(self) -> float:
        """Water-use efficiency [0, 1]; higher → needs less water per week."""
        return self._compute_trait("water_efficiency")

    @property
    def max_lifespan(self) -> int:
        """Maximum lifespan in weeks (40 – 400 weeks, i.e. ~1 – 8 years)."""
        return int(40 + 360 * self._compute_trait("max_lifespan"))

    @property
    def disease_resistance(self) -> float:
        """Disease resistance [0, 1]."""
        return self._compute_trait("disease_resistance")

    @property
    def immune_response(self) -> float:
        """Immune-response strength [0, 1]."""
        return self._compute_trait("immune_response")

    @property
    def stress_tolerance(self) -> float:
        """Tolerance to general environmental stressors [0, 1]."""
        return self._compute_trait("stress_tolerance")

    @property
    def heat_tolerance(self) -> float:
        """Tolerance to high temperatures [0, 1]."""
        return self._compute_trait("heat_tolerance")

    @property
    def cold_tolerance(self) -> float:
        """Tolerance to low temperatures [0, 1]."""
        return self._compute_trait("cold_tolerance")

    @property
    def drought_tolerance(self) -> float:
        """Tolerance to low water availability [0, 1]."""
        return self._compute_trait("drought_tolerance")

    @property
    def hibernation_tendency(self) -> float:
        """Tendency to hibernate during harsh periods [0, 1]."""
        return self._compute_trait("hibernation_tendency")

    @property
    def intelligence(self) -> float:
        """Problem-solving and learning ability [0, 1]."""
        return self._compute_trait("intelligence")

    @property
    def adaptability(self) -> float:
        """Behavioral adaptability [0, 1]."""
        return self._compute_trait("adaptability")

    @property
    def communication(self) -> float:
        """Ability to communicate with conspecifics [0, 1]."""
        return self._compute_trait("communication")

    @property
    def reproduction_likelihood(self) -> float:
        """
        Probability that a compatible mating attempt results in conception [0, 1].

        Distinct from fecundity (litter size) — this is the per-encounter
        conception probability.  Low values model sub-fertility; high values
        model high fertility.  Both are under natural selection pressure.
        """
        return self._compute_trait("reproduction_likelihood")

    @property
    def mutation_rate(self) -> float:
        """
        Per-locus probability of mutation during reproduction [0.001 – 0.05].

        Low values → offspring inherit genes faithfully; high values → more
        genetic noise.  Both are under natural selection pressure.
        """
        return 0.001 + 0.049 * self._compute_trait("mutation_rate")

    @property
    def selectivity(self) -> float:
        """
        How much the compatibility threshold is raised above COMPATIBILITY_FLOOR [0, 1].
        Higher selectivity → harder to find a compatible mate → more genetically protective.
        """
        return self._compute_trait("selectivity")

    @property
    def base_predation_rate(self) -> float:
        """
        Intrinsic per-week probability of death from exogenous causes [0, MAX_BASE_PREDATION_RATE].

        Encodes the creature's inherent vulnerability to all mortality sources not
        modelled explicitly — predation, accidents, disease outbreaks, etc. — rather
        than simulating predator–prey interactions directly.  Shares loci with fecundity,
        so high-fecundity genotypes also tend toward higher vulnerability (r/K tradeoff).
        Density-dependent pressure from the habitat's PREDATION_ALPHA adds on top.
        """
        return self._compute_trait("base_predation_rate") * self.MAX_BASE_PREDATION_RATE

    @property
    def is_sexually_viable(self) -> bool:
        """True when the creature has reached reproductive age."""
        return self.age >= self.weeks_to_sexual_viability

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    def trace_lineage(self, n: int) -> dict:
        """
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
        """
        if n <= 0:
            return {"id": self.creature_id, "parents": []}
        return {
            "id": self.creature_id,
            "parents": [p.trace_lineage(n - 1) for p in self.parents],
        }

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def compatibility_score(self, other: "Creature") -> float:
        """
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
        """
        indices = self.TRAIT_GENE_INDICES["compatibility_genes"]
        v1 = self.genes[indices]
        v2 = other.genes[indices]
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        return float(np.dot(v1, v2) / norm)

    def is_compatible(self, other: "Creature") -> tuple[bool, float, str]:
        """
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
        """
        if self.sex == other.sex:
            return False, 0.0, "same_sex"
        if not self.is_sexually_viable:
            return False, 0.0, "self_not_viable"
        if not other.is_sexually_viable:
            return False, 0.0, "other_not_viable"
        female = self if self.sex == "female" else other
        if female.is_pregnant:
            return False, 0.0, "female_already_pregnant"

        score = self.compatibility_score(other)
        # selectivity trait raises the threshold above COMPATIBILITY_FLOOR by at most 0.15.
        # Using a fixed cap (rather than scaling by 1-FLOOR) prevents OWA bias from
        # pushing the threshold so high that same-species creatures can't mate.
        threshold = self.COMPATIBILITY_FLOOR + 0.15 * ((self.selectivity + other.selectivity) / 2)
        if score < threshold:
            return False, score, f"genetic_incompatibility (score={score:.4f} < threshold={threshold:.4f})"
        return True, score, ""

    def reproduce(self, other: "Creature", rng: np.random.Generator) -> list["Creature"]:
        """
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
        """
        compatible, _, _ = self.is_compatible(other)
        if not compatible:
            return []

        female = self if self.sex == "female" else other
        male = other if self.sex == "female" else self

        # --- Fertility check ---
        # Even compatible pairs may not conceive on every encounter.
        # Conception probability is the average of both parents' traits so
        # both are under selection pressure.
        fertility = (female.reproduction_likelihood + male.reproduction_likelihood) / 2
        if rng.random() > fertility:
            return []

        # Litter size: Poisson draw on female fecundity, min 1
        n_offspring = max(1, int(rng.poisson(female.fecundity)))

        litter: list[Creature] = []
        for _ in range(n_offspring):
            # Independent per-locus parent selection for each sibling
            parent_choice = rng.integers(0, 2, size=GENE_DIMS)
            child_genes = np.where(parent_choice == 0, self.genes, other.genes)

            # Mutation: each locus uses its chosen parent's rate
            mut_rates = np.where(parent_choice == 0, self.mutation_rate, other.mutation_rate)
            mutation_mask = rng.random(GENE_DIMS) < mut_rates
            if mutation_mask.any():
                child_genes[mutation_mask] = rng.standard_normal(int(mutation_mask.sum()))

            litter.append(Creature(genes=child_genes, parents=[self, other]))

        # Mark female as pregnant; store litter until term
        female.is_pregnant = True
        female.weeks_pregnant = 0
        female._pending_offspring = litter

        return litter

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_week(self, environment=None) -> dict:
        """
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
        """
        if not self.is_alive:
            return self._week_log(["creature is already dead"])

        # --- Starvation / dehydration ---
        # Energy and hydration are set by the Habitat before this is called.
        # Checking here also handles standalone simulate_week() use.
        if self.energy <= 0:
            self.is_alive = False
            self.cause_of_death = "starvation"
            return self._week_log(["died of starvation"])
        if self.hydration <= 0:
            self.is_alive = False
            self.cause_of_death = "dehydration"
            return self._week_log(["died of dehydration"])

        self.age += 1
        events: list[str] = []

        # --- Old age ---
        if self.age >= self.max_lifespan:
            self.is_alive = False
            self.cause_of_death = "old_age"
            events.append(f"died of old age at week {self.age}")
            return self._week_log(events)

        # --- Pregnancy progression ---
        if self.is_pregnant:
            self.weeks_pregnant += 1
            if self.weeks_pregnant >= self.reproduction_time:
                events.append("gave_birth")
                self.is_pregnant = False
                self.weeks_pregnant = 0
                # _pending_offspring is collected by the Habitat and cleared there

        return self._week_log(events)

    def _week_log(self, events: list[str]) -> dict:
        return {
            "creature_id": self.creature_id,
            "age": self.age,
            "is_alive": self.is_alive,
            "cause_of_death": self.cause_of_death,
            "events": events,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "alive" if self.is_alive else f"dead ({self.cause_of_death})"
        return f"Creature(id={self.creature_id[:8]}…, age={self.age}, {status}, parents={len(self.parents)})"
