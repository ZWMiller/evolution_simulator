import uuid
from collections import deque
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


# ---------------------------------------------------------------------------
# Vectorized trait computation helpers (used by compute_stats)
# ---------------------------------------------------------------------------

# Maps each LOGGED_TRAITS property name to the key used by _compute_trait.
# All names match: every property calls _compute_trait with its own name.
_TRAIT_INTERNAL_KEY: dict[str, str] = {t: t for t in LOGGED_TRAITS}

# Scaling table: (offset, scale, is_int) for each LOGGED_TRAIT.
# Replicates the property definitions in creature.py exactly.
# is_int=True means floor-truncation is applied per-creature before averaging,
# matching int(offset + scale * raw) rather than the incorrect int(mean).
_TRAIT_SCALING: dict[str, tuple] = {
    "fecundity":                 (1.0,   7.0,   False),
    "reproduction_time":         (1.0,   19.0,  True),
    "weeks_to_sexual_viability": (4.0,   46.0,  True),
    "parental_investment":       (0.0,   1.0,   False),
    "reproduction_likelihood":   (0.0,   1.0,   False),
    "metabolism":                (0.5,   1.5,   False),
    "water_efficiency":          (0.0,   1.0,   False),
    "max_lifespan":              (40.0,  360.0, True),
    "disease_resistance":        (0.0,   1.0,   False),
    "immune_response":           (0.0,   1.0,   False),
    "stress_tolerance":          (0.0,   1.0,   False),
    "heat_tolerance":            (0.0,   1.0,   False),
    "cold_tolerance":            (0.0,   1.0,   False),
    "drought_tolerance":         (0.0,   1.0,   False),
    "hibernation_tendency":      (0.0,   1.0,   False),
    "migration_likelihood":      (0.0,   1.0,   False),
    "risk_tolerance":            (0.0,   1.0,   False),
    "aggression":                (0.0,   1.0,   False),
    "territorial":               (0.0,   1.0,   False),
    "social_tendency":           (0.0,   1.0,   False),
    "nocturnal_tendency":        (0.0,   1.0,   False),
    "size":                      (0.0,   1.0,   False),
    "strength":                  (0.0,   1.0,   False),
    "speed":                     (0.0,   1.0,   False),
    "camouflage":                (0.0,   1.0,   False),
    "foraging_ability":          (0.0,   1.0,   False),
    "intelligence":              (0.0,   1.0,   False),
    "adaptability":              (0.0,   1.0,   False),
    "pack_hunting":              (0.0,   1.0,   False),
    "scavenging_tendency":       (0.0,   1.0,   False),
    "communication":             (0.0,   1.0,   False),
    "mutation_rate":             (0.001, 0.049, False),
    "selectivity":               (0.0,   1.0,   False),
    "base_predation_rate":       (0.0,   0.005, False),
}


def _batch_compute_traits(creatures: list) -> np.ndarray:
    """
    Vectorized OWA trait computation for all LOGGED_TRAITS across a list of creatures.

    Replicates the per-creature _compute_trait + property scaling exactly:
      1. Stack genes into (N, 500).
      2. For each trait: fancy-index the relevant loci, sort descending per row,
         apply normalised OWA weights, sigmoid → raw [0,1] value.
      3. Apply each trait's (offset, scale) and floor-truncate int-valued traits
         per creature before averaging, so mean(int(f(x))) is preserved rather
         than the incorrect int(mean(f(x))).

    Returns (N, len(LOGGED_TRAITS)) float64 array of scaled values.
    Uses the first creature's class for TRAIT_GENE_INDICES and OWA_ALPHA.
    """
    N = len(creatures)
    cls = creatures[0].__class__
    alpha: float = cls.OWA_ALPHA
    gene_matrix: np.ndarray = np.stack([c.genes for c in creatures])  # (N, 500)

    result = np.empty((N, len(LOGGED_TRAITS)), dtype=np.float64)
    for j, trait in enumerate(LOGGED_TRAITS):
        key = _TRAIT_INTERNAL_KEY[trait]
        indices = cls.TRAIT_GENE_INDICES[key]
        k = len(indices)

        vals = gene_matrix[:, indices]                 # (N, k)
        sorted_vals = np.sort(vals, axis=1)[:, ::-1]  # descending per row

        i_arr = np.arange(k, dtype=np.float64)
        weights = alpha * (1.0 - alpha) ** i_arr
        weights /= weights.sum()

        raw = sorted_vals @ weights                    # (N,)
        sigmoid_vals = 1.0 / (1.0 + np.exp(-raw))     # (N,) ∈ [0, 1]

        offset, scale, is_int = _TRAIT_SCALING[trait]
        scaled = offset + scale * sigmoid_vals
        if is_int:
            scaled = np.floor(scaled)
        result[:, j] = scaled

    return result


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


def _gale_shapley(
    score_matrix: np.ndarray,
    male_thresholds: np.ndarray,
    female_thresholds: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Male-proposing Gale-Shapley deferred-acceptance stable matching.

    Finds a stable matching between M males and F females given a pairwise
    compatibility score matrix and per-creature acceptance thresholds.  A
    matching is *stable* if no unmatched (male, female) pair both prefer each
    other over their current partners — i.e. no blocking pair exists.

    Parameters
    ----------
    score_matrix : (M, F) float array
        Pairwise cosine similarity scores in [-1, 1].  Entry [i, j] is the
        compatibility score between male i and female j.  Produced by
        Habitat._build_compatibility_matrix().
    male_thresholds : (M,) float array
        Each male's personal acceptance floor.  He will not propose to any
        female whose score is below this value.
    female_thresholds : (F,) float array
        Each female's personal acceptance floor.  She will auto-reject any
        proposer whose score is below this value.

    Returns
    -------
    list of (male_idx, female_idx) int tuples
        Indices into the original males/females lists passed to
        _mate_stable_matching.  Only matched pairs are returned; unmatched
        individuals are omitted.  Each matched pair is subsequently passed to
        _attempt_mating, which re-checks is_compatible() and calls reproduce().
        The algorithm itself does not trigger conception — it only determines
        who attempts to mate with whom.

    Algorithm
    ---------
    Each free male proposes to his top-ranked remaining candidate.  A free
    female tentatively accepts; an engaged female accepts if the new proposer
    scores higher for her than her current partner (releasing the old partner
    back to the free pool).  Rejected males advance to their next candidate.
    The loop terminates when every free male has exhausted his list.

    Assumptions and baked-in choices
    ---------------------------------
    1.  **Male-proposing direction.**  This gives every male his best possible
        partner across all stable matchings, and every female her worst.  The
        direction is biologically arbitrary — there is no neutral variant.
        Swapping to female-proposing simply flips which side is optimal vs.
        pessimal.

    2.  **Bilateral threshold pre-filtering.**  A female j is included in male
        i's preference list only if score[i,j] >= BOTH male_thresholds[i] AND
        female_thresholds[j].  This is an optimisation: a female will never
        accept a score below her floor, so proposals below it are provably
        wasted.  Skipping them reduces the worst-case proposal count without
        changing the set of possible stable matchings.

    3.  **Tie-breaking via noise.**  Preference lists are ranked on
        score_matrix + N(0, NOISE_SIGMA).  The noise is sampled once before the
        loop so rankings are consistent throughout a single call.  Threshold
        checks always use the *original* score_matrix — noise must never push a
        score across a compatibility floor.  NOISE_SIGMA = 0.005 is well below
        the minimum meaningful score gap (~0.01 between near-identical
        same-species pairs with genome noise = 0.05).

    4.  **Female preference comparison uses noisy scores.**  When a female
        decides whether to swap her current partner for a new proposer, she
        compares noisy[new, j] vs noisy[current, j].  Using the same noisy
        matrix that determined the initial ranking keeps comparisons consistent
        and ensures no two males have exactly equal appeal to the same female.

    5.  **Threshold asymmetry vs is_compatible().**  This function uses
        *individual* thresholds (FLOOR + 0.15 * own_selectivity) for
        pre-filtering and acceptance.  Creature.is_compatible() uses the
        *average* selectivity of the pair.  A pair that clears both individual
        thresholds here can still fail is_compatible() if the pair-average
        raises the combined bar above one partner's score.  _attempt_mating()
        re-checks is_compatible() after the matching, so these edge cases
        produce a mating_event with compatible=False rather than a birth.

    6.  **Deque order (FIFO) affects which stable matching is found.**  When
        multiple males become free in the same round, they are processed in the
        order they were released.  Multiple stable matchings can exist; FIFO
        does not guarantee any specific one.  Combined with noise (assumption 3)
        this introduces stochasticity without biasing toward any fixed outcome.
    """
    M, F = score_matrix.shape
    if M == 0 or F == 0:
        return []

    NOISE_SIGMA = 0.005  # see assumption 3
    noisy = score_matrix + np.random.normal(0, NOISE_SIGMA, score_matrix.shape)

    # Build each male's ordered preference list (assumptions 2 and 3).
    # male_prefs[i] is sorted from most-preferred (index 0) to least-preferred.
    # Only females that clear BOTH individual thresholds are included.
    male_prefs: list[list[int]] = []
    for i in range(M):
        eligible_mask = (
            (score_matrix[i] >= male_thresholds[i]) &   # male's own floor
            (score_matrix[i] >= female_thresholds)       # female's own floor
        )
        eligible_indices = np.where(eligible_mask)[0]
        if eligible_indices.size == 0:
            male_prefs.append([])
            continue
        order = np.argsort(noisy[i, eligible_indices])[::-1]  # highest noisy score first
        male_prefs.append(eligible_indices[order].tolist())

    # Algorithm state ----------------------------------------------------------
    # male_next[i]      : next index into male_prefs[i] for the next proposal.
    #                     Only ever increments — a male never re-proposes to a
    #                     female who has already rejected him.
    # male_partner[i]   : current female partner index, or -1 if unmatched.
    # female_partner[j] : current male partner index, or -1 if free.
    male_next:      list[int] = [0] * M
    male_partner:   list[int] = [-1] * M
    female_partner: list[int] = [-1] * F

    # Seed the free queue with every male who has at least one candidate.
    # Males with empty preference lists (no compatible females) are never queued
    # and remain permanently unmatched, which is correct.
    free: deque[int] = deque(i for i in range(M) if male_prefs[i])

    while free:
        i = free.popleft()

        # A male may have exhausted his list since he was last queued (e.g. he
        # was re-queued after a rejection, then his remaining candidates were
        # all already visited).  Guard prevents index-out-of-bounds.
        if male_next[i] >= len(male_prefs[i]):
            continue

        j = male_prefs[i][male_next[i]]
        male_next[i] += 1  # advance regardless of outcome; i never re-proposes to j

        if female_partner[j] == -1:
            # j is free: tentative acceptance
            female_partner[j] = i
            male_partner[i] = j

        else:
            current = female_partner[j]
            # Female j compares i to her current partner using noisy scores
            # (assumption 4) to break ties consistently
            if noisy[i, j] > noisy[current, j]:
                # j prefers i; release current partner back to the free pool
                male_partner[current] = -1
                female_partner[j] = i
                male_partner[i] = j
                if male_next[current] < len(male_prefs[current]):
                    free.append(current)
            else:
                # j rejects i; re-queue i only if he still has candidates
                if male_next[i] < len(male_prefs[i]):
                    free.append(i)

    # Return matched pairs.  _attempt_mating will re-run is_compatible() on
    # each pair (assumption 5), so pairs that pass Gale-Shapley thresholds but
    # fail the pair-averaged is_compatible() threshold produce a compatible=False
    # event rather than offspring.
    return [(i, male_partner[i]) for i in range(M) if male_partner[i] != -1]


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

    # Initial viability study (481 LHS trials, not extensive): stable outcomes
    # rose from ~14% at low gain (0.10–0.22) to ~22% at high gain (0.35–0.60);
    # estimated viable range 0.35–0.60.
    FOOD_ENERGY_GAIN: float = 0.47
    # Initial viability study (481 LHS trials, not extensive): stable outcomes
    # peaked at ~28% at low cost (0.05–0.14) and dropped to ~8% above 0.32;
    # estimated viable range 0.05–0.22.
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

    # Sharpness multiplier for weighted-matrix mating.
    # Sampling weight = max(0, score - threshold) ^ (1 + MATING_SHARPNESS_K * selectivity)
    # With K=3: low selectivity → exponent ≈ 1 (nearly uniform above floor),
    # high selectivity → exponent ≈ 4 (strongly peaked at best available mate).
    MATING_SHARPNESS_K: float = 3.0

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

    # ------------------------------------------------------------------
    # Mating helpers
    # ------------------------------------------------------------------

    def _mate_zip(
        self,
        viable_males: list,
        viable_females: list,
    ) -> list[dict]:
        """
        Legacy zip-pairing strategy.

        Shuffles all viable males and females habitat-wide, then zips them
        into 1:1 pairs.  Any cross-species pairing that fails the compatibility
        check wastes both individuals' mating opportunity for that week, which
        creates a severe minority-species penalty (Allee effect).
        """
        np.random.shuffle(viable_males)
        np.random.shuffle(viable_females)
        mating_events: list[dict] = []
        for male, female in zip(viable_males, viable_females):
            event = self._attempt_mating(male, female)
            mating_events.append(event)
        return mating_events

    def _mate_species_priority(
        self,
        viable_males: list,
        viable_females: list,
    ) -> list[dict]:
        """
        Species-first priority pairing with cross-species spillover.

        1. Group males and females by species.
        2. Pair each species' own males and females first (shuffled within-species).
        3. Collect all unpaired individuals into a shared spillover pool.
        4. Zip-pair the spillover pool for cross-species hybridisation.

        Eliminates the minority-species Allee effect: each species gets mating
        proportional to its own sex ratio, regardless of relative abundance.
        Hybridisation is still possible for surplus/unmatched individuals.
        """
        from collections import defaultdict

        males_by_species: dict[str, list] = defaultdict(list)
        females_by_species: dict[str, list] = defaultdict(list)
        for m in viable_males:
            males_by_species[m.species].append(m)
        for f in viable_females:
            females_by_species[f.species].append(f)

        mating_events: list[dict] = []
        spillover_males: list = []
        spillover_females: list = []

        all_species = set(males_by_species) | set(females_by_species)
        species_order = list(all_species)
        np.random.shuffle(species_order)

        for sp in species_order:
            sp_males = males_by_species.get(sp, [])
            sp_females = females_by_species.get(sp, [])
            np.random.shuffle(sp_males)
            np.random.shuffle(sp_females)
            for male, female in zip(sp_males, sp_females):
                event = self._attempt_mating(male, female)
                mating_events.append(event)
            # Surplus individuals go to spillover
            n_paired = min(len(sp_males), len(sp_females))
            spillover_males.extend(sp_males[n_paired:])
            spillover_females.extend(sp_females[n_paired:])

        # Cross-species hybridisation pass on leftovers
        np.random.shuffle(spillover_males)
        np.random.shuffle(spillover_females)
        for male, female in zip(spillover_males, spillover_females):
            event = self._attempt_mating(male, female)
            mating_events.append(event)

        return mating_events

    @staticmethod
    def _build_compatibility_matrix(males: list, females: list) -> np.ndarray:
        """
        Vectorized pairwise cosine-similarity matrix on the compatibility_genes subset.

        Returns an (M, F) float64 array with values in [-1, 1], where entry [i, j]
        is the compatibility score between males[i] and females[j].  Shared by
        Shared by weighted-sampling and stable-matching mating strategies.
        """
        from .creature import DEFAULT_TRAIT_GENE_INDICES
        indices = DEFAULT_TRAIT_GENE_INDICES["compatibility_genes"]

        male_mat = np.stack([m.genes[indices] for m in males])     # (M, 245)
        female_mat = np.stack([f.genes[indices] for f in females]) # (F, 245)

        dots = male_mat @ female_mat.T  # (M, F)

        male_norms = np.linalg.norm(male_mat, axis=1, keepdims=True)    # (M, 1)
        female_norms = np.linalg.norm(female_mat, axis=1, keepdims=True) # (F, 1)
        denom = male_norms @ female_norms.T  # (M, F)

        safe = denom > 1e-10
        scores = np.where(safe, dots / np.where(safe, denom, 1.0), 0.0)
        return np.clip(scores, -1.0, 1.0)

    def _mate_weighted_matrix(
        self,
        viable_males: list,
        viable_females: list,
    ) -> list[dict]:
        """
        Full pairwise score matrix with selectivity-weighted sampling.

        1. Build the M×F compatibility score matrix in one vectorized pass.
        2. Iterate females in random order; each female samples a male using:
             weight_j = max(0, score[j,f] − threshold_f) ^ sharpness
           where threshold_f = COMPATIBILITY_FLOOR + 0.15 * female.selectivity
           and sharpness = 1 + MATING_SHARPNESS_K * mean(male.selectivity, female.selectivity)
        3. If no available male clears the female's threshold she goes unmated.
        4. The sampled male is removed from the pool (no double-mating).

        Low selectivity → exponent ≈ 1 → nearly uniform above the floor → liberal
        hybridisation.  High selectivity → exponent ≈ 4 → sharply peaked at the
        highest-scoring available male → near-exclusive same-species mating.
        """
        from .creature import Creature as _Creature

        if not viable_males or not viable_females:
            return []

        score_matrix = self._build_compatibility_matrix(viable_males, viable_females)
        # score_matrix[i, j] = compatibility between males[i] and females[j]

        female_order = list(range(len(viable_females)))
        np.random.shuffle(female_order)

        available_males = list(range(len(viable_males)))
        male_selectivities = np.array([m.selectivity for m in viable_males])
        mating_events: list[dict] = []

        for f_idx in female_order:
            if not available_males:
                break
            female = viable_females[f_idx]

            threshold_f = _Creature.COMPATIBILITY_FLOOR + 0.15 * female.selectivity
            scores_f = score_matrix[available_males, f_idx]
            above = np.maximum(0.0, scores_f - threshold_f)

            if above.max() == 0.0:
                continue

            # Per-pair sharpness: average selectivity of the specific male + this female
            sel_m = male_selectivities[available_males]
            sharpness = 1.0 + self.MATING_SHARPNESS_K * (sel_m + female.selectivity) / 2.0
            weights = above ** sharpness

            total = weights.sum()
            if total == 0.0:
                continue
            probs = weights / total

            chosen_pool_idx = int(np.random.choice(len(available_males), p=probs))
            chosen_male_idx = available_males[chosen_pool_idx]
            male = viable_males[chosen_male_idx]

            event = self._attempt_mating(male, female)
            mating_events.append(event)
            available_males.pop(chosen_pool_idx)

        return mating_events

    def _mate_stable_matching(
        self,
        viable_males: list,
        viable_females: list,
    ) -> list[dict]:
        """
        Mutual-preference stable matching via Gale-Shapley deferred acceptance.

        Builds the full M×F compatibility score matrix, computes each creature's
        personal acceptance threshold from its selectivity trait, runs the
        module-level _gale_shapley() function to find a stable set of pairs, then
        calls _attempt_mating() on each pair.

        See _gale_shapley() for the full algorithm description, assumptions, and
        documented baked-in choices (male-proposing direction, bilateral threshold
        pre-filtering, tie-breaking noise, threshold asymmetry vs is_compatible).

        Unlike weighted_matrix, every matched pair is the best stable outcome for
        the male — no male would prefer an unmatched female who also prefers him.
        Hybridisation occurs only when a cross-species individual genuinely ranks
        above all available same-species candidates for both parties.
        """
        from .creature import Creature as _Creature

        if not viable_males or not viable_females:
            return []

        score_matrix = self._build_compatibility_matrix(viable_males, viable_females)

        # Per-creature thresholds mirror the formula in is_compatible(), but use
        # each creature's OWN selectivity rather than the pair average.  See
        # _gale_shapley assumption 5 for the implications of this asymmetry.
        male_thresholds = np.array([
            _Creature.COMPATIBILITY_FLOOR + 0.15 * m.selectivity
            for m in viable_males
        ])
        female_thresholds = np.array([
            _Creature.COMPATIBILITY_FLOOR + 0.15 * f.selectivity
            for f in viable_females
        ])

        pairs = _gale_shapley(score_matrix, male_thresholds, female_thresholds)

        mating_events: list[dict] = []
        for male_idx, female_idx in pairs:
            event = self._attempt_mating(viable_males[male_idx], viable_females[female_idx])
            mating_events.append(event)

        return mating_events

    @staticmethod
    def _attempt_mating(male, female) -> dict:
        """Run one compatibility check + optional reproduce(); return the event dict."""
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
            litter = male.reproduce(female)
            if litter:
                event["fertilized"] = True
                event["litter_size"] = len(litter)
                event["offspring_ids"] = [c.creature_id for c in litter]
                if male.species != female.species:
                    event["hybridization"] = {
                        "male_species": male.species,
                        "female_species": female.species,
                    }
            else:
                event["reason"] = "infertile"
        else:
            event["reason"] = reason
        return event

    def simulate_week(
        self,
        species_registry=None,
        isolation_probability: float = 0.001,
        mating_strategy: str = "zip",
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

        if mating_strategy == "species_priority":
            mating_events = self._mate_species_priority(viable_males, viable_females)
        elif mating_strategy == "weighted_matrix":
            mating_events = self._mate_weighted_matrix(viable_males, viable_females)
        elif mating_strategy == "stable_matching":
            mating_events = self._mate_stable_matching(viable_males, viable_females)
        else:
            # "zip" is the default / fallback for unknown strategy strings
            mating_events = self._mate_zip(viable_males, viable_females)

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

        # Batch-compute all scaled trait values and generations for the full
        # alive population at once, then slice per-species with index arrays.
        trait_matrix = _batch_compute_traits(alive)  # (N, len(LOGGED_TRAITS))
        generations = np.fromiter(
            (c.generation for c in alive), dtype=np.float64, count=len(alive)
        )

        species_indices: dict[str, list[int]] = {}
        for i, c in enumerate(alive):
            species_indices.setdefault(c.species, []).append(i)

        stats: dict[str, dict] = {}
        for sp_name, idx in species_indices.items():
            idx_arr = np.array(idx, dtype=np.intp)
            n = len(idx_arr)
            sp_means = trait_matrix[idx_arr].mean(axis=0)  # (len(LOGGED_TRAITS),)
            stats[sp_name] = {
                "count": n,
                "mean_food_prob":  round(float(food_probs[idx_arr].mean()),  4),
                "mean_water_prob": round(float(water_probs[idx_arr].mean()), 4),
                "mean_generation": round(float(generations[idx_arr].mean()), 2),
                "mean_traits": {
                    t: round(float(sp_means[j]), 4)
                    for j, t in enumerate(LOGGED_TRAITS)
                },
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
