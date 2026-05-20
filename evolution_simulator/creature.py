import uuid
import numpy as np
from typing import Optional

# Total dimensionality of the genetic vector
GENE_DIMS = 500

# ---------------------------------------------------------------------------
# Trait gene index sets
# ---------------------------------------------------------------------------
# Each trait is determined by a distributed subset of gene indices drawn from
# the full 500-dimensional vector.  This mirrors real polygenic inheritance:
# many loci spread across the genome each contribute a small effect, and the
# same locus can influence multiple traits (pleiotropy).
#
# These index sets are stored as a class attribute on Creature so that:
#   1. Subclasses representing distinct species can override them.
#   2. They are inspectable / evolvable at runtime if needed.
#
# Trait value computation — Ordered Weighted Averaging (OWA):
#   1. Extract genes at trait_indices.
#   2. Sort descending; assign exponentially decaying weights
#      w_i = alpha * (1 - alpha)^i, then normalise so they sum to 1.
#   3. raw = dot(weights, sorted_genes)
#   4. trait_value = sigmoid(raw)  → [0, 1]
#
# OWA gives higher-valued loci more phenotypic weight.  A beneficial mutation
# that pushes a locus to the top of the ranking gains weight proportional to
# OWA_ALPHA (default 0.6), making that mutation immediately visible to
# selection — unlike a plain mean where each locus contributes only 1/N of
# the signal regardless of its value.
# ---------------------------------------------------------------------------

DEFAULT_TRAIT_GENE_INDICES: dict[str, list[int]] = {
    # --- Reproduction ---
    # fecundity: driven by broad reproductive-system loci + general vigor
    "fecundity": [
        0, 9, 23, 24, 25, 46, 125, 200, 201, 202,
        203, 204, 210, 300, 301, 310, 315, 420,
    ],
    # gestation / egg development time
    "reproduction_time": [
        1, 15, 26, 47, 130, 205, 206, 211, 302,
        316, 421, 450, 451,
    ],
    # age at which reproduction becomes possible
    "weeks_to_sexual_viability": [
        2, 16, 27, 48, 131, 207, 212, 303, 317,
        422, 452, 453, 460,
    ],
    # investment of energy / time into offspring after birth
    "parental_investment": [
        3, 17, 28, 49, 132, 208, 213, 304, 318,
        423, 461, 462,
    ],

    # --- Behavior ---
    # aggression shares loci with size and strength (bigger → more aggressive)
    "aggression": [
        4, 10, 18, 32, 33, 50, 51, 100, 101,
        150, 250, 251, 305, 400, 401, 463,
    ],
    # migration likelihood shares loci with speed and risk tolerance
    "migration_likelihood": [
        5, 11, 19, 34, 52, 58, 102, 151, 252,
        253, 306, 319, 402, 464, 465,
    ],
    # territorial tendency overlaps with aggression and size
    "territorial": [
        6, 12, 20, 32, 50, 53, 103, 152, 254,
        307, 320, 403, 466,
    ],
    # social tendency is negatively correlated with territorial in practice
    "social_tendency": [
        7, 13, 21, 54, 76, 104, 153, 255, 256,
        308, 321, 404, 467, 468,
    ],
    # pack hunting overlaps with social tendency and communication
    "pack_hunting": [
        8, 14, 22, 55, 76, 105, 154, 257, 309,
        322, 405, 469,
    ],
    # scavenging vs. active hunting
    "scavenging_tendency": [
        29, 37, 56, 106, 155, 156, 258, 311,
        323, 406, 470, 471,
    ],
    # nocturnal activity pattern
    "nocturnal_tendency": [
        30, 57, 107, 157, 259, 260, 312, 324,
        407, 472, 473,
    ],
    # risk tolerance overlaps with aggression and intelligence
    "risk_tolerance": [
        31, 51, 58, 74, 108, 158, 261, 313,
        325, 408, 474, 475, 476,
    ],

    # --- Physical traits ---
    # size has broad effects: overlaps metabolism, strength, aggression
    "size": [
        32, 59, 109, 159, 160, 262, 314, 326,
        409, 477, 478,
    ],
    # strength shares loci with size
    "strength": [
        32, 33, 60, 110, 161, 263, 264, 327,
        410, 479, 480,
    ],
    # speed overlaps with size (inversely) and metabolism
    "speed": [
        34, 61, 111, 162, 265, 328, 411, 481,
        482, 483,
    ],
    # camouflage is largely independent
    "camouflage": [
        35, 62, 112, 163, 266, 329, 412, 484, 485,
    ],

    # --- Physiology / survival ---
    # metabolism overlaps with size and speed
    "metabolism": [
        32, 34, 36, 63, 113, 164, 165, 267,
        330, 413, 486, 487,
    ],
    # foraging ability overlaps with intelligence and sensory acuity
    "foraging_ability": [
        37, 64, 74, 114, 166, 268, 331, 414,
        488, 489,
    ],
    "water_efficiency": [
        38, 65, 115, 167, 269, 332, 415, 490, 491,
    ],
    # lifespan overlaps with disease resistance and stress tolerance
    "lifespan_factor": [
        39, 40, 42, 66, 116, 168, 270, 271,
        333, 416, 492, 493,
    ],
    # disease resistance overlaps with immune response
    "disease_resistance": [
        40, 41, 67, 117, 169, 272, 334, 417,
        494, 495,
    ],
    "immune_response": [
        40, 41, 68, 118, 170, 273, 335, 418,
        496, 497,
    ],
    # stress tolerance overlaps with heat/cold tolerance
    "stress_tolerance": [
        42, 43, 44, 69, 119, 171, 274, 336,
        419, 498, 499,
    ],

    # --- Environmental adaptation ---
    # heat tolerance overlaps with stress tolerance and drought tolerance
    "heat_tolerance": [
        43, 45, 70, 120, 172, 275, 276, 337,
        424, 440, 441,
    ],
    "cold_tolerance": [
        44, 71, 121, 173, 277, 338, 425, 442,
        443, 444,
    ],
    # drought tolerance overlaps with water efficiency and heat tolerance
    "drought_tolerance": [
        38, 43, 45, 72, 122, 174, 278, 339,
        426, 445, 446, 447,
    ],
    # hibernation tendency overlaps with cold tolerance and metabolism
    "hibernation_tendency": [
        36, 44, 73, 123, 175, 279, 340, 427,
        448, 449, 454,
    ],

    # --- Cognitive / social ---
    # intelligence overlaps with foraging, risk tolerance, adaptability
    "intelligence": [
        31, 37, 74, 124, 176, 280, 341, 428,
        455, 456, 457,
    ],
    # adaptability overlaps with intelligence and stress tolerance
    "adaptability": [
        42, 74, 75, 126, 177, 281, 342, 429,
        458, 459,
    ],
    # communication overlaps with social tendency and intelligence
    "communication": [
        7, 74, 76, 127, 178, 282, 343, 430,
        431, 432,
    ],

    # reproduction_likelihood: probability that a compatible mating attempt
    # results in conception.  Overlaps with fecundity and health loci
    # (pleiotropy: fitness affects both mate appeal and fertility).
    "reproduction_likelihood": [
        0, 3, 39, 80, 81, 82, 140, 141, 142,
        220, 221, 222, 290, 291, 392, 393,
    ],

    # --- Reproduction / genetics ---
    # sex_determination: heritable; sigmoid > 0.5 → female.
    # Kept outside the compatibility_genes index ranges
    # (80-129, 140-189, 220-269, 285-339, 390-429) so that sex loci
    # do not influence mate-compatibility cosine similarity.
    "sex_determination": [
        0, 3, 77, 133, 195, 283, 344, 350,
        351, 352, 433, 434, 435,
    ],
    # compatibility_genes: large spread used for mate-compatibility cosine similarity.
    # Models a major-histocompatibility-complex-like system; creatures from a shared
    # lineage cluster near 1.0 while divergent populations drift apart (speciation).
    "compatibility_genes": (
        list(range(80, 130))    # 50 loci
        + list(range(140, 190)) # 50 loci
        + list(range(220, 270)) # 50 loci
        + list(range(285, 340)) # 55 loci
        + list(range(390, 430)) # 40 loci
    ),  # 245 total loci
    # selectivity: raises the compatibility threshold above COMPATIBILITY_FLOOR.
    # Species with high selectivity are more genetically protective.
    "selectivity": [
        353, 354, 355, 356, 357, 358, 359,
        436, 437, 438, 439,
    ],
    # mutation_rate: how often a gene is randomised at birth.
    # Under selection: low-mutation parents pass genes more faithfully.
    "mutation_rate": [
        360, 361, 362, 363, 364, 365, 366,
        367, 368, 369, 370,
    ],
    # base_predation_rate: intrinsic daily predation vulnerability.
    # Shares loci with fecundity — encodes the r/K tradeoff: creatures
    # with high fecundity genes also tend to be more conspicuous/vulnerable
    # (think mouse vs. elephant).  Unique loci represent body size,
    # activity level, and general conspicuousness.
    "base_predation_rate": [
        # Shared with fecundity (r/K tradeoff)
        0, 9, 25, 200, 201, 203, 210, 420,
        # Unique loci: intrinsic conspicuousness and vulnerability
        371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382,
    ],
}


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
        Explicit ID.  Auto-generated UUID4 if not provided.
    """

    # Class-level attribute so species subclasses can override the mapping
    TRAIT_GENE_INDICES: dict[str, list[int]] = DEFAULT_TRAIT_GENE_INDICES

    # OWA weight-decay rate.  The highest-ranked gene locus receives weight
    # alpha, the next alpha*(1-alpha), and so on (normalised to sum to 1).
    # Higher values → faster decay → top locus dominates more strongly.
    # Range (0, 1); species subclasses can override to tune selectivity.
    OWA_ALPHA: float = 0.6

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
    COMPATIBILITY_FLOOR: float = 0.9

    def __init__(
        self,
        genes: Optional[np.ndarray] = None,
        parents: Optional[list] = None,
        creature_id: Optional[str] = None,
    ):
        if genes is not None:
            if genes.shape != (GENE_DIMS,):
                raise ValueError(
                    f"genes must be a 1-D array of length {GENE_DIMS}, got {genes.shape}"
                )
            self.genes: np.ndarray = genes.astype(float)
        else:
            self.genes = np.random.randn(GENE_DIMS)

        self.creature_id: str = creature_id if creature_id is not None else str(uuid.uuid4())
        self.parents: list["Creature"] = parents if parents is not None else []
        self.generation: int = (max(p.generation for p in self.parents) + 1) if self.parents else 0

        # Must be initialised before any property access (sex determination calls _compute_trait)
        self._trait_cache: dict[str, float] = {}

        # Vital state
        self.age: int = 0
        self.is_alive: bool = True
        self.cause_of_death: Optional[str] = None

        # Sex is fixed at birth from the gene vector (heritable)
        self.sex: str = "female" if self._compute_trait("sex_determination") >= 0.5 else "male"

        # Species is inherited from the first parent; "unknown" for founders.
        # The SpeciesRegistry validates this at birth and triggers a speciation
        # event if the genome has diverged too far from the parent species' progenitor.
        self.species: str = self.parents[0].species if self.parents else "unknown"

        # Reproductive state
        self.is_pregnant: bool = False
        self.weeks_pregnant: int = 0
        # Litter created at fertilisation; released into the population at term
        self._pending_offspring: list["Creature"] = []

        # Resource state (driven by habitat cross-product calculations)
        self.energy: float = 1.0      # 0 = starving, 1 = fully fed
        self.hydration: float = 1.0   # 0 = dehydrated, 1 = fully hydrated

    # ------------------------------------------------------------------
    # Core trait computation
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _compute_trait(self, name: str) -> float:
        """
        Compute the normalized [0, 1] value for a named trait via OWA.

        Gene values at contributing loci are sorted descending and combined
        with exponentially decaying weights (OWA_ALPHA controls the decay),
        then passed through a sigmoid.  Higher-valued loci receive more weight,
        so beneficial mutations are immediately visible to selection.

        Results are memoised in _trait_cache: since genes are immutable after
        birth, each trait value is computed at most once per creature lifetime.

        Subclasses can tune OWA_ALPHA or override TRAIT_GENE_INDICES[name].
        """
        cached = self._trait_cache.get(name)
        if cached is not None:
            return cached
        indices = self.TRAIT_GENE_INDICES[name]
        vals = self.genes[indices]
        n = len(vals)
        alpha = self.__class__.OWA_ALPHA
        i = np.arange(n)
        weights = alpha * (1 - alpha) ** i
        weights /= weights.sum()
        raw = float(np.dot(weights, np.sort(vals)[::-1]))
        result = self._sigmoid(raw)
        self._trait_cache[name] = result
        return result

    def trait_indices(self, name: str) -> list[int]:
        """Return the gene indices that contribute to the named trait."""
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
        """Weeks required to gestate / develop offspring (1 – 52)."""
        return int(1 + 51 * self._compute_trait("reproduction_time"))

    @property
    def weeks_to_sexual_viability(self) -> int:
        """Weeks before the creature can reproduce (4 – 100)."""
        return int(4 + 96 * self._compute_trait("weeks_to_sexual_viability"))

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
        return int(40 + 360 * self._compute_trait("lifespan_factor"))

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

        Returns a value in [-1, 1]:
          -1  completely anti-correlated gene vectors (maximally incompatible)
           0  orthogonal (unrelated, typical for random founding individuals)
          +1  identical gene vectors (perfect genetic match)

        Creatures from a shared lineage cluster near +1; populations that have
        diverged over many generations drift toward 0 or below, producing
        reproductive isolation (speciation).
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
        Full mating-compatibility check.

        Returns
        -------
        (compatible, score, reason)
            compatible : bool   – whether mating can proceed
            score      : float  – cosine similarity in [-1, 1]
            reason     : str    – empty string if compatible, else why not
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
        # Each partner contributes half of the selectivity pressure
        threshold = self.COMPATIBILITY_FLOOR + (1 - self.COMPATIBILITY_FLOOR) * (
            (self.selectivity + other.selectivity) / 2
        )
        if score < threshold:
            return False, score, f"genetic_incompatibility (score={score:.4f} < threshold={threshold:.4f})"
        return True, score, ""

    def reproduce(self, other: "Creature") -> list["Creature"]:
        """
        Attempt to produce a litter with *other*.

        Litter size is a Poisson draw centred on the female's fecundity trait
        (minimum 1 if the draw is zero), so each mating event is stochastic.
        Each offspring receives an independent per-locus Mendelian draw from
        the two parents, and each locus is mutated at the rate of whichever
        parent contributed it — keeping both mutation rates under independent
        selection pressure.

        Returns a (possibly empty) list of Creature objects.
          - Empty list  → compatibility check failed; no mating occurred.
          - Non-empty   → successful mating; the female is marked pregnant and
                          the litter is stored in female._pending_offspring
                          until gestation completes.
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
        if np.random.random() > fertility:
            return []

        # Litter size: Poisson draw on female fecundity, min 1
        n_offspring = max(1, int(np.random.poisson(female.fecundity)))

        litter: list["Creature"] = []
        for _ in range(n_offspring):
            # Independent per-locus parent selection for each sibling
            parent_choice = np.random.randint(0, 2, size=GENE_DIMS)
            child_genes = np.where(parent_choice == 0, self.genes, other.genes)

            # Mutation: each locus uses its chosen parent's rate
            mut_rates = np.where(parent_choice == 0, self.mutation_rate, other.mutation_rate)
            mutation_mask = np.random.random(GENE_DIMS) < mut_rates
            if mutation_mask.any():
                child_genes[mutation_mask] = np.random.randn(int(mutation_mask.sum()))

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

        Parameters
        ----------
        environment : Environment, optional
            The environment the creature currently inhabits.  Reserved for
            future integration: food and water availability will be computed
            via dot products between self.genes and environment.vector.

        Returns
        -------
        dict
            Week log with keys:
            - "creature_id"    : str
            - "age"            : int  (post-increment)
            - "is_alive"       : bool
            - "cause_of_death" : str or None
            - "events"         : list[str]
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
        return (
            f"Creature(id={self.creature_id[:8]}…, age={self.age}, {status}, "
            f"parents={len(self.parents)})"
        )
