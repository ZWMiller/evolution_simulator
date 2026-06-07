"""
Genome layout: the single source of truth for how the 500-dimensional gene
vector is partitioned into trait / resource / compatibility loci.

Everything that needs a gene-index set imports it from here:
  - ``GENE_DIMS``                  total dimensionality of every gene vector
  - ``DEFAULT_TRAIT_GENE_INDICES`` trait name -> contributing loci (polygenic)
  - ``DEFAULT_FOOD_GENE_INDICES``  loci used for food-finding cosine geometry
  - ``DEFAULT_WATER_GENE_INDICES`` loci used for water-finding cosine geometry

It also owns the genotype -> phenotype map itself — the Ordered Weighted
Averaging (OWA) kernel and its default decay constant:
  - ``DEFAULT_OWA_ALPHA``          OWA decay rate (subclasses override via the
                                   ``Creature.OWA_ALPHA`` class attribute)
  - ``owa_aggregate``             vectorized OWA over an (N, k) block of loci
  - ``owa_value``                 scalar OWA for a single creature's loci

``Creature.TRAIT_GENE_INDICES`` and ``Habitat.FOOD_GENE_INDICES`` /
``WATER_GENE_INDICES`` default to the module-level index constants but remain
class attributes so species/habitat subclasses can override them; likewise
``Creature.OWA_ALPHA`` defaults to ``DEFAULT_OWA_ALPHA``.
"""

import numpy as np

# Total dimensionality of the genetic vector.  The habitat environment vector
# shares this dimensionality (it is compared against gene sub-vectors), so
# habitat.py aliases ``HABITAT_VECTOR_DIMS = GENE_DIMS`` rather than redefining.
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

# Default OWA decay rate.  ``Creature.OWA_ALPHA`` defaults to this but is an
# overridable class attribute, so species subclasses can tune the decay.
DEFAULT_OWA_ALPHA: float = 0.6


def owa_aggregate(vals: np.ndarray, alpha: float) -> np.ndarray:
    """
    Vectorised Ordered Weighted Averaging over a block of gene values.

    For each row of *vals*: sort the k gene values descending, combine them
    with normalised exponentially-decaying weights
    ``w_i = alpha * (1 - alpha)^i``, dot-product to get a raw signal, then
    pass through sigmoid to produce a value in [0, 1].

    This is the single OWA kernel shared by ``Creature._compute_trait``
    (single genome), ``traits._batch_compute_traits``, and
    ``traits.compute_phenotype_matrix`` (batches), so the
    genotype → phenotype map is defined in exactly one place.

    Parameters
    ----------
    vals : np.ndarray, shape (N, k)
        Gene values at one trait's loci for N creatures.
    alpha : float
        OWA decay rate in (0, 1).  Higher values concentrate more weight on
        the top-ranked locus (faster decay).  The highest locus receives
        weight ≈ alpha, the next ≈ alpha*(1-alpha), etc., normalised to
        sum to 1.

    Returns
    -------
    np.ndarray, shape (N,)
        Sigmoid-normalised trait values in [0, 1] for each creature.
    """
    vals = np.asarray(vals, dtype=np.float64)
    k = vals.shape[1]
    sorted_vals = np.sort(vals, axis=1)[:, ::-1]  # descending per row
    i = np.arange(k, dtype=np.float64)
    weights = alpha * (1.0 - alpha) ** i
    weights /= weights.sum()
    raw = sorted_vals @ weights  # (N,)
    return 1.0 / (1.0 + np.exp(-raw))


def owa_value(vals: np.ndarray, alpha: float) -> float:
    """
    Scalar OWA for a single genome's loci.

    Convenience wrapper around ``owa_aggregate`` for the common single-creature
    case.  See ``owa_aggregate`` for the full algorithm description.

    Parameters
    ----------
    vals : np.ndarray, shape (k,)
        Gene values at one trait's loci for a single creature.
    alpha : float
        OWA decay rate; see ``owa_aggregate``.

    Returns
    -------
    float
        Sigmoid-normalised trait value in [0, 1].
    """
    return float(owa_aggregate(np.asarray(vals)[np.newaxis, :], alpha)[0])


DEFAULT_TRAIT_GENE_INDICES: dict[str, list[int]] = {
    # --- Reproduction ---
    # fecundity: driven by broad reproductive-system loci + general vigor
    "fecundity": [
        0,
        9,
        23,
        24,
        25,
        46,
        125,
        200,
        201,
        202,
        203,
        204,
        210,
        300,
        301,
        310,
        315,
        420,
    ],
    # gestation / egg development time
    "reproduction_time": [
        1,
        15,
        26,
        47,
        130,
        205,
        206,
        211,
        302,
        316,
        421,
        450,
        451,
    ],
    # age at which reproduction becomes possible
    "weeks_to_sexual_viability": [
        2,
        16,
        27,
        48,
        131,
        207,
        212,
        303,
        317,
        422,
        452,
        453,
        460,
    ],
    # investment of energy / time into offspring after birth
    "parental_investment": [
        3,
        17,
        28,
        49,
        132,
        208,
        213,
        304,
        318,
        423,
        461,
        462,
    ],
    # --- Behavior ---
    # aggression shares loci with size and strength (bigger → more aggressive)
    "aggression": [
        4,
        10,
        18,
        32,
        33,
        50,
        51,
        100,
        101,
        150,
        250,
        251,
        305,
        400,
        401,
        463,
    ],
    # migration likelihood shares loci with speed and risk tolerance
    "migration_likelihood": [
        5,
        11,
        19,
        34,
        52,
        58,
        102,
        151,
        252,
        253,
        306,
        319,
        402,
        464,
        465,
    ],
    # territorial tendency overlaps with aggression and size
    "territorial": [
        6,
        12,
        20,
        32,
        50,
        53,
        103,
        152,
        254,
        307,
        320,
        403,
        466,
    ],
    # social tendency is negatively correlated with territorial in practice
    "social_tendency": [
        7,
        13,
        21,
        54,
        76,
        104,
        153,
        255,
        256,
        308,
        321,
        404,
        467,
        468,
    ],
    # pack hunting overlaps with social tendency and communication
    "pack_hunting": [
        8,
        14,
        22,
        55,
        76,
        105,
        154,
        257,
        309,
        322,
        405,
        469,
    ],
    # scavenging vs. active hunting
    "scavenging_tendency": [
        29,
        37,
        56,
        106,
        155,
        156,
        258,
        311,
        323,
        406,
        470,
        471,
    ],
    # nocturnal activity pattern
    "nocturnal_tendency": [
        30,
        57,
        107,
        157,
        259,
        260,
        312,
        324,
        407,
        472,
        473,
    ],
    # risk tolerance overlaps with aggression and intelligence
    "risk_tolerance": [
        31,
        51,
        58,
        74,
        108,
        158,
        261,
        313,
        325,
        408,
        474,
        475,
        476,
    ],
    # --- Physical traits ---
    # size has broad effects: overlaps metabolism, strength, aggression
    "size": [
        32,
        59,
        109,
        159,
        160,
        262,
        314,
        326,
        409,
        477,
        478,
    ],
    # strength shares loci with size
    "strength": [
        32,
        33,
        60,
        110,
        161,
        263,
        264,
        327,
        410,
        479,
        480,
    ],
    # speed overlaps with size (inversely) and metabolism
    "speed": [
        34,
        61,
        111,
        162,
        265,
        328,
        411,
        481,
        482,
        483,
    ],
    # camouflage is largely independent
    "camouflage": [
        35,
        62,
        112,
        163,
        266,
        329,
        412,
        484,
        485,
    ],
    # --- Physiology / survival ---
    # metabolism overlaps with size and speed
    "metabolism": [
        32,
        34,
        36,
        63,
        113,
        164,
        165,
        267,
        330,
        413,
        486,
        487,
    ],
    # foraging ability overlaps with intelligence and sensory acuity
    "foraging_ability": [
        37,
        64,
        74,
        114,
        166,
        268,
        331,
        414,
        488,
        489,
    ],
    "water_efficiency": [
        38,
        65,
        115,
        167,
        269,
        332,
        415,
        490,
        491,
    ],
    # lifespan overlaps with disease resistance and stress tolerance
    "max_lifespan": [
        39,
        40,
        42,
        66,
        116,
        168,
        270,
        271,
        333,
        416,
        492,
        493,
    ],
    # disease resistance overlaps with immune response
    "disease_resistance": [
        40,
        41,
        67,
        117,
        169,
        272,
        334,
        417,
        494,
        495,
    ],
    "immune_response": [
        40,
        41,
        68,
        118,
        170,
        273,
        335,
        418,
        496,
        497,
    ],
    # stress tolerance overlaps with heat/cold tolerance
    "stress_tolerance": [
        42,
        43,
        44,
        69,
        119,
        171,
        274,
        336,
        419,
        498,
        499,
    ],
    # --- Environmental adaptation ---
    # heat tolerance overlaps with stress tolerance and drought tolerance
    "heat_tolerance": [
        43,
        45,
        70,
        120,
        172,
        275,
        276,
        337,
        424,
        440,
        441,
    ],
    "cold_tolerance": [
        44,
        71,
        121,
        173,
        277,
        338,
        425,
        442,
        443,
        444,
    ],
    # drought tolerance overlaps with water efficiency and heat tolerance
    "drought_tolerance": [
        38,
        43,
        45,
        72,
        122,
        174,
        278,
        339,
        426,
        445,
        446,
        447,
    ],
    # hibernation tendency overlaps with cold tolerance and metabolism
    "hibernation_tendency": [
        36,
        44,
        73,
        123,
        175,
        279,
        340,
        427,
        448,
        449,
        454,
    ],
    # --- Cognitive / social ---
    # intelligence overlaps with foraging, risk tolerance, adaptability
    "intelligence": [
        31,
        37,
        74,
        124,
        176,
        280,
        341,
        428,
        455,
        456,
        457,
    ],
    # adaptability overlaps with intelligence and stress tolerance
    "adaptability": [
        42,
        74,
        75,
        126,
        177,
        281,
        342,
        429,
        458,
        459,
    ],
    # communication overlaps with social tendency and intelligence
    "communication": [
        7,
        74,
        76,
        127,
        178,
        282,
        343,
        430,
        431,
        432,
    ],
    # reproduction_likelihood: probability that a compatible mating attempt
    # results in conception.  Overlaps with fecundity and health loci
    # (pleiotropy: fitness affects both mate appeal and fertility).
    "reproduction_likelihood": [
        0,
        3,
        39,
        80,
        81,
        82,
        140,
        141,
        142,
        220,
        221,
        222,
        290,
        291,
        392,
        393,
    ],
    # --- Reproduction / genetics ---
    # sex_determination: heritable; sigmoid > 0.5 → female.
    # Kept outside the compatibility_genes index ranges
    # (80-129, 140-189, 220-269, 285-339, 390-429) so that sex loci
    # do not influence mate-compatibility cosine similarity.
    "sex_determination": [
        0,
        3,
        77,
        133,
        195,
        283,
        344,
        350,
        351,
        352,
        433,
        434,
        435,
    ],
    # compatibility_genes: large spread used for mate-compatibility cosine similarity.
    # Models a major-histocompatibility-complex-like system; creatures from a shared
    # lineage cluster near 1.0 while divergent populations drift apart (speciation).
    "compatibility_genes": (
        list(range(80, 130))  # 50 loci
        + list(range(140, 190))  # 50 loci
        + list(range(220, 270))  # 50 loci
        + list(range(285, 340))  # 55 loci
        + list(range(390, 430))  # 40 loci
    ),  # 245 total loci
    # selectivity: raises the compatibility threshold above COMPATIBILITY_FLOOR.
    # Species with high selectivity are more genetically protective.
    "selectivity": [
        353,
        354,
        355,
        356,
        357,
        358,
        359,
        436,
        437,
        438,
        439,
    ],
    # mutation_rate: how often a gene is randomised at birth.
    # Under selection: low-mutation parents pass genes more faithfully.
    "mutation_rate": [
        360,
        361,
        362,
        363,
        364,
        365,
        366,
        367,
        368,
        369,
        370,
    ],
    # base_predation_rate: intrinsic daily predation vulnerability.
    # Shares loci with fecundity — encodes the r/K tradeoff: creatures
    # with high fecundity genes also tend to be more conspicuous/vulnerable
    # (think mouse vs. elephant).  Unique loci represent body size,
    # activity level, and general conspicuousness.
    "base_predation_rate": [
        # Shared with fecundity (r/K tradeoff)
        0,
        9,
        25,
        200,
        201,
        203,
        210,
        420,
        # Unique loci: intrinsic conspicuousness and vulnerability
        371,
        372,
        373,
        374,
        375,
        376,
        377,
        378,
        379,
        380,
        381,
        382,
    ],
}


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

DEFAULT_FOOD_GENE_INDICES: list[int] = (
    list(range(37, 80))  # foraging ability, water efficiency, intelligence loci
    + list(range(110, 170))  # size, strength, speed, physiology loci
    + list(range(230, 285))  # broad genomic coverage
)  # 158 total indices

DEFAULT_WATER_GENE_INDICES: list[int] = (
    list(range(38, 78))  # water efficiency, drought tolerance loci
    + list(range(115, 175))  # immune, stress, environmental adaptation loci
    + list(range(270, 345))  # broad genomic coverage
)  # 175 total indices
