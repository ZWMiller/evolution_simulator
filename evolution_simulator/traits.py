"""
Trait catalogs and bulk (population-level) trait measurement.

This module owns *what traits exist and how to measure them across many
creatures*, kept separate from the ``Creature`` class (which only computes a
single trait from its own genes via ``Creature._compute_trait``):

  - ``LOGGED_TRAITS`` / ``_TRAIT_SCALING`` / ``_TRAIT_INTERNAL_KEY`` and
    ``_batch_compute_traits`` — the scaled biological-range traits recorded in
    per-habitat / per-species statistics, computed in one vectorized pass.
  - ``PHENOTYPE_TRAITS`` / ``compute_phenotype_matrix`` / ``compute_phenotype``
    — the raw [0, 1] phenotype axis used for anagenesis / morphological-
    divergence detection.

Layering: this module imports the gene loci and the OWA kernel from
``genetics`` only — it does *not* import ``Creature`` (the OWA decay for a batch
is read from the creatures' own class at call time in ``_batch_compute_traits``).
The DAG is ``genetics <- traits`` and ``genetics <- creature``, with no edge
between ``creature`` and ``traits``.
"""

import numpy as np

from .genetics import DEFAULT_OWA_ALPHA, DEFAULT_TRAIT_GENE_INDICES, owa_aggregate

# ---------------------------------------------------------------------------
# Traits logged in per-habitat and per-species daily statistics
# ---------------------------------------------------------------------------
# These are property names on Creature.  Logged values use the actual scaled
# ranges (fecundity 1-8, metabolism 0.5-2.0, etc.) for interpretability.
LOGGED_TRAITS: tuple[str, ...] = (
    # Reproduction
    "fecundity",
    "reproduction_time",
    "weeks_to_sexual_viability",
    "parental_investment",
    "reproduction_likelihood",
    # Survival / physiology
    "metabolism",
    "water_efficiency",
    "max_lifespan",
    "disease_resistance",
    "immune_response",
    "stress_tolerance",
    # Environmental adaptation
    "heat_tolerance",
    "cold_tolerance",
    "drought_tolerance",
    "hibernation_tendency",
    # Movement / behaviour
    "migration_likelihood",
    "risk_tolerance",
    "aggression",
    "territorial",
    "social_tendency",
    "nocturnal_tendency",
    # Physical
    "size",
    "strength",
    "speed",
    "camouflage",
    # Cognitive / ecological
    "foraging_ability",
    "intelligence",
    "adaptability",
    "pack_hunting",
    "scavenging_tendency",
    "communication",
    # Genetics
    "mutation_rate",
    "selectivity",
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
    "fecundity": (1.0, 7.0, False),
    "reproduction_time": (1.0, 19.0, True),
    "weeks_to_sexual_viability": (4.0, 46.0, True),
    "parental_investment": (0.0, 1.0, False),
    "reproduction_likelihood": (0.0, 1.0, False),
    "metabolism": (0.5, 1.5, False),
    "water_efficiency": (0.0, 1.0, False),
    "max_lifespan": (40.0, 360.0, True),
    "disease_resistance": (0.0, 1.0, False),
    "immune_response": (0.0, 1.0, False),
    "stress_tolerance": (0.0, 1.0, False),
    "heat_tolerance": (0.0, 1.0, False),
    "cold_tolerance": (0.0, 1.0, False),
    "drought_tolerance": (0.0, 1.0, False),
    "hibernation_tendency": (0.0, 1.0, False),
    "migration_likelihood": (0.0, 1.0, False),
    "risk_tolerance": (0.0, 1.0, False),
    "aggression": (0.0, 1.0, False),
    "territorial": (0.0, 1.0, False),
    "social_tendency": (0.0, 1.0, False),
    "nocturnal_tendency": (0.0, 1.0, False),
    "size": (0.0, 1.0, False),
    "strength": (0.0, 1.0, False),
    "speed": (0.0, 1.0, False),
    "camouflage": (0.0, 1.0, False),
    "foraging_ability": (0.0, 1.0, False),
    "intelligence": (0.0, 1.0, False),
    "adaptability": (0.0, 1.0, False),
    "pack_hunting": (0.0, 1.0, False),
    "scavenging_tendency": (0.0, 1.0, False),
    "communication": (0.0, 1.0, False),
    "mutation_rate": (0.001, 0.049, False),
    "selectivity": (0.0, 1.0, False),
    "base_predation_rate": (0.0, 0.005, False),
}


# ---------------------------------------------------------------------------
# Phenotype vector (for anagenesis / morphological-divergence detection)
# ---------------------------------------------------------------------------
# The "phenotype" is the set of ecological/physical/physiological/cognitive
# traits a biologist would actually diagnose — deliberately EXCLUDING the
# genetic-machinery traits (compatibility_genes, sex_determination,
# mutation_rate, selectivity).  Anagenesis is measured as cosine drift of a
# species' living phenotype centroid away from its frozen type, so it captures
# a lineage transforming over time even while it remains interfertile
# (the compatibility subset — and thus reproductive isolation — need not move).
PHENOTYPE_TRAITS: tuple[str, ...] = (
    "fecundity",
    "reproduction_time",
    "weeks_to_sexual_viability",
    "parental_investment",
    "aggression",
    "migration_likelihood",
    "territorial",
    "social_tendency",
    "pack_hunting",
    "scavenging_tendency",
    "nocturnal_tendency",
    "risk_tolerance",
    "size",
    "strength",
    "speed",
    "camouflage",
    "metabolism",
    "foraging_ability",
    "water_efficiency",
    "max_lifespan",
    "disease_resistance",
    "immune_response",
    "stress_tolerance",
    "heat_tolerance",
    "cold_tolerance",
    "drought_tolerance",
    "hibernation_tendency",
    "intelligence",
    "adaptability",
    "communication",
    "reproduction_likelihood",
    "base_predation_rate",
)


def _batch_compute_traits(creatures: list) -> np.ndarray:
    """
    Vectorized OWA trait computation for all LOGGED_TRAITS across a list of creatures.

    For each trait: fancy-index the relevant loci into an (N, k) block, run the
    shared OWA kernel (sort descending, normalised decaying weights, sigmoid),
    then apply the trait's (offset, scale) and floor-truncate int-valued traits
    per creature before averaging, so mean(int(f(x))) is preserved rather than
    the incorrect int(mean(f(x))).

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
        sigmoid_vals = owa_aggregate(gene_matrix[:, indices], alpha)  # (N,) ∈ [0, 1]

        offset, scale, is_int = _TRAIT_SCALING[trait]
        scaled = offset + scale * sigmoid_vals
        if is_int:
            scaled = np.floor(scaled)
        result[:, j] = scaled

    return result


def compute_phenotype_matrix(
    gene_matrix: np.ndarray,
    owa_alpha: float = DEFAULT_OWA_ALPHA,
    trait_gene_indices: dict | None = None,
    traits: tuple[str, ...] = PHENOTYPE_TRAITS,
) -> np.ndarray:
    """
    Vectorized OWA phenotype values for a batch of genomes.

    Returns the RAW sigmoid value in [0, 1] for each trait (not scaled to
    biological ranges) so that cosine comparisons treat every trait on the same
    footing rather than letting wide-range traits like max_lifespan dominate.

    Parameters
    ----------
    gene_matrix : (N, GENE_DIMS) float array
    owa_alpha   : OWA decay rate (defaults to DEFAULT_OWA_ALPHA)
    trait_gene_indices : trait→loci map (defaults to DEFAULT_TRAIT_GENE_INDICES)
    traits      : ordered trait names to include (defaults to PHENOTYPE_TRAITS)

    Returns (N, len(traits)) float64 array of raw [0, 1] values.
    """
    if trait_gene_indices is None:
        trait_gene_indices = DEFAULT_TRAIT_GENE_INDICES
    gene_matrix = np.asarray(gene_matrix, dtype=np.float64)
    n = gene_matrix.shape[0]
    out = np.empty((n, len(traits)), dtype=np.float64)
    for j, trait in enumerate(traits):
        out[:, j] = owa_aggregate(gene_matrix[:, trait_gene_indices[trait]], owa_alpha)
    return out


def compute_phenotype(genes: np.ndarray, owa_alpha: float = DEFAULT_OWA_ALPHA) -> np.ndarray:
    """Phenotype vector (raw [0,1] per PHENOTYPE_TRAITS) for a single genome."""
    return compute_phenotype_matrix(genes[np.newaxis, :], owa_alpha=owa_alpha)[0]
