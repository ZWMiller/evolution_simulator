"""
Species tracking for the evolution simulator.

Each species is defined by a progenitor gene vector.  When a creature is born,
its full genome is compared (cosine similarity) to EVERY registered progenitor.
The creature is assigned to whichever existing species it is most similar to,
provided that similarity exceeds species_threshold.  Only if no existing
species is close enough is a new species declared.

Checking all progenitors (rather than just the parent's species) prevents
false speciation events in two important scenarios:
  1. Drift-back: a lineage briefly diverges, then converges back toward an
     ancestral species — should be re-absorbed rather than logged as a
     second new species.
  2. Convergent evolution: two independent lineages evolve toward the same
     genetic region — should be recognised as the same species.

Using the full 500-dimensional gene vector (vs. the 245-dim mating subset)
makes species detection more sensitive: small directional shifts in many
loci accumulate into detectable divergence before full reproductive isolation.
"""

import random
import tomllib
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .creature import Creature

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH: Path = Path(__file__).parent / "config" / "species_names.toml"


def load_name_config(path: Path = DEFAULT_CONFIG_PATH) -> tuple[list[str], list[str]]:
    """
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
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    adjectives: list[str] = data["adjectives"]
    nouns: list[str] = data["nouns"]

    if len(adjectives) != len(set(adjectives)):
        raise ValueError(f"Duplicate adjectives found in {path}")
    if len(nouns) != len(set(nouns)):
        raise ValueError(f"Duplicate nouns found in {path}")
    if not adjectives or not nouns:
        raise ValueError(f"adjectives and nouns must each be non-empty in {path}")

    return adjectives, nouns


# Module-level lists loaded from the default config.
# Tests and external code can import these directly; the SpeciesRegistry
# also exposes its active lists via .adjectives / .nouns properties.
ADJECTIVES, NOUNS = load_name_config()


# ---------------------------------------------------------------------------
# SpeciesRegistry
# ---------------------------------------------------------------------------

class SpeciesRegistry:
    """
    Tracks species and detects speciation events.

    Usage
    -----
    1. Register founding individuals before the simulation begins:

           registry = SpeciesRegistry()
           name = registry.register_founding_species(founder.genes)
           founder.species = name

    2. At each birth, assign_species() is called automatically by
       Habitat.simulate_day() when a registry is passed:

           result = habitat.simulate_day(species_registry=registry)

       Or call manually:

           registry.assign_species(newborn)

    3. Inspect the history:

           registry.speciation_events  →  list[dict]
           registry.all_species        →  list[str]

    Species detection algorithm
    ---------------------------
    For each newborn, compute full-genome cosine similarity (all 500 dims)
    against EVERY registered progenitor in one vectorised numpy operation.
    The creature is assigned to whichever existing species scores the
    highest, provided that score ≥ species_threshold.

    If NO existing species exceeds the threshold, a speciation event is
    declared: the newborn becomes the progenitor of a new species with a
    randomly drawn name.

    Checking ALL progenitors prevents:
    - False new-species events when a lineage drifts back toward an
      ancestral genetic region (convergence / drift-back).
    - Duplicate species for independently converging lineages.

    Parameters
    ----------
    species_threshold : float
        Full-genome cosine similarity required to be considered the same
        species.  Default 0.95.  Raise to detect speciation earlier;
        lower to allow more genetic variation within a species.
    """

    DEFAULT_SPECIES_THRESHOLD: float = 0.95

    def __init__(
        self,
        species_threshold: float = DEFAULT_SPECIES_THRESHOLD,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ):
        self.species_threshold: float = species_threshold
        self._adjectives, self._nouns = load_name_config(config_path)
        # name → progenitor gene vector (500-dim)
        self._registry: dict[str, np.ndarray] = {}
        # Stacked progenitor matrix and name list kept in sync for fast lookup
        self._progenitor_matrix: Optional[np.ndarray] = None  # (N, 500)
        self._species_order: list[str] = []  # same order as matrix rows
        # Chronological log of every speciation event
        self.speciation_events: list[dict] = []
        # Track used name combinations to avoid duplicates
        self._used_names: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_founding_species(
        self,
        genes: np.ndarray,
        name: Optional[str] = None,
    ) -> str:
        """
        Register a founding species.

        Parameters
        ----------
        genes : np.ndarray (500-dim)
        name  : str, optional.  Auto-generated if not provided.

        Returns the species name; set ``creature.species`` to this on founders.
        """
        if name is None:
            name = self._unique_name()
        elif name in self._registry:
            raise ValueError(f"Species '{name}' is already registered.")
        self._add_to_registry(name, genes)
        return name

    def assign_species(self, creature: "Creature") -> str:
        """
        Validate and assign the species for a newborn creature.

        Compares the creature's full genome against ALL registered progenitors
        in a single vectorised operation and assigns it to the closest match
        (if ≥ species_threshold).  Falls back to declaring a new species only
        when no existing progenitor is close enough.

        Sets ``creature.species`` as a side effect and returns the name.
        """
        if not self._registry:
            # No species registered yet — treat as a new species
            parent_species = creature.species
            name = self._register_new_species(creature, parent_species)
            creature.species = name
            return name

        best_name, best_score = self._closest_species(creature.genes)

        if best_score >= self.species_threshold:
            # Belongs to an existing species (may differ from inherited label)
            creature.species = best_name
            return best_name

        # No existing species is close enough → speciation event
        parent_species = creature.species  # label inherited at birth
        name = self._register_new_species(creature, parent_species)
        creature.species = name
        return name

    def similarity_to_all_progenitors(self, creature: "Creature") -> dict[str, float]:
        """
        Return full-genome cosine similarity between *creature* and every
        registered species' progenitor.  Useful for inspection and logging.
        """
        if not self._registry:
            return {}
        sims = self._batch_similarity(creature.genes)
        return dict(zip(self._species_order, sims.tolist()))

    def progenitor_genes(self, species_name: str) -> Optional[np.ndarray]:
        """Return a copy of the progenitor gene vector for the named species."""
        genes = self._registry.get(species_name)
        return genes.copy() if genes is not None else None

    @property
    def species_count(self) -> int:
        """Number of distinct species currently registered."""
        return len(self._registry)

    @property
    def all_species(self) -> list[str]:
        """Names of all currently registered species."""
        return list(self._registry.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_to_registry(self, name: str, genes: np.ndarray) -> None:
        """Add a new entry and keep the fast-lookup matrix in sync."""
        self._registry[name] = genes.copy()
        self._species_order.append(name)
        self._used_names.add(name)
        # Append row to stacked matrix
        row = genes[np.newaxis, :].copy()  # (1, 500)
        if self._progenitor_matrix is None:
            self._progenitor_matrix = row
        else:
            self._progenitor_matrix = np.vstack([self._progenitor_matrix, row])

    def _batch_similarity(self, genes: np.ndarray) -> np.ndarray:
        """
        Vectorised cosine similarity between *genes* and every progenitor.

        Returns shape (N_species,).
        """
        mat = self._progenitor_matrix          # (N, 500)
        dots = mat @ genes                     # (N,)
        prog_norms = np.linalg.norm(mat, axis=1)   # (N,)
        creature_norm = float(np.linalg.norm(genes))
        denom = prog_norms * creature_norm
        safe = denom > 1e-10
        sims = np.where(safe, dots / np.where(safe, denom, 1.0), 0.0)
        return np.clip(sims, -1.0, 1.0)

    def _closest_species(self, genes: np.ndarray) -> tuple[str, float]:
        """Return (species_name, cosine_similarity) for the nearest progenitor."""
        sims = self._batch_similarity(genes)
        best_idx = int(np.argmax(sims))
        return self._species_order[best_idx], float(sims[best_idx])

    def _register_new_species(
        self, creature: "Creature", parent_species: Optional[str]
    ) -> str:
        name = self._unique_name()
        self._add_to_registry(name, creature.genes)
        self.speciation_events.append({
            "new_species": name,
            "parent_species": parent_species,
            "creature_id": creature.creature_id,
        })
        return name

    def _unique_name(self) -> str:
        """Draw a random adjective + noun pair not yet used."""
        max_attempts = len(self._adjectives) * len(self._nouns)
        for _ in range(max_attempts):
            name = f"{random.choice(self._adjectives)} {random.choice(self._nouns)}"
            if name not in self._used_names:
                self._used_names.add(name)
                return name
        raise RuntimeError(
            f"Exhausted all {len(self._adjectives) * len(self._nouns)} unique species "
            "name combinations."
        )

    def __repr__(self) -> str:
        return (
            f"SpeciesRegistry({self.species_count} species, "
            f"{len(self.speciation_events)} speciation events)"
        )
