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
       Habitat.simulate_week() when a registry is passed:

           result = habitat.simulate_week(species_registry=registry)

       Or call manually:

           registry.assign_species(newborn)

    3. After all habitats have run each week, call promote_candidates():

           all_alive = [c for hab in habitats.values() for c in hab.alive_creatures]
           registry.promote_candidates(all_alive, current_week)

    4. Inspect the history:

           registry.speciation_events  →  list[dict]
           registry.all_species        →  list[str]

    Species detection algorithm (two-stage)
    ----------------------------------------
    Detection: for each newborn whose genome falls below species_threshold for
    all confirmed progenitors, a candidate is created (or the creature joins the
    nearest existing candidate if similar enough).  The creature keeps its
    parent's species label during the candidate period.

    Promotion: each week, promote_candidates() checks every live candidate.
    A candidate is promoted to a confirmed species only when it has at least
    min_species_population living members AND has existed for at least
    min_species_weeks weeks.  On promotion all living members are renamed.
    Candidates whose members all die before promotion are silently evaporated.

    Checking ALL confirmed progenitors prevents:
    - False new-species events when a lineage drifts back toward an
      ancestral genetic region (convergence / drift-back).
    - Duplicate species for independently converging lineages.

    Parameters
    ----------
    species_threshold : float
        Full-genome cosine similarity required to be considered the same
        species.  Default 0.75.
    min_species_population : int
        Minimum number of living candidate members required for promotion.
    min_species_weeks : int
        Minimum number of weeks a candidate must exist before promotion.
    """

    DEFAULT_SPECIES_THRESHOLD: float = 0.75
    DEFAULT_MIN_SPECIES_POPULATION: int = 3
    DEFAULT_MIN_SPECIES_WEEKS: int = 5

    def __init__(
        self,
        species_threshold: float = DEFAULT_SPECIES_THRESHOLD,
        config_path: Path = DEFAULT_CONFIG_PATH,
        min_species_population: int = DEFAULT_MIN_SPECIES_POPULATION,
        min_species_weeks: int = DEFAULT_MIN_SPECIES_WEEKS,
    ):
        self.species_threshold: float = species_threshold
        self.min_species_population: int = min_species_population
        self.min_species_weeks: int = min_species_weeks
        self._adjectives, self._nouns = load_name_config(config_path)
        # name → progenitor gene vector (500-dim)
        self._registry: dict[str, np.ndarray] = {}
        # Stacked progenitor matrix and name list kept in sync for fast lookup
        self._progenitor_matrix: Optional[np.ndarray] = None  # (N, 500)
        self._species_order: list[str] = []  # same order as matrix rows
        # Chronological log of every confirmed speciation event
        self.speciation_events: list[dict] = []
        # Track used name combinations to avoid duplicates
        self._used_names: set[str] = set()
        # Two-stage speciation: candidates waiting for promotion.
        # cid → {genes, parent_species, detected_week, members: set[str],
        #         first_creature_id, peak_members: int}
        self._candidates: dict[str, dict] = {}
        self._next_candidate_id: int = 0
        # Set by SimulationRunner each week before simulate_week calls
        self.current_week: int = 0
        # Candidates that were declared but whose members all died before
        # promotion.  Each entry is a snapshot of the candidate at evaporation
        # time, for lineage visualisation and diagnostics.
        self.failed_speciation_attempts: list[dict] = []

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
        Assign or update the species for a newborn creature (two-stage).

        1. Bootstrap (empty registry): immediately registers a confirmed species.
        2. Confirmed match: creature joins the nearest confirmed species.
        3. Below threshold: creature joins the nearest candidate (if similar
           enough) or creates a new one.  In both cases the creature retains its
           inherited parent species label — no confirmed species is created yet.

        Sets ``creature.species`` as a side effect and returns the name.
        """
        if not self._registry:
            parent_species = creature.species
            name = self._register_new_species(creature, parent_species)
            creature.species = name
            return name

        best_name, best_score = self._closest_species(creature.genes)

        if best_score >= self.species_threshold:
            creature.species = best_name
            return best_name

        # No confirmed species is close enough — enter candidate stage.
        parent_species = creature.species  # inherited label; kept throughout candidate period
        cid, cand_score = self._closest_candidate(creature.genes)

        if cid is not None and cand_score >= self.species_threshold:
            cand = self._candidates[cid]
            cand["members"].add(creature.creature_id)
            cand["peak_members"] = max(cand["peak_members"], len(cand["members"]))
        else:
            new_cid = f"cand_{self._next_candidate_id}"
            self._next_candidate_id += 1
            self._candidates[new_cid] = {
                "genes": creature.genes.copy(),
                "parent_species": parent_species,
                "detected_week": self.current_week,
                "members": {creature.creature_id},
                "first_creature_id": creature.creature_id,
                "peak_members": 1,
            }

        # Creature keeps its parent species label during the candidate period.
        return creature.species

    def promote_candidates(
        self, alive_creatures: "list[Creature]", current_week: int
    ) -> list[dict]:
        """
        Promote any candidates that meet both population and age criteria.

        Call once per simulation step, after all habitats have run and
        migrations have been applied.

        Parameters
        ----------
        alive_creatures : list of all living Creature objects (all habitats)
        current_week    : the current simulation week number

        Returns the list of new speciation events created this call (also
        appended to self.speciation_events).
        """
        alive_ids = {c.creature_id for c in alive_creatures}
        creature_map = {c.creature_id: c for c in alive_creatures}

        promoted_events: list[dict] = []
        to_remove: list[str] = []

        for cid, cand in self._candidates.items():
            alive_members = cand["members"] & alive_ids
            if not alive_members:
                # All members died before promotion — record the failed attempt.
                self.failed_speciation_attempts.append({
                    "parent_species": cand["parent_species"],
                    "detected_week": cand["detected_week"],
                    "failed_week": current_week,
                    "peak_members": cand["peak_members"],
                    "first_creature_id": cand["first_creature_id"],
                })
                to_remove.append(cid)
                continue

            cand["members"] = alive_members
            weeks_elapsed = current_week - cand["detected_week"]

            if (
                len(alive_members) >= self.min_species_population
                and weeks_elapsed >= self.min_species_weeks
            ):
                new_name = self._unique_name()
                self._add_to_registry(new_name, cand["genes"])
                ev = {
                    "new_species": new_name,
                    "parent_species": cand["parent_species"],
                    "creature_id": cand["first_creature_id"],
                    "week": current_week,
                }
                self.speciation_events.append(ev)
                promoted_events.append(ev)
                for creature_id in alive_members:
                    if creature_id in creature_map:
                        creature_map[creature_id].species = new_name
                to_remove.append(cid)

        for cid in to_remove:
            del self._candidates[cid]

        return promoted_events

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
        """Return (species_name, cosine_similarity) for the nearest confirmed progenitor."""
        sims = self._batch_similarity(genes)
        best_idx = int(np.argmax(sims))
        return self._species_order[best_idx], float(sims[best_idx])

    def _closest_candidate(self, genes: np.ndarray) -> tuple[Optional[str], float]:
        """
        Return (candidate_id, cosine_similarity) for the nearest unconfirmed candidate.

        Iterates all pending candidates and computes cosine similarity between
        *genes* and each candidate's progenitor gene vector.  Returns the best
        match and its score.  If no candidates exist, returns (None, 0.0).

        The caller is responsible for checking whether the returned score meets
        species_threshold before deciding to join the candidate.
        """
        if not self._candidates:
            return None, 0.0
        best_cid: Optional[str] = None
        best_score = -2.0
        norm_g = float(np.linalg.norm(genes))
        for cid, cand in self._candidates.items():
            cg = cand["genes"]
            denom = float(np.linalg.norm(cg)) * norm_g
            sim = float(np.clip(np.dot(cg, genes) / denom, -1.0, 1.0)) if denom > 1e-10 else 0.0
            if sim > best_score:
                best_score = sim
                best_cid = cid
        return best_cid, best_score

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
            f"SpeciesRegistry({self.species_count} confirmed species, "
            f"{len(self._candidates)} candidates, "
            f"{len(self.speciation_events)} speciation events)"
        )
