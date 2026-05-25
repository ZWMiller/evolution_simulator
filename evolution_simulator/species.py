"""
Species tracking for the evolution simulator.

Speciation is detected on the **same signal that governs mating**: the
245-locus ``compatibility_genes`` subset (see Creature.compatibility_score).
A newborn's compatibility sub-vector is compared by cosine similarity against
the **living centroid** of every species that currently has living members.
The creature joins the nearest species if that similarity meets
``compatibility_threshold``; otherwise it enters a candidate stage that may be
promoted to a confirmed species.

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
    ("the name follows the type") and is reserved for the future *anagenesis*
    axis, which will measure how far a lineage has drifted from its own past.
  - **living centroid** (245-dim compatibility subset, refreshed periodically):
    the reproductive reference used for C2 detection here.
"""

import random
import tomllib
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .creature import DEFAULT_TRAIT_GENE_INDICES

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
    Tracks species and detects speciation events on the mating-compatibility signal.

    Usage
    -----
    1. Register founding individuals before the simulation begins:

           registry = SpeciesRegistry()
           name = registry.register_founding_species(founder.genes)
           founder.species = name

    2. At each birth, assign_species() is called automatically by
       Habitat.simulate_week() when a registry is passed:

           result = habitat.simulate_week(species_registry=registry)

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
    """

    DEFAULT_COMPATIBILITY_THRESHOLD: float = 0.65
    DEFAULT_MIN_SPECIES_POPULATION: int = 3
    DEFAULT_MIN_SPECIES_WEEKS: int = 5

    def __init__(
        self,
        compatibility_threshold: float = DEFAULT_COMPATIBILITY_THRESHOLD,
        config_path: Path = DEFAULT_CONFIG_PATH,
        min_species_population: int = DEFAULT_MIN_SPECIES_POPULATION,
        min_species_weeks: int = DEFAULT_MIN_SPECIES_WEEKS,
        compat_indices: Optional[list[int]] = None,
    ):
        self.compatibility_threshold: float = compatibility_threshold
        self.min_species_population: int = min_species_population
        self.min_species_weeks: int = min_species_weeks
        if compat_indices is None:
            compat_indices = DEFAULT_TRAIT_GENE_INDICES["compatibility_genes"]
        self._compat_indices: np.ndarray = np.asarray(compat_indices, dtype=int)

        self._adjectives, self._nouns = load_name_config(config_path)

        # name → frozen full-genome TYPE (500-dim).  Anchors the name and is
        # reserved for the future anagenesis axis.  Spans all species ever.
        self._registry: dict[str, np.ndarray] = {}
        # Stacked frozen-type matrix + name list, kept in sync (all species).
        self._progenitor_matrix: Optional[np.ndarray] = None  # (N, 500)
        self._species_order: list[str] = []

        # name → LIVING centroid (compat subset, 245-dim).  Only species with
        # living members appear here after a refresh; this is the C2 detection set.
        self._centroids: dict[str, np.ndarray] = {}
        self._centroid_matrix: Optional[np.ndarray] = None  # (M, 245)
        self._centroid_order: list[str] = []

        # Chronological log of every confirmed speciation event (each carries
        # an "event_type": currently always "cladogenesis").
        self.speciation_events: list[dict] = []
        # Track used name combinations to avoid duplicates
        self._used_names: set[str] = set()
        # Two-stage speciation: candidates waiting for promotion.
        # cid → {genes (full), compat (245-dim seed), parent_species,
        #         detected_week, members: set[str], first_creature_id, peak_members}
        self._candidates: dict[str, dict] = {}
        self._next_candidate_id: int = 0
        # Set by SimulationRunner each week before simulate_week calls
        self.current_week: int = 0
        # Candidates declared but whose members all died before promotion.
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

        Stores the full genome as the frozen type and seeds the species' living
        centroid from the genome's compatibility subset.

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

        1. No living centroids (empty/extinct registry): register a confirmed
           species immediately (bootstrap).
        2. Compatible match: the creature joins the nearest living species whose
           centroid is within compatibility_threshold.
        3. Reproductively isolated: the creature joins the nearest candidate (if
           close enough) or founds a new one.  In both cases it retains its
           inherited parent species label until/unless the candidate is promoted.

        Sets ``creature.species`` as a side effect and returns the name.
        """
        compat = self._compat(creature.genes)

        if self._centroid_matrix is None:
            # No living population to compare against — bootstrap a species.
            name = self._register_new_species(creature, creature.species)
            creature.species = name
            return name

        best_name, best_score = self._closest_centroid(compat)
        if best_score >= self.compatibility_threshold:
            creature.species = best_name
            return best_name

        # Reproductively isolated from every living species — candidate stage.
        parent_species = creature.species  # inherited label; kept through candidacy
        cid, cand_score = self._closest_candidate(compat)

        if cid is not None and cand_score >= self.compatibility_threshold:
            cand = self._candidates[cid]
            cand["members"].add(creature.creature_id)
            cand["peak_members"] = max(cand["peak_members"], len(cand["members"]))
        else:
            new_cid = f"cand_{self._next_candidate_id}"
            self._next_candidate_id += 1
            self._candidates[new_cid] = {
                "genes": creature.genes.copy(),
                "compat": compat.copy(),
                "parent_species": parent_species,
                "detected_week": self.current_week,
                "members": {creature.creature_id},
                "first_creature_id": creature.creature_id,
                "peak_members": 1,
            }

        return creature.species

    def refresh_centroids(self, alive_creatures: "list[Creature]") -> None:
        """
        Recompute each species' living centroid from its current members.

        Call periodically (every ``respeciate_every`` weeks) after migrations.
        The centroid of a species is the mean compatibility sub-vector of its
        living members, EXCLUDING creatures that currently belong to a pending
        candidate — candidate members are incipiently divergent and must not
        drag the parent species' reproductive centre toward themselves.

        Species with no qualifying living members are dropped from the
        comparison set (you cannot breed with an extinct population); they
        remain in the historical registry.
        """
        candidate_member_ids: set[str] = set()
        for cand in self._candidates.values():
            candidate_member_ids |= cand["members"]

        groups: dict[str, list[np.ndarray]] = {}
        for c in alive_creatures:
            if c.creature_id in candidate_member_ids:
                continue
            if c.species in self._registry:
                groups.setdefault(c.species, []).append(self._compat(c.genes))

        self._centroids = {
            name: np.mean(np.stack(vecs), axis=0) for name, vecs in groups.items()
        }
        self._rebuild_centroid_matrix()

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
                # Living centroid seeded from the promoted members' compat mean;
                # full-genome type frozen from the candidate's seed genome.
                member_compat = [
                    self._compat(creature_map[mid].genes)
                    for mid in alive_members
                    if mid in creature_map
                ]
                centroid = (
                    np.mean(np.stack(member_compat), axis=0)
                    if member_compat
                    else cand["compat"]
                )
                self._add_to_registry(new_name, cand["genes"], centroid=centroid)
                ev = {
                    "new_species": new_name,
                    "parent_species": cand["parent_species"],
                    "creature_id": cand["first_creature_id"],
                    "week": current_week,
                    "event_type": "cladogenesis",
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
        Full-genome cosine similarity between *creature* and every species'
        frozen TYPE genome.  Inspection/diagnostics helper; this is the
        full-genome axis the future anagenesis detector will build on, not the
        compatibility signal used for C2 detection.
        """
        if self._progenitor_matrix is None:
            return {}
        mat = self._progenitor_matrix
        dots = mat @ creature.genes
        prog_norms = np.linalg.norm(mat, axis=1)
        creature_norm = float(np.linalg.norm(creature.genes))
        denom = prog_norms * creature_norm
        safe = denom > 1e-10
        sims = np.where(safe, dots / np.where(safe, denom, 1.0), 0.0)
        sims = np.clip(sims, -1.0, 1.0)
        return dict(zip(self._species_order, sims.tolist()))

    def progenitor_genes(self, species_name: str) -> Optional[np.ndarray]:
        """Return a copy of the frozen TYPE genome for the named species."""
        genes = self._registry.get(species_name)
        return genes.copy() if genes is not None else None

    def centroid(self, species_name: str) -> Optional[np.ndarray]:
        """Return a copy of the living compatibility centroid, or None if extinct."""
        c = self._centroids.get(species_name)
        return c.copy() if c is not None else None

    @property
    def species_count(self) -> int:
        """Number of distinct species ever registered (including extinct)."""
        return len(self._registry)

    @property
    def living_species_count(self) -> int:
        """Number of species with living members as of the last centroid refresh."""
        return len(self._centroid_order)

    @property
    def all_species(self) -> list[str]:
        """Names of all species ever registered."""
        return list(self._registry.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compat(self, genes: np.ndarray) -> np.ndarray:
        """Slice the compatibility subset (245-dim) out of a full genome."""
        return genes[self._compat_indices]

    def _add_to_registry(
        self, name: str, genes: np.ndarray, centroid: Optional[np.ndarray] = None
    ) -> None:
        """
        Register a new species: store its frozen full-genome type and seed its
        living compatibility centroid.

        centroid : optional explicit compat centroid (used at promotion, seeded
            from living members).  If omitted, the centroid is seeded from the
            type genome's compatibility subset (used at founding/bootstrap).
        """
        self._registry[name] = genes.copy()
        self._species_order.append(name)
        self._used_names.add(name)

        row = genes[np.newaxis, :].copy()  # (1, 500)
        if self._progenitor_matrix is None:
            self._progenitor_matrix = row
        else:
            self._progenitor_matrix = np.vstack([self._progenitor_matrix, row])

        seed = centroid if centroid is not None else self._compat(genes)
        self._centroids[name] = seed.copy()
        self._rebuild_centroid_matrix()

    def _rebuild_centroid_matrix(self) -> None:
        """Restack the living-centroid matrix from self._centroids."""
        if not self._centroids:
            self._centroid_matrix = None
            self._centroid_order = []
            return
        self._centroid_order = list(self._centroids.keys())
        self._centroid_matrix = np.stack(
            [self._centroids[n] for n in self._centroid_order]
        )

    def _closest_centroid(self, compat: np.ndarray) -> tuple[str, float]:
        """Return (species_name, cosine_similarity) for the nearest living centroid."""
        mat = self._centroid_matrix  # (M, 245)
        dots = mat @ compat
        norms = np.linalg.norm(mat, axis=1) * float(np.linalg.norm(compat))
        safe = norms > 1e-10
        sims = np.where(safe, dots / np.where(safe, norms, 1.0), 0.0)
        sims = np.clip(sims, -1.0, 1.0)
        best_idx = int(np.argmax(sims))
        return self._centroid_order[best_idx], float(sims[best_idx])

    def _closest_candidate(self, compat: np.ndarray) -> tuple[Optional[str], float]:
        """
        Return (candidate_id, cosine_similarity) for the nearest candidate in
        compatibility space.  (None, 0.0) if no candidates exist.
        """
        if not self._candidates:
            return None, 0.0
        best_cid: Optional[str] = None
        best_score = -2.0
        norm_c = float(np.linalg.norm(compat))
        for cid, cand in self._candidates.items():
            cv = cand["compat"]
            denom = float(np.linalg.norm(cv)) * norm_c
            sim = (
                float(np.clip(np.dot(cv, compat) / denom, -1.0, 1.0))
                if denom > 1e-10
                else 0.0
            )
            if sim > best_score:
                best_score = sim
                best_cid = cid
        return best_cid, best_score

    def _register_new_species(
        self, creature: "Creature", parent_species: Optional[str]
    ) -> str:
        """Bootstrap path: immediately confirm a species from a single creature."""
        name = self._unique_name()
        self._add_to_registry(name, creature.genes)
        self.speciation_events.append({
            "new_species": name,
            "parent_species": parent_species,
            "creature_id": creature.creature_id,
            "event_type": "cladogenesis",
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
            f"{self.living_species_count} living, "
            f"{len(self._candidates)} candidates, "
            f"{len(self.speciation_events)} speciation events)"
        )
