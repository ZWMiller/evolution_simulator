"""
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
"""

import random
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .creature import (
    DEFAULT_TRAIT_GENE_INDICES,
    Creature,
    compute_phenotype,
    compute_phenotype_matrix,
)

if TYPE_CHECKING:
    pass

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
    DEFAULT_SPLIT_MAX_K: int = 5
    DEFAULT_SPLIT_ISOLATION_THRESHOLD: float = Creature.COMPATIBILITY_FLOOR
    # Centred phenotype-centroid-vs-type cosine.  A 12k-week (~370-generation)
    # run showed phenotype drift PLATEAUS: the most-transformed living lineages
    # bottom out around ~0.92–0.94 (median ~0.97), so ~0.93 fires only for the
    # genuinely transformed and stays rare.  anagenesis_weeks then requires the
    # transformation to PERSIST before a descendant is minted, so a transient
    # dip that rebounds is not named.
    DEFAULT_ANAGENESIS_THRESHOLD: float = 0.93
    DEFAULT_ANAGENESIS_WEEKS: int = 60

    def __init__(
        self,
        compatibility_threshold: float = DEFAULT_COMPATIBILITY_THRESHOLD,
        config_path: Path = DEFAULT_CONFIG_PATH,
        min_species_population: int = DEFAULT_MIN_SPECIES_POPULATION,
        min_species_weeks: int = DEFAULT_MIN_SPECIES_WEEKS,
        compat_indices: list[int] | None = None,
        split_max_k: int = DEFAULT_SPLIT_MAX_K,
        split_isolation_threshold: float = DEFAULT_SPLIT_ISOLATION_THRESHOLD,
        anagenesis_threshold: float = DEFAULT_ANAGENESIS_THRESHOLD,
        anagenesis_weeks: int = DEFAULT_ANAGENESIS_WEEKS,
    ):
        self.compatibility_threshold: float = compatibility_threshold
        self.min_species_population: int = min_species_population
        self.min_species_weeks: int = min_species_weeks
        # Sub-cluster split (cladogenesis) detection
        self.split_max_k: int = split_max_k
        self.split_isolation_threshold: float = split_isolation_threshold
        # Anagenesis (phenotype drift from frozen type) detection
        self.anagenesis_threshold: float = anagenesis_threshold
        self.anagenesis_weeks: int = anagenesis_weeks
        if compat_indices is None:
            compat_indices = DEFAULT_TRAIT_GENE_INDICES["compatibility_genes"]
        self._compat_indices: np.ndarray = np.asarray(compat_indices, dtype=int)

        self._adjectives, self._nouns = load_name_config(config_path)

        # name → frozen full-genome TYPE (500-dim).  Anchors the name and is
        # reserved for the future anagenesis axis.  Spans all species ever.
        self._registry: dict[str, np.ndarray] = {}
        # Stacked frozen-type matrix + name list, kept in sync (all species).
        self._progenitor_matrix: np.ndarray | None = None  # (N, 500)
        self._species_order: list[str] = []

        # name → LIVING centroid (compat subset, 245-dim).  Only species with
        # living members appear here after a refresh; this is the C2 detection set.
        self._centroids: dict[str, np.ndarray] = {}
        self._centroid_matrix: np.ndarray | None = None  # (M, 245)
        self._centroid_order: list[str] = []

        # name → frozen TYPE phenotype vector (raw [0,1] PHENOTYPE_TRAITS).
        # The anchored reference for the anagenesis axis.
        self._type_phenotype: dict[str, np.ndarray] = {}
        # species name → first week its phenotype centroid was seen below the
        # anagenesis threshold (the persistence clock; cleared if it rebounds).
        self._anagenesis_pending: dict[str, int] = {}

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
        name: str | None = None,
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
        Assign the species for a newborn creature.

        1. No living centroids (empty/extinct registry): register a confirmed
           species immediately (bootstrap).
        2. Otherwise: assign to the nearest living species by centroid cosine
           similarity, regardless of whether the score clears
           compatibility_threshold.  Genuine reproductive isolation is detected
           at the population level by detect_subcluster_splits(), not per-newborn.

        Sets ``creature.species`` as a side effect and returns the name.
        """
        compat = self._compat(creature.genes)

        if self._centroid_matrix is None:
            # No living population to compare against — bootstrap a species.
            name = self._register_new_species(creature, creature.species)
            creature.species = name
            return name

        best_name, _ = self._closest_centroid(compat)
        creature.species = best_name
        return best_name

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

        self._centroids = {name: np.mean(np.stack(vecs), axis=0) for name, vecs in groups.items()}
        self._rebuild_centroid_matrix()

    def promote_candidates(self, alive_creatures: "list[Creature]", current_week: int) -> list[dict]:
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
                self.failed_speciation_attempts.append(
                    {
                        "parent_species": cand["parent_species"],
                        "detected_week": cand["detected_week"],
                        "failed_week": current_week,
                        "peak_members": cand["peak_members"],
                        "first_creature_id": cand["first_creature_id"],
                    }
                )
                to_remove.append(cid)
                continue

            cand["members"] = alive_members
            weeks_elapsed = current_week - cand["detected_week"]

            if len(alive_members) >= self.min_species_population and weeks_elapsed >= self.min_species_weeks:
                new_name = self._unique_name()
                # Living centroid seeded from the promoted members' compat mean;
                # full-genome type frozen from the candidate's seed genome.
                member_compat = [
                    self._compat(creature_map[mid].genes) for mid in alive_members if mid in creature_map
                ]
                centroid = np.mean(np.stack(member_compat), axis=0) if member_compat else cand["compat"]
                self._add_to_registry(new_name, cand["genes"], centroid=centroid)
                origin = cand.get("origin", "kmeans_subcluster")
                ev = {
                    "new_species": new_name,
                    "parent_species": cand["parent_species"],
                    "creature_id": cand["first_creature_id"],
                    "week": current_week,
                    "event_type": f"cladogenesis_{origin}",
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

    def detect_subcluster_splits(self, alive_creatures: "list[Creature]", current_week: int) -> None:
        """
        Detect reproductive-isolation splits WITHIN a species (cladogenesis).

        Run periodically, BEFORE refresh_centroids, so any sub-cluster it seeds
        as a candidate is excluded from the parent species' refreshed centroid.
        For each species, its living members (excluding creatures already in a
        candidate) are clustered in compatibility space; if they fall into 2+
        mutually reproductively-isolated sub-clusters, the cluster closest to the
        frozen type keeps the species name and each other cluster seeds (or
        extends) a candidate.  Promotion/aging/relabelling then flow through the
        existing two-stage gate (promote_candidates).

        This is what catches allopatric/sympatric divergence that a single
        species centroid masks: two habitat-adapted populations sharing one label
        sit symmetrically around their midpoint centroid, so newborns never
        individually fall below threshold — but their members form two clusters,
        which this detects directly.
        """
        candidate_member_ids: set[str] = set()
        for cand in self._candidates.values():
            candidate_member_ids |= cand["members"]

        by_species: dict[str, list] = {}
        for c in alive_creatures:
            if c.creature_id in candidate_member_ids:
                continue
            if c.species in self._registry:
                by_species.setdefault(c.species, []).append(c)

        for sp, members in by_species.items():
            if len(members) < 2 * self.min_species_population:
                continue
            compat = np.stack([self._compat(c.genes) for c in members])
            units = self._unit_rows(compat)
            labels, centroids = self._select_clusters(units, seed=current_week)
            if centroids.shape[0] == 1:
                continue

            type_unit = self._unit_rows(self._compat(self._registry[sp])[np.newaxis, :])[0]
            primary = int(np.argmax(centroids @ type_unit))

            for j in range(centroids.shape[0]):
                if j == primary:
                    continue
                idx = [i for i, lab in enumerate(labels) if lab == j]
                if len(idx) < self.min_species_population:
                    continue
                sub_members = [members[i] for i in idx]
                sub_centroid = centroids[j]

                cid, score = self._closest_candidate(sub_centroid)
                if cid is not None and score >= self.compatibility_threshold:
                    cand = self._candidates[cid]
                    for m in sub_members:
                        cand["members"].add(m.creature_id)
                    cand["peak_members"] = max(cand["peak_members"], len(cand["members"]))
                else:
                    rep = max(
                        sub_members,
                        key=lambda m: float(
                            self._unit_rows(self._compat(m.genes)[np.newaxis, :])[0] @ sub_centroid
                        ),
                    )
                    new_cid = f"cand_{self._next_candidate_id}"
                    self._next_candidate_id += 1
                    self._candidates[new_cid] = {
                        "genes": rep.genes.copy(),
                        "compat": self._compat(rep.genes).copy(),
                        "parent_species": sp,
                        "detected_week": current_week,
                        "members": {m.creature_id for m in sub_members},
                        "first_creature_id": rep.creature_id,
                        "peak_members": len(sub_members),
                        "origin": "kmeans_subcluster",
                    }

    def detect_anagenesis(
        self,
        alive_creatures: "list[Creature]",
        current_week: int,
        pending_only: bool = False,
    ) -> list[dict]:
        """
        Detect a lineage that has TRANSFORMED over time without splitting
        (anagenesis / chronospecies).

        For each living species — skipping any with an active split candidate
        this cycle — compare its living phenotype centroid to its frozen type
        phenotype.  If cosine has fallen below anagenesis_threshold, mint a
        descendant species and respeciate living members by closest phenotype:
        members nearer the new centroid join the descendant; members still nearer
        the type keep the old name (the name follows the type).  If every member
        moves, the ancestor simply has no living members (pure anagenesis /
        lineage replacement); dead ancestors keep their labels — the log is the
        fossil record.

        pending_only : if True, only process species already in
            _anagenesis_pending (rebound-check or fire); skip species not yet
            tracked.  Used for the weekly between-cadence poll so the persistence
            clock is week-precise rather than rounded to respeciate_every.

        Returns the list of anagenesis events created this call.
        """
        split_parents = {cand["parent_species"] for cand in self._candidates.values()}

        by_species: dict[str, list] = {}
        for c in alive_creatures:
            if c.species in self._registry:
                if pending_only and c.species not in self._anagenesis_pending:
                    continue
                by_species.setdefault(c.species, []).append(c)

        events: list[dict] = []
        for sp, members in by_species.items():
            if sp in split_parents:
                self._anagenesis_pending.pop(sp, None)
                continue
            if len(members) < self.min_species_population:
                continue
            ph = compute_phenotype_matrix(np.stack([m.genes for m in members]))
            centroid = ph.mean(axis=0)
            type_ph = self._type_phenotype[sp]
            # Phenotype values live in [0, 1] (the positive orthant), where raw
            # cosine is compressed toward 1 and a threshold like 0.80 is almost
            # unreachable.  Centre on 0.5 so the comparison measures the trait
            # DEVIATION pattern (correlation-like, full [-1, 1] range): a lineage
            # that shifts many traits in a consistent direction then reads as a
            # large angle away from its ancestral form.
            phc = ph - 0.5
            cc = centroid - 0.5
            tc = type_ph - 0.5
            if self._cos(cc, tc) >= self.anagenesis_threshold:
                self._anagenesis_pending.pop(sp, None)  # rebounded → reset clock
                continue

            # Persistence gate: the transformation must hold below threshold for
            # anagenesis_weeks before a descendant is named, so a transient dip
            # that rebounds is not mistaken for a chronospecies boundary.
            # In pending_only mode we only update the clock for tracked species;
            # in full mode setdefault registers the first dip week.
            if pending_only:
                first_seen = self._anagenesis_pending.get(sp)
                if first_seen is None:
                    continue
            else:
                first_seen = self._anagenesis_pending.setdefault(sp, current_week)
            if current_week - first_seen < self.anagenesis_weeks:
                continue

            mover_pairs = [
                (i, m) for i, m in enumerate(members) if self._cos(phc[i], cc) > self._cos(phc[i], tc)
            ]
            if len(mover_pairs) < self.min_species_population:
                continue

            mover_indices = [i for i, _ in mover_pairs]
            mover_centroid_ph = ph[mover_indices].mean(axis=0)
            mover_centroid_ph_c = mover_centroid_ph - 0.5

            # Deduplication guard: if the mover phenotype centroid is already
            # within anagenesis_threshold of an existing species' frozen type
            # phenotype, the movers have drifted into a region already occupied
            # by a confirmed species.  Reassign them there instead of minting a
            # redundant new lineage (prevents cascade re-registration of the same
            # population shift across multiple detection cycles).
            existing_match: str | None = None
            for existing_sp, existing_type_ph in self._type_phenotype.items():
                if existing_sp == sp:
                    continue
                if self._cos(mover_centroid_ph_c, existing_type_ph - 0.5) >= self.anagenesis_threshold:
                    existing_match = existing_sp
                    break
            if existing_match is not None:
                for _, m in mover_pairs:
                    m.species = existing_match
                self._anagenesis_pending.pop(sp, None)
                continue

            rep = max(mover_pairs, key=lambda im: self._cos(phc[im[0]], cc))[1]
            movers = [m for _, m in mover_pairs]
            mover_compat = np.mean(np.stack([self._compat(m.genes) for m in movers]), axis=0)
            new_name = self._unique_name()
            self._add_to_registry(new_name, rep.genes, centroid=mover_compat)
            # Anchor the anagenesis type phenotype to the actual mover population
            # centroid rather than to compute_phenotype(rep.genes).  Because OWA
            # is nonlinear, a single individual's phenotype != the population mean
            # phenotype; using the individual creates an immediate discrepancy that
            # restarts the persistence clock and produces cascade re-fires at the
            # minimum possible interval (2 × respeciate_every).
            self._type_phenotype[new_name] = mover_centroid_ph
            for m in movers:
                m.species = new_name
            ev = {
                "new_species": new_name,
                "parent_species": sp,
                "creature_id": rep.creature_id,
                "week": current_week,
                "event_type": "anagenesis",
            }
            self.speciation_events.append(ev)
            events.append(ev)
            self._anagenesis_pending.pop(sp, None)

        return events

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
        return dict(zip(self._species_order, sims.tolist(), strict=False))

    def progenitor_genes(self, species_name: str) -> np.ndarray | None:
        """Return a copy of the frozen TYPE genome for the named species."""
        genes = self._registry.get(species_name)
        return genes.copy() if genes is not None else None

    def centroid(self, species_name: str) -> np.ndarray | None:
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

    def _add_to_registry(self, name: str, genes: np.ndarray, centroid: np.ndarray | None = None) -> None:
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

        # Anchor the anagenesis reference: the type genome's phenotype.
        self._type_phenotype[name] = compute_phenotype(genes)

    def _rebuild_centroid_matrix(self) -> None:
        """Restack the living-centroid matrix from self._centroids."""
        if not self._centroids:
            self._centroid_matrix = None
            self._centroid_order = []
            return
        self._centroid_order = list(self._centroids.keys())
        self._centroid_matrix = np.stack([self._centroids[n] for n in self._centroid_order])

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

    def _closest_candidate(self, compat: np.ndarray) -> tuple[str | None, float]:
        """
        Return (candidate_id, cosine_similarity) for the nearest candidate in
        compatibility space.  (None, 0.0) if no candidates exist.
        """
        if not self._candidates:
            return None, 0.0
        best_cid: str | None = None
        best_score = -2.0
        norm_c = float(np.linalg.norm(compat))
        for cid, cand in self._candidates.items():
            cv = cand["compat"]
            denom = float(np.linalg.norm(cv)) * norm_c
            sim = float(np.clip(np.dot(cv, compat) / denom, -1.0, 1.0)) if denom > 1e-10 else 0.0
            if sim > best_score:
                best_score = sim
                best_cid = cid
        return best_cid, best_score

    # ------------------------------------------------------------------
    # Spherical k-means (for sub-cluster split detection)
    # ------------------------------------------------------------------
    # We need to partition a species' members in COSINE geometry, because
    # cosine is the metric that governs mating compatibility.  "Spherical"
    # k-means is ordinary k-means run on L2-normalized vectors: once every
    # point lies on the unit sphere, squared Euclidean distance and cosine are
    # equivalent objectives —
    #     ||x - c||^2 = 2 - 2 (x . c)     for unit x, c
    # so minimizing Euclidean distortion is the same as maximizing cosine
    # similarity to the assigned centre.  Implemented in numpy rather than
    # scikit-learn: at this scale (small per-species n, dim 245, K <=
    # split_max_k, run only every respeciate_every weeks) the optimised library
    # buys nothing and would add a heavy sklearn+scipy dependency.

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two vectors in [-1, 1] (0 if either is ~zero)."""
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

    @staticmethod
    def _unit_rows(mat: np.ndarray) -> np.ndarray:
        """
        Project each row onto the unit sphere (L2-normalize).

        This is the step that turns ordinary k-means into *spherical* k-means:
        on unit vectors, the dot product X @ Cᵀ IS the cosine similarity, so all
        downstream assignment and centroid maths operate in cosine geometry.
        Zero-length rows are left unscaled to avoid division by zero.
        """
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return mat / norms

    @staticmethod
    def _kmeanspp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """
        Choose k initial centres with k-means++ (cosine variant).

        k-means++ spreads the seeds out so Lloyd's iterations are far less
        likely to land in a poor local optimum than uniform-random seeding:

          1. Pick the first centre uniformly at random from the points.
          2. For every point compute its cosine distance (1 - cosine) to the
             NEAREST centre chosen so far.
          3. Pick the next centre at random with probability proportional to
             that distance — points far from all current centres are the most
             likely to be chosen.
          4. Repeat 2-3 until k centres are chosen.

        X is assumed to have unit rows, so ``X @ centresᵀ`` is cosine similarity.
        """
        n = X.shape[0]
        centroids = [X[rng.integers(n)]]
        for _ in range(1, k):
            sims = X @ np.stack(centroids).T  # (n, chosen) cosine
            dist = np.clip(1.0 - sims.max(axis=1), 0.0, None)  # dist to nearest centre
            total = float(dist.sum())
            if total <= 1e-12:  # all points coincide
                centroids.append(X[rng.integers(n)])
            else:
                centroids.append(X[rng.choice(n, p=dist / total)])
        return np.stack(centroids)

    def _spherical_kmeans(
        self, X: np.ndarray, k: int, seed: int, n_init: int = 3, max_iter: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cluster unit-row matrix X into k clusters with spherical k-means.

        Runs Lloyd's algorithm ``n_init`` times from independent k-means++
        seedings and keeps the best result, where "best" maximizes the total
        cosine similarity of points to their assigned centre (the spherical
        analogue of minimizing k-means inertia).

        Each Lloyd iteration:
          - **Assign**: label every point by its most-similar centre
            (``argmax`` of the cosine matrix ``X @ centresᵀ``).
          - **Update**: recompute each centre as the *spherical mean* of its
            members — sum the member unit vectors and re-normalize, which is the
            point on the sphere maximizing summed cosine to the cluster.
          - An emptied cluster is reseeded to a random point so k clusters are
            always returned.
          - Iteration stops once labels stop changing (converged) or after
            ``max_iter`` sweeps.

        ``seed`` makes the result deterministic (the whole simulation is
        seeded), so repeated runs reproduce the same partition.

        Returns
        -------
        (labels, centroids) : labels is (n,) int; centroids is (k, d) with UNIT
        rows, so callers can take dot products as cosines directly.
        """
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        best_labels: np.ndarray | None = None
        best_centroids: np.ndarray | None = None
        best_inertia = -np.inf  # total cosine-to-centre; maximize
        for _ in range(n_init):
            centroids = self._kmeanspp_init(X, k, rng)
            labels = np.full(n, -1, dtype=int)
            for _ in range(max_iter):
                new_labels = np.argmax(X @ centroids.T, axis=1)  # assign
                for j in range(k):  # update
                    pts = X[new_labels == j]
                    if len(pts) == 0:
                        centroids[j] = X[rng.integers(n)]  # reseed empty cluster
                    else:
                        c = pts.sum(axis=0)
                        nrm = float(np.linalg.norm(c))
                        centroids[j] = c / nrm if nrm > 1e-12 else pts[0]  # spherical mean
                if np.array_equal(new_labels, labels):
                    labels = new_labels
                    break  # converged
                labels = new_labels
            inertia = float((X @ centroids.T)[np.arange(n), labels].sum())
            if inertia > best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centroids = centroids.copy()
        return best_labels, best_centroids

    def _select_clusters(self, X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Pick the number of reproductively-isolated sub-clusters in X.

        Sweeps K = 1..split_max_k and returns ``(labels, unit centroids)`` for
        the LARGEST K that is a *valid* partition, where valid means:
          - every cluster has at least ``min_species_population`` members, and
          - all cluster centres are mutually below ``split_isolation_threshold``
            (i.e. every pair of sub-clusters has stopped interbreeding).

        Taking the largest valid K makes "K" equal the number of genuinely
        isolated groups: over-splitting a real group produces two centres that
        are still *above* the isolation threshold, which fails the test, so the
        sweep settles back to the true count.  K = 1 is the default when no
        larger K qualifies — a single coherent (still-interbreeding) population
        is therefore never split, because any 2-way split of it leaves the two
        halves above the isolation threshold.
        """
        n = X.shape[0]
        mean = X.sum(axis=0)
        nrm = float(np.linalg.norm(mean))
        mean = mean / nrm if nrm > 1e-12 else X[0]
        best = (np.zeros(n, dtype=int), mean[np.newaxis, :])  # K = 1 fallback

        kmax = min(self.split_max_k, n // self.min_species_population)
        for k in range(2, kmax + 1):
            labels, centroids = self._spherical_kmeans(X, k, seed + k)
            if (np.bincount(labels, minlength=k) < self.min_species_population).any():
                continue  # a cluster too small
            pairwise = centroids @ centroids.T  # cosine (unit centres)
            iu = np.triu_indices(k, k=1)
            if float(pairwise[iu].max()) < self.split_isolation_threshold:
                best = (labels, centroids)  # mutually isolated → valid
        return best

    def _register_new_species(self, creature: "Creature", parent_species: str | None) -> str:
        """Bootstrap path: immediately confirm a species from a single creature."""
        name = self._unique_name()
        self._add_to_registry(name, creature.genes)
        self.speciation_events.append(
            {
                "new_species": name,
                "parent_species": parent_species,
                "creature_id": creature.creature_id,
                "event_type": "cladogenesis_bootstrap",
            }
        )
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
            f"Exhausted all {len(self._adjectives) * len(self._nouns)} unique species name combinations."
        )

    def __repr__(self) -> str:
        return (
            f"SpeciesRegistry({self.species_count} species, "
            f"{self.living_species_count} living, "
            f"{len(self._candidates)} candidates, "
            f"{len(self.speciation_events)} speciation events)"
        )
