"""
SimulationRunner — orchestrates a multi-habitat evolutionary simulation.

Usage
-----
    from pathlib import Path
    from evolution_simulator.simulation import SimulationRunner

    runner = SimulationRunner(Path("my_config.toml"))
    log_dir = runner.setup()   # initialise habitats, species, population
    runner.run()               # simulate all weeks and write logs

Or step-by-step for custom control:

    runner.setup()
    for week in range(100):
        week_log = runner.step()   # returns the week's full log dict

Log layout
----------
    simulation_logs/
    └── YYYY-MM-DD_HH-MM-SS/
        ├── config.toml         # copy of the config used for this run
        ├── metadata.json       # habitat topology, seed, parameters
        ├── week_00001.json     # per-week event log
        ├── week_00002.json
        ├── ...
        └── summary.json       # final state, speciation history

Each week_NNNNN.json contains every birth, death, mating attempt, migration,
speciation event, and habitat isolation — enough to replay the simulation in
a visualiser.
"""

import json
import logging
import random
import shutil
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from .creature import GENE_DIMS, Creature
from .habitat import LOGGED_TRAITS, Habitat
from .habitats import HABITAT_TYPE_REGISTRY
from .species import SpeciesRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimulationRunner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """
    Reads a TOML config, builds the simulation world, and drives the main loop.

    Attributes
    ----------
    config      : dict  – parsed TOML config
    habitats    : dict[str, Habitat]  – keyed by habitat id
    species_registry : SpeciesRegistry
    week        : int   – current simulation week (0 before any steps)
    log_dir     : Path  – output directory for this run (set by setup())
    """

    def __init__(self, config_path: Path):
        config_path = Path(config_path)
        with open(config_path, "rb") as fh:
            self.config: dict = tomllib.load(fh)
        self.config_path: Path = config_path

        self.habitats: dict[str, Habitat] = {}
        self.habitat_types: dict[str, str] = {}  # id → type name string
        self.species_registry: SpeciesRegistry | None = None
        self.week: int = 0
        self.log_dir: Path | None = None
        self._total_hybridization_events: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> Path:
        """
        Initialise the simulation from config.

        1. Creates the timestamped log directory and copies the config into it.
        2. Builds Habitat instances and connects them.
        3. Seeds each habitat with creatures grouped around founding genomes,
           ensuring both sexes are present and intra-group compatibility is high.
        4. Registers each founding genome as a species in the SpeciesRegistry.
        5. Writes metadata.json.

        Returns the Path to the log directory.
        """
        sim_cfg = self.config["simulation"]

        # --- Log directory ---
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(sim_cfg.get("output_dir", "simulation_logs"))
        self.log_dir = output_dir / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.config_path, self.log_dir / "config.toml")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logger.info("Log directory: %s", self.log_dir)

        # --- Species registry ---
        species_cfg = self.config.get("species", {})
        compat_threshold = species_cfg.get(
            "compatibility_threshold", SpeciesRegistry.DEFAULT_COMPATIBILITY_THRESHOLD
        )
        min_pop = species_cfg.get("min_species_population", SpeciesRegistry.DEFAULT_MIN_SPECIES_POPULATION)
        min_weeks = species_cfg.get("min_species_weeks", SpeciesRegistry.DEFAULT_MIN_SPECIES_WEEKS)
        split_max_k = species_cfg.get("split_max_k", SpeciesRegistry.DEFAULT_SPLIT_MAX_K)
        split_isolation = species_cfg.get(
            "split_isolation_threshold", SpeciesRegistry.DEFAULT_SPLIT_ISOLATION_THRESHOLD
        )
        anagenesis_threshold = species_cfg.get(
            "anagenesis_threshold", SpeciesRegistry.DEFAULT_ANAGENESIS_THRESHOLD
        )
        anagenesis_weeks = species_cfg.get("anagenesis_weeks", SpeciesRegistry.DEFAULT_ANAGENESIS_WEEKS)
        self.species_registry = SpeciesRegistry(
            compatibility_threshold=compat_threshold,
            min_species_population=min_pop,
            min_species_weeks=min_weeks,
            split_max_k=split_max_k,
            split_isolation_threshold=split_isolation,
            anagenesis_threshold=anagenesis_threshold,
            anagenesis_weeks=anagenesis_weeks,
        )

        # --- Habitats ---
        for inst in self.config["habitats"]["instances"]:
            hab_class = HABITAT_TYPE_REGISTRY[inst["type"]]
            hab = hab_class(
                habitat_id=inst["id"],
                name=inst.get("name", inst["id"]),
                instance_seed=inst.get("seed"),
                population_support=inst.get("population_support"),
            )
            self.habitats[inst["id"]] = hab
            self.habitat_types[inst["id"]] = inst["type"]

        for pair in self.config["habitats"].get("connections", []):
            self.habitats[pair[0]].add_neighbor(self.habitats[pair[1]])

        # --- Initial population ---
        global_n_species = sim_cfg.get("initial_species_per_habitat", 3)
        global_n_per_species = sim_cfg.get("creatures_per_species", 10)
        genome_noise = sim_cfg.get("initial_genome_noise", 0.05)
        # Bias (0–1): 0 = pure random genome, 1 = fully aligned to habitat vector.
        # A small bias (0.3) raises initial food/water probability from ~50% to ~65%
        # so the founding population survives the first generation with random metabolism.
        # Descendants still need to evolve; the bias only closes the viability gap.
        habitat_bias = sim_cfg.get("founding_habitat_bias", 0.0)
        seed = sim_cfg.get("seed", None)
        rng = np.random.default_rng(seed)
        # Seed the global RNGs so the full run is reproducible from `seed`.
        # habitat.py and creature.py use the global np.random.* singleton and
        # stdlib random; without this they are OS-initialised and differ between
        # runs even with the same config seed.
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._founders_by_hab: dict[str, list[dict]] = {hid: [] for hid in self.habitats}
        mirror = sim_cfg.get("mirror_founding_population", False)

        if mirror:
            # Draw founding genomes once and register each as a species, then seed
            # every habitat from those same genomes.  This lets isolated populations
            # diverge from an identical starting point.
            # habitat_bias is intentionally ignored: one founding genome cannot be
            # simultaneously aligned to multiple different habitat vectors.
            founding_specs: list[tuple[np.ndarray, str]] = []
            for _ in range(global_n_species):
                fg = rng.standard_normal(GENE_DIMS)
                founding_specs.append((fg, self.species_registry.register_founding_species(fg)))

            for hab_id, hab in self.habitats.items():
                n_per = self._habitat_instance_config(hab_id).get(
                    "creatures_per_species", global_n_per_species
                )
                for fg, sp_name in founding_specs:
                    for i in range(n_per):
                        genes = fg + rng.standard_normal(GENE_DIMS) * genome_noise
                        creature = Creature(genes=genes)
                        creature.sex = "female" if i % 2 == 0 else "male"
                        creature.species = sp_name
                        creature.age = creature.weeks_to_sexual_viability + 1
                        hab.add_creature(creature)
                        self._founders_by_hab[hab_id].append(
                            {
                                "creature_id": creature.creature_id,
                                "species": sp_name,
                                "sex": creature.sex,
                                "generation": 0,
                            }
                        )
        else:
            for hab_id, hab in self.habitats.items():
                inst_cfg = self._habitat_instance_config(hab_id)
                n_species = inst_cfg.get("initial_species_per_habitat", global_n_species)
                n_per = inst_cfg.get("creatures_per_species", global_n_per_species)

                for _ in range(n_species):
                    random_genes = rng.standard_normal(GENE_DIMS)
                    if habitat_bias > 0.0:
                        # Mix random genes with a scaled habitat direction so creatures
                        # start partially aligned to local resources.
                        hab_dir = hab.vector / (np.linalg.norm(hab.vector) + 1e-10)
                        hab_scaled = hab_dir * np.sqrt(GENE_DIMS)
                        founding_genes = (1.0 - habitat_bias) * random_genes + habitat_bias * hab_scaled
                    else:
                        founding_genes = random_genes
                    species_name = self.species_registry.register_founding_species(founding_genes)

                    for i in range(n_per):
                        genes = founding_genes + rng.standard_normal(GENE_DIMS) * genome_noise
                        creature = Creature(genes=genes)
                        # Assign a 50/50 sex split directly so mating is possible
                        # regardless of what the founding genome's sex loci encode.
                        creature.sex = "female" if i % 2 == 0 else "male"
                        creature.species = species_name
                        # Start at sexual maturity so mating begins on week 1.
                        creature.age = creature.weeks_to_sexual_viability + 1
                        hab.add_creature(creature)
                        self._founders_by_hab[hab_id].append(
                            {
                                "creature_id": creature.creature_id,
                                "species": species_name,
                                "sex": creature.sex,
                                "generation": 0,
                            }
                        )

        total_pop = sum(h.population_size for h in self.habitats.values())
        logger.info(
            "Setup complete: %d habitats, %d creatures, %d founding species",
            len(self.habitats),
            total_pop,
            self.species_registry.species_count,
        )

        self._write_metadata()
        return self.log_dir

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Simulate all weeks specified in config, writing a log file per week.

        Halts early if the global population reaches zero (extinction), in
        which case summary.json records ``"extinct": true``.
        """
        weeks = self.config["simulation"]["weeks"]
        logger.info("Starting simulation: %d weeks", weeks)
        extinct = False
        for w in range(weeks):
            self.step()
            pop = sum(h.population_size for h in self.habitats.values())
            if pop == 0:
                logger.info("Global extinction on week %d — halting early.", self.week)
                extinct = True
                break
            if (w + 1) % 50 == 0 or w == 0:
                logger.info(
                    "Week %d: population=%d, species=%d",
                    self.week,
                    pop,
                    self.species_registry.species_count,
                )
        self._write_summary(extinct=extinct)
        logger.info("Simulation complete. Logs in: %s", self.log_dir)

    def step(self) -> dict:
        """
        Advance the simulation by one week.

        Always (every week):
          1. Runs simulate_week() on every habitat.
          2. Applies all pending migrations (after all habitats are processed,
             so no creature is simulated twice in one week).
          3. Stamps the current week onto any new speciation events.
          4. Tallies hybridization events.
          5. Computes cheap global aggregates (population, species count,
             births/deaths counts) needed for progress display + extinction.

        Two independent cadences decide what (if anything) is written this week:
          - stats_every  gates the expensive per-species/per-habitat statistics
            (compute_stats()).
          - events_every gates the per-event detail (individual births, deaths,
            mating attempts, migrations, speciation events, isolations).
        A week_NNNNN.json is written if EITHER is due. Week 1 and the final week
        always get at least a statistics snapshot.

        Returns a dict. When a file is written it is the (possibly partial) week
        log. When nothing is due it is a lightweight summary. In all cases the
        returned dict contains every field runner.py / run() reads:
            week, logged, stats_logged, events_logged,
            global_population, global_species_count,
            births_this_week, deaths_this_week
        """
        self.week += 1
        sim_cfg = self.config["simulation"]
        iso_prob = sim_cfg.get("isolation_probability", 0.001)

        habitat_results: dict[str, dict] = {}
        pending_migrations: list[tuple] = []  # (creature, from_hab_id, dest_habitat)

        prev_speciation_count = len(self.species_registry.speciation_events)
        # current_week must be set before simulate_week so assign_species records
        # the correct detected_week on any new candidates.
        self.species_registry.current_week = self.week

        mating_strategy = sim_cfg.get("mating_strategy", "zip")
        for hab_id, hab in self.habitats.items():
            result = hab.simulate_week(
                species_registry=self.species_registry,
                isolation_probability=iso_prob,
                mating_strategy=mating_strategy,
            )
            habitat_results[hab_id] = result
            for creature, dest_hab in result["migrations"]:
                pending_migrations.append((creature, hab_id, dest_hab))

        # Apply migrations after all habitats have run
        migration_log: list[dict] = []
        for creature, from_id, dest_hab in pending_migrations:
            dest_hab.add_creature(creature)
            migration_log.append(
                {
                    "creature_id": creature.creature_id,
                    "species": creature.species,
                    "from_habitat": from_id,
                    "to_habitat": dest_hab.habitat_id,
                }
            )

        # Refresh living centroids (the reproductive reference for speciation)
        # and promote candidates that have met population + age criteria.  Both
        # must happen after migrations so migrated creatures count toward their
        # species centroid and toward candidate membership.  Promotions append
        # to speciation_events, so they're included in new_speciations below.
        all_alive = [c for hab in self.habitats.values() for c in hab.alive_creatures]
        respeciate_every = sim_cfg.get("respeciate_every", 10)
        if self.week == 1 or (respeciate_every >= 1 and self.week % respeciate_every == 0):
            # Order matters: seed split candidates first so their members are
            # excluded from the parent centroid in the refresh that follows;
            # then detect whole-lineage transformation (anagenesis).
            self.species_registry.detect_subcluster_splits(all_alive, self.week)
            self.species_registry.refresh_centroids(all_alive)
            self.species_registry.detect_anagenesis(all_alive, self.week)
        self.species_registry.promote_candidates(all_alive, self.week)
        # Weekly poll for species already in the anagenesis pending state: check
        # every week whether the clock has expired (fire) or the population has
        # rebounded (reset).  On cadence weeks the full detect_anagenesis above
        # already processed pending species, so this is effectively a no-op then.
        if self.species_registry._anagenesis_pending:
            self.species_registry.detect_anagenesis(all_alive, self.week, pending_only=True)

        new_speciations = self.species_registry.speciation_events[prev_speciation_count:]
        for ev in new_speciations:
            if "week" not in ev:
                ev["week"] = self.week

        for result in habitat_results.values():
            for ev in result.get("mating_events", []):
                if "hybridization" in ev:
                    self._total_hybridization_events += 1

        # --- Cheap aggregates, computed EVERY week (progress + extinction) ---
        total_pop = sum(h.population_size for h in self.habitats.values())
        births_this_week = sum(len(r["births"]) for r in habitat_results.values())
        deaths_this_week = sum(
            len(r["deaths"]) + len(r.get("predation_deaths", [])) for r in habitat_results.values()
        )

        # --- Decide what to write this week ---
        stats_every = sim_cfg.get("stats_every", 1)
        events_every = sim_cfg.get("events_every", 1)
        total_weeks = sim_cfg.get("weeks", 0)

        is_endpoint = (self.week == 1) or (self.week == total_weeks)
        stats_due = is_endpoint or (stats_every >= 1 and self.week % stats_every == 0)
        events_due = events_every >= 1 and self.week % events_every == 0

        if stats_due or events_due:
            week_log = self._build_week_log(
                habitat_results,
                migration_log,
                new_speciations,
                include_stats=stats_due,
                include_events=events_due,
            )
            self._write_week_log(week_log)
            week_log["logged"] = True
        else:
            week_log = {
                "week": self.week,
                "logged": False,
                "global_population": total_pop,
                "global_species_count": self.species_registry.species_count,
            }

        week_log["stats_logged"] = stats_due
        week_log["events_logged"] = events_due
        week_log["births_this_week"] = births_this_week
        week_log["deaths_this_week"] = deaths_this_week
        return week_log

    # ------------------------------------------------------------------
    # Log construction
    # ------------------------------------------------------------------

    def _build_week_log(
        self,
        habitat_results: dict[str, dict],
        migration_log: list[dict],
        new_speciations: list[dict],
        *,
        include_stats: bool = True,
        include_events: bool = True,
    ) -> dict:
        # ------------------------------------------------------------------
        # (A) Cheap global header — ALWAYS present (visualizer hard-requires
        #     week / global_population / global_species_count).
        # ------------------------------------------------------------------
        species_counts: Counter = Counter()
        total_pop = 0
        for hab in self.habitats.values():
            for c in hab.alive_creatures:
                species_counts[c.species] += 1
                total_pop += 1

        week_log: dict = {
            "week": self.week,
            "timestamp": datetime.now().isoformat(),
            "global_population": total_pop,
            "global_species_count": self.species_registry.species_count,
        }

        # ------------------------------------------------------------------
        # (B) Cheap per-habitat skeleton — ALWAYS present: id/type/name,
        #     population, species_distribution. Event detail added in (D).
        # ------------------------------------------------------------------
        habitats_log: dict = {}
        for hab_id, result in habitat_results.items():
            hab = self.habitats[hab_id]
            habitats_log[hab_id] = {
                "habitat_id": result["habitat_id"],
                "habitat_type": self.habitat_types.get(hab_id, "Unknown"),
                "habitat_name": hab.name,
                "population": result["population"],
                "species_distribution": dict(Counter(c.species for c in hab.alive_creatures)),
            }

        # ------------------------------------------------------------------
        # (C) EXPENSIVE statistics — only when include_stats.
        # ------------------------------------------------------------------
        if include_stats:
            week_log["global_species_distribution"] = dict(species_counts)

            habitat_stats: dict = {}
            sp_total: dict[str, int] = {}
            sp_hab_counts: dict[str, dict[str, int]] = {}
            sp_trait_sums: dict[str, dict[str, float]] = {}
            sp_generation_sums: dict[str, float] = {}

            for hab_id, hab in self.habitats.items():
                stats = hab.compute_stats()
                habitat_stats[hab_id] = {
                    "habitat_id": hab_id,
                    "habitat_name": hab.name,
                    "habitat_type": self.habitat_types.get(hab_id, "Unknown"),
                    "by_species": stats,
                }
                for sp_name, sp_data in stats.items():
                    n = sp_data["count"]
                    if sp_name not in sp_total:
                        sp_total[sp_name] = 0
                        sp_hab_counts[sp_name] = {}
                        sp_trait_sums[sp_name] = {t: 0.0 for t in LOGGED_TRAITS}
                        sp_generation_sums[sp_name] = 0.0
                    sp_total[sp_name] += n
                    sp_hab_counts[sp_name][hab_id] = n
                    for t in LOGGED_TRAITS:
                        sp_trait_sums[sp_name][t] += sp_data["mean_traits"][t] * n
                    sp_generation_sums[sp_name] += sp_data.get("mean_generation", 0.0) * n

            species_stats: dict = {}
            for sp_name, n in sp_total.items():
                species_stats[sp_name] = {
                    "total_count": n,
                    "habitat_counts": sp_hab_counts[sp_name],
                    "mean_generation": round(sp_generation_sums[sp_name] / n, 2),
                    "mean_traits": {t: round(sp_trait_sums[sp_name][t] / n, 4) for t in LOGGED_TRAITS},
                }

            week_log["habitat_stats"] = habitat_stats
            week_log["species_stats"] = species_stats

        # ------------------------------------------------------------------
        # (D) EXPENSIVE event detail — only when include_events.
        # ------------------------------------------------------------------
        if include_events:
            for hab_id, result in habitat_results.items():
                all_death_ids = result["deaths"] + result.get("predation_deaths", [])
                deaths_detail = []
                for cid in all_death_ids:
                    creature_log = result["week_results"].get(cid, {})
                    deaths_detail.append(
                        {
                            "creature_id": cid,
                            "cause": creature_log.get("cause_of_death"),
                            "age": creature_log.get("age"),
                        }
                    )

                births_detail = [
                    {
                        "creature_id": c.creature_id,
                        "species": c.species,
                        "sex": c.sex,
                        "generation": c.generation,
                        "parents": [p.creature_id for p in c.parents] if c.parents else [],
                    }
                    for c in result["births"]
                ]

                migrations_out = [
                    {
                        "creature_id": c.creature_id,
                        "species": c.species,
                        "to_habitat": dest.habitat_id,
                    }
                    for c, dest in result["migrations"]
                ]

                entry = habitats_log[hab_id]
                entry["births"] = births_detail
                entry["deaths"] = deaths_detail
                entry["mating_events"] = result.get("mating_events", [])
                entry["migrations_out"] = migrations_out
                entry["isolations"] = result["isolations"]

            week_log["speciation_events"] = new_speciations
            week_log["migrations"] = migration_log

        week_log["habitats"] = habitats_log
        return week_log

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _write_week_log(self, week_log: dict) -> None:
        path = self.log_dir / f"week_{self.week:05d}.json"
        with open(path, "w") as fh:
            json.dump(week_log, fh)

    def _write_metadata(self) -> None:
        sim_cfg = self.config["simulation"]
        metadata = {
            "simulation_start": datetime.now().isoformat(),
            "parameters": {
                "weeks": sim_cfg["weeks"],
                "seed": sim_cfg.get("seed"),
                "initial_species_per_habitat": sim_cfg.get("initial_species_per_habitat", 3),
                "creatures_per_species": sim_cfg.get("creatures_per_species", 10),
                "initial_genome_noise": sim_cfg.get("initial_genome_noise", 0.05),
                "isolation_probability": sim_cfg.get("isolation_probability", 0.001),
                "compatibility_threshold": self.config.get("species", {}).get(
                    "compatibility_threshold",
                    SpeciesRegistry.DEFAULT_COMPATIBILITY_THRESHOLD,
                ),
                "respeciate_every": sim_cfg.get("respeciate_every", 10),
                "split_max_k": self.config.get("species", {}).get(
                    "split_max_k", SpeciesRegistry.DEFAULT_SPLIT_MAX_K
                ),
                "split_isolation_threshold": self.config.get("species", {}).get(
                    "split_isolation_threshold",
                    SpeciesRegistry.DEFAULT_SPLIT_ISOLATION_THRESHOLD,
                ),
                "anagenesis_threshold": self.config.get("species", {}).get(
                    "anagenesis_threshold", SpeciesRegistry.DEFAULT_ANAGENESIS_THRESHOLD
                ),
                "anagenesis_weeks": self.config.get("species", {}).get(
                    "anagenesis_weeks", SpeciesRegistry.DEFAULT_ANAGENESIS_WEEKS
                ),
            },
            "habitats": [
                {
                    "id": inst["id"],
                    "type": inst["type"],
                    "name": inst.get("name", inst["id"]),
                    "seed": inst.get("seed"),
                }
                for inst in self.config["habitats"]["instances"]
            ],
            "connections": self.config["habitats"].get("connections", []),
            "founding_species": self.species_registry.all_species,
            "founders_by_hab": getattr(self, "_founders_by_hab", {}),
        }
        with open(self.log_dir / "metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2)

    def _write_summary(self, extinct: bool = False) -> None:
        summary = {
            "simulation_end": datetime.now().isoformat(),
            "weeks_simulated": self.week,
            "extinct": extinct,
            "final_population": sum(h.population_size for h in self.habitats.values()),
            "total_species_ever": self.species_registry.species_count,
            "total_speciation_events": len(self.species_registry.speciation_events),
            "total_hybridization_events": self._total_hybridization_events,
            "all_speciation_events": self.species_registry.speciation_events,
            "all_failed_speciation_attempts": self.species_registry.failed_speciation_attempts,
            "final_species_distribution": {
                hab_id: dict(Counter(c.species for c in hab.alive_creatures))
                for hab_id, hab in self.habitats.items()
            },
            "final_population_per_habitat": {
                hab_id: hab.population_size for hab_id, hab in self.habitats.items()
            },
        }
        with open(self.log_dir / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _habitat_instance_config(self, hab_id: str) -> dict:
        """Return the TOML [[habitats.instances]] block for the given id."""
        for inst in self.config["habitats"]["instances"]:
            if inst["id"] == hab_id:
                return inst
        return {}
