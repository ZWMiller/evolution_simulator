"""
SimulationRunner — orchestrates a multi-habitat evolutionary simulation.

Usage
-----
    from pathlib import Path
    from evolution_simulator.simulation import SimulationRunner

    runner = SimulationRunner(Path("my_config.toml"))
    log_dir = runner.setup()   # initialise habitats, species, population
    runner.run()               # simulate all days and write logs

Or step-by-step for custom control:

    runner.setup()
    for day in range(100):
        day_log = runner.step()   # returns the day's full log dict

Log layout
----------
    simulation_logs/
    └── YYYY-MM-DD_HH-MM-SS/
        ├── config.toml        # copy of the config used for this run
        ├── metadata.json      # habitat topology, seed, parameters
        ├── day_00001.json     # per-day event log
        ├── day_00002.json
        ├── ...
        └── summary.json      # final state, speciation history

Each day_NNNNN.json contains every birth, death, mating attempt, migration,
speciation event, and habitat isolation — enough to replay the simulation in
a visualiser.
"""

import json
import logging
import shutil
import tomllib
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from .creature import Creature, DEFAULT_TRAIT_GENE_INDICES, GENE_DIMS
from .habitat import Habitat, LOGGED_TRAITS
from .habitats import HABITAT_TYPE_REGISTRY
from .species import SpeciesRegistry

logger = logging.getLogger(__name__)

# Gene indices forced in founding genomes to bootstrap viable life-history traits.
_SEX_GENE_INDICES: list[int] = DEFAULT_TRAIT_GENE_INDICES["sex_determination"]
_MATURITY_GENE_INDICES: list[int] = DEFAULT_TRAIT_GENE_INDICES["days_to_sexual_viability"]
_GESTATION_GENE_INDICES: list[int] = DEFAULT_TRAIT_GENE_INDICES["reproduction_time"]


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
    day         : int   – current simulation day (0 before any steps)
    log_dir     : Path  – output directory for this run (set by setup())
    """

    def __init__(self, config_path: Path):
        config_path = Path(config_path)
        with open(config_path, "rb") as fh:
            self.config: dict = tomllib.load(fh)
        self.config_path: Path = config_path

        self.habitats: dict[str, Habitat] = {}
        self.habitat_types: dict[str, str] = {}   # id → type name string
        self.species_registry: Optional[SpeciesRegistry] = None
        self.day: int = 0
        self.log_dir: Optional[Path] = None

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
        threshold = self.config.get("species", {}).get("threshold", 0.95)
        self.species_registry = SpeciesRegistry(species_threshold=threshold)

        # --- Habitats ---
        for inst in self.config["habitats"]["instances"]:
            hab_class = HABITAT_TYPE_REGISTRY[inst["type"]]
            hab = hab_class(
                habitat_id=inst["id"],
                name=inst.get("name", inst["id"]),
                instance_seed=inst.get("seed"),
            )
            self.habitats[inst["id"]] = hab
            self.habitat_types[inst["id"]] = inst["type"]

        for pair in self.config["habitats"].get("connections", []):
            self.habitats[pair[0]].add_neighbor(self.habitats[pair[1]])

        # --- Initial population ---
        global_n_species = sim_cfg.get("initial_species_per_habitat", 3)
        global_n_per_species = sim_cfg.get("creatures_per_species", 10)
        genome_noise = sim_cfg.get("initial_genome_noise", 0.05)
        seed = sim_cfg.get("seed", None)
        rng = np.random.default_rng(seed)

        for hab_id, hab in self.habitats.items():
            inst_cfg = self._habitat_instance_config(hab_id)
            n_species = inst_cfg.get("initial_species_per_habitat", global_n_species)
            n_per = inst_cfg.get("creatures_per_species", global_n_per_species)

            for sp_idx in range(n_species):
                founding_genes = rng.standard_normal(GENE_DIMS)
                species_name = self.species_registry.register_founding_species(
                    founding_genes
                )

                for i in range(n_per):
                    genes = founding_genes + rng.standard_normal(GENE_DIMS) * genome_noise
                    # Force a 50/50 sex split within each group so mating is
                    # always possible regardless of the founding genome's sex loci.
                    if i < n_per // 2:
                        genes[_SEX_GENE_INDICES] = 10.0   # female
                    else:
                        genes[_SEX_GENE_INDICES] = -10.0  # male
                    # Force fast maturation and gestation so the founding population
                    # can reproduce quickly and establish a next generation.  These
                    # loci are inherited by offspring, so the fast life-history
                    # traits propagate until mutation drifts them upward.
                    genes[_MATURITY_GENE_INDICES] = -10.0   # min days_to_sexual_viability (~30d)
                    genes[_GESTATION_GENE_INDICES] = -10.0  # min reproduction_time (~10d)

                    creature = Creature(genes=genes)
                    creature.species = species_name
                    # Start founding creatures at sexual maturity so mating can
                    # happen immediately without waiting out the maturity period.
                    creature.age = creature.days_to_sexual_viability + 1
                    hab.add_creature(creature)

        total_pop = sum(h.population_size for h in self.habitats.values())
        logger.info(
            "Setup complete: %d habitats, %d creatures, %d founding species",
            len(self.habitats), total_pop, self.species_registry.species_count,
        )

        self._write_metadata()
        return self.log_dir

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Simulate all days specified in config, writing a log file per day.

        Halts early if the global population reaches zero (extinction), in
        which case summary.json records ``"extinct": true``.
        """
        days = self.config["simulation"]["days"]
        logger.info("Starting simulation: %d days", days)
        extinct = False
        for d in range(days):
            self.step()
            pop = sum(h.population_size for h in self.habitats.values())
            if pop == 0:
                logger.info("Global extinction on day %d — halting early.", self.day)
                extinct = True
                break
            if (d + 1) % 50 == 0 or d == 0:
                logger.info(
                    "Day %d: population=%d, species=%d",
                    self.day, pop, self.species_registry.species_count,
                )
        self._write_summary(extinct=extinct)
        logger.info("Simulation complete. Logs in: %s", self.log_dir)

    def step(self) -> dict:
        """
        Advance the simulation by one day.

        1. Runs simulate_day() on every habitat.
        2. Applies all pending migrations (creatures move after all habitats
           have been processed, so no creature is simulated twice in one day).
        3. Builds and writes the day's JSON log.

        Returns the full day log dict.
        """
        self.day += 1
        iso_prob = self.config["simulation"].get("isolation_probability", 0.001)

        habitat_results: dict[str, dict] = {}
        pending_migrations: list[tuple] = []  # (creature, from_hab_id, dest_habitat)

        prev_speciation_count = len(self.species_registry.speciation_events)

        for hab_id, hab in self.habitats.items():
            result = hab.simulate_day(
                species_registry=self.species_registry,
                isolation_probability=iso_prob,
            )
            habitat_results[hab_id] = result
            for creature, dest_hab in result["migrations"]:
                pending_migrations.append((creature, hab_id, dest_hab))

        # Apply migrations after all habitats have run
        migration_log: list[dict] = []
        for creature, from_id, dest_hab in pending_migrations:
            dest_hab.add_creature(creature)
            migration_log.append({
                "creature_id": creature.creature_id,
                "species": creature.species,
                "from_habitat": from_id,
                "to_habitat": dest_hab.habitat_id,
            })

        new_speciations = self.species_registry.speciation_events[prev_speciation_count:]
        day_log = self._build_day_log(habitat_results, migration_log, new_speciations)
        self._write_day_log(day_log)
        return day_log

    # ------------------------------------------------------------------
    # Log construction
    # ------------------------------------------------------------------

    def _build_day_log(
        self,
        habitat_results: dict[str, dict],
        migration_log: list[dict],
        new_speciations: list[dict],
    ) -> dict:
        # Global tallies
        species_counts: Counter = Counter()
        total_pop = 0
        for hab in self.habitats.values():
            for c in hab.alive_creatures:
                species_counts[c.species] += 1
                total_pop += 1

        # ------------------------------------------------------------------
        # Habitat stats + species stats (computed after migrations applied)
        # ------------------------------------------------------------------
        habitat_stats: dict = {}
        # Accumulators for species-level aggregation
        sp_total: dict[str, int] = {}
        sp_hab_counts: dict[str, dict[str, int]] = {}
        sp_trait_sums: dict[str, dict[str, float]] = {}

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
                sp_total[sp_name] += n
                sp_hab_counts[sp_name][hab_id] = n
                for t in LOGGED_TRAITS:
                    sp_trait_sums[sp_name][t] += sp_data["mean_traits"][t] * n

        species_stats: dict = {}
        for sp_name, n in sp_total.items():
            species_stats[sp_name] = {
                "total_count": n,
                "habitat_counts": sp_hab_counts[sp_name],
                "mean_traits": {
                    t: round(sp_trait_sums[sp_name][t] / n, 4)
                    for t in LOGGED_TRAITS
                },
            }

        # ------------------------------------------------------------------
        # Per-habitat event log
        # ------------------------------------------------------------------
        habitats_log: dict = {}
        for hab_id, result in habitat_results.items():
            hab = self.habitats[hab_id]
            hab_species = Counter(c.species for c in hab.alive_creatures)

            # Combine resource/age deaths and predation deaths into one list
            all_death_ids = result["deaths"] + result.get("predation_deaths", [])
            deaths_detail = []
            for cid in all_death_ids:
                creature_log = result["day_results"].get(cid, {})
                deaths_detail.append({
                    "creature_id": cid,
                    "cause": creature_log.get("cause_of_death"),
                    "age": creature_log.get("age"),
                })

            births_detail = [
                {
                    "creature_id": c.creature_id,
                    "species": c.species,
                    "sex": c.sex,
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

            habitats_log[hab_id] = {
                "habitat_id": result["habitat_id"],
                "habitat_type": self.habitat_types.get(hab_id, "Unknown"),
                "habitat_name": hab.name,
                "population": result["population"],
                "species_distribution": dict(hab_species),
                "births": births_detail,
                "deaths": deaths_detail,
                "mating_events": result.get("mating_events", []),
                "migrations_out": migrations_out,
                "isolations": result["isolations"],
            }

        return {
            "day": self.day,
            "timestamp": datetime.now().isoformat(),
            "global_population": total_pop,
            "global_species_count": self.species_registry.species_count,
            "global_species_distribution": dict(species_counts),
            "speciation_events": new_speciations,
            "migrations": migration_log,
            "habitat_stats": habitat_stats,
            "species_stats": species_stats,
            "habitats": habitats_log,
        }

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _write_day_log(self, day_log: dict) -> None:
        path = self.log_dir / f"day_{self.day:05d}.json"
        with open(path, "w") as fh:
            json.dump(day_log, fh)

    def _write_metadata(self) -> None:
        sim_cfg = self.config["simulation"]
        metadata = {
            "simulation_start": datetime.now().isoformat(),
            "parameters": {
                "days": sim_cfg["days"],
                "seed": sim_cfg.get("seed"),
                "initial_species_per_habitat": sim_cfg.get("initial_species_per_habitat", 3),
                "creatures_per_species": sim_cfg.get("creatures_per_species", 10),
                "initial_genome_noise": sim_cfg.get("initial_genome_noise", 0.05),
                "isolation_probability": sim_cfg.get("isolation_probability", 0.001),
                "species_threshold": self.config.get("species", {}).get("threshold", 0.95),
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
        }
        with open(self.log_dir / "metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2)

    def _write_summary(self, extinct: bool = False) -> None:
        summary = {
            "simulation_end": datetime.now().isoformat(),
            "days_simulated": self.day,
            "extinct": extinct,
            "final_population": sum(h.population_size for h in self.habitats.values()),
            "total_species_ever": self.species_registry.species_count,
            "total_speciation_events": len(self.species_registry.speciation_events),
            "all_speciation_events": self.species_registry.speciation_events,
            "final_species_distribution": {
                hab_id: dict(Counter(c.species for c in hab.alive_creatures))
                for hab_id, hab in self.habitats.items()
            },
            "final_population_per_habitat": {
                hab_id: hab.population_size
                for hab_id, hab in self.habitats.items()
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
