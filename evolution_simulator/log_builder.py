"""
Log construction helpers for ``SimulationRunner``.

Currently houses ``StatsAccumulator`` — the interval-statistics buffer that
collects per-week births/deaths/migrations/matings/isolations between logging
flushes.  It was previously a dozen loose ``_bin_*`` Counter/int attributes on
``SimulationRunner``; gathering them into one object with ``record_week`` /
``flush_interval`` / ``reset`` keeps the runner focused on orchestration and
makes the accumulation logic testable in isolation.

(The per-week log *assembly* and file I/O — ``_build_week_log`` / ``_write_*`` —
remain on ``SimulationRunner``: they read pervasive runner state, so relocating
them here would couple this module back to the runner rather than decoupling.)
"""

from collections import Counter


class StatsAccumulator:
    """
    Buffers interval statistics between logging flushes.

    ``SimulationRunner.step`` calls :meth:`record_week` every week.  When the
    stats logging cadence fires, :meth:`flush_interval` returns the
    ``interval_stats`` dict and resets the buffer to start a fresh interval at
    the next week.
    """

    def __init__(self, start_week: int = 1) -> None:
        self.reset(start_week)

    def reset(self, start_week: int) -> None:
        """Clear all counters and begin a fresh interval at ``start_week``."""
        self.start_week: int = start_week
        self.deaths_by_cause: Counter = Counter()
        self.deaths_by_species: Counter = Counter()
        self.births_by_habitat: Counter = Counter()
        self.births_by_species: Counter = Counter()
        self.births_by_habitat_species: Counter = Counter()  # (hab_id, species) → count
        self.migrations_by_route: Counter = Counter()  # (from_id, to_id) → count
        self.mating_attempts: int = 0
        self.fertilizations: int = 0
        self.hybrid_conceptions: int = 0
        self.isolation_events: list[dict] = []

    def record_week(self, week: int, habitat_results: dict, migration_log: list) -> list[dict]:
        """
        Accumulate one week of habitat results + migrations into the buffer.

        Returns the list of isolation-event dicts created this call, so the
        caller can also append them to its all-time isolation log.
        """
        new_isolation_events: list[dict] = []
        for hab_id, result in habitat_results.items():
            for cid in result.get("deaths", []) + result.get("predation_deaths", []):
                wr = result.get("week_results", {}).get(cid, {})
                self.deaths_by_cause[wr.get("cause_of_death", "unknown")] += 1
                self.deaths_by_species[wr.get("species", "unknown")] += 1
            for creature in result.get("births", []):
                self.births_by_habitat[hab_id] += 1
                self.births_by_species[creature.species] += 1
                self.births_by_habitat_species[(hab_id, creature.species)] += 1
            for ev in result.get("mating_events", []):
                self.mating_attempts += 1
                if ev.get("fertilized"):
                    self.fertilizations += 1
                if "hybridization" in ev:
                    self.hybrid_conceptions += 1
            for blocked_neighbor in result.get("isolations", []):
                event = {
                    "week": week,
                    "from_habitat": hab_id,
                    "blocked_neighbor": blocked_neighbor,
                }
                self.isolation_events.append(event)
                new_isolation_events.append(event)
        for entry in migration_log:
            self.migrations_by_route[(entry["from_habitat"], entry["to_habitat"])] += 1
        return new_isolation_events

    def build_interval_stats(self, end_week: int) -> dict:
        """Assemble the ``interval_stats`` dict covering [start_week, end_week]."""
        births_by_hab_species: dict = {}
        for (hid, sp), count in self.births_by_habitat_species.items():
            births_by_hab_species.setdefault(hid, {})[sp] = count

        return {
            "weeks_covered": [self.start_week, end_week],
            "deaths": {
                "total": sum(self.deaths_by_cause.values()),
                "by_cause": dict(self.deaths_by_cause),
                "by_species": dict(self.deaths_by_species),
            },
            "births": {
                "total": sum(self.births_by_habitat.values()),
                "by_habitat": dict(self.births_by_habitat),
                "by_species": dict(self.births_by_species),
                "by_habitat_and_species": births_by_hab_species,
            },
            "migrations": {
                "total": sum(self.migrations_by_route.values()),
                "by_route": {f"{f}→{t}": n for (f, t), n in self.migrations_by_route.items()},
            },
            "mating": {
                "total_pairings": self.mating_attempts,
                "fertilizations": self.fertilizations,
                "hybrid_conceptions": self.hybrid_conceptions,
            },
            "isolation_events": self.isolation_events,
        }

    def flush_interval(self, end_week: int) -> dict:
        """Build the interval_stats dict, then reset to start at ``end_week + 1``."""
        stats = self.build_interval_stats(end_week)
        self.reset(end_week + 1)
        return stats
