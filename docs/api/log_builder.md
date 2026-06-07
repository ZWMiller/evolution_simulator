<a id="evolution_simulator.log_builder"></a>

# evolution\_simulator.log\_builder

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

<a id="evolution_simulator.log_builder.StatsAccumulator"></a>

## StatsAccumulator Objects

```python
class StatsAccumulator()
```

Buffers interval statistics between logging flushes.

``SimulationRunner.step`` calls :meth:`record_week` every week.  When the
stats logging cadence fires, :meth:`flush_interval` returns the
``interval_stats`` dict and resets the buffer to start a fresh interval at
the next week.

<a id="evolution_simulator.log_builder.StatsAccumulator.reset"></a>

#### reset

```python
def reset(start_week: int) -> None
```

Clear all counters and begin a fresh accumulation interval.

Parameters
----------
start_week : int
    The first week of the new interval; stored as ``self.start_week``
    and included in the ``weeks_covered`` field of the next
    ``flush_interval()`` output.

<a id="evolution_simulator.log_builder.StatsAccumulator.record_week"></a>

#### record\_week

```python
def record_week(week: int, habitat_results: dict,
                migration_log: list) -> list[dict]
```

Accumulate one week of habitat results and migrations into the buffer.

Parameters
----------
week : int
    The current simulation week number, stamped on any isolation events
    created this call.
habitat_results : dict[str, dict]
    Mapping of ``habitat_id → result`` as returned by
    ``Habitat.simulate_week()``.  Deaths, births, mating events, and
    isolation events are extracted from each result.
migration_log : list[dict]
    List of migration event dicts (``creature_id``, ``from_habitat``,
    ``to_habitat``) produced by ``SimulationRunner.step()``.

Returns
-------
list[dict]
    Isolation-event dicts created this call (``week``,
    ``from_habitat``, ``blocked_neighbor``), so the caller can also
    append them to its all-time isolation log.

<a id="evolution_simulator.log_builder.StatsAccumulator.build_interval_stats"></a>

#### build\_interval\_stats

```python
def build_interval_stats(end_week: int) -> dict
```

Assemble the ``interval_stats`` dict without resetting the buffer.

Parameters
----------
end_week : int
    The last week covered by this interval; stored alongside
    ``self.start_week`` in the ``weeks_covered`` field.

Returns
-------
dict
    Interval summary with top-level keys: ``weeks_covered``,
    ``deaths``, ``births``, ``migrations``, ``mating``, and
    ``isolation_events``.

<a id="evolution_simulator.log_builder.StatsAccumulator.flush_interval"></a>

#### flush\_interval

```python
def flush_interval(end_week: int) -> dict
```

Build the interval_stats dict, then reset the buffer.

Parameters
----------
end_week : int
    The last week of the interval being closed; passed to
    ``build_interval_stats()``.  The buffer is then reset so the
    next interval starts at ``end_week + 1``.

Returns
-------
dict
    The completed interval stats dict; see ``build_interval_stats()``.
