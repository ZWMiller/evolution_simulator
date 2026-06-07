<a id="evolution_simulator.simulation"></a>

# evolution\_simulator.simulation

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

<a id="evolution_simulator.simulation.SimulationRunner"></a>

## SimulationRunner Objects

```python
class SimulationRunner()
```

Reads a TOML config, builds the simulation world, and drives the main loop.

Attributes
----------
config      : dict  – parsed TOML config
habitats    : dict[str, Habitat]  – keyed by habitat id
species_registry : SpeciesRegistry
week        : int   – current simulation week (0 before any steps)
log_dir     : Path  – output directory for this run (set by setup())

<a id="evolution_simulator.simulation.SimulationRunner.__init__"></a>

#### \_\_init\_\_

```python
def __init__(config_path: Path)
```

Parameters
----------
config_path : Path
    Path to the TOML configuration file.  The config is parsed
    immediately; call :meth:`setup` to build the world and
    :meth:`run` (or :meth:`step` in a loop) to simulate.

<a id="evolution_simulator.simulation.SimulationRunner.setup"></a>

#### setup

```python
def setup() -> Path
```

Initialise the simulation from config.

1. Creates the timestamped log directory and copies the config into it.
2. Builds Habitat instances and connects them.
3. Seeds each habitat with creatures grouped around founding genomes,
   ensuring both sexes are present and intra-group compatibility is high.
4. Registers each founding genome as a species in the SpeciesRegistry.
5. Writes metadata.json.

Returns the Path to the log directory.

<a id="evolution_simulator.simulation.SimulationRunner.run"></a>

#### run

```python
def run() -> None
```

Simulate all weeks specified in config, writing a log file per week.

Halts early if the global population reaches zero (extinction), in
which case summary.json records ``"extinct": true``.

<a id="evolution_simulator.simulation.SimulationRunner.step"></a>

#### step

```python
def step() -> dict
```

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
