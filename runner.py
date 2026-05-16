"""
runner.py — stand-alone simulation entry point.

Usage
-----
    python runner.py                           # use bundled default config
    python runner.py path/to/my_config.toml   # use a custom config
    python runner.py --weeks 100              # override the number of weeks

The script prints live progress to stdout and writes structured JSON logs to
the output directory specified in the config (default: simulation_logs/).
"""

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolution Simulator — run a genetic evolution simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py
  python runner.py my_config.toml
  python runner.py my_config.toml --days 500
""",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to a TOML config file.  Defaults to the bundled simulation.toml.",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=None,
        help="Override the number of weeks specified in the config.",
    )
    args = parser.parse_args()

    # Resolve config path
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[error] Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
    else:
        config_path = (
            Path(__file__).parent
            / "evolution_simulator"
            / "config"
            / "simulation.toml"
        )

    # Import here so startup errors surface cleanly
    from evolution_simulator.simulation import SimulationRunner

    runner = SimulationRunner(config_path)

    # Allow --weeks to override config without mutating the file
    if args.weeks is not None:
        runner.config["simulation"]["weeks"] = args.weeks

    weeks = runner.config["simulation"]["weeks"]

    print()
    print("  Evolution Simulator")
    print(f"  Config : {config_path}")
    print(f"  Weeks  : {weeks}")
    print()

    log_dir = runner.setup()
    print(f"  Log dir: {log_dir}")
    print(f"  Habitats : {len(runner.habitats)}")
    print(
        f"  Founding species : {runner.species_registry.species_count}"
    )
    total_pop = sum(h.population_size for h in runner.habitats.values())
    print(f"  Starting population : {total_pop}")
    print()
    print(f"  {'Week':>6}  {'Pop':>7}  {'Species':>8}  {'Births':>7}  {'Deaths':>7}  {'Time/wk':>9}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*9}")

    total_births = 0
    total_deaths = 0
    extinct = False

    for _ in range(weeks):
        t0 = time.perf_counter()
        week_log = runner.step()
        elapsed = time.perf_counter() - t0

        week_births = sum(len(h["births"]) for h in week_log["habitats"].values())
        week_deaths = sum(len(h["deaths"]) for h in week_log["habitats"].values())
        total_births += week_births
        total_deaths += week_deaths

        # Print a summary row every 10 weeks (and always on week 1)
        if runner.week == 1 or runner.week % 10 == 0 or runner.week == weeks:
            print(
                f"  {runner.week:>6}  "
                f"{week_log['global_population']:>7}  "
                f"{week_log['global_species_count']:>8}  "
                f"{total_births:>7}  "
                f"{total_deaths:>7}  "
                f"{elapsed*1000:>7.1f}ms"
            )

        if week_log["global_population"] == 0:
            extinct = True
            print(f"\n  !! Global extinction on week {runner.week} — halting early.")
            break

    runner._write_summary(extinct=extinct)

    print()
    print("  ── Simulation complete ──────────────────────────────")
    print(f"  Weeks simulated  : {runner.week}")
    print(f"  Outcome          : {'EXTINCTION' if extinct else 'completed'}")
    print(f"  Final population : {sum(h.population_size for h in runner.habitats.values())}")
    print(f"  Total species    : {runner.species_registry.species_count}")
    print(f"  Speciation events: {len(runner.species_registry.speciation_events)}")
    print(f"  Total births     : {total_births}")
    print(f"  Total deaths     : {total_deaths}")
    print(f"  Logs written to  : {log_dir}")
    print()


if __name__ == "__main__":
    main()
