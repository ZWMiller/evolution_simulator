"""
Evolution Simulator — advanced interactive log visualiser.

Usage
-----
    poetry run python visualizer_advanced.py
    poetry run python visualizer_advanced.py simulation_logs/2026-05-14_21-51-40
    poetry run python visualizer_advanced.py --port 8052

Opens a Dash app at http://127.0.0.1:8052
"""

import argparse
from pathlib import Path

from visualizer import load_run, latest_run, make_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced evolution simulator log visualiser")
    parser.add_argument(
        "log_dir", nargs="?", default=None,
        help="Path to a simulation log directory (default: most recent in simulation_logs/)",
    )
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else latest_run(Path("simulation_logs"))
    print(f"Loading: {log_dir}", flush=True)

    run = load_run(log_dir)
    print(
        f"  {run['days_simulated']} days  |  "
        f"{run['summary']['total_species_ever']} species  |  "
        f"{'EXTINCT' if run['extinct'] else 'alive'}",
        flush=True,
    )

    app = make_app(run)
    print(f"\nOpen http://{args.host}:{args.port}/\n", flush=True)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
