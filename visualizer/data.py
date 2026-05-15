"""
Load a completed simulation run from its log directory into a structured dict
that every other module can read without re-parsing files.
"""

import json
import math
import random
from pathlib import Path


def load_run(log_dir: Path) -> dict:
    log_dir = Path(log_dir)

    with open(log_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(log_dir / "summary.json") as f:
        summary = json.load(f)

    config_text = ""
    config_name = "N/A"
    cfg_path = log_dir / "config.toml"
    if cfg_path.exists():
        config_text = cfg_path.read_text()
        config_name = cfg_path.name

    hab_ids     = [h["id"]   for h in metadata["habitats"]]
    hab_names   = {h["id"]: h["name"] for h in metadata["habitats"]}
    hab_types   = {h["id"]: h["type"] for h in metadata["habitats"]}
    connections = metadata.get("connections", [])

    days_data: dict[int, dict] = {}
    for p in sorted(log_dir.glob("day_*.json")):
        with open(p) as f:
            d = json.load(f)
        days_data[d["day"]] = d

    all_days = sorted(days_data.keys())
    all_species: set[str] = set()
    global_series: list[dict]                         = []
    hab_pop:       dict[str, dict[int, int]]          = {h: {} for h in hab_ids}
    species_global: dict[str, list[dict]]             = {}
    species_per_hab: dict[str, dict[str, list[dict]]] = {h: {} for h in hab_ids}
    migrations_by_day:  dict[int, list[dict]]         = {}
    migrations_by_edge: dict[tuple, list[dict]]       = {}
    speciation_by_day:  dict[int, list[dict]]         = {}

    from visualizer.style import ALL_TRAITS  # avoid circular at module level

    for day_n in all_days:
        d = days_data[day_n]
        all_species.update(d.get("global_species_distribution", {}).keys())

        global_series.append({
            "day":           day_n,
            "population":    d["global_population"],
            "species_count": d["global_species_count"],
        })

        for hab_id in hab_ids:
            hab_log = d.get("habitats", {}).get(hab_id, {})
            hab_pop[hab_id][day_n] = hab_log.get("population", 0)

        for sp, sp_data in d.get("species_stats", {}).items():
            row: dict = {"day": day_n, "count": sp_data["total_count"]}
            row.update(sp_data.get("mean_traits", {}))
            species_global.setdefault(sp, []).append(row)

        for hab_id, hab_data in d.get("habitat_stats", {}).items():
            for sp, sp_data in hab_data.get("by_species", {}).items():
                row = {
                    "day":             day_n,
                    "count":           sp_data["count"],
                    "mean_food_prob":  sp_data.get("mean_food_prob"),
                    "mean_water_prob": sp_data.get("mean_water_prob"),
                }
                row.update(sp_data.get("mean_traits", {}))
                species_per_hab.setdefault(hab_id, {}).setdefault(sp, []).append(row)

        migs = d.get("migrations", [])
        migrations_by_day[day_n] = migs
        for m in migs:
            key = (m["from_habitat"], m["to_habitat"])
            migrations_by_edge.setdefault(key, []).append({**m, "day": day_n})

        speciation_by_day[day_n] = d.get("speciation_events", [])

    node_positions = _spring_layout(hab_ids, connections)

    return {
        "run_id":              log_dir.name,
        "metadata":            metadata,
        "summary":             summary,
        "config_text":         config_text,
        "config_name":         config_name,
        "hab_ids":             hab_ids,
        "hab_names":           hab_names,
        "hab_types":           hab_types,
        "connections":         connections,
        "all_days":            all_days,
        "all_species":         sorted(all_species),
        "days_data":           days_data,
        "global_series":       global_series,
        "hab_pop":             hab_pop,
        "species_global":      species_global,
        "species_per_hab":     species_per_hab,
        "migrations_by_day":   migrations_by_day,
        "migrations_by_edge":  migrations_by_edge,
        "speciation_by_day":   speciation_by_day,
        "node_positions":      node_positions,
        "days_simulated":      summary["days_simulated"],
        "extinct":             summary["extinct"],
    }


def _spring_layout(hab_ids: list, connections: list, w: int = 560, h: int = 420) -> dict:
    n = len(hab_ids)
    pos = {
        hab: [
            w / 2 + (w * 0.38) * math.cos(2 * math.pi * i / n),
            h / 2 + (h * 0.38) * math.sin(2 * math.pi * i / n),
        ]
        for i, hab in enumerate(hab_ids)
    }
    conn_set = {(a, b) for a, b in connections} | {(b, a) for a, b in connections}

    for _ in range(200):
        for hab in hab_ids:
            fx = fy = 0.0
            for other in hab_ids:
                if other == hab:
                    continue
                dx = pos[hab][0] - pos[other][0]
                dy = pos[hab][1] - pos[other][1]
                d2 = max(1.0, dx * dx + dy * dy)
                d  = d2 ** 0.5
                fx += dx / d2 * 4000
                fy += dy / d2 * 4000
                if (hab, other) in conn_set:
                    fx -= dx / d * 1.0
                    fy -= dy / d * 1.0
            pos[hab][0] = max(80, min(w - 80, pos[hab][0] + fx * 0.008))
            pos[hab][1] = max(80, min(h - 80, pos[hab][1] + fy * 0.008))

    return {hab: {"x": pos[hab][0], "y": pos[hab][1]} for hab in hab_ids}


def latest_run(base: Path) -> Path:
    runs = sorted(base.glob("*/day_00001.json"))
    if not runs:
        raise FileNotFoundError(f"No simulation run found under {base}")
    return runs[-1].parent
