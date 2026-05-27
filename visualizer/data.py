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

    weeks_data: dict[int, dict] = {}
    for p in sorted(log_dir.glob("week_*.json")):
        with open(p) as f:
            d = json.load(f)
        weeks_data[d["week"]] = d

    all_weeks = sorted(weeks_data.keys())
    all_species: set[str] = set()
    global_series: list[dict]                         = []
    hab_pop:       dict[str, dict[int, int]]          = {h: {} for h in hab_ids}
    species_global: dict[str, list[dict]]             = {}
    species_per_hab: dict[str, dict[str, list[dict]]] = {h: {} for h in hab_ids}
    migrations_by_week:    dict[int, list[dict]]         = {}
    migrations_by_edge:   dict[tuple, list[dict]]       = {}
    speciation_by_week:   dict[int, list[dict]]         = {}
    hybridization_by_week: dict[int, list[dict]]        = {}

    # creature_births: creature_id → {week, species, parents, sex, generation, hab_id}
    creature_births: dict[str, dict] = {}

    from visualizer.style import ALL_TRAITS  # avoid circular at module level

    # Seed creature_births with founders from metadata (generation 0)
    for hab_id, founders in metadata.get("founders_by_hab", {}).items():
        for f in founders:
            creature_births[f["creature_id"]] = {
                "week": 0,
                "species": f["species"],
                "parents": [],
                "sex": f["sex"],
                "generation": 0,
                "hab_id": hab_id,
            }

    for week_n in all_weeks:
        d = weeks_data[week_n]
        all_species.update(d.get("global_species_distribution", {}).keys())

        global_series.append({
            "week":          week_n,
            "population":    d["global_population"],
            "species_count": d["global_species_count"],
        })

        for hab_id in hab_ids:
            hab_log = d.get("habitats", {}).get(hab_id, {})
            hab_pop[hab_id][week_n] = hab_log.get("population", 0)

        for sp, sp_data in d.get("species_stats", {}).items():
            row: dict = {
                "week": week_n,
                "count": sp_data["total_count"],
                "mean_generation": sp_data.get("mean_generation"),
            }
            row.update(sp_data.get("mean_traits", {}))
            species_global.setdefault(sp, []).append(row)

        for hab_id, hab_data in d.get("habitat_stats", {}).items():
            for sp, sp_data in hab_data.get("by_species", {}).items():
                row = {
                    "week":             week_n,
                    "count":            sp_data["count"],
                    "mean_food_prob":   sp_data.get("mean_food_prob"),
                    "mean_water_prob":  sp_data.get("mean_water_prob"),
                    "mean_generation":  sp_data.get("mean_generation"),
                }
                row.update(sp_data.get("mean_traits", {}))
                species_per_hab.setdefault(hab_id, {}).setdefault(sp, []).append(row)

        migs = d.get("migrations", [])
        migrations_by_week[week_n] = migs
        for m in migs:
            key = (m["from_habitat"], m["to_habitat"])
            migrations_by_edge.setdefault(key, []).append({**m, "week": week_n})

        speciation_by_week[week_n] = d.get("speciation_events", [])

        hybrids = []
        for hab_id in hab_ids:
            for ev in d.get("habitats", {}).get(hab_id, {}).get("mating_events", []):
                if "hybridization" in ev:
                    hybrids.append({**ev["hybridization"], "hab_id": hab_id})
        hybridization_by_week[week_n] = hybrids

        # Index births for family tree
        for hab_id in hab_ids:
            hab_log = d.get("habitats", {}).get(hab_id, {})
            for birth in hab_log.get("births", []):
                creature_births[birth["creature_id"]] = {
                    "week": week_n,
                    "species": birth["species"],
                    "parents": birth.get("parents", []),
                    "sex": birth.get("sex", "?"),
                    "generation": birth.get("generation", 1),
                    "hab_id": hab_id,
                }

    # ── Species lineage (for phylogeny) ───────────────────────────────────────
    # Always built from summary.json so the phylogeny is complete regardless
    # of whether per-week speciation_events were logged (events_every cadence).
    all_speciation_events: list[dict] = summary.get("all_speciation_events", [])
    failed_speciation_attempts: list[dict] = summary.get("all_failed_speciation_attempts", [])
    speciated_names = {ev["new_species"] for ev in all_speciation_events}

    founding_species: list[str] = metadata.get(
        "founding_species",
        [sp for sp in all_species if sp not in speciated_names],
    )

    species_lineage: dict[str, dict] = {}
    for sp in founding_species:
        species_lineage[sp] = {"parent": None, "week": 0, "children": []}

    for ev in all_speciation_events:
        new_sp = ev["new_species"]
        parent_sp = ev.get("parent_species")
        species_lineage[new_sp] = {
            "parent": parent_sp,
            "week": ev.get("week", 0),
            "creature_id": ev.get("creature_id"),
            # "cladogenesis" (a split) vs "anagenesis" (in-place transformation).
            "event_type": ev.get("event_type", "cladogenesis"),
            "children": [],
        }
        if parent_sp and parent_sp in species_lineage:
            species_lineage[parent_sp]["children"].append(new_sp)

    # Ensure any species without lineage info is treated as a root
    for sp in all_species:
        if sp not in species_lineage:
            species_lineage[sp] = {"parent": None, "week": 0, "children": []}

    # Reverse parent→children index (for descendant traversal in family tree)
    creature_children: dict[str, list[str]] = {}
    for cid, cdata in creature_births.items():
        for parent_id in cdata.get("parents", []):
            creature_children.setdefault(parent_id, []).append(cid)

    # Peak population per species (across all weeks)
    species_peak_population: dict[str, int] = {
        sp: max((r["count"] for r in rows), default=0)
        for sp, rows in species_global.items()
    }

    # Lifespan per species: weeks from first appearance to last logged presence.
    # Species still alive at run end get credit through weeks_simulated.
    weeks_simulated = summary["weeks_simulated"]
    final_dist = summary.get("final_species_distribution", {})
    final_living = {sp for hab_dist in final_dist.values() for sp in hab_dist}
    species_lifespan: dict[str, int] = {}
    for sp, data in species_lineage.items():
        birth = data.get("week", 0)
        if sp in final_living:
            species_lifespan[sp] = weeks_simulated - birth
        elif sp in species_global:
            last_seen = max(r["week"] for r in species_global[sp] if r["count"] > 0)
            species_lifespan[sp] = last_seen - birth
        else:
            species_lifespan[sp] = 0

    node_positions = _spring_layout(hab_ids, connections)

    return {
        "run_id":                  log_dir.name,
        "metadata":                metadata,
        "summary":                 summary,
        "config_text":             config_text,
        "config_name":             config_name,
        "hab_ids":                 hab_ids,
        "hab_names":               hab_names,
        "hab_types":               hab_types,
        "connections":             connections,
        "all_weeks":               all_weeks,
        "all_species":             sorted(all_species),
        "weeks_data":              weeks_data,
        "global_series":           global_series,
        "hab_pop":                 hab_pop,
        "species_global":          species_global,
        "species_per_hab":         species_per_hab,
        "migrations_by_week":       migrations_by_week,
        "migrations_by_edge":      migrations_by_edge,
        "speciation_by_week":      speciation_by_week,
        "hybridization_by_week":   hybridization_by_week,
        "creature_births":         creature_births,
        "creature_children":       creature_children,
        "species_lineage":              species_lineage,
        "species_peak_population":      species_peak_population,
        "species_lifespan":             species_lifespan,
        "failed_speciation_attempts":   failed_speciation_attempts,
        "node_positions":               node_positions,
        "weeks_simulated":              summary["weeks_simulated"],
        "extinct":                      summary["extinct"],
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
    runs = sorted(base.glob("*/week_00001.json"))
    if not runs:
        raise FileNotFoundError(f"No simulation run found under {base}")
    return runs[-1].parent
