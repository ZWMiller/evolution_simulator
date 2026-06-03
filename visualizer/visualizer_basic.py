"""
Evolution Simulator — basic interactive log visualiser.

Usage
-----
    poetry run python visualizer/visualizer_basic.py
    poetry run python visualizer/visualizer_basic.py simulation_logs/2026-05-14_21-51-40
    poetry run python visualizer/visualizer_basic.py --port 8051

Opens a Dash web app at http://127.0.0.1:8050
"""

import argparse
import json
from pathlib import Path

import dash
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

ALL_TRAITS = [
    "fecundity",
    "reproduction_time",
    "weeks_to_sexual_viability",
    "parental_investment",
    "reproduction_likelihood",
    "metabolism",
    "water_efficiency",
    "max_lifespan",
    "disease_resistance",
    "immune_response",
    "stress_tolerance",
    "heat_tolerance",
    "cold_tolerance",
    "drought_tolerance",
    "hibernation_tendency",
    "migration_likelihood",
    "risk_tolerance",
    "aggression",
    "territorial",
    "social_tendency",
    "nocturnal_tendency",
    "size",
    "strength",
    "speed",
    "camouflage",
    "foraging_ability",
    "intelligence",
    "adaptability",
    "pack_hunting",
    "scavenging_tendency",
    "communication",
    "mutation_rate",
    "selectivity",
    "base_predation_rate",
]

TRAIT_OPTIONS = [{"label": t.replace("_", " ").title(), "value": t} for t in ALL_TRAITS]


def load_run(log_dir: Path) -> dict:
    """Load all JSON files from a simulation run directory into memory."""
    log_dir = Path(log_dir)

    with open(log_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(log_dir / "summary.json") as f:
        summary = json.load(f)

    week_files = sorted(log_dir.glob("week_*.json"))
    days = []
    for p in week_files:
        with open(p) as f:
            days.append(json.load(f))

    habitat_ids = [h["id"] for h in metadata["habitats"]]
    habitat_names = {h["id"]: h["name"] for h in metadata["habitats"]}
    habitat_types = {h["id"]: h["type"] for h in metadata["habitats"]}

    all_species: set[str] = set()
    for d in days:
        all_species.update(d.get("global_species_distribution", {}).keys())

    global_series = []
    habitat_series = []
    species_global = []
    species_per_hab = []
    speciation_events = []

    for d in days:
        week_n = d["week"]

        global_series.append(
            {
                "week": week_n,
                "population": d["global_population"],
                "species_count": d["global_species_count"],
            }
        )

        for ev in d.get("speciation_events", []):
            speciation_events.append(
                {
                    "week": week_n,
                    "new_species": ev["new_species"],
                    "parent_species": ev["parent_species"],
                    "event_type": ev.get("event_type", "cladogenesis"),
                }
            )

        for hab_id in habitat_ids:
            hab_log = d.get("habitats", {}).get(hab_id, {})
            habitat_series.append(
                {
                    "week": week_n,
                    "hab_id": hab_id,
                    "hab_name": habitat_names.get(hab_id, hab_id),
                    "population": hab_log.get("population", 0),
                }
            )

        for sp_name, sp_data in d.get("species_stats", {}).items():
            row = {
                "week": week_n,
                "species": sp_name,
                "total_count": sp_data["total_count"],
            }
            for t in ALL_TRAITS:
                row[t] = sp_data["mean_traits"].get(t)
            species_global.append(row)

        for hab_id, hab_data in d.get("habitat_stats", {}).items():
            for sp_name, sp_data in hab_data.get("by_species", {}).items():
                row = {
                    "week": week_n,
                    "hab_id": hab_id,
                    "hab_name": habitat_names.get(hab_id, hab_id),
                    "species": sp_name,
                    "count": sp_data["count"],
                    "mean_food_prob": sp_data.get("mean_food_prob"),
                    "mean_water_prob": sp_data.get("mean_water_prob"),
                }
                for t in ALL_TRAITS:
                    row[t] = sp_data["mean_traits"].get(t)
                species_per_hab.append(row)

    last_data_week = max((r["week"] for r in species_global), default=1)

    # Per-trait global min/max across all weeks and species — used by the heatmap
    # so that each cell's color reflects absolute position in the trait's full
    # historical range, not relative rank among species on a single week.
    trait_global_ranges: dict[str, tuple[float, float]] = {}
    for t in ALL_TRAITS:
        vals = [r[t] for r in species_global if r.get(t) is not None]
        if vals:
            trait_global_ranges[t] = (min(vals), max(vals))
        else:
            trait_global_ranges[t] = (0.0, 1.0)

    # Interval / bin stats: keyed by the logged week the snapshot covers up to.
    interval_stats_by_week: dict[int, dict] = {}
    for d in days:
        if "interval_stats" in d:
            interval_stats_by_week[d["week"]] = d["interval_stats"]

    return {
        "metadata": metadata,
        "summary": summary,
        "habitat_ids": habitat_ids,
        "habitat_names": habitat_names,
        "habitat_types": habitat_types,
        "all_species": sorted(all_species),
        "global_series": global_series,
        "habitat_series": habitat_series,
        "species_global": species_global,
        "species_per_hab": species_per_hab,
        "speciation_events": speciation_events,
        "weeks_simulated": summary["weeks_simulated"],
        "last_data_week": last_data_week,
        "trait_global_ranges": trait_global_ranges,
        "extinct": summary["extinct"],
        "interval_stats_by_week": interval_stats_by_week,
        "weeks_with_interval_stats": sorted(interval_stats_by_week.keys()),
    }


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _speciation_shapes(events: list[dict], y0: float = 0, y1: float = 1, yref: str = "paper"):
    shapes, annotations = [], []
    for ev in events:
        # Anagenesis (in-place transformation) is drawn amber to distinguish it
        # from cladogenesis (a split), which stays neutral grey.
        is_ana = ev.get("event_type") == "anagenesis"
        line_color = "rgba(232,162,61,0.55)" if is_ana else "rgba(150,150,150,0.4)"
        text_color = "rgba(232,162,61,0.8)" if is_ana else "rgba(120,120,120,0.7)"
        shapes.append(
            dict(
                type="line",
                x0=ev["week"],
                x1=ev["week"],
                y0=y0,
                y1=y1,
                yref=yref,
                line=dict(color=line_color, width=1, dash="dot"),
            )
        )
        annotations.append(
            dict(
                x=ev["week"],
                y=y1,
                yref=yref,
                xanchor="left",
                text=ev["new_species"],
                showarrow=False,
                font=dict(size=8, color=text_color),
                textangle=-60,
            )
        )
    return shapes, annotations


def fig_global_overview(run: dict) -> go.Figure:
    gs = run["global_series"]
    weeks = [r["week"] for r in gs]
    pop = [r["population"] for r in gs]
    sp_count = [r["species_count"] for r in gs]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=pop,
            name="Global Population",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.1)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=sp_count,
            name="Species Count",
            line=dict(color="#FF9800", width=2, dash="dash"),
            yaxis="y2",
        )
    )

    shapes, annotations = _speciation_shapes(run["speciation_events"])
    fig.update_layout(
        title="Global Population & Species Richness",
        xaxis_title="Week",
        yaxis=dict(title="Population", rangemode="tozero"),
        yaxis2=dict(title="Species Count", overlaying="y", side="right", rangemode="tozero"),
        hovermode="x unified",
        shapes=shapes,
        annotations=annotations,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=40, b=40),
    )
    return fig


def fig_habitat_population(run: dict) -> go.Figure:
    hs = run["habitat_series"]
    hab_ids = run["habitat_ids"]
    hab_names = run["habitat_names"]
    colors = px.colors.qualitative.Set2

    by_hab: dict = {h: {"weeks": [], "pop": []} for h in hab_ids}
    for r in hs:
        by_hab[r["hab_id"]]["weeks"].append(r["week"])
        by_hab[r["hab_id"]]["pop"].append(r["population"])

    fig = go.Figure()
    for i, hid in enumerate(hab_ids):
        fig.add_trace(
            go.Scatter(
                x=by_hab[hid]["weeks"],
                y=by_hab[hid]["pop"],
                name=f"{hab_names[hid]} ({run['habitat_types'][hid]})",
                line=dict(color=colors[i % len(colors)], width=2),
                stackgroup="one",
                fill="tonexty",
            )
        )

    shapes, annotations = _speciation_shapes(run["speciation_events"])
    fig.update_layout(
        title="Population per Habitat (stacked)",
        xaxis_title="Week",
        yaxis_title="Population",
        hovermode="x unified",
        shapes=shapes,
        annotations=annotations,
        margin=dict(t=40, b=40),
    )
    return fig


def fig_species_population(run: dict, selected_species: list[str]) -> go.Figure:
    if not selected_species:
        return go.Figure().update_layout(title="Select at least one species above")

    by_sp: dict = {}
    for r in run["species_global"]:
        if r["species"] not in selected_species:
            continue
        sp = r["species"]
        if sp not in by_sp:
            by_sp[sp] = {"weeks": [], "count": []}
        by_sp[sp]["weeks"].append(r["week"])
        by_sp[sp]["count"].append(r["total_count"])

    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, sp in enumerate(selected_species):
        if sp not in by_sp:
            continue
        fig.add_trace(
            go.Scatter(
                x=by_sp[sp]["weeks"],
                y=by_sp[sp]["count"],
                name=sp,
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    shapes, annotations = _speciation_shapes(run["speciation_events"])
    fig.update_layout(
        title="Species Population Over Time",
        xaxis_title="Week",
        yaxis_title="Population",
        hovermode="x unified",
        shapes=shapes,
        annotations=annotations,
        margin=dict(t=40, b=40),
    )
    return fig


def fig_species_trait(run: dict, selected_species: list[str], trait: str) -> go.Figure:
    if not selected_species or not trait:
        return go.Figure().update_layout(title="Select species and trait above")

    by_sp: dict = {}
    for r in run["species_global"]:
        if r["species"] not in selected_species:
            continue
        if r.get(trait) is None:
            continue
        sp = r["species"]
        if sp not in by_sp:
            by_sp[sp] = {"weeks": [], "vals": []}
        by_sp[sp]["weeks"].append(r["week"])
        by_sp[sp]["vals"].append(r[trait])

    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, sp in enumerate(selected_species):
        if sp not in by_sp:
            continue
        fig.add_trace(
            go.Scatter(
                x=by_sp[sp]["weeks"],
                y=by_sp[sp]["vals"],
                name=sp,
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    fig.update_layout(
        title=f"Mean {trait.replace('_', ' ').title()} Over Time",
        xaxis_title="Week",
        yaxis_title=trait.replace("_", " ").title(),
        hovermode="x unified",
        margin=dict(t=40, b=40),
    )
    return fig


def fig_adaptation_by_habitat(run: dict, hab_id: str) -> go.Figure:
    """Food and water adaptation probability over time for all species in one habitat."""
    rows = [r for r in run["species_per_hab"] if r["hab_id"] == hab_id]
    if not rows:
        return go.Figure().update_layout(title="No data for this habitat")

    sp_food: dict = {}
    sp_water: dict = {}
    for r in rows:
        sp = r["species"]
        if sp not in sp_food:
            sp_food[sp] = {"weeks": [], "vals": []}
            sp_water[sp] = {"weeks": [], "vals": []}
        if r.get("mean_food_prob") is not None:
            sp_food[sp]["weeks"].append(r["week"])
            sp_food[sp]["vals"].append(r["mean_food_prob"])
        if r.get("mean_water_prob") is not None:
            sp_water[sp]["weeks"].append(r["week"])
            sp_water[sp]["vals"].append(r["mean_water_prob"])

    colors = px.colors.qualitative.Plotly
    species_list = sorted(sp_food.keys())

    fig = go.Figure()
    for i, sp in enumerate(species_list):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=sp_food[sp]["weeks"],
                y=sp_food[sp]["vals"],
                name=f"{sp} (food)",
                line=dict(color=color, width=1.5),
                legendgroup=sp,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sp_water[sp]["weeks"],
                y=sp_water[sp]["vals"],
                name=f"{sp} (water)",
                line=dict(color=color, width=1.5, dash="dot"),
                legendgroup=sp,
                showlegend=False,
            )
        )

    hab_name = run["habitat_names"].get(hab_id, hab_id)
    hab_type = run["habitat_types"].get(hab_id, "")
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="unadapted baseline (0.5)")
    fig.update_layout(
        title=f"Habitat Adaptation: {hab_name} ({hab_type})  — solid=food, dotted=water",
        xaxis_title="Week",
        yaxis=dict(title="Resource Discovery Probability", range=[0, 1]),
        hovermode="x unified",
        margin=dict(t=50, b=40),
    )
    return fig


def fig_rk_tradeoff(run: dict, week: int) -> go.Figure:
    """Scatter of mean fecundity vs mean base_predation_rate at a given week."""
    rows = [r for r in run["species_global"] if r["week"] == week]
    if not rows:
        return go.Figure().update_layout(title=f"No species data for week {week}")

    species = [r["species"] for r in rows]
    fecundity = [r.get("fecundity") or 0 for r in rows]
    pred_rate = [r.get("base_predation_rate") or 0 for r in rows]
    size = [max(5, r["total_count"]) for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fecundity,
            y=pred_rate,
            mode="markers+text",
            text=species,
            textposition="top center",
            marker=dict(
                size=[s**0.5 * 3 for s in size],
                color=size,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Population"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Fecundity: %{x:.3f}<br>"
                "Predation Rate: %{y:.5f}<br>"
                "Population: %{marker.color}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"r/K Tradeoff — Week {week}  (bubble size ∝ √population)",
        xaxis_title="Mean Fecundity",
        yaxis_title="Mean Base Predation Rate",
        margin=dict(t=50, b=40),
    )
    return fig


def fig_trait_heatmap(run: dict, week: int, hab_id: str | None = None) -> go.Figure:
    """
    Heatmap of mean trait values across all living species on a given week,
    row-normalised so different-scale traits are visually comparable.
    """
    if hab_id:
        rows = [r for r in run["species_per_hab"] if r["hab_id"] == hab_id and r["week"] == week]
    else:
        rows = [r for r in run["species_global"] if r["week"] == week]

    if not rows:
        return go.Figure().update_layout(title=f"No species data for week {week}")

    species = [r["species"] for r in rows]
    ranges = run["trait_global_ranges"]
    matrix = []
    for t in ALL_TRAITS:
        mn, mx = ranges[t]
        rng = mx - mn
        vals = [r.get(t) or 0 for r in rows]
        normed = [(v - mn) / rng if rng > 0 else 0.5 for v in vals]
        matrix.append(normed)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=species,
            y=[t.replace("_", " ") for t in ALL_TRAITS],
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            hovertemplate="Species: %{x}<br>Trait: %{y}<br>Normalised: %{z:.3f}<extra></extra>",
        )
    )
    title_suffix = f" in {run['habitat_names'].get(hab_id, hab_id)}" if hab_id else " (global)"
    fig.update_layout(
        title=f"Trait snapshot — Week {week}{title_suffix}  (normalised to full-run range)",
        xaxis=dict(tickangle=-45),
        height=900,
        margin=dict(t=60, l=180, b=120, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Interval / bin stats figures and display builder
# ---------------------------------------------------------------------------


def fig_bin_deaths_by_cause(ist: dict) -> go.Figure:
    by_cause = ist["deaths"].get("by_cause", {})
    causes = list(by_cause.keys())
    counts = [by_cause[c] for c in causes]
    fig = go.Figure(go.Bar(x=counts, y=causes, orientation="h", marker_color="#ef5350"))
    fig.update_layout(
        title=f"Deaths by Cause  (total: {ist['deaths'].get('total', 0)})",
        height=max(200, 36 * len(causes) + 70),
        margin=dict(t=36, b=20, l=110, r=20),
    )
    return fig


def fig_bin_deaths_by_species(ist: dict) -> go.Figure:
    by_sp = ist["deaths"].get("by_species", {})
    top = sorted(by_sp.items(), key=lambda x: -x[1])[:12]
    species = [s for s, _ in top]
    counts = [c for _, c in top]
    fig = go.Figure(go.Bar(x=counts, y=species, orientation="h", marker_color="#ab47bc"))
    fig.update_layout(
        title="Deaths by Species",
        height=max(220, 24 * len(species) + 70),
        margin=dict(t=36, b=20, l=160, r=20),
    )
    return fig


def fig_bin_births_by_habitat(ist: dict, run: dict) -> go.Figure:
    by_hab = ist["births"].get("by_habitat", {})
    habs = sorted(by_hab.keys(), key=lambda h: -by_hab[h])
    names = [run["habitat_names"].get(h, h) for h in habs]
    counts = [by_hab[h] for h in habs]
    fig = go.Figure(go.Bar(x=counts, y=names, orientation="h", marker_color="#42a5f5"))
    fig.update_layout(
        title=f"Births by Habitat  (total: {ist['births'].get('total', 0)})",
        height=max(200, 32 * len(habs) + 70),
        margin=dict(t=36, b=20, l=160, r=20),
    )
    return fig


def fig_bin_births_by_species(ist: dict) -> go.Figure:
    by_sp = ist["births"].get("by_species", {})
    top = sorted(by_sp.items(), key=lambda x: -x[1])[:12]
    species = [s for s, _ in top]
    counts = [c for _, c in top]
    fig = go.Figure(go.Bar(x=counts, y=species, orientation="h", marker_color="#26a69a"))
    fig.update_layout(
        title="Births by Species",
        height=max(220, 24 * len(species) + 70),
        margin=dict(t=36, b=20, l=160, r=20),
    )
    return fig


def fig_bin_migrations(ist: dict, run: dict) -> go.Figure:
    by_route = ist["migrations"].get("by_route", {})
    if not by_route:
        fig = go.Figure()
        fig.update_layout(title="No migrations this bin", height=120, margin=dict(t=36, b=20))
        return fig
    sorted_routes = sorted(by_route.items(), key=lambda x: -x[1])
    labels = []
    for route_key, _ in sorted_routes:
        if "→" in route_key:
            f, t = route_key.split("→", 1)
            f_name = run["habitat_names"].get(f.strip(), f.strip())
            t_name = run["habitat_names"].get(t.strip(), t.strip())
            labels.append(f"{f_name} → {t_name}")
        else:
            labels.append(route_key)
    counts = [c for _, c in sorted_routes]
    fig = go.Figure(go.Bar(x=counts, y=labels, orientation="h", marker_color="#ffa726"))
    fig.update_layout(
        title=f"Migrations by Route  (total: {ist['migrations'].get('total', 0)})",
        height=max(200, 28 * len(sorted_routes) + 70),
        margin=dict(t=36, b=20, l=220, r=20),
    )
    return fig


def fig_bin_habitat_births(ist: dict, hab_id: str, hab_name: str) -> go.Figure:
    sp_counts = ist["births"].get("by_habitat_and_species", {}).get(hab_id, {})
    if not sp_counts:
        fig = go.Figure()
        fig.update_layout(title=f"{hab_name}: no births", height=100, margin=dict(t=36, b=10))
        return fig
    top = sorted(sp_counts.items(), key=lambda x: -x[1])[:10]
    species = [s for s, _ in top]
    counts = [c for _, c in top]
    fig = go.Figure(go.Bar(x=counts, y=species, orientation="h", marker_color="#66bb6a"))
    fig.update_layout(
        title=hab_name,
        height=max(160, 24 * len(species) + 60),
        margin=dict(t=36, b=20, l=150, r=20),
    )
    return fig


def build_bin_stats_display(ist: dict, run: dict, week: int) -> html.Div:
    """Render the full interval-stats panel for one logged week."""
    weeks_covered = ist.get("weeks_covered", [week, week])
    deaths = ist.get("deaths", {})
    births = ist.get("births", {})
    mating = ist.get("mating", {})
    migrations = ist.get("migrations", {})
    isolation_events = ist.get("isolation_events", [])

    total_pairings = mating.get("total_pairings", 0)
    fertilizations = mating.get("fertilizations", 0)
    hybrid = mating.get("hybrid_conceptions", 0)
    fert_pct = f"{100 * fertilizations / total_pairings:.0f}%" if total_pairings else "—"

    summary_line = (
        f"Weeks {weeks_covered[0]}–{weeks_covered[1]}  |  "
        f"Deaths: {deaths.get('total', 0)}  |  "
        f"Births: {births.get('total', 0)}  |  "
        f"Migrations: {migrations.get('total', 0)}  |  "
        f"Matings: {total_pairings}  ({fertilizations} fertilized = {fert_pct},  {hybrid} hybrid conceptions)"
    )

    # Isolation alert block — shows exact week each route closed.
    iso_block = []
    if isolation_events:
        items = []
        for ev in isolation_events:
            from_name = run["habitat_names"].get(ev["from_habitat"], ev["from_habitat"])
            to_name = run["habitat_names"].get(ev["blocked_neighbor"], ev["blocked_neighbor"])
            items.append(html.Li(f"Week {ev['week']}: {from_name} → {to_name} route closed"))
        iso_block = [
            html.Div(
                [
                    html.Strong("⚠ Route Isolations this bin:"),
                    html.Ul(items, style={"margin": "4px 0 0 16px", "padding": 0}),
                ],
                style={
                    "backgroundColor": "#fff3e0",
                    "border": "1px solid #ffb300",
                    "borderRadius": 4,
                    "padding": "8px 16px",
                    "marginBottom": 12,
                    "fontSize": "13px",
                    "color": "#e65100",
                },
            )
        ]

    hab_birth_charts = [
        html.Div(
            dcc.Graph(figure=fig_bin_habitat_births(ist, hid, run["habitat_names"].get(hid, hid))),
            style={"flex": "1", "minWidth": 240},
        )
        for hid in run["habitat_ids"]
    ]

    return html.Div(
        [
            html.Div(
                summary_line,
                style={
                    "padding": "10px 14px",
                    "backgroundColor": "#e3f2fd",
                    "fontFamily": "monospace",
                    "fontSize": "13px",
                    "borderRadius": 4,
                    "marginBottom": 12,
                    "whiteSpace": "pre-wrap",
                },
            ),
            *iso_block,
            html.H4("Deaths this bin", style={"marginBottom": 4}),
            html.Div(
                [
                    html.Div(dcc.Graph(figure=fig_bin_deaths_by_cause(ist)), style={"flex": "1"}),
                    html.Div(dcc.Graph(figure=fig_bin_deaths_by_species(ist)), style={"flex": "1"}),
                ],
                style={"display": "flex", "gap": 12, "marginBottom": 12},
            ),
            html.H4("Births this bin", style={"marginBottom": 4}),
            html.Div(
                [
                    html.Div(dcc.Graph(figure=fig_bin_births_by_habitat(ist, run)), style={"flex": "1"}),
                    html.Div(dcc.Graph(figure=fig_bin_births_by_species(ist)), style={"flex": "1"}),
                ],
                style={"display": "flex", "gap": 12, "marginBottom": 12},
            ),
            html.H4("Births by Species per Habitat", style={"marginBottom": 4}),
            html.Div(
                hab_birth_charts, style={"display": "flex", "flexWrap": "wrap", "gap": 12, "marginBottom": 12}
            ),
            html.H4("Migrations this bin", style={"marginBottom": 4}),
            dcc.Graph(figure=fig_bin_migrations(ist, run)),
        ],
        style={"padding": "0 0 24px"},
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def build_layout(run: dict) -> html.Div:
    summary = run["summary"]
    extinct_badge = (
        html.Span("EXTINCT", style={"color": "red", "fontWeight": "bold"})
        if summary["extinct"]
        else html.Span("Survived", style={"color": "green", "fontWeight": "bold"})
    )

    hab_options = [
        {"label": f"{run['habitat_names'][h]} ({run['habitat_types'][h]})", "value": h}
        for h in run["habitat_ids"]
    ]
    species_options = [{"label": s, "value": s} for s in run["all_species"]]

    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.H2("Evolution Simulator — Log Visualiser", style={"margin": "0 0 4px 0"}),
                    html.Div(
                        [
                            html.Span(f"Habitats: {len(run['habitat_ids'])}  ·  "),
                            html.Span(f"Weeks: {run['weeks_simulated']}  ·  "),
                            html.Span(f"Species ever: {summary['total_species_ever']}  ·  "),
                            html.Span(f"Speciations: {summary['total_speciation_events']}  ·  "),
                            html.Span(f"Isolations: {len(summary.get('all_isolation_events', []))}  ·  "),
                            html.Span("Status: "),
                            extinct_badge,
                        ],
                        style={"fontSize": "13px", "color": "#555"},
                    ),
                ],
                style={
                    "padding": "16px 24px 8px",
                    "borderBottom": "1px solid #ddd",
                    "backgroundColor": "#f8f9fa",
                },
            ),
            dcc.Tabs(
                id="tabs",
                value="overview",
                children=[
                    # Tab 1: Overview
                    dcc.Tab(
                        label="Overview",
                        value="overview",
                        children=[
                            dcc.Graph(id="fig-global", figure=fig_global_overview(run)),
                            dcc.Graph(id="fig-hab-pop", figure=fig_habitat_population(run)),
                        ],
                    ),
                    # Tab 2: Habitat Adaptation
                    dcc.Tab(
                        label="Habitat Adaptation",
                        value="adaptation",
                        children=[
                            html.Div(
                                [
                                    html.Label("Habitat:", style={"fontWeight": "bold", "marginRight": 8}),
                                    dcc.Dropdown(
                                        id="adapt-hab-select",
                                        options=hab_options,
                                        value=run["habitat_ids"][0],
                                        clearable=False,
                                        style={"width": 320},
                                    ),
                                ],
                                style={"padding": "12px 24px 0"},
                            ),
                            dcc.Graph(id="fig-adaptation"),
                        ],
                    ),
                    # Tab 3: Species Tracker
                    dcc.Tab(
                        label="Species Tracker",
                        value="species",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "Species (multi-select):",
                                                style={"fontWeight": "bold", "marginRight": 8},
                                            ),
                                            dcc.Dropdown(
                                                id="sp-select",
                                                options=species_options,
                                                value=run["all_species"][:3],
                                                multi=True,
                                                style={"width": 600},
                                            ),
                                        ],
                                        style={"marginBottom": 8},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "Trait to plot:",
                                                style={"fontWeight": "bold", "marginRight": 8},
                                            ),
                                            dcc.Dropdown(
                                                id="sp-trait-select",
                                                options=TRAIT_OPTIONS,
                                                value="fecundity",
                                                clearable=False,
                                                style={"width": 320},
                                            ),
                                        ]
                                    ),
                                ],
                                style={"padding": "12px 24px 0"},
                            ),
                            dcc.Graph(id="fig-sp-pop"),
                            dcc.Graph(id="fig-sp-trait"),
                        ],
                    ),
                    # Tab 4: r/K Tradeoff
                    dcc.Tab(
                        label="r/K Tradeoff",
                        value="rk",
                        children=[
                            html.Div(
                                [
                                    html.Label(
                                        f"Week (1 – {run['weeks_simulated']}):",
                                        style={"fontWeight": "bold", "marginRight": 8},
                                    ),
                                    dcc.Slider(
                                        id="rk-week-slider",
                                        min=1,
                                        max=run["weeks_simulated"],
                                        step=1,
                                        value=min(32, run["weeks_simulated"]),
                                        marks={
                                            w: str(w)
                                            for w in range(
                                                0,
                                                run["weeks_simulated"] + 1,
                                                max(1, run["weeks_simulated"] // 10),
                                            )
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ],
                                style={"padding": "12px 24px 0"},
                            ),
                            dcc.Graph(id="fig-rk"),
                        ],
                    ),
                    # Tab 5: Trait Heatmap
                    dcc.Tab(
                        label="Trait Heatmap",
                        value="heatmap",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "Scope:", style={"fontWeight": "bold", "marginRight": 8}
                                            ),
                                            dcc.Dropdown(
                                                id="heatmap-scope",
                                                options=[
                                                    {"label": "Global (all species)", "value": "__global__"}
                                                ]
                                                + hab_options,
                                                value="__global__",
                                                clearable=False,
                                                style={"width": 380},
                                            ),
                                        ],
                                        style={"marginBottom": 8},
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                f"Week (1 – {run['last_data_week']}):",
                                                style={"fontWeight": "bold", "marginRight": 8},
                                            ),
                                            dcc.Slider(
                                                id="heatmap-week-slider",
                                                min=1,
                                                max=run["last_data_week"],
                                                step=1,
                                                value=run["last_data_week"],
                                                marks={
                                                    w: str(w)
                                                    for w in range(
                                                        0,
                                                        run["last_data_week"] + 1,
                                                        max(1, run["last_data_week"] // 10),
                                                    )
                                                },
                                                tooltip={"placement": "bottom", "always_visible": True},
                                            ),
                                        ]
                                    ),
                                ],
                                style={"padding": "12px 24px 0"},
                            ),
                            dcc.Graph(id="fig-heatmap"),
                        ],
                    ),
                    # Tab 6: Bin / Interval Stats
                    dcc.Tab(
                        label="Bin Stats",
                        value="binstats",
                        children=[
                            html.Div(
                                [
                                    html.Label(
                                        "Logged week:",
                                        style={"fontWeight": "bold", "marginRight": 8},
                                    ),
                                    dcc.Dropdown(
                                        id="bin-week-select",
                                        options=[
                                            {"label": f"Week {w}", "value": w}
                                            for w in run["weeks_with_interval_stats"]
                                        ],
                                        value=(
                                            run["weeks_with_interval_stats"][-1]
                                            if run["weeks_with_interval_stats"]
                                            else None
                                        ),
                                        clearable=False,
                                        style={"width": 200},
                                    ),
                                ],
                                style={"padding": "12px 24px 8px", "display": "flex", "alignItems": "center"},
                            ),
                            html.Div(id="bin-stats-content", style={"padding": "4px 24px"}),
                        ],
                    ),
                ],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# App + callbacks
# ---------------------------------------------------------------------------


def make_app(run: dict) -> dash.Dash:
    app = dash.Dash(__name__, title="Evo Sim Visualiser")
    app.layout = build_layout(run)

    @app.callback(
        Output("fig-adaptation", "figure"),
        Input("adapt-hab-select", "value"),
    )
    def update_adaptation(hab_id):
        return fig_adaptation_by_habitat(run, hab_id)

    @app.callback(
        Output("fig-sp-pop", "figure"),
        Output("fig-sp-trait", "figure"),
        Input("sp-select", "value"),
        Input("sp-trait-select", "value"),
    )
    def update_species(selected, trait):
        selected = selected or []
        return fig_species_population(run, selected), fig_species_trait(run, selected, trait)

    @app.callback(
        Output("fig-rk", "figure"),
        Input("rk-week-slider", "value"),
    )
    def update_rk(week):
        return fig_rk_tradeoff(run, week)

    @app.callback(
        Output("fig-heatmap", "figure"),
        Input("heatmap-scope", "value"),
        Input("heatmap-week-slider", "value"),
    )
    def update_heatmap(scope, week):
        hab_id = None if scope == "__global__" else scope
        return fig_trait_heatmap(run, week, hab_id)

    @app.callback(
        Output("bin-stats-content", "children"),
        Input("bin-week-select", "value"),
    )
    def update_bin_stats(week):
        if not run["weeks_with_interval_stats"]:
            return html.P(
                "No interval stats found in this run. "
                "This run predates the interval_stats feature, or stats_every=0.",
                style={"color": "#888", "padding": "16px 0"},
            )
        if week is None:
            return html.P("Select a logged week above.", style={"color": "#888"})
        ist = run["interval_stats_by_week"].get(week)
        if ist is None:
            return html.P(f"No interval stats for week {week}.", style={"color": "#888"})
        return build_bin_stats_display(ist, run, week)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _latest_run(base: Path) -> Path:
    runs = sorted(base.glob("*/week_00001.json"))
    if not runs:
        raise FileNotFoundError(f"No simulation run found under {base}")
    return runs[-1].parent


def main():
    parser = argparse.ArgumentParser(description="Visualise an evolution simulator log")
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=None,
        help="Path to a simulation log directory (default: most recent under simulation_logs/)",
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = _latest_run(Path("simulation_logs"))

    print(f"Loading run from: {log_dir}", flush=True)
    run = load_run(log_dir)
    print(
        f"  {run['weeks_simulated']} weeks  |  "
        f"{run['summary']['total_species_ever']} species  |  "
        f"{'EXTINCT' if run['extinct'] else 'survived'}",
        flush=True,
    )

    app = make_app(run)
    print(f"\nOpen http://{args.host}:{args.port}/ in your browser\n", flush=True)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
