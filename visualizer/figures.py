"""
Plotly figure builders used inside panel components and main-canvas views.
"""

import math

import plotly.graph_objects as go

from visualizer.style import BG, BORDER, DIMTEXT, FONT, HABITAT_COLORS, PANEL_BG, TEXT


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout(**overrides) -> dict:
    base = dict(
        paper_bgcolor=PANEL_BG,
        plot_bgcolor="#0a0e0a",
        font=dict(color=TEXT, family=FONT, size=10),
        margin=dict(t=28, b=28, l=40, r=12),
        xaxis=dict(
            color=DIMTEXT,
            gridcolor=BORDER,
            zeroline=False,
            title_text="week",
            rangemode="tozero",
        ),
        yaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False),
        showlegend=False,
        height=180,
    )
    base.update(overrides)
    return base


def _nice_tick(max_val: float) -> float:
    """Return a 'nice' round number ≈ max_val/6 for use as one tick interval."""
    if max_val <= 0:
        return 1.0
    raw = max_val / 6
    exp = 10 ** math.floor(math.log10(raw))
    return min([exp, 2 * exp, 5 * exp], key=lambda v: abs(v - raw))


def _hab_border(run: dict, hab_id: str | None) -> str:
    if hab_id is None:
        return "#3a7a3a"
    hab_type = run["hab_types"].get(hab_id, "Forest")
    _, border, _ = HABITAT_COLORS.get(hab_type, ("#1a1a1a", "#5a5a5a", "#888"))
    return border


def _empty_fig(message: str = "no data", height: int = 200) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=DIMTEXT, size=12, family=FONT),
    )
    fig.update_layout(
        paper_bgcolor=PANEL_BG,
        plot_bgcolor="#0a0e0a",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=10, b=10, l=10, r=10),
        height=height,
    )
    return fig


# ── Bin / interval stats figures ─────────────────────────────────────────────


def bin_deaths_by_cause(ist: dict) -> go.Figure:
    """Compact horizontal bar: deaths by cause for one stats bin."""
    by_cause = ist.get("deaths", {}).get("by_cause", {})
    if not by_cause:
        return _empty_fig("no death data", height=120)
    cause_order = ["starvation", "dehydration", "old_age", "predation"]
    causes = [c for c in cause_order if c in by_cause] + [c for c in by_cause if c not in cause_order]
    counts = [by_cause[c] for c in causes]
    fig = go.Figure(
        go.Bar(
            x=counts,
            y=[c.replace("_", " ") for c in causes],
            orientation="h",
            marker_color=["#ef5350", "#ef8a50", "#ab47bc", "#ff7043"],
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="deaths by cause", font=dict(size=10, color=DIMTEXT), x=0),
            height=max(120, 28 * len(causes) + 50),
            margin=dict(t=28, b=10, l=80, r=12),
            xaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False, title_text=""),
            yaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False),
        )
    )
    return fig


def bin_habitat_births(ist: dict, hab_id: str, hab_name: str, run: dict) -> go.Figure:
    """Compact horizontal bar: births by species in one habitat for one stats bin."""
    sp_counts = ist.get("births", {}).get("by_habitat_and_species", {}).get(hab_id, {})
    if not sp_counts:
        return _empty_fig(f"no births in {hab_name}", height=100)
    top = sorted(sp_counts.items(), key=lambda x: -x[1])[:10]
    species = [s for s, _ in top]
    counts = [c for _, c in top]
    border = _hab_border(run, hab_id)
    fig = go.Figure(go.Bar(x=counts, y=species, orientation="h", marker_color=border))
    fig.update_layout(
        **_base_layout(
            title=dict(text=f"births · {hab_name}", font=dict(size=10, color=DIMTEXT), x=0),
            height=max(120, 24 * len(species) + 50),
            margin=dict(t=28, b=10, l=140, r=12),
            xaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False, title_text=""),
            yaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False),
        )
    )
    return fig


# ── Panel chart figures ───────────────────────────────────────────────────────


def species_population(run: dict, species: str, hab_id: str | None) -> go.Figure:
    rows = (
        run["species_per_hab"].get(hab_id, {}).get(species, [])
        if hab_id
        else run["species_global"].get(species, [])
    )
    days = [r["week"] for r in rows]
    vals = [r["count"] for r in rows]
    color = _hab_border(run, hab_id)
    scope = run["hab_names"].get(hab_id, "global") if hab_id else "global"

    fig = go.Figure()
    if days:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=vals,
                mode="lines",
                line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(color, 0.13),
            )
        )
    fig.update_layout(
        **_base_layout(
            title=dict(text=f"population  [{scope}]", font=dict(color=DIMTEXT, size=10, family=FONT)),
            height=160,
        )
    )
    return fig


def species_trait(run: dict, species: str, trait: str, hab_id: str | None) -> go.Figure:
    rows = (
        run["species_per_hab"].get(hab_id, {}).get(species, [])
        if hab_id
        else run["species_global"].get(species, [])
    )
    days = [r["week"] for r in rows if r.get(trait) is not None]
    vals = [r[trait] for r in rows if r.get(trait) is not None]
    color = _hab_border(run, hab_id)
    scope = run["hab_names"].get(hab_id, "global") if hab_id else "global"

    fig = go.Figure()
    if days:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=vals,
                mode="lines",
                line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(color, 0.12),
            )
        )
    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"{trait.replace('_', ' ')}  [{scope}]", font=dict(color=DIMTEXT, size=10, family=FONT)
            ),
        )
    )
    return fig


def resource_probability(run: dict, species: str, hab_id: str) -> go.Figure:
    """Food and water probability over time for a species in a specific habitat."""
    rows = run["species_per_hab"].get(hab_id, {}).get(species, [])
    weeks = [r["week"] for r in rows]
    food = [r.get("mean_food_prob") for r in rows]
    water = [r.get("mean_water_prob") for r in rows]

    if not weeks:
        return _empty_fig("no resource data for this species/habitat", height=160)

    hab_type = run["hab_types"].get(hab_id, "Forest")
    _, border, _ = HABITAT_COLORS.get(hab_type, ("#1a1a1a", "#5a5a5a", "#888"))

    fig = go.Figure()
    fig.add_hline(y=0.5, line=dict(color="#444", dash="dot", width=1))
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=food,
            mode="lines",
            name="food prob",
            line=dict(color="#56c456", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=water,
            mode="lines",
            name="water prob",
            line=dict(color="#2ea8e0", width=1.5),
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(
                text="resource adaptation  [0.5 = unadapted baseline]",
                font=dict(color=DIMTEXT, size=10, family=FONT),
            ),
            yaxis=dict(
                color=DIMTEXT, gridcolor=BORDER, zeroline=False, range=[0, 1], title_text="probability"
            ),
            showlegend=True,
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            height=160,
        )
    )
    return fig


def generation_adaptation(run: dict, species: str, hab_id: str | None) -> go.Figure:
    """Mean generation (left) vs food/water probability (right) over time."""
    rows = (
        run["species_per_hab"].get(hab_id, {}).get(species, [])
        if hab_id
        else run["species_global"].get(species, [])
    )
    weeks = [r["week"] for r in rows]
    gens = [r.get("mean_generation") for r in rows]
    food = [r.get("mean_food_prob") for r in rows] if hab_id else [None] * len(rows)
    water = [r.get("mean_water_prob") for r in rows] if hab_id else [None] * len(rows)

    if not weeks:
        return _empty_fig("no generation data", height=180)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=gens,
            mode="lines",
            name="mean generation",
            line=dict(color="#c4dcc4", width=1.5),
            yaxis="y1",
        )
    )
    if hab_id and any(v is not None for v in food):
        fig.add_hline(y=0.5, line=dict(color="#333", dash="dot", width=1), yref="y2")
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=food,
                mode="lines",
                name="food prob",
                line=dict(color="#56c456", width=1.2, dash="dash"),
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=water,
                mode="lines",
                name="water prob",
                line=dict(color="#2ea8e0", width=1.2, dash="dash"),
                yaxis="y2",
            )
        )
    fig.update_layout(
        **_base_layout(
            title=dict(text="generation vs adaptation", font=dict(color=DIMTEXT, size=10, family=FONT)),
            yaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False, title_text="generation"),
            yaxis2=dict(
                color="#5a9a5a",
                overlaying="y",
                side="right",
                zeroline=False,
                title_text="prob",
                range=[0, 1],
            ),
            showlegend=True,
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            height=180,
        )
    )
    return fig


def global_overview(run: dict) -> go.Figure:
    gs = run["global_series"]
    days = [r["week"] for r in gs]
    pop = [r["population"] for r in gs]
    spc = [r["species_count"] for r in gs]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days,
            y=pop,
            name="population",
            line=dict(color="#3a7a3a", width=1.5),
            fill="tozeroy",
            fillcolor=_hex_to_rgba("#3a7a3a", 0.13),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=spc,
            name="species",
            line=dict(color="#7a5a3a", width=1.5, dash="dash"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        **_base_layout(
            yaxis2=dict(
                color="#7a5a3a",
                overlaying="y",
                side="right",
                zeroline=False,
                title_text="species",
            ),
            yaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False, title_text="pop"),
            hovermode="x unified",
            showlegend=True,
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            height=200,
        )
    )
    return fig


def edge_migration(run: dict, src: str, tgt: str) -> go.Figure:
    all_fwd = run["migrations_by_edge"].get((src, tgt), [])
    all_rev = run["migrations_by_edge"].get((tgt, src), [])

    fwd_by_week: dict[int, int] = {}
    for m in all_fwd:
        fwd_by_week[m["week"]] = fwd_by_week.get(m["week"], 0) + 1
    rev_by_week: dict[int, int] = {}
    for m in all_rev:
        rev_by_week[m["week"]] = rev_by_week.get(m["week"], 0) + 1

    src_name = run["hab_names"].get(src, src)
    tgt_name = run["hab_names"].get(tgt, tgt)

    fig = go.Figure()
    if fwd_by_week:
        fig.add_trace(
            go.Scatter(
                x=list(fwd_by_week.keys()),
                y=list(fwd_by_week.values()),
                name=f"{src_name}→{tgt_name}",
                mode="lines",
                line=dict(color="#3a7a3a", width=1.2),
            )
        )
    if rev_by_week:
        fig.add_trace(
            go.Scatter(
                x=list(rev_by_week.keys()),
                y=list(rev_by_week.values()),
                name=f"{tgt_name}→{src_name}",
                mode="lines",
                line=dict(color="#7a5a3a", width=1.2, dash="dot"),
            )
        )
    fig.update_layout(
        **_base_layout(
            showlegend=True,
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        )
    )
    return fig


# ── Main-canvas views ─────────────────────────────────────────────────────────


def species_phylogeny(run: dict, min_weeks: int = 0) -> go.Figure:
    """
    Speciation tree: nodes = species, edges = parent→child speciation.
    X-axis = week of first appearance; Y-axis = branching layout position.

    min_weeks: hide any species whose lifespan (last seen - birth week) is
    shorter than this value. Children of hidden nodes are reparented to their
    nearest surviving ancestor so the visible tree stays connected.
    """
    lineage = run.get("species_lineage", {})
    peak_pop = run.get("species_peak_population", {})

    # ── Optional lifespan filter ─────────────────────────────────────────────
    if min_weeks > 0:
        lifespan = run.get("species_lifespan", {})
        kept = {sp for sp in lineage if lifespan.get(sp, 0) >= min_weeks}

        # Reparent: find nearest surviving ancestor for each filtered-out node
        def _surviving_ancestor(sp: str) -> str | None:
            parent = lineage[sp].get("parent")
            while parent is not None:
                if parent in kept:
                    return parent
                parent = lineage[parent].get("parent")
            return None

        filtered_lineage: dict[str, dict] = {}
        for sp in kept:
            entry = dict(lineage[sp])
            orig_parent = entry.get("parent")
            if orig_parent is not None and orig_parent not in kept:
                entry = dict(entry)
                entry["parent"] = _surviving_ancestor(sp)
            # Rebuild children list to only include kept species
            entry["children"] = [c for c in entry.get("children", []) if c in kept]
            filtered_lineage[sp] = entry
        lineage = filtered_lineage

    if not lineage:
        return _empty_fig("no speciation data", height=600)

    # ── Tree layout: in-order traversal assigns y positions ─────────────────
    children_map: dict[str, list[str]] = {sp: lineage[sp].get("children", []) for sp in lineage}
    roots = [sp for sp in lineage if lineage[sp].get("parent") is None]

    positions: dict[str, tuple[float, float]] = {}
    counter = [0]

    def _assign(sp: str) -> None:
        kids = sorted(children_map.get(sp, []), key=lambda k: lineage[k].get("week", 0))
        if not kids:
            positions[sp] = (lineage[sp].get("week", 0), float(counter[0]))
            counter[0] += 1
        else:
            for kid in kids:
                _assign(kid)
            ys = [positions[k][1] for k in kids if k in positions]
            positions[sp] = (lineage[sp].get("week", 0), sum(ys) / len(ys))

    for root in sorted(roots, key=lambda r: lineage[r].get("week", 0)):
        _assign(root)
        counter[0] += 1  # vertical gap between independent trees

    # ── Edge traces ──────────────────────────────────────────────────────────
    # Three edge styles by event type:
    #   cladogenesis_newborn      – solid green  (newborn fell outside all species)
    #   cladogenesis_kmeans_subcluster – solid blue   (k-means detected bimodal split)
    #   anagenesis                – dashed orange (in-place lineage transformation)
    #   anything else / legacy    – solid green  (backward compat with old logs)
    clad_newborn_x: list = []
    clad_newborn_y: list = []
    clad_kmeans_x: list = []
    clad_kmeans_y: list = []
    ana_x: list = []
    ana_y: list = []
    for sp, data in lineage.items():
        parent = data.get("parent")
        if parent and parent in positions and sp in positions:
            px, py = positions[parent]
            cx, cy = positions[sp]
            # L-shaped cladogram edge: horizontal from parent, then vertical to child
            et = data.get("event_type", "")
            if et == "anagenesis":
                ana_x.extend([px, cx, cx, None])
                ana_y.extend([py, py, cy, None])
            elif et == "cladogenesis_kmeans_subcluster":
                clad_kmeans_x.extend([px, cx, cx, None])
                clad_kmeans_y.extend([py, py, cy, None])
            else:
                clad_newborn_x.extend([px, cx, cx, None])
                clad_newborn_y.extend([py, py, cy, None])

    fig = go.Figure()

    if clad_newborn_x:
        fig.add_trace(
            go.Scatter(
                x=clad_newborn_x,
                y=clad_newborn_y,
                mode="lines",
                line=dict(color="#2a5a2a", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if clad_kmeans_x:
        fig.add_trace(
            go.Scatter(
                x=clad_kmeans_x,
                y=clad_kmeans_y,
                mode="lines",
                line=dict(color="#2a5a8a", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if ana_x:
        fig.add_trace(
            go.Scatter(
                x=ana_x,
                y=ana_y,
                mode="lines",
                line=dict(color="#b5792a", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # ── Node trace ───────────────────────────────────────────────────────────
    node_x = [positions[sp][0] for sp in lineage if sp in positions]
    node_y = [positions[sp][1] for sp in lineage if sp in positions]
    node_sp = [sp for sp in lineage if sp in positions]
    node_sizes = [
        max(8, min(24, 8 + 16 * peak_pop.get(sp, 0) / max(1, max(peak_pop.values())))) for sp in node_sp
    ]
    node_labels = [
        f"{sp}<br>week: {lineage[sp].get('week', 0)}"
        f"<br>peak pop: {peak_pop.get(sp, 0)}"
        f"<br>parent: {lineage[sp].get('parent') or 'founder'}"
        + (
            ""
            if lineage[sp].get("parent") is None
            else f"<br>via: {lineage[sp].get('event_type', 'cladogenesis')}"
        )
        for sp in node_sp
    ]
    node_customdata = [{"species": sp} for sp in node_sp]

    # Colour by origin: founder / newborn isolation / k-means split / anagenesis
    def _node_color(sp: str) -> str:
        d = lineage[sp]
        if d.get("parent") is None:
            return "#e4f4a0"
        et = d.get("event_type", "")
        if et == "anagenesis":
            return "#e8a23d"
        if et == "cladogenesis_kmeans_subcluster":
            return "#4a90d9"
        return "#56c456"  # newborn isolation, bootstrap, or legacy logs

    node_colors = [_node_color(sp) for sp in node_sp]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(color="#3a6a3a", width=1),
            ),
            text=[sp.split()[0] for sp in node_sp],  # first word of name as label
            textposition="top center",
            textfont=dict(size=8, color=DIMTEXT),
            hovertext=node_labels,
            hoverinfo="text",
            customdata=node_customdata,
            showlegend=False,
        )
    )

    # ── Legend annotations (only for event types that actually appear) ────────
    color_set = set(node_colors)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="#e4f4a0", size=10),
            name="founder species",
        )
    )
    if "#56c456" in color_set:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="#56c456", size=10),
                name="cladogenesis: newborn isolation (legacy)",
            )
        )
    if "#4a90d9" in color_set:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="#4a90d9", size=10),
                name="cladogenesis: k-means subcluster split",
            )
        )
    if "#e8a23d" in color_set:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="#e8a23d", size=10),
                name="anagenesis (in-place transformation)",
            )
        )

    all_weeks = run.get("all_weeks", [0])
    max_w = max(all_weeks)
    x_left = -_nice_tick(max_w)  # one tick of breathing room so legend doesn't overlap
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor="#050805",
        font=dict(color=TEXT, family=FONT, size=10),
        margin=dict(t=40, b=40, l=60, r=20),
        xaxis=dict(
            color=DIMTEXT,
            gridcolor=BORDER,
            zeroline=False,
            title_text="week of first appearance",
            range=[x_left, max_w * 1.05],
        ),
        yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(
            font=dict(size=10, family=FONT),
            bgcolor="rgba(0,0,0,0)",
            x=0.01,
            y=0.99,
        ),
        hovermode="closest",
        clickmode="event",
        height=600,
        title=dict(
            text="SPECIATION PHYLOGENY",
            font=dict(color=DIMTEXT, size=13, family=FONT),
            x=0.01,
        ),
    )
    return fig


def family_tree_wheel(run: dict, creature_id: str) -> go.Figure:
    """
    Radial ancestry wheel centred on creature_id.
    Ancestors radiate outward in concentric semicircles (upper half).
    Descendants spread downward as a tree (lower half).
    Click any node to re-centre.
    """
    births = run.get("creature_births", {})
    children_map = run.get("creature_children", {})

    if creature_id not in births and creature_id not in children_map:
        return _empty_fig("select a creature to view its family tree", height=550)

    RING_RADIUS = 2.0  # distance between rings
    MAX_PER_RING = 14  # cap ancestors per generation ring

    # ── Collect ancestors (up to 3 generations back) ─────────────────────────
    ancestors_by_gen: dict[int, list[str]] = {}
    frontier = [creature_id]
    for depth in range(1, 4):
        nxt: list[str] = []
        for fid in frontier:
            nxt.extend(births.get(fid, {}).get("parents", []))
        # deduplicate, keep known creatures only
        seen: set = set()
        deduped = [c for c in nxt if c not in seen and not seen.add(c)]  # type: ignore[func-returns-value]
        deduped = deduped[:MAX_PER_RING]
        if not deduped:
            break
        ancestors_by_gen[depth] = deduped
        frontier = deduped

    # ── Collect descendants (up to 3 generations forward) ────────────────────
    descendants_by_gen: dict[int, list[str]] = {}
    frontier = [creature_id]
    for depth in range(1, 4):
        nxt = []
        for fid in frontier:
            nxt.extend(children_map.get(fid, []))
        seen = set()
        deduped = [c for c in nxt if c not in seen and not seen.add(c)]  # type: ignore[func-returns-value]
        deduped = deduped[:20]
        if not deduped:
            break
        descendants_by_gen[depth] = deduped
        frontier = deduped

    # ── Compute positions ─────────────────────────────────────────────────────
    positions: dict[str, tuple[float, float]] = {creature_id: (0.0, 0.0)}

    # Ancestors: upper semicircle rings (angles 160° → 20°)
    for gen, ids in ancestors_by_gen.items():
        radius = gen * RING_RADIUS
        n = len(ids)
        for i, cid in enumerate(ids):
            angle_deg = 160.0 - 140.0 * i / max(1, n - 1) if n > 1 else 90.0
            angle_rad = math.radians(angle_deg)
            positions[cid] = (radius * math.cos(angle_rad), radius * math.sin(angle_rad))

    # Descendants: lower half tree (spread horizontally, drop vertically)
    for gen, ids in descendants_by_gen.items():
        n = len(ids)
        spread = min(n, 10) * (RING_RADIUS * 0.6)
        for i, cid in enumerate(ids):
            x = (-spread / 2 + spread * i / max(1, n - 1)) if n > 1 else 0.0
            positions[cid] = (x, -gen * RING_RADIUS)

    all_cids = set(positions.keys())

    # ── Build edges ───────────────────────────────────────────────────────────
    edge_x: list = []
    edge_y: list = []
    for cid in all_cids:
        cx, cy = positions[cid]
        for parent_id in births.get(cid, {}).get("parents", []):
            if parent_id in all_cids:
                px, py = positions[parent_id]
                edge_x.extend([px, cx, None])
                edge_y.extend([py, cy, None])

    # ── Categorise nodes ──────────────────────────────────────────────────────
    ancestor_set = {c for ids in ancestors_by_gen.values() for c in ids}
    descendant_set = {c for ids in descendants_by_gen.values() for c in ids}

    STYLE: dict[str, dict] = {
        "focus": {"color": "#e4f4e4", "size": 18, "sym": "circle"},
        "ancestor": {"color": "#56c456", "size": 11, "sym": "circle"},
        "descendant": {"color": "#2ea8e0", "size": 11, "sym": "circle"},
    }

    def _cat(cid: str) -> str:
        if cid == creature_id:
            return "focus"
        if cid in ancestor_set:
            return "ancestor"
        return "descendant"

    fig = go.Figure()

    # Edge trace
    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="#253a25", width=1),
                hoverinfo="skip",
            )
        )

    # Node traces (one per category for legend)
    for cat, style in STYLE.items():
        if cat == "focus":
            cids_in_cat = [creature_id]
        elif cat == "ancestor":
            cids_in_cat = [c for c in all_cids if c in ancestor_set]
        else:
            cids_in_cat = [c for c in all_cids if c in descendant_set]

        if not cids_in_cat:
            continue

        xs = [positions[c][0] for c in cids_in_cat]
        ys = [positions[c][1] for c in cids_in_cat]
        hover = []
        customdata = []
        for cid in cids_in_cat:
            info = births.get(cid, {})
            hover.append(
                f"id: {cid[:8]}…<br>"
                f"species: {info.get('species', '?')}<br>"
                f"sex: {info.get('sex', '?')}<br>"
                f"generation: {info.get('generation', '?')}<br>"
                f"born: week {info.get('week', '?')}<br>"
                f"<i>click to re-centre</i>"
            )
            customdata.append({"creature_id": cid})

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    color=style["color"],
                    size=style["size"],
                    line=dict(color="#3a6a3a", width=1),
                    symbol=style["sym"],
                ),
                text=hover,
                hoverinfo="text",
                customdata=customdata,
                name=cat,
            )
        )

    # ── Sex labels on close relatives ─────────────────────────────────────────
    label_cids = [creature_id] + ancestors_by_gen.get(1, []) + descendants_by_gen.get(1, [])
    lx = [positions[c][0] for c in label_cids if c in positions]
    ly = [positions[c][1] for c in label_cids if c in positions]
    lt = [births.get(c, {}).get("sex", "?")[0].upper() for c in label_cids if c in positions]

    fig.add_trace(
        go.Scatter(
            x=lx,
            y=ly,
            mode="text",
            text=lt,
            textfont=dict(size=7, color="#0a0e0a"),
            hoverinfo="skip",
        )
    )

    # ── Ring / tier labels ────────────────────────────────────────────────────
    gen_labels_anc = {1: "parents", 2: "grandparents", 3: "great-grandparents"}
    gen_labels_des = {1: "children", 2: "grandchildren", 3: "great-grandchildren"}
    for gen in sorted(ancestors_by_gen):
        fig.add_annotation(
            x=0,
            y=gen * RING_RADIUS + 0.35,
            text=gen_labels_anc.get(gen, f"anc gen {gen}"),
            showarrow=False,
            font=dict(size=9, color=DIMTEXT, family=FONT),
        )
    for gen in sorted(descendants_by_gen):
        fig.add_annotation(
            x=0,
            y=-gen * RING_RADIUS - 0.35,
            text=gen_labels_des.get(gen, f"des gen {gen}"),
            showarrow=False,
            font=dict(size=9, color=DIMTEXT, family=FONT),
        )

    # Focus label
    focus_info = births.get(creature_id, {})
    focus_label = (
        f"{focus_info.get('species', 'unknown')}<br>"
        f"gen {focus_info.get('generation', '?')}  "
        f"week {focus_info.get('week', '?')}"
    )
    fig.add_annotation(
        x=0,
        y=-0.45,
        text=focus_label,
        showarrow=False,
        font=dict(size=9, color=TEXT, family=FONT),
    )

    max_r = max(len(ancestors_by_gen), len(descendants_by_gen), 1) * RING_RADIUS + 1
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor="#050805",
        font=dict(color=TEXT, family=FONT, size=10),
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(visible=False, range=[-max_r * 1.1, max_r * 1.1]),
        yaxis=dict(visible=False, range=[-max_r * 1.1, max_r * 1.1], scaleanchor="x"),
        showlegend=True,
        legend=dict(
            font=dict(size=10, family=FONT),
            bgcolor="rgba(0,0,0,0)",
            x=0.01,
            y=0.99,
        ),
        hovermode="closest",
        clickmode="event",
        height=600,
    )
    return fig


# ── Trait comparison ──────────────────────────────────────────────────────────

_TC_COLORS = [
    "#56c456",
    "#4a90d9",
    "#e8a23d",
    "#e86a6a",
    "#a060d0",
    "#60d0d0",
    "#d0a060",
    "#d06090",
    "#90d060",
    "#6090d0",
    "#d09060",
    "#60d0a0",
    "#d0c040",
    "#c060c0",
    "#40c0d0",
    "#d06060",
    "#70b870",
    "#b07030",
    "#7070d0",
    "#d07070",
]
_TC_DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


def _tc_metric_series(run: dict, species: str, metric: str):
    """Return (weeks, vals) for a species/metric pair from the run data."""
    if metric in ("mean_food_prob", "mean_water_prob"):
        by_week: dict[int, list] = {}
        for sp_data in run["species_per_hab"].values():
            for row in sp_data.get(species, []):
                w = row["week"]
                v = row.get(metric)
                n = row.get("count", 0)
                if v is not None and n > 0:
                    by_week.setdefault(w, [0.0, 0])
                    by_week[w][0] += v * n
                    by_week[w][1] += n
        weeks = sorted(by_week)
        vals = [by_week[w][0] / by_week[w][1] if by_week[w][1] > 0 else None for w in weeks]
        return weeks, vals
    rows = run["species_global"].get(species, [])
    weeks = [r["week"] for r in rows]
    vals = [r.get(metric) for r in rows]
    return weeks, vals


def _tc_anagenesis_descendants(run: dict, species: str) -> list[str]:
    """All anagenesis-type children/grandchildren of species (BFS)."""
    lineage = run.get("species_lineage", {})
    result, queue = [], [species]
    while queue:
        sp = queue.pop(0)
        for child in lineage.get(sp, {}).get("children", []):
            if lineage.get(child, {}).get("event_type") == "anagenesis":
                result.append(child)
                queue.append(child)
    return result


def trait_comparison(
    run: dict,
    selected_species: list[str],
    selected_metrics: list[str],
    include_anagenesis: bool,
) -> go.Figure:
    """
    Multi-species multi-metric line chart for divergence analysis.

    Color encodes species identity; line dash encodes metric (when > 1 metric).
    Anagenesis descendants (if requested) use the same color as their base
    species but at reduced opacity, starting from their appearance week.
    """
    if not selected_species or not selected_metrics:
        return _empty_fig("select at least one species and one metric above", height=500)

    max_w = max(run["all_weeks"])

    # Build the full list of (species, base_species) pairs to plot.
    plot_pairs: list[tuple[str, str]] = []
    for sp in selected_species:
        plot_pairs.append((sp, sp))
        if include_anagenesis:
            for desc in _tc_anagenesis_descendants(run, sp):
                plot_pairs.append((desc, sp))

    base_color = {sp: _TC_COLORS[i % len(_TC_COLORS)] for i, sp in enumerate(selected_species)}
    multi_metric = len(selected_metrics) > 1

    fig = go.Figure()

    for sp, base_sp in plot_pairs:
        color = base_color[base_sp]
        is_desc = sp != base_sp
        opacity = 0.55 if is_desc else 1.0
        width = 1.2 if is_desc else 1.8

        for m_idx, metric in enumerate(selected_metrics):
            weeks, vals = _tc_metric_series(run, sp, metric)
            if not weeks:
                continue

            metric_label = metric.replace("_", " ")
            if multi_metric:
                trace_name = f"{'↳ ' if is_desc else ''}{sp} — {metric_label}"
            else:
                trace_name = f"{'↳ ' if is_desc else ''}{sp}"

            fig.add_trace(
                go.Scatter(
                    x=weeks,
                    y=vals,
                    mode="lines",
                    name=trace_name,
                    line=dict(
                        color=color,
                        width=width,
                        dash=_TC_DASHES[m_idx % len(_TC_DASHES)],
                    ),
                    opacity=opacity,
                    hovertemplate=(
                        f"<b>{sp}</b><br>"
                        f"metric: {metric_label}<br>"
                        "week: %{x}<br>"
                        "value: %{y:.4f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor="#050805",
        font=dict(color=TEXT, family=FONT, size=10),
        margin=dict(t=20, b=48, l=60, r=20),
        xaxis=dict(
            color=DIMTEXT,
            gridcolor=BORDER,
            zeroline=False,
            title_text="week",
            range=[0, max_w * 1.02],
        ),
        yaxis=dict(
            color=DIMTEXT,
            gridcolor=BORDER,
            zeroline=False,
            title_text=selected_metrics[0].replace("_", " ") if not multi_metric else "value",
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=10, family=FONT),
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor=BORDER,
            borderwidth=1,
        ),
        hovermode="x unified",
        height=500,
        title=dict(
            text="TRAIT COMPARISON",
            font=dict(color=DIMTEXT, size=13, family=FONT),
            x=0.01,
        ),
    )
    return fig
