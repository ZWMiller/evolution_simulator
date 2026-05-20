"""
Plotly figure builders used inside panel components and main-canvas views.
"""

import math

import plotly.graph_objects as go

from visualizer.style import BG, PANEL_BG, BORDER, TEXT, DIMTEXT, FONT, HABITAT_COLORS


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
        xaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False, title_text="week"),
        yaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False),
        showlegend=False,
        height=180,
    )
    base.update(overrides)
    return base


def _hab_border(run: dict, hab_id: str | None) -> str:
    if hab_id is None:
        return "#3a7a3a"
    hab_type = run["hab_types"].get(hab_id, "Forest")
    _, border, _ = HABITAT_COLORS.get(hab_type, ("#1a1a1a", "#5a5a5a", "#888"))
    return border


def _empty_fig(message: str = "no data", height: int = 200) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(color=DIMTEXT, size=12, family=FONT),
    )
    fig.update_layout(
        paper_bgcolor=PANEL_BG, plot_bgcolor="#0a0e0a",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(t=10, b=10, l=10, r=10), height=height,
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
        fig.add_trace(go.Scatter(
            x=days, y=vals, mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.13),
        ))
    fig.update_layout(**_base_layout(
        title=dict(text=f"population  [{scope}]",
                   font=dict(color=DIMTEXT, size=10, family=FONT)),
        height=160,
    ))
    return fig


def species_trait(run: dict, species: str, trait: str, hab_id: str | None) -> go.Figure:
    rows = (
        run["species_per_hab"].get(hab_id, {}).get(species, [])
        if hab_id
        else run["species_global"].get(species, [])
    )
    days = [r["week"] for r in rows if r.get(trait) is not None]
    vals = [r[trait]  for r in rows if r.get(trait) is not None]
    color = _hab_border(run, hab_id)
    scope = run["hab_names"].get(hab_id, "global") if hab_id else "global"

    fig = go.Figure()
    if days:
        fig.add_trace(go.Scatter(
            x=days, y=vals, mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.12),
        ))
    fig.update_layout(**_base_layout(
        title=dict(text=f"{trait.replace('_', ' ')}  [{scope}]",
                   font=dict(color=DIMTEXT, size=10, family=FONT)),
    ))
    return fig


def resource_probability(run: dict, species: str, hab_id: str) -> go.Figure:
    """Food and water probability over time for a species in a specific habitat."""
    rows = run["species_per_hab"].get(hab_id, {}).get(species, [])
    weeks = [r["week"] for r in rows]
    food  = [r.get("mean_food_prob")  for r in rows]
    water = [r.get("mean_water_prob") for r in rows]

    if not weeks:
        return _empty_fig("no resource data for this species/habitat", height=160)

    hab_type = run["hab_types"].get(hab_id, "Forest")
    _, border, _ = HABITAT_COLORS.get(hab_type, ("#1a1a1a", "#5a5a5a", "#888"))

    fig = go.Figure()
    fig.add_hline(y=0.5, line=dict(color="#444", dash="dot", width=1))
    fig.add_trace(go.Scatter(
        x=weeks, y=food, mode="lines", name="food prob",
        line=dict(color="#56c456", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=weeks, y=water, mode="lines", name="water prob",
        line=dict(color="#2ea8e0", width=1.5),
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="resource adaptation  [0.5 = unadapted baseline]",
                   font=dict(color=DIMTEXT, size=10, family=FONT)),
        yaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False,
                   range=[0, 1], title_text="probability"),
        showlegend=True,
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        height=160,
    ))
    return fig


def generation_adaptation(run: dict, species: str, hab_id: str | None) -> go.Figure:
    """Mean generation (left) vs food/water probability (right) over time."""
    rows = (
        run["species_per_hab"].get(hab_id, {}).get(species, [])
        if hab_id
        else run["species_global"].get(species, [])
    )
    weeks = [r["week"] for r in rows]
    gens  = [r.get("mean_generation") for r in rows]
    food  = [r.get("mean_food_prob")  for r in rows] if hab_id else [None] * len(rows)
    water = [r.get("mean_water_prob") for r in rows] if hab_id else [None] * len(rows)

    if not weeks:
        return _empty_fig("no generation data", height=180)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weeks, y=gens, mode="lines", name="mean generation",
        line=dict(color="#c4dcc4", width=1.5),
        yaxis="y1",
    ))
    if hab_id and any(v is not None for v in food):
        fig.add_hline(y=0.5, line=dict(color="#333", dash="dot", width=1), yref="y2")
        fig.add_trace(go.Scatter(
            x=weeks, y=food, mode="lines", name="food prob",
            line=dict(color="#56c456", width=1.2, dash="dash"),
            yaxis="y2",
        ))
        fig.add_trace(go.Scatter(
            x=weeks, y=water, mode="lines", name="water prob",
            line=dict(color="#2ea8e0", width=1.2, dash="dash"),
            yaxis="y2",
        ))
    fig.update_layout(**_base_layout(
        title=dict(text="generation vs adaptation",
                   font=dict(color=DIMTEXT, size=10, family=FONT)),
        yaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False,
                   title_text="generation"),
        yaxis2=dict(
            color="#5a9a5a", overlaying="y", side="right",
            zeroline=False, title_text="prob", range=[0, 1],
        ),
        showlegend=True,
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        height=180,
    ))
    return fig


def global_overview(run: dict) -> go.Figure:
    gs   = run["global_series"]
    days = [r["week"] for r in gs]
    pop  = [r["population"]    for r in gs]
    spc  = [r["species_count"] for r in gs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=pop, name="population",
        line=dict(color="#3a7a3a", width=1.5),
        fill="tozeroy", fillcolor=_hex_to_rgba("#3a7a3a", 0.13),
    ))
    fig.add_trace(go.Scatter(
        x=days, y=spc, name="species",
        line=dict(color="#7a5a3a", width=1.5, dash="dash"),
        yaxis="y2",
    ))
    fig.update_layout(**_base_layout(
        yaxis2=dict(
            color="#7a5a3a", overlaying="y", side="right",
            zeroline=False, title_text="species",
        ),
        yaxis=dict(color=DIMTEXT, gridcolor=BORDER, zeroline=False, title_text="pop"),
        hovermode="x unified",
        showlegend=True,
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        height=200,
    ))
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
        fig.add_trace(go.Scatter(
            x=list(fwd_by_week.keys()), y=list(fwd_by_week.values()),
            name=f"{src_name}→{tgt_name}", mode="lines",
            line=dict(color="#3a7a3a", width=1.2),
        ))
    if rev_by_week:
        fig.add_trace(go.Scatter(
            x=list(rev_by_week.keys()), y=list(rev_by_week.values()),
            name=f"{tgt_name}→{src_name}", mode="lines",
            line=dict(color="#7a5a3a", width=1.2, dash="dot"),
        ))
    fig.update_layout(**_base_layout(
        showlegend=True,
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


# ── Main-canvas views ─────────────────────────────────────────────────────────

def species_phylogeny(run: dict) -> go.Figure:
    """
    Speciation tree: nodes = species, edges = parent→child speciation.
    X-axis = week of first appearance; Y-axis = branching layout position.
    """
    lineage = run.get("species_lineage", {})
    peak_pop = run.get("species_peak_population", {})

    if not lineage:
        return _empty_fig("no speciation data", height=600)

    # ── Tree layout: in-order traversal assigns y positions ─────────────────
    children_map: dict[str, list[str]] = {sp: lineage[sp].get("children", []) for sp in lineage}
    roots = [sp for sp in lineage if lineage[sp].get("parent") is None]

    positions: dict[str, tuple[float, float]] = {}
    counter = [0]

    def _assign(sp: str) -> None:
        kids = sorted(children_map.get(sp, []),
                      key=lambda k: lineage[k].get("week", 0))
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
    edge_x: list = []
    edge_y: list = []
    for sp, data in lineage.items():
        parent = data.get("parent")
        if parent and parent in positions and sp in positions:
            px, py = positions[parent]
            cx, cy = positions[sp]
            # L-shaped cladogram edge: horizontal from parent, then vertical to child
            edge_x.extend([px, cx, cx, None])
            edge_y.extend([py, py, cy, None])

    fig = go.Figure()

    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color="#2a5a2a", width=1),
            hoverinfo="skip",
        ))

    # ── Node trace ───────────────────────────────────────────────────────────
    node_x = [positions[sp][0] for sp in lineage if sp in positions]
    node_y = [positions[sp][1] for sp in lineage if sp in positions]
    node_sp = [sp for sp in lineage if sp in positions]
    node_sizes = [
        max(8, min(24, 8 + 16 * peak_pop.get(sp, 0) / max(1, max(peak_pop.values()))))
        for sp in node_sp
    ]
    node_labels = [
        f"{sp}<br>week: {lineage[sp].get('week', 0)}"
        f"<br>peak pop: {peak_pop.get(sp, 0)}"
        f"<br>parent: {lineage[sp].get('parent') or 'founder'}"
        for sp in node_sp
    ]
    node_customdata = [{"species": sp} for sp in node_sp]

    # Colour founders differently from speciated species
    node_colors = [
        "#e4f4a0" if lineage[sp].get("parent") is None else "#56c456"
        for sp in node_sp
    ]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
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
        name="",
    ))

    # ── Legend annotations ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#e4f4a0", size=10),
        name="founder species",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="#56c456", size=10),
        name="speciated species",
    ))

    all_weeks = run.get("all_weeks", [0])
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor="#050805",
        font=dict(color=TEXT, family=FONT, size=10),
        margin=dict(t=40, b=40, l=60, r=20),
        xaxis=dict(
            color=DIMTEXT, gridcolor=BORDER, zeroline=False,
            title_text="week of first appearance",
            range=[-5, max(all_weeks) * 1.05],
        ),
        yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(
            font=dict(size=10, family=FONT),
            bgcolor="rgba(0,0,0,0)",
            x=0.01, y=0.99,
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

    RING_RADIUS = 2.0   # distance between rings
    MAX_PER_RING = 14   # cap ancestors per generation ring

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
            positions[cid] = (radius * math.cos(angle_rad),
                              radius * math.sin(angle_rad))

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
        "focus":      {"color": "#e4f4e4", "size": 18, "sym": "circle"},
        "ancestor":   {"color": "#56c456", "size": 11, "sym": "circle"},
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
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color="#253a25", width=1),
            hoverinfo="skip",
        ))

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

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
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
        ))

    # ── Sex labels on close relatives ─────────────────────────────────────────
    label_cids = (
        [creature_id]
        + ancestors_by_gen.get(1, [])
        + descendants_by_gen.get(1, [])
    )
    lx = [positions[c][0] for c in label_cids if c in positions]
    ly = [positions[c][1] for c in label_cids if c in positions]
    lt = [births.get(c, {}).get("sex", "?")[0].upper() for c in label_cids if c in positions]

    fig.add_trace(go.Scatter(
        x=lx, y=ly, mode="text", text=lt,
        textfont=dict(size=7, color="#0a0e0a"),
        hoverinfo="skip",
    ))

    # ── Ring / tier labels ────────────────────────────────────────────────────
    gen_labels_anc = {1: "parents", 2: "grandparents", 3: "great-grandparents"}
    gen_labels_des = {1: "children", 2: "grandchildren", 3: "great-grandchildren"}
    for gen in sorted(ancestors_by_gen):
        fig.add_annotation(
            x=0, y=gen * RING_RADIUS + 0.35,
            text=gen_labels_anc.get(gen, f"anc gen {gen}"),
            showarrow=False,
            font=dict(size=9, color=DIMTEXT, family=FONT),
        )
    for gen in sorted(descendants_by_gen):
        fig.add_annotation(
            x=0, y=-gen * RING_RADIUS - 0.35,
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
        x=0, y=-0.45,
        text=focus_label, showarrow=False,
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
            x=0.01, y=0.99,
        ),
        hovermode="closest",
        clickmode="event",
        height=600,
    )
    return fig
