"""
Plotly figure builders used inside panel components.
"""

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
