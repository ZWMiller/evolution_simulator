"""
Top-level Dash layout builder.

The Store for panel state is injected by make_app() after the fact so
that it sits at the root level (required by Dash).
"""

import json

import dash_cytoscape as cyto
from dash import dcc, html

from visualizer.network import build_cyto_elements, CYTO_STYLESHEET
from visualizer.style import BG, PANEL_BG, BORDER, TEXT, DIMTEXT, BRIGHT, FONT


def _mode_btn(label: str, mode: str, active: bool = False) -> html.Button:
    return html.Button(
        label,
        id={"type": "mode-btn", "mode": mode},
        n_clicks=0,
        style=dict(
            background="#1a2a1a" if active else "none",
            fontFamily=FONT,
            fontSize="11px",
            letterSpacing="0.12em",
            cursor="pointer",
            padding="4px 12px",
            borderRadius="2px",
            border=f"1px solid {'#3a7a3a' if active else BORDER}",
            color=BRIGHT if active else DIMTEXT,
        ),
    )


def build(run: dict) -> html.Div:
    all_weeks = run["all_weeks"]
    min_week  = all_weeks[0]
    max_week  = all_weeks[-1]
    summary   = run["summary"]
    all_species = run["all_species"]

    status_color = "#aa2020" if run["extinct"] else "#3a7a3a"
    status_label = "EXTINCT" if run["extinct"] else "ALIVE"

    _btn_style = dict(
        background="none", border="none", cursor="pointer",
        fontFamily=FONT, letterSpacing="0.06em", fontSize="12px",
    )

    header = html.Div(
        [
            html.Span(
                f"SIM: {run['run_id']}",
                id={"type": "panel-action", "action": json.dumps({"kind": "meta"})},
                n_clicks=0,
                style=dict(**_btn_style, color=DIMTEXT, marginRight="24px"),
            ),
            html.Span(
                f"WEEKS: {run['weeks_simulated']}",
                style=dict(color=DIMTEXT, marginRight="24px", fontSize="12px"),
            ),
            html.Span(
                f"SPECIES: {summary['total_species_ever']}",
                style=dict(color=DIMTEXT, marginRight="24px", fontSize="12px"),
            ),
            html.Span(
                status_label,
                style=dict(color=status_color, marginRight="24px",
                           fontSize="12px", fontWeight="bold"),
            ),
            html.Span(
                f"CFG: {run['config_name']}",
                id={"type": "panel-action", "action": json.dumps({"kind": "config"})},
                n_clicks=0,
                style=dict(**_btn_style, color=DIMTEXT, marginRight="24px"),
            ),
            # ── Mode switcher ─────────────────────────────────────────────────
            html.Div(
                [
                    _mode_btn("HABITAT",       "habitat",            active=True),
                    _mode_btn("PHYLOGENY",     "phylogeny"),
                    _mode_btn("FAMILY TREE",   "family_tree"),
                    _mode_btn("TRAIT COMPARE", "trait_comparison"),
                ],
                id="mode-switcher",
                style=dict(
                    display="flex", gap="6px", marginLeft="auto", marginRight="24px",
                ),
            ),
            html.Span(
                id="day-display",
                children=f"WEEK: {min_week:04d}",
                style=dict(color=BRIGHT, fontSize="13px", letterSpacing="0.1em"),
            ),
        ],
        style=dict(
            display="flex", alignItems="center", height="46px",
            padding="0 20px", backgroundColor=BG,
            borderBottom=f"1px solid {BORDER}",
            fontFamily=FONT, boxSizing="border-box",
        ),
    )

    # ── Habitat canvas (cytoscape — existing behaviour) ───────────────────────
    habitat_canvas = html.Div(
        cyto.Cytoscape(
            id="cyto-graph",
            elements=build_cyto_elements(run, min_week),
            layout={"name": "preset"},
            stylesheet=CYTO_STYLESHEET,
            style={"width": "100%", "height": "100%", "backgroundColor": BG},
            userZoomingEnabled=True,
            userPanningEnabled=True,
            boxSelectionEnabled=False,
        ),
        id="habitat-canvas",
        style=dict(
            position="absolute", inset="0",
            display="block", overflow="hidden",
        ),
    )

    # ── Phylogeny canvas ──────────────────────────────────────────────────────
    phylogeny_canvas = html.Div(
        [
            # Control bar
            html.Div(
                [
                    html.Span("min weeks survived:", style=dict(
                        color=DIMTEXT, fontSize="11px", fontFamily=FONT,
                        alignSelf="center", marginRight="8px", whiteSpace="nowrap",
                    )),
                    dcc.Input(
                        id="phylo-min-weeks",
                        type="number",
                        min=0,
                        step=1,
                        value=0,
                        debounce=True,
                        placeholder="0",
                        style=dict(
                            width="90px", backgroundColor="#0b0f0b",
                            color=TEXT, fontFamily=FONT, fontSize="12px",
                            border=f"1px solid {BORDER}", borderRadius="2px",
                            padding="2px 6px",
                        ),
                    ),
                    html.Span(
                        "  (filters out species with a shorter lifespan)",
                        style=dict(
                            color=DIMTEXT, fontSize="10px", fontFamily=FONT,
                            alignSelf="center", marginLeft="8px",
                        ),
                    ),
                ],
                style=dict(
                    display="flex", alignItems="center",
                    padding="8px 16px", height="46px",
                    backgroundColor=BG, borderBottom=f"1px solid {BORDER}",
                    boxSizing="border-box", flexShrink="0",
                ),
            ),
            dcc.Graph(
                id="phylogeny-graph",
                figure={},
                config={"displayModeBar": "hover", "scrollZoom": True},
                responsive=True,
                style={"flex": "1", "minHeight": "0"},
                clear_on_unhover=True,
            ),
        ],
        id="phylogeny-canvas",
        style=dict(
            position="absolute", inset="0",
            display="none", overflow="hidden",
            flexDirection="column",
        ),
    )

    # ── Family tree canvas ────────────────────────────────────────────────────
    family_tree_canvas = html.Div(
        [
            # Control bar: species selector + creature selector
            html.Div(
                [
                    html.Span("species:", style=dict(
                        color=DIMTEXT, fontSize="11px", fontFamily=FONT,
                        alignSelf="center", marginRight="8px",
                    )),
                    dcc.Dropdown(
                        id="ft-species-dropdown",
                        options=[{"label": sp, "value": sp} for sp in all_species],
                        placeholder="select species…",
                        clearable=True,
                        style=dict(
                            width="220px", backgroundColor="#0b0f0b",
                            color=TEXT, fontFamily=FONT, fontSize="12px",
                            border=f"1px solid {BORDER}", borderRadius="2px",
                        ),
                    ),
                    html.Span("creature:", style=dict(
                        color=DIMTEXT, fontSize="11px", fontFamily=FONT,
                        alignSelf="center", marginLeft="20px", marginRight="8px",
                    )),
                    dcc.Dropdown(
                        id="ft-creature-dropdown",
                        options=[],
                        placeholder="select creature…",
                        clearable=True,
                        style=dict(
                            width="320px", backgroundColor="#0b0f0b",
                            color=TEXT, fontFamily=FONT, fontSize="12px",
                            border=f"1px solid {BORDER}", borderRadius="2px",
                        ),
                    ),
                    html.Span(
                        "click any node to re-centre",
                        style=dict(
                            color=DIMTEXT, fontSize="10px", fontFamily=FONT,
                            marginLeft="20px", alignSelf="center",
                        ),
                    ),
                ],
                style=dict(
                    display="flex", alignItems="center",
                    padding="8px 16px", height="46px",
                    backgroundColor=BG, borderBottom=f"1px solid {BORDER}",
                    boxSizing="border-box", flexShrink="0",
                ),
            ),
            # Wheel figure
            dcc.Graph(
                id="family-tree-graph",
                figure={},
                config={"displayModeBar": "hover", "scrollZoom": True},
                responsive=True,
                style={"flex": "1", "minHeight": "0"},
                clear_on_unhover=True,
            ),
        ],
        id="family-tree-canvas",
        style=dict(
            position="absolute", inset="0",
            display="none", overflow="hidden",
            flexDirection="column",
        ),
    )

    # ── Trait comparison canvas ───────────────────────────────────────────────
    # Build metric options from actual logged fields so the list stays current
    # even if new traits are added to the simulation.
    _sample_rows = next(iter(run["species_global"].values()), [])
    _sample_row  = _sample_rows[0] if _sample_rows else {}
    _logged_traits = sorted(
        k for k in _sample_row
        if k not in ("week", "count", "mean_generation")
    )
    tc_metric_options = [
        {"label": "── Population ──",   "value": "__h_pop",    "disabled": True},
        {"label": "Population count",   "value": "count"},
        {"label": "Mean generation",    "value": "mean_generation"},
        {"label": "── Resource ──",     "value": "__h_res",    "disabled": True},
        {"label": "Food probability",   "value": "mean_food_prob"},
        {"label": "Water probability",  "value": "mean_water_prob"},
        {"label": "── Traits ──",       "value": "__h_traits", "disabled": True},
    ] + [{"label": t.replace("_", " "), "value": t} for t in _logged_traits]

    _dd_style = dict(
        backgroundColor="#0b0f0b", color=TEXT, fontFamily=FONT,
        fontSize="12px", border=f"1px solid {BORDER}", borderRadius="2px",
    )
    _lbl_style = dict(
        color=DIMTEXT, fontSize="11px", fontFamily=FONT,
        alignSelf="center", marginRight="8px", whiteSpace="nowrap",
    )

    trait_comparison_canvas = html.Div(
        [
            # ── Control bar ───────────────────────────────────────────────────
            html.Div(
                [
                    html.Span("species:", style=_lbl_style),
                    dcc.Dropdown(
                        id="tc-species-dropdown",
                        options=[{"label": sp, "value": sp} for sp in all_species],
                        multi=True,
                        placeholder="select one or more species…",
                        style={**_dd_style, "width": "380px"},
                    ),
                    html.Span("metric:", style={**_lbl_style, "marginLeft": "20px"}),
                    dcc.Dropdown(
                        id="tc-metric-dropdown",
                        options=tc_metric_options,
                        multi=True,
                        placeholder="select one or more metrics…",
                        style={**_dd_style, "width": "340px"},
                    ),
                    dcc.Checklist(
                        id="tc-anagenesis-check",
                        options=[{"label": " include anagenesis descendants",
                                  "value": "yes"}],
                        value=[],
                        style=dict(
                            marginLeft="24px", color=DIMTEXT,
                            fontFamily=FONT, fontSize="11px",
                            alignSelf="center",
                        ),
                        inputStyle={"marginRight": "6px", "accentColor": "#3a7a3a"},
                    ),
                ],
                style=dict(
                    display="flex", alignItems="center",
                    padding="8px 16px", height="54px",
                    backgroundColor=BG, borderBottom=f"1px solid {BORDER}",
                    boxSizing="border-box", flexShrink="0",
                    gap="4px",
                ),
            ),
            # ── Chart ─────────────────────────────────────────────────────────
            dcc.Graph(
                id="trait-comparison-graph",
                figure={},
                config={"displayModeBar": "hover", "scrollZoom": True},
                responsive=True,
                style={"flex": "1", "minHeight": "0"},
            ),
        ],
        id="trait-comparison-canvas",
        style=dict(
            position="absolute", inset="0",
            display="none", overflow="hidden",
            flexDirection="column",
        ),
    )

    # ── Canvas container (all four layers stacked) ────────────────────────────
    canvas_area = html.Div(
        [habitat_canvas, phylogeny_canvas, family_tree_canvas,
         trait_comparison_canvas],
        id="canvas-area",
        style=dict(
            flex="1", position="relative", overflow="hidden", minWidth="200px",
        ),
    )

    # Drag handle between canvas and panel
    resize_handle = html.Div(
        id="resize-handle",
        style=dict(
            width="5px",
            cursor="ew-resize",
            backgroundColor=BORDER,
            flexShrink="0",
            zIndex="10",
            transition="background-color 0.15s",
        ),
    )

    panel = html.Div(
        [
            html.Div(
                html.Button(
                    "×",
                    id={"type": "panel-action", "action": json.dumps({"kind": "close"})},
                    n_clicks=0,
                    style=dict(background="none", border="none", color=DIMTEXT,
                               fontFamily=FONT, fontSize="18px", cursor="pointer",
                               padding="4px 10px", lineHeight="1"),
                ),
                style=dict(backgroundColor=PANEL_BG,
                           borderBottom=f"1px solid {BORDER}",
                           padding="6px 10px", display="flex",
                           justifyContent="flex-end"),
            ),
            html.Div(
                id="panel-body",
                style=dict(padding="12px 16px", overflowY="auto",
                           flex="1", fontFamily=FONT, color=TEXT),
            ),
        ],
        id="panel-container",
        style=dict(
            width="0", display="flex", flexDirection="column",
            backgroundColor=PANEL_BG, overflow="hidden",
            transition="width 0.15s ease",
            minWidth="0",
        ),
    )

    main_area = html.Div(
        [canvas_area, resize_handle, panel],
        id="main-area",
        style=dict(display="flex", flex="1", overflow="hidden"),
    )

    # Build marks from the actual logged weeks only.  With step=None the slider
    # snaps exclusively to these positions, so dragging never lands on a week
    # that has no data.  Show ~10 evenly-spaced labels to keep the track readable.
    n_labels = 10
    label_every = max(1, len(all_weeks) // n_labels)
    label_set = {all_weeks[i] for i in range(0, len(all_weeks), label_every)}
    label_set.add(max_week)
    marks = {w: (str(w) if w in label_set else "") for w in all_weeks}

    _step_btn = dict(
        background="none", border=f"1px solid {BORDER}",
        color=DIMTEXT, fontFamily=FONT, fontSize="16px",
        cursor="pointer", padding="0 10px",
        alignSelf="center", lineHeight="1",
        flexShrink="0", height="28px",
    )

    timeline = html.Div(
        [
            html.Button(
                "GLOBAL STATS",
                id={"type": "panel-action", "action": json.dumps({"kind": "global"})},
                n_clicks=0,
                style=dict(
                    background="none", border=f"1px solid {BORDER}",
                    color=DIMTEXT, fontFamily=FONT, fontSize="11px",
                    cursor="pointer", padding="4px 10px",
                    letterSpacing="0.1em", whiteSpace="nowrap",
                    alignSelf="center", marginRight="16px", flexShrink="0",
                ),
            ),
            html.Button("◀", id="prev-week-btn", n_clicks=0, style=_step_btn),
            html.Div(
                dcc.Slider(
                    id="timeline-slider",
                    min=min_week, max=max_week, step=None,
                    value=min_week, marks=marks,
                    tooltip={"placement": "top", "always_visible": True},
                    updatemode="drag",
                ),
                style=dict(flex="1", alignSelf="center", padding="0 8px"),
            ),
            html.Button("▶", id="next-week-btn", n_clicks=0, style=_step_btn),
        ],
        id="timeline",
        style=dict(
            display="flex", height="72px", padding="0 20px",
            backgroundColor=BG, borderTop=f"1px solid {BORDER}",
            alignItems="center", boxSizing="border-box", gap="4px",
        ),
    )

    return html.Div(
        [header, main_area, timeline],
        style=dict(
            display="flex", flexDirection="column",
            height="100vh", backgroundColor=BG,
            margin="0", padding="0", overflow="hidden",
            fontFamily=FONT,
        ),
    )
