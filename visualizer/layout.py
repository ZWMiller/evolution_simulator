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


def build(run: dict) -> html.Div:
    all_days = run["all_days"]
    min_day  = all_days[0]
    max_day  = all_days[-1]
    step     = max(1, len(all_days) // 10)
    summary  = run["summary"]

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
                f"DAYS: {run['days_simulated']}",
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
            html.Span(
                id="day-display",
                children=f"DAY: {min_day:04d}",
                style=dict(color=BRIGHT, marginLeft="auto",
                           fontSize="13px", letterSpacing="0.1em"),
            ),
        ],
        style=dict(
            display="flex", alignItems="center", height="46px",
            padding="0 20px", backgroundColor=BG,
            borderBottom=f"1px solid {BORDER}",
            fontFamily=FONT, boxSizing="border-box",
        ),
    )

    graph_area = html.Div(
        cyto.Cytoscape(
            id="cyto-graph",
            elements=build_cyto_elements(run, min_day),
            layout={"name": "preset"},
            stylesheet=CYTO_STYLESHEET,
            style={"width": "100%", "height": "100%", "backgroundColor": BG},
            userZoomingEnabled=True,
            userPanningEnabled=True,
            boxSelectionEnabled=False,
        ),
        id="graph-container",
        style=dict(flex="1", overflow="hidden", position="relative", minWidth="200px"),
    )

    # Drag handle between graph and panel
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
        [graph_area, resize_handle, panel],
        id="main-area",
        style=dict(display="flex", flex="1", overflow="hidden"),
    )

    marks = {d: str(d) for d in range(min_day, max_day + 1, step)}
    marks[max_day] = str(max_day)

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
            html.Button("◀", id="prev-day-btn", n_clicks=0, style=_step_btn),
            html.Div(
                dcc.Slider(
                    id="timeline-slider",
                    min=min_day, max=max_day, step=1,
                    value=min_day, marks=marks,
                    tooltip={"placement": "top", "always_visible": True},
                    updatemode="drag",
                ),
                style=dict(flex="1", alignSelf="center", padding="0 8px"),
            ),
            html.Button("▶", id="next-day-btn", n_clicks=0, style=_step_btn),
        ],
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
