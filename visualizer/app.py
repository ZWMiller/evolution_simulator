"""
Dash application factory + all callbacks.
"""

import json

import dash
import dash_cytoscape as cyto
from dash import dcc, html, Input, Output, State, ALL, callback_context, no_update

from visualizer.layout import build
from visualizer.network import build_cyto_elements
from visualizer.panels import render
import visualizer.figures as figs
from visualizer.style import BG, PANEL_BG, BORDER, FONT

cyto.load_extra_layouts()

_PANEL_OPEN_STYLE = dict(
    width="460px", display="flex", flexDirection="column",
    backgroundColor=PANEL_BG, overflow="hidden",
    transition="width 0.15s ease", minWidth="0",
)
_PANEL_CLOSED_STYLE = dict(
    width="0", display="flex", flexDirection="column",
    backgroundColor=PANEL_BG, overflow="hidden",
    transition="width 0.15s ease", minWidth="0",
)

_CANVAS_SHOW = dict(position="absolute", inset="0", display="flex",
                    flexDirection="column", overflow="hidden")
_CANVAS_HIDE = dict(position="absolute", inset="0", display="none",
                    overflow="hidden")

# Resize-handle drag behaviour injected into the page HTML.
_RESIZE_JS = """
<script>
(function() {
  function init() {
    var handle = document.getElementById('resize-handle');
    var panel  = document.getElementById('panel-container');
    if (!handle || !panel) { setTimeout(init, 150); return; }

    var dragging = false, startX = 0, startW = 0;

    handle.addEventListener('mousedown', function(e) {
      dragging = true;
      startX = e.clientX;
      startW = panel.offsetWidth;
      document.body.style.cursor    = 'ew-resize';
      document.body.style.userSelect = 'none';
      handle.style.backgroundColor   = '#3a6a3a';
      e.preventDefault();
    });

    document.addEventListener('mousemove', function(e) {
      if (!dragging) return;
      var newW = Math.max(280, Math.min(900, startW + (startX - e.clientX)));
      panel.style.width      = newW + 'px';
      panel.style.transition = 'none';
    });

    document.addEventListener('mouseup', function() {
      if (!dragging) return;
      dragging = false;
      document.body.style.cursor    = '';
      document.body.style.userSelect = '';
      handle.style.backgroundColor   = '';
      panel.style.transition = 'width 0.15s ease';
      window.dispatchEvent(new Event('resize'));
    });

    handle.addEventListener('mouseenter', function() {
      if (!dragging) handle.style.backgroundColor = '#2a4a2a';
    });
    handle.addEventListener('mouseleave', function() {
      if (!dragging) handle.style.backgroundColor = '';
    });
  }
  init();
})();
</script>
"""

_INDEX_HTML = f"""
<!DOCTYPE html>
<html>
<head>
{{%metas%}}
<title>{{%title%}}</title>
{{%favicon%}}
{{%css%}}
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 0; background: {BG}; }}
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: {BG}; }}
  ::-webkit-scrollbar-thumb {{ background: #2a3a2a; border-radius: 2px; }}
  .rc-slider-track {{ background-color: #3a6a3a !important; }}
  .rc-slider-handle {{ border-color: #3a6a3a !important; background: #1a2a1a !important; }}
  .rc-slider-rail  {{ background-color: #1c281c !important; }}
  /* Dropdown theming */
  .Select-control {{ background-color: #0b0f0b !important; border-color: #1c281c !important; }}
  .Select-menu-outer {{ background-color: #0b0f0b !important; border-color: #1c281c !important; }}
  .Select-option {{ background-color: #0b0f0b !important; color: #c4dcc4 !important; }}
  .Select-option.is-focused {{ background-color: #1a2a1a !important; }}
  .Select-value-label {{ color: #c4dcc4 !important; }}
  .Select-placeholder {{ color: #a0b8a0 !important; }}
  button:hover {{ opacity: 0.8; }}
  #resize-handle:hover {{ background-color: #2a4a2a !important; }}
</style>
</head>
<body>
{{%app_entry%}}
<footer>
{{%config%}}
{{%scripts%}}
{{%renderer%}}
</footer>
{_RESIZE_JS}
</body>
</html>
"""

_BORDER_VAL = BORDER  # used in mode-button style callbacks


def make_app(run: dict) -> dash.Dash:
    app = dash.Dash(__name__, title="EvoSim // LOG VIEWER")
    app.index_string = _INDEX_HTML

    layout = build(run)
    # Stores injected at root level (required by Dash pattern-matching callbacks)
    layout.children.insert(0, dcc.Store(id="panel-state",       data=None))
    layout.children.insert(1, dcc.Store(id="tap-store",         data=None))
    layout.children.insert(2, dcc.Store(id="view-mode",         data="habitat"))
    layout.children.insert(3, dcc.Store(id="family-tree-focus", data=None))
    app.layout = layout

    # ── Network graph + day label ──────────────────────────────────────────────
    @app.callback(
        Output("cyto-graph",  "elements"),
        Output("day-display", "children"),
        Input("timeline-slider", "value"),
    )
    def update_graph(day: int):
        return build_cyto_elements(run, day), f"WEEK: {day:04d}"

    # ── Step buttons ───────────────────────────────────────────────────────────
    @app.callback(
        Output("timeline-slider", "value"),
        Input("prev-week-btn", "n_clicks"),
        Input("next-week-btn", "n_clicks"),
        State("timeline-slider", "value"),
        prevent_initial_call=True,
    )
    def step_week(prev_n, next_n, week):
        all_weeks = run["all_weeks"]
        try:
            idx = all_weeks.index(week)
        except ValueError:
            idx = min(range(len(all_weeks)), key=lambda i: abs(all_weeks[i] - week))
        if "prev" in callback_context.triggered[0]["prop_id"]:
            return all_weeks[max(0, idx - 1)]
        return all_weeks[min(len(all_weeks) - 1, idx + 1)]

    # ── Graph tap capture ──────────────────────────────────────────────────────
    @app.callback(
        Output("tap-store", "data"),
        Input("cyto-graph", "tapNodeData"),
        Input("cyto-graph", "tapEdgeData"),
        State("tap-store",  "data"),
        prevent_initial_call=True,
    )
    def record_tap(node_data, edge_data, prev):
        ctx  = callback_context
        prop = ctx.triggered[0]["prop_id"]
        seq  = (prev or {}).get("seq", 0) + 1
        if "tapNodeData" in prop and node_data:
            return {"kind": "node", "data": node_data, "seq": seq}
        if "tapEdgeData" in prop and edge_data:
            return {"kind": "edge", "data": edge_data, "seq": seq}
        return no_update

    # ── View-mode switching ────────────────────────────────────────────────────
    @app.callback(
        Output("view-mode", "data"),
        Input({"type": "mode-btn", "mode": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def switch_mode(clicks):
        ctx = callback_context
        if not ctx.triggered or not ctx.triggered[0]["value"]:
            return no_update
        tid = ctx.triggered_id
        return tid.get("mode", "habitat") if isinstance(tid, dict) else no_update

    # ── Canvas visibility + timeline visibility ────────────────────────────────
    @app.callback(
        Output("habitat-canvas",    "style"),
        Output("phylogeny-canvas",  "style"),
        Output("family-tree-canvas","style"),
        Output("timeline",          "style"),
        Input("view-mode", "data"),
    )
    def update_canvas_visibility(mode: str):
        _show_block = dict(position="absolute", inset="0", display="block", overflow="hidden")
        _show_flex  = dict(position="absolute", inset="0", display="flex",
                           flexDirection="column", overflow="hidden")
        _hide       = dict(position="absolute", inset="0", display="none", overflow="hidden")

        _timeline_show = dict(
            display="flex", height="72px", padding="0 20px",
            backgroundColor=BG, borderTop=f"1px solid {BORDER}",
            alignItems="center", boxSizing="border-box", gap="4px",
        )
        _timeline_hide = dict(display="none")

        if mode == "phylogeny":
            return _hide, _show_block, _hide, _timeline_hide
        if mode == "family_tree":
            return _hide, _hide, _show_flex, _timeline_hide
        # default: habitat
        return _show_block, _hide, _hide, _timeline_show

    # ── Mode button highlight ──────────────────────────────────────────────────
    @app.callback(
        Output({"type": "mode-btn", "mode": ALL}, "style"),
        Input("view-mode", "data"),
    )
    def highlight_mode_btn(mode: str):
        modes = ["habitat", "phylogeny", "family_tree"]
        labels = {"habitat": "HABITAT", "phylogeny": "PHYLOGENY",
                  "family_tree": "FAMILY TREE"}
        styles = []
        for m in modes:
            active = (m == mode)
            styles.append(dict(
                background="none",
                fontFamily=FONT,
                fontSize="11px",
                letterSpacing="0.12em",
                cursor="pointer",
                padding="4px 12px",
                borderRadius="2px",
                border=f"1px solid {'#3a7a3a' if active else BORDER}",
                color="#e4f4e4" if active else "#a0b8a0",
                backgroundColor="#1a2a1a" if active else "transparent",
            ))
        return styles

    # ── Phylogeny figure ───────────────────────────────────────────────────────
    @app.callback(
        Output("phylogeny-graph", "figure"),
        Input("view-mode", "data"),
    )
    def render_phylogeny(mode: str):
        if mode != "phylogeny":
            return no_update
        return figs.species_phylogeny(run)

    # ── Phylogeny click → panel state (open species panel) ────────────────────
    @app.callback(
        Output("panel-state", "data", allow_duplicate=True),
        Input("phylogeny-graph", "clickData"),
        State("panel-state", "data"),
        prevent_initial_call=True,
    )
    def phylogeny_click_to_panel(click_data, current):
        if not click_data:
            return no_update
        points = click_data.get("points", [])
        if not points:
            return no_update
        cd = points[0].get("customdata", {})
        sp = cd.get("species") if isinstance(cd, dict) else None
        if not sp:
            return no_update
        return {"type": "species_global", "species": sp, "back": current}

    # ── Family tree — species dropdown → creature options ──────────────────────
    @app.callback(
        Output("ft-creature-dropdown", "options"),
        Output("ft-creature-dropdown", "value"),
        Input("ft-species-dropdown", "value"),
    )
    def update_creature_options(species: str):
        if not species:
            return [], None

        births = run.get("creature_births", {})
        sp_creatures = [
            (cid, data) for cid, data in births.items()
            if data.get("species") == species
        ]
        if not sp_creatures:
            return [], None

        # Sort: founders first (generation 0), then by week desc for recent ones
        founders = [(cid, d) for cid, d in sp_creatures if d.get("generation", 1) == 0]
        others   = sorted(
            [(cid, d) for cid, d in sp_creatures if d.get("generation", 1) > 0],
            key=lambda x: -x[1].get("week", 0),
        )
        ordered = founders + others[:80]  # max ~100 options

        options = [
            {
                "label": (
                    f"[gen {d.get('generation', '?')}] "
                    f"wk {d.get('week', '?')}  "
                    f"{d.get('sex', '?')[0].upper()}  "
                    f"{cid[:8]}…"
                ),
                "value": cid,
            }
            for cid, d in ordered
        ]
        # Default to first founder (or first creature if none)
        default = ordered[0][0] if ordered else None
        return options, default

    # ── Family tree focus store (dropdown or wheel-click) ─────────────────────
    @app.callback(
        Output("family-tree-focus", "data"),
        Input("ft-creature-dropdown",  "value"),
        Input("family-tree-graph",     "clickData"),
        prevent_initial_call=True,
    )
    def update_family_tree_focus(dropdown_val, click_data):
        ctx  = callback_context
        prop = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        if "family-tree-graph.clickData" in prop and click_data:
            points = click_data.get("points", [])
            if points:
                cd = points[0].get("customdata", {})
                cid = cd.get("creature_id") if isinstance(cd, dict) else None
                if cid:
                    return cid

        if "ft-creature-dropdown" in prop and dropdown_val:
            return dropdown_val

        return no_update

    # ── Family tree figure ─────────────────────────────────────────────────────
    @app.callback(
        Output("family-tree-graph", "figure"),
        Input("family-tree-focus", "data"),
        Input("view-mode",          "data"),
    )
    def render_family_tree(creature_id: str, mode: str):
        if mode != "family_tree":
            return no_update
        if not creature_id:
            return figs._empty_fig("select a species and creature above", height=550)
        return figs.family_tree_wheel(run, creature_id)

    # ── Panel state machine ────────────────────────────────────────────────────
    @app.callback(
        Output("panel-state", "data"),
        Input("tap-store",                            "data"),
        Input({"type": "panel-action", "action": ALL}, "n_clicks"),
        Input({"type": "sp-btn",       "index":  ALL}, "n_clicks"),
        Input({"type": "trait-btn",    "index":  ALL}, "n_clicks"),
        State("timeline-slider", "value"),
        State("panel-state",     "data"),
        prevent_initial_call=True,
    )
    def update_panel_state(tap_store, _pa, _sb, _tb, day, current):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        prop = ctx.triggered[0]["prop_id"]
        tid  = ctx.triggered_id
        val  = ctx.triggered[0].get("value")

        # Graph tap (via decoupled store)
        if prop == "tap-store.data" and tap_store:
            kind = tap_store.get("kind")
            data = tap_store.get("data", {})
            if kind == "node":
                return {"type": "habitat", "hab_id": data["id"], "back": None}
            if kind == "edge":
                return {
                    "type":   "edge",
                    "source": tap_store["data"]["source"],
                    "target": tap_store["data"]["target"],
                    "back":   None,
                }

        if not isinstance(tid, dict):
            return no_update

        if not val:
            return no_update

        kind = tid.get("type")

        if kind == "panel-action":
            action = json.loads(tid.get("action", "{}"))
            k = action.get("kind")
            if k == "close":
                return None
            if k in ("config", "meta"):
                return {"type": "config", "back": current}
            if k == "global":
                return {"type": "global", "back": None}
            if k == "back":
                return current.get("back") if current else None

        if kind == "sp-btn":
            data = json.loads(tid.get("index", "{}"))
            sp_type = data.get("type", "species_in_hab")
            if sp_type == "species_global":
                return {"type": "species_global", "species": data["species"], "back": current}
            return {
                "type":    "species_in_hab",
                "hab_id":  data["hab_id"],
                "species": data["species"],
                "back":    current,
            }

        if kind == "trait-btn":
            return json.loads(tid.get("index", "{}"))

        return no_update

    # ── Panel rendering ────────────────────────────────────────────────────────
    @app.callback(
        Output("panel-body",      "children"),
        Output("panel-container", "style"),
        Input("panel-state",      "data"),
        Input("timeline-slider",  "value"),
    )
    def render_panel(state, day: int):
        if not state:
            return [], _PANEL_CLOSED_STYLE
        return render(state, run, day), _PANEL_OPEN_STYLE

    return app
