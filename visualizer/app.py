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

# Resize-handle drag behaviour injected into the page HTML.
# Polls until the DOM is ready, then wires up mousedown/mousemove/mouseup.
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
      // Nudge Plotly to reflow any charts inside the resized panel
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


def make_app(run: dict) -> dash.Dash:
    app = dash.Dash(__name__, title="EvoSim // LOG VIEWER")
    app.index_string = _INDEX_HTML

    layout = build(run)
    # Stores must live at the root level to be available to all callbacks
    layout.children.insert(0, dcc.Store(id="panel-state", data=None))
    layout.children.insert(1, dcc.Store(id="tap-store",   data=None))
    app.layout = layout

    # ── Network graph + day label ──────────────────────────────────────────────
    @app.callback(
        Output("cyto-graph",  "elements"),
        Output("day-display", "children"),
        Input("timeline-slider", "value"),
    )
    def update_graph(day: int):
        return build_cyto_elements(run, day), f"DAY: {day:04d}"

    # ── Step buttons ───────────────────────────────────────────────────────────
    @app.callback(
        Output("timeline-slider", "value"),
        Input("prev-day-btn", "n_clicks"),
        Input("next-day-btn", "n_clicks"),
        State("timeline-slider", "value"),
        prevent_initial_call=True,
    )
    def step_day(prev_n, next_n, day):
        all_days = run["all_days"]
        try:
            idx = all_days.index(day)
        except ValueError:
            idx = min(range(len(all_days)), key=lambda i: abs(all_days[i] - day))
        if "prev" in callback_context.triggered[0]["prop_id"]:
            return all_days[max(0, idx - 1)]
        return all_days[min(len(all_days) - 1, idx + 1)]

    # ── Capture graph taps into a store (decouples from element updates) ───────
    # When cytoscape elements are rebuilt (timeline drag), tapNodeData/tapEdgeData
    # may re-fire with stale data, which would incorrectly reset panel state.
    # Using an intermediate store + ignoring null values prevents that.
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
        return no_update  # null tap from element refresh — ignore

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

        # Guard against ghost fires: dynamically created buttons start with
        # n_clicks=0 and briefly trigger pattern-matching callbacks when the
        # panel re-renders. Only process genuine clicks (n_clicks > 0).
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
