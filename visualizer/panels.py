"""
Panel content renderers.  Each public function receives the current panel
state dict (plus the full run dict and selected day) and returns a list of
Dash components that are dropped into the scrollable panel body.

Panel state shapes:
  None                                           → panel closed
  {"type": "config"}                             → raw config text
  {"type": "global"}                             → global stats + species list
  {"type": "habitat",       "hab_id": str}       → habitat detail + species list
  {"type": "species_in_hab","hab_id": str,
                             "species": str,
                             "trait": str|None}  → species stats within habitat
  {"type": "species_global","species": str,
                             "trait": str|None}  → species stats globally
  {"type": "edge",          "source": str,
                             "target": str}      → migration event detail
Every state may carry "back": <previous_state> for breadcrumb navigation.
"""

import json

from dash import dcc, html

import visualizer.figures as figs
from visualizer.style import (
    BG, PANEL_BG, BORDER, TEXT, DIMTEXT, BRIGHT, FONT,
    HABITAT_COLORS, KEY_TRAITS, QUICK_TRAITS,
)


# ── Component primitives ──────────────────────────────────────────────────────

def _s(**kw) -> dict:
    return kw


def _section(title: str, children: list) -> html.Div:
    return html.Div([
        html.Div(
            f"── {title} ",
            style=_s(color=DIMTEXT, fontSize="11px", letterSpacing="0.1em",
                     marginTop="16px", marginBottom="7px",
                     paddingBottom="4px", borderBottom=f"1px solid {BORDER}"),
        ),
        *children,
    ])


def _kv(key: str, val) -> html.Div:
    return html.Div([
        html.Span(f"{key}: ", style=_s(color=DIMTEXT, fontSize="13px")),
        html.Span(str(val),   style=_s(color=TEXT,    fontSize="13px")),
    ], style=_s(marginBottom="5px"))


def _species_button(label: str, index_data: dict) -> html.Button:
    return html.Button(
        f"  {label}",
        id={"type": "sp-btn", "index": json.dumps(index_data)},
        n_clicks=0,
        style=_s(
            display="block", width="100%", textAlign="left",
            background="none", border="none",
            borderBottom=f"1px solid {BORDER}",
            color=TEXT, fontFamily=FONT, fontSize="13px",
            padding="6px 4px", cursor="pointer",
        ),
    )


def _trait_button(trait: str, panel_state: dict) -> html.Button:
    active  = panel_state.get("trait") == trait
    new_state = {**panel_state, "trait": trait}
    return html.Button(
        trait.replace("_", " "),
        id={"type": "trait-btn", "index": json.dumps(new_state)},
        n_clicks=0,
        style=_s(
            background="#1a2a1a" if active else "none",
            border=f"1px solid {'#3a6a3a' if active else BORDER}",
            color="#88ee88" if active else DIMTEXT,
            fontFamily=FONT, fontSize="12px",
            padding="4px 8px", cursor="pointer", margin="2px 2px 2px 0",
            borderRadius="2px",
        ),
    )


def _back_button(state: dict) -> html.Button | None:
    if not state.get("back"):
        return None
    return html.Button(
        "< BACK",
        id={"type": "panel-action", "action": json.dumps({"kind": "back"})},
        n_clicks=0,
        style=_s(
            background="none", border="none",
            color=DIMTEXT, fontFamily=FONT, fontSize="12px",
            cursor="pointer", padding="0 0 10px 0", letterSpacing="0.08em",
        ),
    )


def _graph(figure, margin_bottom="4px") -> html.Div:
    return html.Div(
        dcc.Graph(figure=figure, config={"displayModeBar": False},
                  responsive=True),
        style=_s(marginBottom=margin_bottom),
    )


# ── Public dispatcher ─────────────────────────────────────────────────────────

def render(state: dict | None, run: dict, day: int) -> list:
    if state is None:
        return []
    dispatch = {
        "config":         _config,
        "global":         _global,
        "habitat":        _habitat,
        "species_in_hab": _species,
        "species_global": _species,
        "edge":           _edge,
    }
    fn = dispatch.get(state.get("type"))
    return fn(state, run, day) if fn else []


# ── Individual panel renderers ────────────────────────────────────────────────

def _config(state: dict, run: dict, day: int) -> list:
    back = _back_button(state)
    items = [back] if back else []
    items += [
        html.Div("CONFIG", style=_s(color=BRIGHT, fontSize="12px",
                                    letterSpacing="0.15em", fontWeight="bold",
                                    marginBottom="10px")),
        html.Pre(
            run["config_text"] or "(no config.toml found in log directory)",
            style=_s(color=DIMTEXT, fontSize="12px", whiteSpace="pre-wrap",
                     overflowX="auto", lineHeight="1.6",
                     background="#050805", padding="10px", borderRadius="2px"),
        ),
    ]
    return items


def _global(state: dict, run: dict, day: int) -> list:
    d       = run["days_data"].get(day, {})
    gs      = run["global_series"]
    snapshot = next((r for r in reversed(gs) if r["day"] <= day), gs[0] if gs else {})
    summary = run["summary"]

    deaths_today = sum(
        len(run["days_data"][day].get("habitats", {}).get(h, {}).get("deaths", []))
        for h in run["hab_ids"]
        if day in run["days_data"]
    )
    spec_so_far = [
        ev
        for dn, evs in run["speciation_by_day"].items()
        for ev in evs
        if dn <= day
    ]

    back = _back_button(state)
    items = [back] if back else []
    items += [
        html.Div("GLOBAL STATS", style=_s(color=BRIGHT, fontSize="15px",
                                           letterSpacing="0.15em", fontWeight="bold",
                                           marginBottom="10px")),
        _graph(figs.global_overview(run)),
        _section("SNAPSHOT", [
            _kv("day",               day),
            _kv("global population", snapshot.get("population", 0)),
            _kv("living species",    snapshot.get("species_count", 0)),
            _kv("species ever",      summary["total_species_ever"]),
            _kv("speciations",       len(spec_so_far)),
            _kv("deaths today",      deaths_today),
            _kv("status",            "EXTINCT" if run["extinct"] else "alive"),
        ]),
    ]

    # Species active on this day
    dist   = d.get("global_species_distribution", {})
    active = sorted(dist.items(), key=lambda x: -x[1])
    if active:
        items.append(_section("SPECIES ON THIS DAY", []))
        for sp, cnt in active:
            items.append(
                _species_button(f"{sp}  ({cnt})", {"type": "species_global", "species": sp})
            )

    # Recent speciations
    recent = spec_so_far[-10:]
    if recent:
        items.append(_section(f"RECENT SPECIATIONS (last {len(recent)})", [
            html.Div(
                f"day {ev.get('day', ev_day):>5}  {ev['new_species']}  ← {ev['parent_species']}",
                style=_s(color=DIMTEXT, fontSize="12px", marginBottom="3px",
                         fontFamily=FONT, whiteSpace="nowrap",
                         overflow="hidden", textOverflow="ellipsis"),
            )
            for ev_day, evs in run["speciation_by_day"].items()
            for ev in evs
            if ev_day <= day
        ][-10:]))

    return items


def _habitat(state: dict, run: dict, day: int) -> list:
    hab_id   = state["hab_id"]
    hab_name = run["hab_names"].get(hab_id, hab_id)
    hab_type = run["hab_types"].get(hab_id, "")
    pop      = run["hab_pop"][hab_id].get(day, 0)
    _, border, text_c = HABITAT_COLORS.get(hab_type, ("#1a1a1a", "#4a4a4a", "#888888"))

    d       = run["days_data"].get(day, {})
    hab_log = d.get("habitats",     {}).get(hab_id, {})
    births  = hab_log.get("births",         [])
    deaths  = hab_log.get("deaths",         [])
    migs    = hab_log.get("migrations_out", [])
    isos    = hab_log.get("isolations",     [])
    sp_dist = hab_log.get("species_distribution", {})

    conns = [
        b if a == hab_id else a
        for a, b in run["connections"]
        if a == hab_id or b == hab_id
    ]

    back  = _back_button(state)
    items = [back] if back else []
    items += [
        html.Div(hab_name.upper(),
                 style=_s(color=text_c, fontSize="15px", letterSpacing="0.12em",
                          fontWeight="bold", marginBottom="2px")),
        html.Div(hab_type,
                 style=_s(color=DIMTEXT, fontSize="12px", letterSpacing="0.1em",
                          marginBottom="10px")),
        _section("STATUS", [
            _kv("population",   pop),
            _kv("species",      len([s for s, c in sp_dist.items() if c > 0])),
            _kv("connected to", ", ".join(run["hab_names"].get(c, c) for c in conns)),
            _kv("births today", len(births)),
            _kv("deaths today", len(deaths)),
            _kv("migrants out", len(migs)),
            _kv("isolations",   len(isos)),
        ]),
    ]

    active = sorted(sp_dist.items(), key=lambda x: -x[1]) if sp_dist else []
    if active:
        items.append(_section("SPECIES", []))
        for sp, cnt in active:
            items.append(
                _species_button(
                    f"{sp}  ({cnt})",
                    {"type": "species_in_hab", "hab_id": hab_id, "species": sp},
                )
            )

    return items


def _species(state: dict, run: dict, day: int) -> list:
    species   = state["species"]
    hab_scoped = state["type"] == "species_in_hab"
    hab_id    = state.get("hab_id") if hab_scoped else None
    hab_name  = run["hab_names"].get(hab_id, "global") if hab_id else "global"
    hab_type  = run["hab_types"].get(hab_id, "Forest") if hab_id else "Forest"
    _, border, text_c = HABITAT_COLORS.get(hab_type, ("#1a1a1a", "#4a4a4a", "#888888"))
    selected_trait = state.get("trait")

    # Stats for this day
    if hab_scoped and hab_id:
        day_rows = [
            r for r in run["species_per_hab"].get(hab_id, {}).get(species, [])
            if r["day"] == day
        ]
    else:
        day_rows = [r for r in run["species_global"].get(species, []) if r["day"] == day]

    count  = day_rows[0]["count"] if day_rows else 0
    traits = day_rows[0] if day_rows else {}

    # Daily events
    births = migs_in = migs_out = []
    if day in run["days_data"]:
        d = run["days_data"][day]
        if hab_id:
            hl     = d.get("habitats", {}).get(hab_id, {})
            births = [b for b in hl.get("births",         []) if b.get("species") == species]
            migs_out = [m for m in hl.get("migrations_out", []) if m.get("species") == species]
        migs_in = [
            m for m in d.get("migrations", [])
            if m["species"] == species and (not hab_id or m["to_habitat"] == hab_id)
        ]

    back  = _back_button(state)
    items = [back] if back else []
    items += [
        html.Div(species.upper(),
                 style=_s(color=text_c, fontSize="15px", letterSpacing="0.1em",
                          fontWeight="bold", marginBottom="2px")),
        html.Div(f"in {hab_name}" if hab_scoped else "global",
                 style=_s(color=DIMTEXT, fontSize="12px", marginBottom="10px")),
        _section("SNAPSHOT", [
            _kv("count",        count),
            _kv("births today", len(births)),
            _kv("migrants in",  len(migs_in)),
            _kv("migrants out", len(migs_out)),
        ]),
    ]

    trait_rows = [
        _kv(t.replace("_", " "), f"{traits[t]:.4f}")
        for t in KEY_TRAITS
        if traits.get(t) is not None
    ]
    if trait_rows:
        items.append(_section("TRAITS", trait_rows))

    # Population chart (always shown)
    items.append(_section("POPULATION OVER TIME", [
        _graph(figs.species_population(run, species, hab_id if hab_scoped else None)),
    ]))

    # Trait evolution — buttons + optional chart
    items.append(_section("TRAIT EVOLUTION", [
        html.Div(
            [_trait_button(t, state) for t in QUICK_TRAITS],
            style=_s(marginBottom="6px", lineHeight="2"),
        ),
    ]))
    if selected_trait:
        items.append(
            _graph(figs.species_trait(run, species, selected_trait, hab_id if hab_scoped else None))
        )

    # Birth events
    if births:
        items.append(_section(f"BORN TODAY ({len(births)})", [
            html.Div(
                f"{b.get('sex', '?')[0].upper()}  id={b['creature_id'][:8]}…",
                style=_s(color=DIMTEXT, fontSize="12px", marginBottom="3px", fontFamily=FONT),
            )
            for b in births[:12]
        ]))

    if migs_in or migs_out:
        mig_items = [
            html.Div(f"→ IN   from {m['from_habitat']}",
                     style=_s(color="#50c050", fontSize="12px", marginBottom="3px"))
            for m in migs_in[:8]
        ] + [
            html.Div(f"← OUT  to   {m.get('to_habitat', '?')}",
                     style=_s(color="#c08040", fontSize="12px", marginBottom="3px"))
            for m in migs_out[:8]
        ]
        items.append(_section("MIGRATIONS TODAY", mig_items))

    return items


def _edge(state: dict, run: dict, day: int) -> list:
    src      = state["source"]
    tgt      = state["target"]
    src_name = run["hab_names"].get(src, src)
    tgt_name = run["hab_names"].get(tgt, tgt)

    migs_fwd = [m for m in run["migrations_by_day"].get(day, [])
                if m["from_habitat"] == src and m["to_habitat"] == tgt]
    migs_rev = [m for m in run["migrations_by_day"].get(day, [])
                if m["from_habitat"] == tgt and m["to_habitat"] == src]

    back  = _back_button(state)
    items = [back] if back else []
    items += [
        html.Div("MIGRATION ROUTE",
                 style=_s(color=BRIGHT, fontSize="15px", letterSpacing="0.12em",
                          fontWeight="bold", marginBottom="4px")),
        html.Div(f"{src_name.upper()}  ↔  {tgt_name.upper()}",
                 style=_s(color=DIMTEXT, fontSize="12px", marginBottom="10px")),
        _section("TODAY", [
            _kv(f"→  {src_name} → {tgt_name}", len(migs_fwd)),
            _kv(f"←  {tgt_name} → {src_name}", len(migs_rev)),
        ]),
        _section("HISTORY", [_graph(figs.edge_migration(run, src, tgt))]),
    ]

    for direction, mig_list, label in [
        (f"→ {src_name} → {tgt_name} TODAY ({len(migs_fwd)})", migs_fwd, "from_habitat"),
        (f"← {tgt_name} → {src_name} TODAY ({len(migs_rev)})", migs_rev, "from_habitat"),
    ]:
        if mig_list:
            items.append(_section(direction, [
                html.Div(
                    f"{m['species']}  id={m['creature_id'][:8]}…",
                    style=_s(color=DIMTEXT, fontSize="12px", marginBottom="3px"),
                )
                for m in mig_list[:15]
            ]))

    return items
