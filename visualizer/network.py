"""
Cytoscape element builder and stylesheet for the habitat network graph.
"""

from visualizer.style import HABITAT_COLORS, FONT


def build_cyto_elements(run: dict, day: int) -> list:
    elements = []

    for hab_id in run["hab_ids"]:
        hab_type = run["hab_types"][hab_id]
        hab_name = run["hab_names"][hab_id]
        pop      = run["hab_pop"][hab_id].get(day, 0)
        bg, border, text_c = HABITAT_COLORS.get(hab_type, ("#1a1a1a", "#4a4a4a", "#888888"))
        pos  = run["node_positions"][hab_id]
        size = max(70, min(130, 60 + pop ** 0.42))

        short = hab_name.upper()
        elements.append({
            "data": {
                "id":           hab_id,
                "label":        f"{short}\n{hab_type}  {pop:,}",
                "hab_type":     hab_type,
                "population":   pop,
                "bg_color":     bg,
                "border_color": border,
                "text_color":   text_c,
                "node_size":    size,
            },
            "position": pos,
        })

    for src, tgt in run["connections"]:
        elements.append({
            "data": {
                "id":     f"{src}__{tgt}",
                "source": src,
                "target": tgt,
            }
        })

    return elements


CYTO_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label":            "data(label)",
            "text-valign":      "center",
            "text-halign":      "center",
            "font-family":      "Courier New, Courier, monospace",
            "font-size":        "9px",
            "color":            "data(text_color)",
            "background-color": "data(bg_color)",
            "border-color":     "data(border_color)",
            "border-width":     2,
            "width":            "data(node_size)",
            "height":           "data(node_size)",
            "text-wrap":        "wrap",
            "text-max-width":   "68px",
            "shape":            "round-rectangle",
        },
    },
    {
        "selector": "edge",
        "style": {
            "line-color":         "#2a3a2a",
            "width":              1.5,
            "curve-style":        "bezier",
            "target-arrow-shape": "none",
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "border-color":      "#88ee88",
            "border-width":      3,
            "overlay-color":     "#88ee88",
            "overlay-opacity":   0.08,
        },
    },
    {
        "selector": "edge:selected",
        "style": {
            "line-color": "#88ee88",
            "width":      2.5,
        },
    },
]
