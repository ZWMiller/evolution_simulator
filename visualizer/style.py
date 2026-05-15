"""
Shared style tokens, color palette, and trait constants.
"""

BG       = "#080c08"
PANEL_BG = "#0b0f0b"
BORDER   = "#1c281c"
TEXT     = "#c4dcc4"
DIMTEXT  = "#a0b8a0"
BRIGHT   = "#e4f4e4"
FONT     = "'Courier New', Courier, monospace"

# (node_bg, node_border, node_text)
HABITAT_COLORS: dict[str, tuple[str, str, str]] = {
    "Forest":     ("#1a3c1a", "#56c456", "#88ee88"),
    "Plains":     ("#3a2c0a", "#cc9830", "#f0c860"),
    "Tundra":     ("#0e2438", "#2ea8e0", "#60ccff"),
    "Alpine":     ("#241840", "#7850d8", "#b090ff"),
    "Rainforest": ("#0c3020", "#20c878", "#50f0a8"),
    "Desert":     ("#3c1c06", "#d86020", "#ff8840"),
}

ALL_TRAITS: list[str] = [
    "fecundity", "reproduction_time", "days_to_sexual_viability",
    "parental_investment", "reproduction_likelihood", "metabolism",
    "water_efficiency", "max_lifespan", "disease_resistance", "immune_response",
    "stress_tolerance", "heat_tolerance", "cold_tolerance", "drought_tolerance",
    "hibernation_tendency", "migration_likelihood", "risk_tolerance", "aggression",
    "territorial", "social_tendency", "nocturnal_tendency", "size", "strength",
    "speed", "camouflage", "foraging_ability", "intelligence", "adaptability",
    "pack_hunting", "scavenging_tendency", "communication", "mutation_rate",
    "selectivity", "base_predation_rate",
]

QUICK_TRAITS: list[str] = [
    "fecundity", "metabolism", "size", "speed", "intelligence",
    "mutation_rate", "aggression", "disease_resistance",
    "migration_likelihood", "water_efficiency",
]

KEY_TRAITS: list[str] = [
    "fecundity", "metabolism", "size", "speed", "intelligence",
    "mutation_rate", "disease_resistance", "water_efficiency",
    "migration_likelihood", "max_lifespan",
]
