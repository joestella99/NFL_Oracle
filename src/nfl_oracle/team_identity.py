"""Canonical team-history keys across franchise relocations.

Apply only to historical team features and PBP joins. Preserve original
schedule abbreviations, odds matching, game IDs and displayed team names.
"""

ALIASES = {
    "STL": "LA",    # Rams, relocated 2016
    "SD": "LAC",    # Chargers, relocated 2017
    "OAK": "LV",    # Raiders, relocated 2020
    "WSH": "WAS",   # Washington abbreviation variants
    "JAC": "JAX",   # Jacksonville abbreviation variants
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
}


def franchise_key(code: str) -> str:
    """Return a stable team history key without conflating LA and LAC."""
    label = str(code).strip().upper()
    return ALIASES.get(label, label)
