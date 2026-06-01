"""
goods_dictionary.py
-------------------
Comprehensive goods synonym dictionary mapping UK/US variants to canonical forms.
Covers all goods from the U.S. Department of Labor TVPRA list linked to child labour.

HACE: Data Changing Child Labour
MSc Data Science - University of Manchester (2024-2025)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL GOODS DICTIONARY
# Key   = canonical UK-English form
# Value = list of all known synonyms and alternate spellings
# ─────────────────────────────────────────────────────────────────────────────

GOODS_DICTIONARY = {
    # Mining & Metals
    "gold": ["gold", "golden", "au metal", "gold ore", "gold mining", "artisanal gold"],
    "silver": ["silver", "silver ore", "ag metal", "silver mining"],
    "cobalt": ["cobalt", "cobalt ore", "cobalt mining", "cobalt metal"],
    "tin": ["tin", "tin ore", "cassiterite", "tin mining", "tinplate"],
    "tungsten": ["tungsten", "wolframite", "tungsten ore"],
    "tantalum": ["tantalum", "coltan", "columbite-tantalite"],
    "diamonds": ["diamond", "diamonds", "rough diamond", "conflict diamond", "blood diamond"],
    "iron": ["iron ore", "iron mining", "ferrous ore", "pig iron"],
    "nickel": ["nickel", "nickel ore", "nickel mining"],
    "coal": ["coal", "coal mining", "coking coal", "thermal coal"],
    "copper": ["copper", "copper ore", "copper mining"],
    "mica": ["mica", "mica mining", "muscovite", "phlogopite"],

    # Agricultural Products
    "sugarcane": ["sugarcane", "sugar cane", "cane sugar", "sugar production"],
    "cotton": ["cotton", "raw cotton", "cotton farming", "cotton picking", "cotton gin"],
    "cocoa": ["cocoa", "cacao", "cocoa bean", "cocoa farming", "cocoa pod"],
    "coffee": ["coffee", "coffee bean", "coffee farming", "coffee cherry"],
    "tobacco": ["tobacco", "tobacco leaf", "tobacco farming", "tobacco plant"],
    "rice": ["rice", "rice paddy", "rice farming", "paddy field"],
    "wheat": ["wheat", "wheat farming", "wheat harvest"],
    "maize": ["maize", "corn", "maize farming", "corn farming"],
    "rubber": ["rubber", "natural rubber", "rubber tree", "rubber tapping", "latex"],
    "timber": ["timber", "wood", "logging", "deforestation", "lumber", "hardwood", "softwood"],
    "palm oil": ["palm oil", "palm fruit", "oil palm", "crude palm oil", "palm kernel oil"],
    "bananas": ["banana", "bananas", "banana farming", "banana plantation"],
    "tea": ["tea", "tea leaf", "tea farming", "tea plantation", "tea picking"],
    "bricks": ["brick", "bricks", "brick kiln", "brick making", "clay brick"],
    "garments": ["garment", "garments", "clothing", "apparel", "textile", "ready-made garments", "RMG"],
    "textiles": ["textile", "textiles", "fabric", "woven fabric", "knitted fabric"],
    "fireworks": ["fireworks", "firecracker", "pyrotechnics"],
    "pornography": ["pornography", "child pornography"],
    "salt": ["salt", "salt mining", "salt production", "salt harvesting"],
    "fish": ["fish", "fishing", "fish processing", "seafood", "aquaculture"],
    "cattle": ["cattle", "beef", "livestock", "cattle ranching"],
    "sesame": ["sesame", "sesame seed", "sesame farming"],
    "vanilla": ["vanilla", "vanilla bean", "vanilla farming"],
    "cashews": ["cashew", "cashews", "cashew nut", "cashew farming"],

    # Downstream / Manufactured Goods
    "electronics": ["electronics", "electronic component", "circuit board", "PCB"],
    "smartphones": ["smartphone", "mobile phone", "cell phone", "handset"],
    "computers": ["computer", "laptop", "desktop computer", "PC"],
    "batteries": ["battery", "batteries", "lithium battery", "rechargeable battery"],
    "electric vehicles": ["electric vehicle", "EV", "electric car", "e-vehicle"],
    "solar panels": ["solar panel", "photovoltaic", "PV panel", "solar module"],
    "jewellery": ["jewellery", "jewelry", "gold jewellery", "diamond ring", "gem"],
    "chocolate": ["chocolate", "dark chocolate", "cocoa product", "chocolate bar"],
    "cooking oil": ["cooking oil", "palm cooking oil", "vegetable oil", "edible oil"],
    "cosmetics": ["cosmetic", "cosmetics", "beauty product", "personal care", "skincare"],
    "tyres": ["tyre", "tire", "rubber tyre", "vehicle tyre"],
    "gloves": ["glove", "gloves", "rubber glove", "latex glove"],
    "footwear": ["footwear", "shoe", "shoes", "boot", "sandal"],
}

# Ambiguous goods that require contextual classification
AMBIGUOUS_GOODS = [
    "gold", "silver", "rubber", "timber", "tin", "nickel", "diamonds", "iron"
    # Note: "lead" deliberately excluded — overwhelmingly used as verb
]


def get_canonical_form(synonym: str) -> str:
    """
    Given a synonym, return its canonical UK-English form.

    Args:
        synonym: Any known synonym string

    Returns:
        Canonical form string, or None if not found
    """
    synonym_lower = synonym.lower().strip()
    for canonical, synonyms in GOODS_DICTIONARY.items():
        if synonym_lower in [s.lower() for s in synonyms]:
            return canonical
    return None


def get_all_synonyms() -> list:
    """
    Return flat list of all synonyms across all goods.

    Returns:
        List of all synonym strings
    """
    all_syns = []
    for synonyms in GOODS_DICTIONARY.values():
        all_syns.extend(synonyms)
    return all_syns


def get_reverse_mapping() -> dict:
    """
    Build reverse mapping: synonym → canonical form.

    Returns:
        Dict mapping each synonym to its canonical form
    """
    reverse = {}
    for canonical, synonyms in GOODS_DICTIONARY.items():
        for syn in synonyms:
            reverse[syn.lower()] = canonical
    return reverse


def is_ambiguous(canonical: str) -> bool:
    """Check if a canonical good is in the ambiguous goods list."""
    return canonical in AMBIGUOUS_GOODS


if __name__ == "__main__":
    print(f"Total canonical goods: {len(GOODS_DICTIONARY)}")
    print(f"Total synonyms: {len(get_all_synonyms())}")
    print(f"Ambiguous goods: {AMBIGUOUS_GOODS}")
    print(f"\nExample lookup 'cacao' → {get_canonical_form('cacao')}")
    print(f"Example lookup 'eggplant' → {get_canonical_form('eggplant')}")
