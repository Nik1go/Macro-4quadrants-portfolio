import re
from typing import Optional

def is_crypto_price_market(title: str, slug: str = "") -> bool:
    """Strictly filter for crypto price markets: [Asset] above [Strike] on [Date]."""
    if not isinstance(title, str) or not isinstance(slug, str):
        return False
        
    t = title.lower()
    s = slug.lower()
    
    # 1. Asset check (exact whitelist)
    allowed_assets = ["bitcoin", "ethereum", "xrp", "btc", "eth"]
    if not any(a in t for a in allowed_assets) and not any(a in s for a in allowed_assets):
        return False

    # 2. Pattern check
    has_keywords = "above" in s or "above" in t
    has_date_indicator = "on" in s or "on" in t
    
    # Noise rejection
    noise = ["gta", "movie", "album", "ceasefire", "trump", "election", "war", "announced", "trailer"]
    if any(n in s for n in noise) or any(n in t for n in noise):
        return False

    # 3. Final validation
    match_regex = re.search(r"above|below|hit|reach", s + " " + t)
    return bool(match_regex and has_date_indicator)

test_cases = [
    ("Will Bitcoin be above $70,000 on April 05?", "bitcoin-above-70000-on-april-05", True),
    ("Will Ethereum hit $4000 on March 30?", "ethereum-hit-4000-on-march-30", True),
    ("Will XRP reach $1.00 on April 10?", "xrp-reach-1-on-april-10", True),
    ("Will Bitcoin hit 1M before GTA VI", "will-bitcoin-hit-1m-before-gta-vi-872", False),
    ("Russia-Ukraine ceasefire before GTA VI", "russia-ukraine-ceasefire-before-gta-vi-554", False),
    ("New Rihanna album before GTA VI", "new-rhianna-album-before-gta-vi-926", False),
    ("Will Trump win the election?", "trump-win-election-2024", False)
]

print("--- Testing Filter Logic ---")
for title, slug, expected in test_cases:
    result = is_crypto_price_market(title, slug)
    status = "PASS" if result == expected else "FAIL"
    print(f"[{status}] Title: {title[:40]}... -> Result: {result}")
