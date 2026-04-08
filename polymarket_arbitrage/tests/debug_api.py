import httpx
import asyncio
import json
from datetime import datetime, timedelta

async def debug_polymarket():
    base_url = "https://gamma-api.polymarket.com"
    asset_name = "Bitcoin"
    target_date = datetime.now() + timedelta(days=2) # April 5
    date_str = target_date.strftime("%B %d").replace(" 0", " ")
    q = f"{asset_name} above on {date_str}"
    
    params = {"active": "true", "closed": "false", "query": q, "limit": 5}
    print(f"[*] Querying: {q}")
    print(f"[*] URL: {base_url}/markets")
    print(f"[*] Params: {params}")
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{base_url}/markets", params=params)
        print(f"[*] Status: {response.status_code}")
        data = response.json()
        print(f"[*] Found {len(data)} items")
        
        for i, market in enumerate(data):
            print(f"\n--- Market {i+1} ---")
            print(f"Title: {market.get('question') or market.get('title')}")
            print(f"Slug: {market.get('slug')}")
            print(f"CLOB IDs: {market.get('clobTokenIds')}")
            
if __name__ == "__main__":
    asyncio.run(debug_polymarket())
